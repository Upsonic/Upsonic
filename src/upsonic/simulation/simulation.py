"""
Main Simulation orchestrator class.

This module provides the core Simulation class that manages the execution
of LLM-powered time-series simulations.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from upsonic.simulation.base import BaseSimulationObject, SimulationConfig, SimulationStepOutput
from upsonic.simulation.time_step import TimeStep, TimeStepManager
from upsonic.simulation.result import SimulationResult, SimulationStepRecord


class Simulation:
    """
    LLM-powered simulation orchestrator.
    
    This class manages the execution of time-series simulations by:
    - Iterating through time steps
    - Building prompts with previous state context
    - Calling the LLM to predict next values
    - Tracking metrics throughout the simulation
    - Generating comprehensive results and reports
    
    Example:
        ```python
        from upsonic.simulation import Simulation
        from upsonic.simulation.scenarios import MerchantRevenueForecastSimulation
        
        simulation = Simulation(
            MerchantRevenueForecastSimulation(
                merchant_name="TechCo",
                shareholders=["Alice", "Bob"],
                sector="E-commerce",
                location="San Francisco",
                current_monthly_revenue_usd=50000
            ),
            model="openai/gpt-4o",
            time_step="daily",
            simulation_duration=100,
            metrics_to_track=["monthly recurring revenue"]
        )
        
        # Run the simulation
        result = simulation.run()
        
        # Access reports
        result.report("summary").to_pdf("summary.pdf")
        result.report("visual").show()
        ```
    """
    
    def __init__(
        self,
        simulation_object: BaseSimulationObject,
        model: str = "openai/gpt-4o",
        time_step: str = "daily",
        simulation_duration: int = 100,
        metrics_to_track: Optional[List[str]] = None,
        temperature: float = 0.7,
        retry_on_error: bool = True,
        max_retries: int = 3,
        show_progress: bool = True,
        start_date: Optional[datetime] = None
    ):
        """
        Initialize the simulation.
        
        Args:
            simulation_object: The simulation scenario object
            model: LLM model to use (e.g., "openai/gpt-4o")
            time_step: Time unit for each step ("daily", "weekly", "monthly", etc.)
            simulation_duration: Number of steps to simulate
            metrics_to_track: List of metric names to track
            temperature: LLM temperature (0.0-2.0)
            retry_on_error: Whether to retry failed LLM calls
            max_retries: Maximum retries per step
            show_progress: Whether to display progress
            start_date: Simulation start date (defaults to now)
        """
        self._simulation_object = simulation_object
        self._model_name = model
        self._time_step = TimeStep.from_string(time_step)
        self._simulation_duration = simulation_duration
        self._metrics_to_track = metrics_to_track or []
        self._temperature = temperature
        self._retry_on_error = retry_on_error
        self._max_retries = max_retries
        self._show_progress = show_progress
        
        # Initialize time step manager
        self._time_manager = TimeStepManager(
            time_step=self._time_step,
            start_date=start_date or datetime.now()
        )
        
        # Internal state
        self._simulation_id: str = str(uuid.uuid4())
        self._model: Any = None
        self._direct: Any = None
        self._is_running: bool = False
        self._steps: List[SimulationStepRecord] = []
        self._current_state: Dict[str, Any] = {}
        
        # Validate configuration
        self._config = SimulationConfig(
            simulation_object=simulation_object,
            model=model,
            time_step=time_step,
            simulation_duration=simulation_duration,
            metrics_to_track=self._metrics_to_track,
            temperature=temperature,
            retry_on_error=retry_on_error,
            max_retries=max_retries,
            show_progress=show_progress
        )
    
    @property
    def simulation_id(self) -> str:
        """Get the unique simulation ID."""
        return self._simulation_id
    
    @property
    def simulation_object(self) -> BaseSimulationObject:
        """Get the simulation scenario object."""
        return self._simulation_object
    
    @property
    def duration(self) -> int:
        """Get the simulation duration in steps."""
        return self._simulation_duration
    
    @property
    def time_step(self) -> TimeStep:
        """Get the time step granularity."""
        return self._time_step
    
    @property
    def metrics_to_track(self) -> List[str]:
        """Get the list of metrics being tracked."""
        return self._metrics_to_track
    
    @property
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self._is_running
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model for the simulation."""
        from upsonic.direct import Direct
        
        self._direct = Direct(model=self._model_name)
        self._model = self._direct._prepare_model()
    
    def _build_step_prompt(
        self,
        step: int,
        previous_state: Dict[str, Any]
    ) -> str:
        """
        Build the prompt for a simulation step.
        
        Args:
            step: Current step number
            previous_state: State from the previous step
            
        Returns:
            str: The complete prompt for the LLM
        """
        # Get base prompt from simulation object
        base_prompt = self._simulation_object.build_step_prompt(
            step=step,
            previous_state=previous_state,
            metrics_to_track=self._metrics_to_track,
            time_step_unit=self._time_step.singular_unit
        )
        
        # Get time context
        time_context = self._time_manager.get_time_context(step)
        
        # Get additional context from simulation object
        additional_context = self._simulation_object.get_context_for_step(step)
        
        # Build complete prompt
        prompt_parts = [
            base_prompt,
            f"\nTime Context: {time_context['timestamp']} ({time_context['day_of_week']})",
        ]
        
        if additional_context:
            prompt_parts.append(f"\nAdditional Context: {additional_context}")
        
        prompt_parts.append(
            f"\n\nPlease predict the following metrics for step {step}: "
            f"{', '.join(self._metrics_to_track)}"
        )
        
        return "\n".join(prompt_parts)
    
    async def _execute_step(
        self,
        step: int,
        previous_state: Dict[str, Any]
    ) -> SimulationStepRecord:
        """
        Execute a single simulation step.
        
        Args:
            step: Current step number
            previous_state: State from the previous step
            
        Returns:
            SimulationStepRecord: The record for this step
        """
        from upsonic.tasks.tasks import Task
        
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_step_prompt(step, previous_state)
        
        # Get output schema from simulation object
        output_schema = self._simulation_object.get_step_output_schema()
        
        # Create task with structured output
        task = Task(
            description=prompt,
            response_format=output_schema
        )
        
        # Execute with retries
        attempt = 0
        last_error: Optional[Exception] = None
        result: Optional[BaseModel] = None
        
        while attempt <= self._max_retries:
            try:
                result = await self._direct.do_async(task)
                break
            except Exception as e:
                last_error = e
                attempt += 1
                if not self._retry_on_error or attempt > self._max_retries:
                    raise
                await asyncio.sleep(0.5 * attempt)  # Exponential backoff
        
        end_time = time.time()
        
        if result is None:
            raise RuntimeError(
                f"Failed to execute step {step} after {self._max_retries} retries: {last_error}"
            )
        
        # Extract metrics from result
        metrics = self._simulation_object.extract_metrics(result, self._metrics_to_track)
        metrics = self._simulation_object.validate_metrics(metrics, step)
        
        # Create step record
        record = SimulationStepRecord(
            step=step,
            timestamp=self._time_manager.format_timestamp(step),
            prompt=prompt,
            raw_response=result.model_dump_json() if hasattr(result, 'model_dump_json') else str(result),
            parsed_response=result,
            metrics=metrics,
            execution_time=end_time - start_time,
            success=True,
            error=None
        )
        
        return record
    
    async def _run_simulation_async(self) -> SimulationResult:
        """
        Execute the simulation asynchronously.
        
        Returns:
            SimulationResult: The complete simulation result
        """
        self._is_running = True
        simulation_start_time = time.time()
        
        try:
            # Initialize model
            self._initialize_model()
            
            # Get initial state
            self._current_state = self._simulation_object.get_initial_state()
            self._steps = []
            
            # Create initial step record (step 0)
            initial_record = SimulationStepRecord(
                step=0,
                timestamp=self._time_manager.format_timestamp(0),
                prompt="",
                raw_response="",
                parsed_response=None,
                metrics=self._current_state.copy(),
                execution_time=0.0,
                success=True,
                error=None
            )
            self._steps.append(initial_record)
            
            # Run simulation steps
            if self._show_progress:
                self._print_progress_start()
            
            for step in range(1, self._simulation_duration + 1):
                if self._show_progress:
                    self._print_progress_step(step)
                
                try:
                    # Execute step
                    record = await self._execute_step(step, self._current_state)
                    self._steps.append(record)
                    
                    # Update current state with new metrics
                    self._current_state.update(record.metrics)
                    
                except Exception as e:
                    # Record failed step
                    error_record = SimulationStepRecord(
                        step=step,
                        timestamp=self._time_manager.format_timestamp(step),
                        prompt=self._build_step_prompt(step, self._current_state),
                        raw_response="",
                        parsed_response=None,
                        metrics={},
                        execution_time=0.0,
                        success=False,
                        error=str(e)
                    )
                    self._steps.append(error_record)
                    
                    if not self._retry_on_error:
                        raise
            
            if self._show_progress:
                self._print_progress_complete()
            
            simulation_end_time = time.time()
            
            # Create result
            result = SimulationResult(
                simulation_id=self._simulation_id,
                simulation_object=self._simulation_object,
                config=self._config,
                steps=self._steps,
                start_time=simulation_start_time,
                end_time=simulation_end_time,
                time_manager=self._time_manager,
                metrics_to_track=self._metrics_to_track
            )
            
            return result
            
        finally:
            self._is_running = False
    
    def run(self) -> SimulationResult:
        """
        Execute the simulation synchronously.
        
        Returns:
            SimulationResult: The complete simulation result
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._run_simulation_async())
                return future.result()
        except RuntimeError:
            return asyncio.run(self._run_simulation_async())
    
    async def arun(self) -> SimulationResult:
        """
        Execute the simulation asynchronously.
        
        Returns:
            SimulationResult: The complete simulation result
        """
        return await self._run_simulation_async()
    
    def _print_progress_start(self) -> None:
        """Print simulation start message."""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        console.print(Panel(
            f"[bold cyan]Starting Simulation[/bold cyan]\n\n"
            f"[yellow]Scenario:[/yellow] {self._simulation_object.name}\n"
            f"[yellow]Description:[/yellow] {self._simulation_object.description}\n"
            f"[yellow]Duration:[/yellow] {self._simulation_duration} {self._time_step.plural_unit}\n"
            f"[yellow]Metrics:[/yellow] {', '.join(self._metrics_to_track)}\n"
            f"[yellow]Model:[/yellow] {self._model_name}",
            title="🔮 Simulation",
            border_style="cyan"
        ))
    
    def _print_progress_step(self, step: int) -> None:
        """Print step progress."""
        from rich.console import Console
        
        console = Console()
        progress = (step / self._simulation_duration) * 100
        console.print(
            f"  [dim]Step {step}/{self._simulation_duration}[/dim] "
            f"[cyan]({self._time_manager.format_timestamp(step)})[/cyan] "
            f"[green]{progress:.1f}%[/green]",
            end="\r"
        )
    
    def _print_progress_complete(self) -> None:
        """Print simulation completion message."""
        from rich.console import Console
        
        console = Console()
        console.print(
            f"\n  [bold green]✓ Simulation complete![/bold green] "
            f"({self._simulation_duration} steps)"
        )

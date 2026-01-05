"""
Tests for the Agent Pipeline Architecture
"""

import pytest
from upsonic.agent.pipeline import (
    Step, StepResult, StepStatus,
    PipelineManager,
    InitializationStep, CacheCheckStep, UserPolicyStep, ModelSelectionStep,
    ValidationStep, FinalizationStep
)
from upsonic.run.agent.context import AgentRunContext


class MockTask:
    """Mock task for testing."""
    def __init__(self, description="Test task", enable_cache=False):
        self.description = description
        self.enable_cache = enable_cache
        self.is_paused = False
        self._response = None
        self.response = None
        self.not_main_task = False
        self.price_id = "test-123"
        self._original_input = None
        self.cache_method = "simple"
        self.cache_threshold = 0.9
        self.cache_duration_minutes = 60
        self.cache_embedding_provider = None
        self.attachments = []  # Add attachments attribute

    def task_start(self, agent):
        """Mock task_start method."""
        pass

    def task_end(self):
        pass

    def set_cache_manager(self, manager):
        pass

    async def get_cached_response(self, input_text, model):
        return None


class MockModel:
    """Mock model for testing."""
    def __init__(self, name="test-model"):
        self.model_name = name


class MockAgent:
    """Mock agent for testing."""
    def __init__(self):
        from upsonic.run.agent.output import AgentRunOutput
        from upsonic.run.base import RunStatus
        
        self.debug = False
        self.user_policy = None
        self.agent_policy = None
        self.reflection_processor = None
        self.reflection = False
        self.model = MockModel()
        self._cache_manager = None
        self.tool_call_count = 0
        self._tool_call_count = 0
        self._tool_limit_reached = False
        self.run_id = "test-run-id"
        self.agent_id = "test-agent"
        self.name = "test-agent"
        self.session_id = "test-session"
        self.user_id = None
        self._agent_run_output = AgentRunOutput(
            run_id=self.run_id,
            agent_id=self.agent_id,
            agent_name=self.name,
            session_id=self.session_id,
            status=RunStatus.running
        )
        self._agent_run_context = None
        self._run_result = type('obj', (object,), {
            'start_new_run': lambda self: None,
            'output': None
        })()
        
    def get_agent_id(self):
        return "test-agent"
        
    def _setup_tools(self, task):
        pass
    
    async def _build_model_request(self, task, memory_handler, state):
        return []
    
    def _build_model_request_parameters(self, task):
        return None
    
    async def _execute_with_guardrail(self, task, memory_handler, state):
        return None
    
    async def _handle_model_response(self, response, messages):
        return response
    
    def _extract_output(self, response, task):
        return "Test output"
    
    async def _apply_agent_policy(self, task):
        return task


# ============================================================================
# Test Step Base Class
# ============================================================================

class TestStep:
    """Test the base Step class."""
    
    def test_step_interface(self):
        """Test that Step is an abstract base class."""
        with pytest.raises(TypeError):
            Step()
    
    @pytest.mark.asyncio
    async def test_custom_step(self):
        """Test creating a custom step."""
        class CustomStep(Step):
            @property
            def name(self) -> str:
                return "custom"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                return StepResult(
                    status=StepStatus.COMPLETED,
                    execution_time=0.0
                )
        
        step = CustomStep()
        assert step.name == "custom"
        
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        result = await step.run(context, task, agent, model, step_number=0)
        
        assert result.status == StepStatus.COMPLETED
        assert result.execution_time >= 0.0  # Time should be set
    
    @pytest.mark.asyncio
    async def test_step_error_handling(self):
        """Test step error handling."""
        class ErrorStep(Step):
            @property
            def name(self) -> str:
                return "error"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                raise ValueError("Test error")
        
        step = ErrorStep()
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        with pytest.raises(ValueError):
            await step.run(context, task, agent, model, step_number=0)


# ============================================================================
# Test AgentRunContext
# ============================================================================

class TestAgentRunContext:
    """Test the AgentRunContext model."""
    
    def test_context_creation(self):
        """Test creating a context."""
        task = MockTask()
        context = AgentRunContext(
            run_id="test-run",
            session_id="test-session",
            task=task
        )
        
        assert context.task is not None
        assert context.run_id == "test-run"
        assert context.session_id == "test-session"
        assert len(context.messages) == 0


# ============================================================================
# Test PipelineManager
# ============================================================================

class TestPipelineManager:
    """Test the PipelineManager."""
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Test basic pipeline execution."""
        class Step1(Step):
            @property
            def name(self) -> str:
                return "step1"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                return StepResult(
                    status=StepStatus.COMPLETED,
                    execution_time=0.0
                )
        
        class Step2(Step):
            @property
            def name(self) -> str:
                return "step2"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                return StepResult(
                    status=StepStatus.COMPLETED,
                    execution_time=0.0
                )
        
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        pipeline = PipelineManager(steps=[Step1(), Step2()], task=task, agent=agent, model=model)
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        result = await pipeline.execute(context)
        
        # Both steps should have executed
        stats = pipeline.get_execution_stats()
        assert stats['executed_steps'] == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self):
        """Test pipeline stops on error and propagates it."""
        class Step1(Step):
            @property
            def name(self) -> str:
                return "step1"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                raise ValueError("Step 1 error")
        
        class Step2(Step):
            @property
            def name(self) -> str:
                return "step2"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                return StepResult(status=StepStatus.COMPLETED, execution_time=0.0)
        
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        pipeline = PipelineManager(steps=[Step1(), Step2()], task=task, agent=agent, model=model)
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        with pytest.raises(ValueError):
            await pipeline.execute(context)
        
        # Only step1 should have been attempted
        stats = pipeline.get_execution_stats()
        assert stats['executed_steps'] == 0  # Error occurred before completion
    
    def test_pipeline_step_management(self):
        """Test adding/removing steps."""
        class DummyStep(Step):
            def __init__(self, step_name):
                self.step_name = step_name
            
            @property
            def name(self) -> str:
                return self.step_name
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                return StepResult(status=StepStatus.COMPLETED, execution_time=0.0)
        
        pipeline = PipelineManager()
        
        # Add steps
        pipeline.add_step(DummyStep("step1"))
        pipeline.add_step(DummyStep("step2"))
        assert len(pipeline.steps) == 2
        
        # Insert step
        pipeline.insert_step(1, DummyStep("step_middle"))
        assert len(pipeline.steps) == 3
        assert pipeline.steps[1].name == "step_middle"
        
        # Remove step
        removed = pipeline.remove_step("step_middle")
        assert removed is True
        assert len(pipeline.steps) == 2
        
        # Get step
        step = pipeline.get_step("step1")
        assert step is not None
        assert step.name == "step1"
    
    @pytest.mark.asyncio
    async def test_pipeline_statistics(self):
        """Test pipeline execution statistics."""
        class SuccessStep(Step):
            @property
            def name(self) -> str:
                return "success"
            
            async def execute(self, context: AgentRunContext, task, agent, model) -> StepResult:
                return StepResult(status=StepStatus.COMPLETED, execution_time=0.0)
        
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        pipeline = PipelineManager(steps=[SuccessStep()], task=task, agent=agent, model=model)
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        await pipeline.execute(context)
        
        stats = pipeline.get_execution_stats()
        assert stats['total_steps'] == 1
        assert stats['executed_steps'] == 1
        assert 'success' in stats['step_results']
        assert stats['step_results']['success']['execution_time'] >= 0.0


# ============================================================================
# Test Built-in Steps
# ============================================================================

class TestBuiltinSteps:
    """Test the built-in pipeline steps."""
    
    @pytest.mark.asyncio
    async def test_initialization_step(self):
        """Test initialization step."""
        step = InitializationStep()
        agent = MockAgent()
        task = MockTask()
        model = MockModel()
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        result = await step.run(context, task, agent, model, step_number=0)
        
        assert result.status == StepStatus.COMPLETED
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_model_selection_step(self):
        """Test model selection step."""
        step = ModelSelectionStep()
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        pipeline = PipelineManager(steps=[step], task=task, agent=agent, model=model)
        result = await step.run(context, task, agent, model, step_number=0)
        
        assert result.status == StepStatus.COMPLETED
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_validation_step(self):
        """Test validation step."""
        step = ValidationStep()
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        
        result = await step.run(context, task, agent, model, step_number=0)
        
        # Should succeed for valid task
        assert result.status == StepStatus.COMPLETED
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_finalization_step(self):
        """Test finalization step."""
        step = FinalizationStep()
        task = MockTask()
        task.response = "Test response"
        agent = MockAgent()
        model = MockModel()
        
        context = AgentRunContext(run_id="test-run", session_id="test-session", task=task)
        context.final_output = "Test output"
        
        result = await step.run(context, task, agent, model, step_number=0)
        
        assert result.status == StepStatus.COMPLETED
        assert result.execution_time >= 0.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test a complete pipeline execution."""
        task = MockTask()
        agent = MockAgent()
        model = MockModel()
        pipeline = PipelineManager(
            steps=[
                InitializationStep(),
                ModelSelectionStep(),
                ValidationStep(),
                FinalizationStep(),
            ],
            task=task,
            agent=agent,
            model=model
        )
        
        context = AgentRunContext(
            run_id="test-run",
            session_id="test-session",
            task=task
        )
        
        result = await pipeline.execute(context)
        
        # Should complete successfully
        assert result is not None
        
        # Should have execution statistics
        stats = pipeline.get_execution_stats()
        assert stats['total_steps'] == 4
        assert stats['executed_steps'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

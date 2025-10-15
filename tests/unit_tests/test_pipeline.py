"""
Tests for the Agent Pipeline Architecture
"""

import pytest
from upsonic.agent.pipeline import (
    Step, StepResult, StepStatus, StepContext,
    PipelineManager,
    InitializationStep, CacheCheckStep, UserPolicyStep, ModelSelectionStep,
    ValidationStep, FinalizationStep
)


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
        self.debug = False
        self.user_policy = None
        self.agent_policy = None
        self.reflection_processor = None
        self.reflection = False
        self.model = MockModel()
        self._cache_manager = None
        self.tool_call_count = 0
        self._tool_call_count = 0
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
            
            async def execute(self, context: StepContext) -> StepResult:
                return StepResult(
                    status=StepStatus.SUCCESS,
                    execution_time=0.0
                )
        
        step = CustomStep()
        assert step.name == "custom"
        
        context = StepContext(task=MockTask(), agent=MockAgent())
        result = await step.run(context)
        
        assert result.status == StepStatus.SUCCESS
        assert result.execution_time >= 0.0  # Time should be set
    
    @pytest.mark.asyncio
    async def test_step_error_handling(self):
        """Test step error handling."""
        class ErrorStep(Step):
            @property
            def name(self) -> str:
                return "error"
            
            async def execute(self, context: StepContext) -> StepResult:
                raise ValueError("Test error")
        
        step = ErrorStep()
        context = StepContext(task=MockTask(), agent=MockAgent())
        
        with pytest.raises(ValueError):
            await step.run(context)


# ============================================================================
# Test StepContext
# ============================================================================

class TestStepContext:
    """Test the StepContext model."""
    
    def test_context_creation(self):
        """Test creating a context."""
        context = StepContext(
            task=MockTask(),
            agent=MockAgent()
        )
        
        assert context.task is not None
        assert context.agent is not None
        assert context.model is None  # Optional
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
            
            async def execute(self, context: StepContext) -> StepResult:
                return StepResult(
                    status=StepStatus.SUCCESS,
                    execution_time=0.0
                )
        
        class Step2(Step):
            @property
            def name(self) -> str:
                return "step2"
            
            async def execute(self, context: StepContext) -> StepResult:
                return StepResult(
                    status=StepStatus.SUCCESS,
                    execution_time=0.0
                )
        
        pipeline = PipelineManager(steps=[Step1(), Step2()])
        context = StepContext(task=MockTask(), agent=MockAgent())
        
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
            
            async def execute(self, context: StepContext) -> StepResult:
                raise ValueError("Step 1 error")
        
        class Step2(Step):
            @property
            def name(self) -> str:
                return "step2"
            
            async def execute(self, context: StepContext) -> StepResult:
                return StepResult(status=StepStatus.SUCCESS, execution_time=0.0)
        
        pipeline = PipelineManager(steps=[Step1(), Step2()])
        context = StepContext(task=MockTask(), agent=MockAgent())
        
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
            
            async def execute(self, context: StepContext) -> StepResult:
                return StepResult(status=StepStatus.SUCCESS, execution_time=0.0)
        
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
            
            async def execute(self, context: StepContext) -> StepResult:
                return StepResult(status=StepStatus.SUCCESS, execution_time=0.0)
        
        pipeline = PipelineManager(steps=[SuccessStep()])
        context = StepContext(task=MockTask(), agent=MockAgent())
        
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
        context = StepContext(task=task, agent=agent)
        
        result = await step.run(context)
        
        assert result.status == StepStatus.SUCCESS
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_model_selection_step(self):
        """Test model selection step."""
        step = ModelSelectionStep()
        context = StepContext(
            task=MockTask(),
            agent=MockAgent(),
            model=None
        )
        
        result = await step.run(context)
        
        assert result.status == StepStatus.SUCCESS
        assert context.model is not None
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_validation_step(self):
        """Test validation step."""
        step = ValidationStep()
        context = StepContext(
            task=MockTask(),
            agent=MockAgent()
        )
        
        result = await step.run(context)
        
        # Should succeed for valid task
        assert result.status == StepStatus.SUCCESS
        assert result.execution_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_finalization_step(self):
        """Test finalization step."""
        step = FinalizationStep()
        task = MockTask()
        task.response = "Test response"
        agent = MockAgent()
        
        context = StepContext(
            task=task,
            agent=agent
        )
        context.final_output = "Test output"
        
        result = await step.run(context)
        
        assert result.status == StepStatus.SUCCESS
        assert result.execution_time >= 0.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test a complete pipeline execution."""
        pipeline = PipelineManager(
            steps=[
                InitializationStep(),
                ModelSelectionStep(),
                ValidationStep(),
                FinalizationStep(),
            ]
        )
        
        context = StepContext(
            task=MockTask(),
            agent=MockAgent()
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

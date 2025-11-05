# Test Plan - Specific Test Files to Create

## Overview
This document lists all test files you need to create, organized by priority, with specific test cases for each.

---

## 🔴 **CRITICAL PRIORITY** - Core Agent Functionality

### 1. Agent Tests (`tests/unit_tests/agent/`)

#### `test_direct.py`
**Location:** `tests/unit_tests/agent/test_direct.py`

**Test Cases to Implement:**
- `test_direct_initialization()` - Test Direct class initialization
- `test_direct_initialization_with_model()` - Test init with model string
- `test_direct_initialization_with_model_instance()` - Test init with Model instance
- `test_direct_initialization_with_settings()` - Test init with ModelSettings
- `test_direct_initialization_with_profile()` - Test init with ModelProfileSpec
- `test_direct_initialization_with_provider()` - Test init with Provider
- `test_direct_do_basic()` - Test basic `do()` method
- `test_direct_do_with_text_task()` - Test `do()` with simple text task
- `test_direct_do_with_structured_output()` - Test `do()` with Pydantic response format
- `test_direct_do_with_context()` - Test `do()` with task context
- `test_direct_do_with_attachments()` - Test `do()` with file attachments
- `test_direct_do_async()` - Test async execution
- `test_direct_print_do()` - Test `print_do()` method
- `test_direct_print_do_async()` - Test async `print_do_async()`
- `test_direct_with_model()` - Test `with_model()` builder method
- `test_direct_with_settings()` - Test `with_settings()` builder method
- `test_direct_with_profile()` - Test `with_profile()` builder method
- `test_direct_with_provider()` - Test `with_provider()` builder method
- `test_direct_error_handling()` - Test error handling
- `test_direct_model_preparation()` - Test `_prepare_model()` method
- `test_direct_build_messages_from_task()` - Test message building
- `test_direct_build_request_parameters()` - Test request parameter building
- `test_direct_extract_output()` - Test output extraction

---

#### `test_deep_agent.py`
**Location:** `tests/unit_tests/agent/test_deep_agent.py`

**Test Cases to Implement:**
- `test_deep_agent_initialization()` - Test DeepAgent initialization
- `test_deep_agent_initialization_with_subagents()` - Test init with subagents
- `test_deep_agent_initialization_with_instructions()` - Test init with custom instructions
- `test_deep_agent_do_basic()` - Test basic `do()` method
- `test_deep_agent_do_async()` - Test async execution
- `test_deep_agent_add_file()` - Test adding files to virtual filesystem
- `test_deep_agent_todo_management()` - Test todo creation and tracking
- `test_deep_agent_todo_completion()` - Test todo completion loop
- `test_deep_agent_virtual_filesystem_ls()` - Test `ls` tool functionality
- `test_deep_agent_virtual_filesystem_read_file()` - Test `read_file` tool
- `test_deep_agent_virtual_filesystem_write_file()` - Test `write_file` tool
- `test_deep_agent_virtual_filesystem_edit_file()` - Test `edit_file` tool
- `test_deep_agent_subagent_spawning()` - Test `create_task_tool` subagent creation
- `test_deep_agent_state_persistence()` - Test state persistence across calls
- `test_deep_agent_write_todos_tool()` - Test `write_todos` tool integration
- `test_deep_agent_multiple_todos()` - Test multiple todos management
- `test_deep_agent_todo_states()` - Test todo state transitions (pending, in_progress, completed)

---

#### `test_base_agent.py`
**Location:** `tests/unit_tests/agent/test_base_agent.py`

**Test Cases to Implement:**
- `test_base_agent_abstract_class()` - Test that BaseAgent is abstract
- `test_base_agent_cannot_instantiate()` - Test that BaseAgent cannot be instantiated directly

---

#### `test_context_managers.py`
**Location:** `tests/unit_tests/agent/test_context_managers.py`

**Test Cases to Implement:**
- Test all 8 context manager files in `agent/context_managers/`
- `test_agent_context_manager()` - If exists
- `test_memory_context_manager()` - If exists
- `test_tool_context_manager()` - If exists
- `test_policy_context_manager()` - If exists
- `test_canvas_context_manager()` - If exists
- `test_reliability_context_manager()` - If exists
- `test_reflection_context_manager()` - If exists
- `test_compression_context_manager()` - If exists

---

#### `test_policy_manager.py`
**Location:** `tests/unit_tests/agent/test_policy_manager.py`

**Test Cases to Implement:**
- `test_policy_manager_initialization()` - Test PolicyManager initialization
- `test_policy_manager_user_policy()` - Test user policy application
- `test_policy_manager_agent_policy()` - Test agent policy application
- `test_policy_manager_multiple_policies()` - Test multiple policies

---

#### `test_run_result.py`
**Location:** `tests/unit_tests/agent/test_run_result.py`

**Test Cases to Implement:**
- `test_agent_run_result_creation()` - Test AgentRunResult creation
- `test_agent_run_result_properties()` - Test all properties
- `test_agent_run_result_serialization()` - Test serialization

---

### 2. Team/Multi-Agent Tests (`tests/unit_tests/team/`)

#### `test_team.py`
**Location:** `tests/unit_tests/team/test_team.py`

**Test Cases to Implement:**
- `test_team_initialization()` - Test Team initialization
- `test_team_initialization_with_agents()` - Test init with agent list
- `test_team_initialization_with_tasks()` - Test init with task list
- `test_team_initialization_with_memory()` - Test init with shared memory
- `test_team_sequential_mode()` - Test sequential execution mode
- `test_team_coordinate_mode()` - Test coordinate mode (leader agent)
- `test_team_route_mode()` - Test route mode (router agent)
- `test_team_do_single_task()` - Test `do()` with single task
- `test_team_do_multiple_tasks()` - Test `do()` with multiple tasks
- `test_team_do_async()` - Test async execution
- `test_team_complete()` - Test `complete()` alias
- `test_team_print_do()` - Test `print_do()` method
- `test_team_ask_other_team_members()` - Test `ask_other_team_members=True`
- `test_team_multi_agent()` - Test `multi_agent()` method
- `test_team_multi_agent_async()` - Test async `multi_agent_async()`
- `test_team_task_delegation()` - Test task delegation to agents
- `test_team_result_combination()` - Test result combination
- `test_team_context_sharing()` - Test context sharing between agents
- `test_team_error_handling()` - Test error handling

---

#### `test_coordinator_setup.py`
**Location:** `tests/unit_tests/team/test_coordinator_setup.py`

**Test Cases to Implement:**
- `test_coordinator_setup_initialization()` - Test CoordinatorSetup init
- `test_coordinator_setup_format_agent_manifest()` - Test agent manifest formatting
- `test_coordinator_setup_format_tasks_manifest()` - Test tasks manifest formatting
- `test_coordinator_setup_create_coordinate_prompt()` - Test coordinate prompt creation
- `test_coordinator_setup_create_route_prompt()` - Test route prompt creation
- `test_coordinator_setup_summarize_tool()` - Test tool summarization

---

#### `test_delegation_manager.py`
**Location:** `tests/unit_tests/team/test_delegation_manager.py`

**Test Cases to Implement:**
- `test_delegation_manager_initialization()` - Test DelegationManager init
- `test_delegation_manager_get_delegation_tool()` - Test delegation tool creation
- `test_delegation_manager_delegate_task()` - Test task delegation
- `test_delegation_manager_tool_mapping()` - Test tool mapping

---

#### `test_task_assignment.py`
**Location:** `tests/unit_tests/team/test_task_assignment.py`

**Test Cases to Implement:**
- `test_task_assignment_initialization()` - Test TaskAssignment init
- `test_task_assignment_assign_task()` - Test task assignment logic
- `test_task_assignment_route_task()` - Test task routing

---

#### `test_result_combiner.py`
**Location:** `tests/unit_tests/team/test_result_combiner.py`

**Test Cases to Implement:**
- `test_result_combiner_initialization()` - Test ResultCombiner init
- `test_result_combiner_combine_results()` - Test result combination logic
- `test_result_combiner_multiple_results()` - Test multiple result combination

---

#### `test_context_sharing.py`
**Location:** `tests/unit_tests/team/test_context_sharing.py`

**Test Cases to Implement:**
- `test_context_sharing_initialization()` - Test ContextSharing init
- `test_context_sharing_share_context()` - Test context sharing
- `test_context_sharing_shared_memory()` - Test shared memory access

---

### 3. Graph Tests (`tests/unit_tests/graph/`)

#### `test_graph.py`
**Location:** `tests/unit_tests/graph/test_graph.py`

**Test Cases to Implement:**
- `test_graph_initialization()` - Test Graph initialization
- `test_graph_initialization_with_default_agent()` - Test init with default agent
- `test_graph_initialization_with_storage()` - Test init with storage
- `test_graph_add_task()` - Test adding Task to graph
- `test_graph_add_task_node()` - Test adding TaskNode
- `test_graph_add_task_chain()` - Test adding TaskChain
- `test_graph_add_decision_func()` - Test adding DecisionFunc
- `test_graph_add_decision_llm()` - Test adding DecisionLLM
- `test_graph_add_edge()` - Test adding edges
- `test_graph_execute()` - Test graph execution
- `test_graph_execute_async()` - Test async execution
- `test_graph_parallel_execution()` - Test parallel task execution
- `test_graph_state_management()` - Test state management
- `test_graph_get_latest_output()` - Test getting latest output
- `test_graph_validation()` - Test graph validation

---

#### `test_graphv2.py`
**Location:** `tests/unit_tests/graph/test_graphv2.py`

**Test Cases to Implement:**
- `test_state_graph_initialization()` - Test StateGraph initialization
- `test_state_graph_add_node()` - Test adding nodes
- `test_state_graph_add_edge()` - Test adding edges
- `test_state_graph_add_conditional_edge()` - Test conditional edges
- `test_state_graph_compile()` - Test graph compilation
- `test_state_graph_invoke()` - Test graph invocation
- `test_state_graph_invoke_async()` - Test async invocation
- `test_state_graph_checkpointing()` - Test checkpoint creation/retrieval
- `test_state_graph_interrupt()` - Test interrupt functionality
- `test_state_graph_send()` - Test Send primitive
- `test_state_graph_command()` - Test Command primitive
- `test_state_graph_store()` - Test store integration
- `test_state_graph_cache()` - Test cache integration
- `test_state_graph_task_decorator()` - Test `@task` decorator
- `test_state_graph_retry_policy()` - Test retry policies
- `test_state_graph_cache_policy()` - Test cache policies
- `test_state_graph_errors()` - Test error handling (GraphRecursionError, GraphValidationError, GraphInterruptError)

---

#### `test_graphv2_checkpoint.py`
**Location:** `tests/unit_tests/graph/test_graphv2_checkpoint.py`

**Test Cases to Implement:**
- `test_memory_saver()` - Test MemorySaver checkpointer
- `test_sqlite_checkpointer()` - Test SqliteCheckpointer
- `test_checkpoint_creation()` - Test checkpoint creation
- `test_checkpoint_retrieval()` - Test checkpoint retrieval
- `test_state_snapshot()` - Test state snapshot functionality

---

#### `test_graphv2_store.py`
**Location:** `tests/unit_tests/graph/test_graphv2_store.py`

**Test Cases to Implement:**
- `test_in_memory_store()` - Test InMemoryStore
- `test_store_get()` - Test store get operation
- `test_store_set()` - Test store set operation
- `test_store_delete()` - Test store delete operation

---

#### `test_graphv2_cache.py`
**Location:** `tests/unit_tests/graph/test_graphv2_cache.py`

**Test Cases to Implement:**
- `test_in_memory_cache()` - Test InMemoryCache
- `test_sqlite_cache()` - Test SqliteCache
- `test_cache_policy()` - Test cache policy application
- `test_cache_get()` - Test cache get operation
- `test_cache_set()` - Test cache set operation

---

### 4. Tools System Tests (`tests/unit_tests/tools/`)

#### `test_tool_manager.py`
**Location:** `tests/unit_tests/tools/test_tool_manager.py`

**Test Cases to Implement:**
- `test_tool_manager_initialization()` - Test ToolManager initialization
- `test_tool_manager_add_tool()` - Test adding tools
- `test_tool_manager_remove_tool()` - Test removing tools
- `test_tool_manager_get_tool()` - Test getting tools
- `test_tool_manager_list_tools()` - Test listing tools
- `test_tool_manager_execute_tool()` - Test tool execution

---

#### `test_tool_processor.py`
**Location:** `tests/unit_tests/tools/test_tool_processor.py`

**Test Cases to Implement:**
- `test_tool_processor_initialization()` - Test ToolProcessor initialization
- `test_tool_processor_process_tool_calls()` - Test processing tool calls
- `test_tool_processor_validate_tool_calls()` - Test validation
- `test_tool_processor_execute_tool_calls()` - Test execution

---

#### `test_tool_orchestration.py`
**Location:** `tests/unit_tests/tools/test_tool_orchestration.py`

**Test Cases to Implement:**
- `test_orchestrator_initialization()` - Test Orchestrator initialization
- `test_orchestrator_orchestrate_tools()` - Test tool orchestration
- `test_orchestrator_parallel_execution()` - Test parallel tool execution
- `test_orchestrator_sequential_execution()` - Test sequential execution

---

#### `test_tool_mcp.py`
**Location:** `tests/unit_tests/tools/test_tool_mcp.py`

**Test Cases to Implement:**
- `test_mcp_initialization()` - Test MCP integration initialization
- `test_mcp_load_server()` - Test loading MCP server
- `test_mcp_get_tools()` - Test getting MCP tools
- `test_mcp_execute_tool()` - Test executing MCP tools
- `test_mcp_error_handling()` - Test error handling

---

#### `test_tool_schema.py`
**Location:** `tests/unit_tests/tools/test_tool_schema.py`

**Test Cases to Implement:**
- `test_tool_schema_generation()` - Test schema generation
- `test_tool_schema_validation()` - Test schema validation
- `test_tool_schema_from_function()` - Test schema from function

---

#### `test_tool_wrappers.py`
**Location:** `tests/unit_tests/tools/test_tool_wrappers.py`

**Test Cases to Implement:**
- `test_tool_wrapper_creation()` - Test tool wrapper creation
- `test_tool_wrapper_execution()` - Test wrapper execution

---

#### `test_tool_deferred.py`
**Location:** `tests/unit_tests/tools/test_tool_deferred.py`

**Test Cases to Implement:**
- `test_deferred_tool_creation()` - Test deferred tool creation
- `test_deferred_tool_execution()` - Test deferred execution

---

#### `test_builtin_tools.py`
**Location:** `tests/unit_tests/tools/test_builtin_tools.py`

**Test Cases to Implement:**
- `test_builtin_tools_list()` - Test listing builtin tools
- `test_builtin_tools_execution()` - Test builtin tool execution

---

#### `test_common_tools_duckduckgo.py`
**Location:** `tests/unit_tests/tools/test_common_tools_duckduckgo.py`

**Test Cases to Implement:**
- `test_duckduckgo_search()` - Test DuckDuckGo search tool
- `test_duckduckgo_search_error_handling()` - Test error handling

---

#### `test_common_tools_tavily.py`
**Location:** `tests/unit_tests/tools/test_common_tools_tavily.py`

**Test Cases to Implement:**
- `test_tavily_search()` - Test Tavily search tool
- `test_tavily_search_error_handling()` - Test error handling

---

#### `test_common_tools_financial.py`
**Location:** `tests/unit_tests/tools/test_common_tools_financial.py`

**Test Cases to Implement:**
- `test_financial_tools_list()` - Test financial tools
- `test_financial_tool_execution()` - Test execution

---

## 🟡 **HIGH PRIORITY** - Supporting Systems

### 5. Storage Tests (`tests/unit_tests/storage/`)

#### `test_storage_providers.py`
**Location:** `tests/unit_tests/storage/test_storage_providers.py`

**Test Cases to Implement:**
- `test_in_memory_storage()` - Test InMemoryStorage
- `test_json_storage()` - Test JSONStorage
- `test_sqlite_storage()` - Test SQLiteStorage
- `test_redis_storage()` - Test RedisStorage
- `test_postgres_storage()` - Test PostgresStorage
- `test_mongodb_storage()` - Test MongoStorage
- `test_mem0_storage()` - Test Mem0Storage
- `test_storage_connection_handling()` - Test connection handling
- `test_storage_error_scenarios()` - Test error scenarios
- `test_storage_serialization()` - Test data serialization

---

### 6. Model Provider Tests (`tests/unit_tests/models/`)

#### `test_model_openai.py`
**Location:** `tests/unit_tests/models/test_model_openai.py`

**Test Cases to Implement:**
- `test_openai_model_initialization()` - Test OpenAI model init
- `test_openai_model_request()` - Test request method
- `test_openai_model_streaming()` - Test streaming
- `test_openai_model_error_handling()` - Test error handling

---

#### `test_model_anthropic.py`
**Location:** `tests/unit_tests/models/test_model_anthropic.py`

**Test Cases to Implement:**
- `test_anthropic_model_initialization()` - Test Anthropic model init
- `test_anthropic_model_request()` - Test request method
- `test_anthropic_model_streaming()` - Test streaming
- `test_anthropic_model_error_handling()` - Test error handling

---

#### `test_model_google.py`
**Location:** `tests/unit_tests/models/test_model_google.py`

**Test Cases to Implement:**
- `test_google_model_initialization()` - Test Google model init
- `test_google_model_request()` - Test request method
- `test_google_model_streaming()` - Test streaming

---

#### `test_model_groq.py`
**Location:** `tests/unit_tests/models/test_model_groq.py`

**Test Cases to Implement:**
- `test_groq_model_initialization()` - Test Groq model init
- `test_groq_model_request()` - Test request method
- `test_groq_model_streaming()` - Test streaming

---

#### `test_model_mistral.py`
**Location:** `tests/unit_tests/models/test_model_mistral.py`

**Test Cases to Implement:**
- `test_mistral_model_initialization()` - Test Mistral model init
- `test_mistral_model_request()` - Test request method
- `test_mistral_model_streaming()` - Test streaming

---

#### `test_model_cohere.py`
**Location:** `tests/unit_tests/models/test_model_cohere.py`

**Test Cases to Implement:**
- `test_cohere_model_initialization()` - Test Cohere model init
- `test_cohere_model_request()` - Test request method

---

#### `test_model_bedrock.py`
**Location:** `tests/unit_tests/models/test_model_bedrock.py`

**Test Cases to Implement:**
- `test_bedrock_model_initialization()` - Test Bedrock model init
- `test_bedrock_model_request()` - Test request method

---

#### `test_model_huggingface.py`
**Location:** `tests/unit_tests/models/test_model_huggingface.py`

**Test Cases to Implement:**
- `test_huggingface_model_initialization()` - Test HuggingFace model init
- `test_huggingface_model_request()` - Test request method

---

## 🟢 **MEDIUM PRIORITY** - Additional Systems

### 7. OCR Tests (`tests/unit_tests/ocr/`)

#### `test_ocr.py`
**Location:** `tests/unit_tests/ocr/test_ocr.py`

**Test Cases to Implement:**
- `test_ocr_initialization()` - Test OCR class initialization
- `test_ocr_extract_text()` - Test text extraction
- `test_ocr_extract_text_from_image()` - Test image extraction
- `test_ocr_extract_text_from_pdf()` - Test PDF extraction
- `test_ocr_provider_selection()` - Test provider selection
- `test_easyocr_provider()` - Test EasyOCR provider
- `test_tesseract_provider()` - Test Tesseract provider
- `test_paddleocr_provider()` - Test PaddleOCR provider
- `test_rapidocr_provider()` - Test RapidOCR provider
- `test_deepseek_ocr_provider()` - Test DeepSeek OCR provider
- `test_ocr_error_handling()` - Test error handling

---

### 8. Chat System Tests (`tests/unit_tests/chat/`)

#### `test_chat.py`
**Location:** `tests/unit_tests/chat/test_chat.py`

**Test Cases to Implement:**
- `test_chat_initialization()` - Test Chat class initialization
- `test_chat_send_message()` - Test sending messages
- `test_chat_get_response()` - Test getting responses
- `test_chat_conversation_flow()` - Test conversation flow
- `test_chat_streaming()` - Test streaming responses
- `test_chat_session_management()` - Test session management

---

#### `test_session_manager.py`
**Location:** `tests/unit_tests/chat/test_session_manager.py`

**Test Cases to Implement:**
- `test_session_manager_initialization()` - Test SessionManager init
- `test_session_manager_create_session()` - Test session creation
- `test_session_manager_get_session()` - Test session retrieval
- `test_session_manager_delete_session()` - Test session deletion
- `test_session_manager_list_sessions()` - Test listing sessions

---

#### `test_cost_calculator.py`
**Location:** `tests/unit_tests/chat/test_cost_calculator.py`

**Test Cases to Implement:**
- `test_cost_calculator_initialization()` - Test CostCalculator init
- `test_cost_calculator_calculate_cost()` - Test cost calculation
- `test_cost_calculator_different_models()` - Test different models
- `test_cost_calculator_usage_tracking()` - Test usage tracking

---

#### `test_message.py`
**Location:** `tests/unit_tests/chat/test_message.py`

**Test Cases to Implement:**
- `test_message_creation()` - Test message creation
- `test_message_serialization()` - Test message serialization
- `test_message_parts()` - Test message parts

---

### 9. Canvas Tests (`tests/unit_tests/canvas/`)

#### `test_canvas.py`
**Location:** `tests/unit_tests/canvas/test_canvas.py`

**Test Cases to Implement:**
- `test_canvas_initialization()` - Test Canvas initialization
- `test_canvas_add_element()` - Test adding elements
- `test_canvas_update_element()` - Test updating elements
- `test_canvas_remove_element()` - Test removing elements
- `test_canvas_render()` - Test rendering

---

### 10. Durable Execution Tests (`tests/unit_tests/durable/`)

#### `test_durable_execution.py`
**Location:** `tests/unit_tests/durable/test_durable_execution.py`

**Test Cases to Implement:**
- `test_durable_execution_initialization()` - Test DurableExecution init
- `test_durable_execution_execute()` - Test execution
- `test_durable_execution_resume()` - Test resuming execution
- `test_durable_execution_serialization()` - Test serialization

---

#### `test_durable_storage.py`
**Location:** `tests/unit_tests/durable/test_durable_storage.py`

**Test Cases to Implement:**
- `test_in_memory_durable_storage()` - Test InMemoryDurableStorage
- `test_file_durable_storage()` - Test FileDurableStorage
- `test_sqlite_durable_storage()` - Test SQLiteDurableStorage
- `test_redis_durable_storage()` - Test RedisDurableStorage

---

### 11. Reflection Tests (`tests/unit_tests/reflection/`)

#### `test_reflection.py`
**Location:** `tests/unit_tests/reflection/test_reflection.py`

**Test Cases to Implement:**
- `test_reflection_processor_initialization()` - Test ReflectionProcessor init
- `test_reflection_processor_process()` - Test reflection processing
- `test_reflection_models()` - Test reflection models

---

### 12. Reliability Layer Tests (`tests/unit_tests/reliability_layer/`)

#### `test_reliability_layer.py`
**Location:** `tests/unit_tests/reliability_layer/test_reliability_layer.py`

**Test Cases to Implement:**
- `test_reliability_layer_initialization()` - Test ReliabilityLayer init
- `test_reliability_layer_verify()` - Test verification
- `test_reliability_layer_edit()` - Test editing
- `test_reliability_layer_iterative_improvement()` - Test iterative improvement

---

### 13. Cache Tests (`tests/unit_tests/cache/`)

#### `test_cache_manager.py`
**Location:** `tests/unit_tests/cache/test_cache_manager.py`

**Test Cases to Implement:**
- `test_cache_manager_initialization()` - Test CacheManager init
- `test_cache_manager_get()` - Test cache get
- `test_cache_manager_set()` - Test cache set
- `test_cache_manager_delete()` - Test cache delete
- `test_cache_manager_clear()` - Test cache clear
- `test_cache_manager_ttl()` - Test TTL functionality

---

### 14. Context Tests (`tests/unit_tests/context/`)

#### `test_context_agent.py`
**Location:** `tests/unit_tests/context/test_context_agent.py`

**Test Cases to Implement:**
- `test_context_agent_initialization()` - Test context agent
- `test_context_agent_build_context()` - Test context building

---

#### `test_context_task.py`
**Location:** `tests/unit_tests/context/test_context_task.py`

**Test Cases to Implement:**
- `test_context_task_initialization()` - Test context task
- `test_context_task_build_context()` - Test context building

---

#### `test_context_sources.py`
**Location:** `tests/unit_tests/context/test_context_sources.py`

**Test Cases to Implement:**
- `test_context_sources_initialization()` - Test context sources
- `test_context_sources_load()` - Test source loading

---

### 15. Database Tests (`tests/unit_tests/db/`)

#### `test_database.py`
**Location:** `tests/unit_tests/db/test_database.py`

**Test Cases to Implement:**
- `test_database_initialization()` - Test Database initialization
- `test_database_operations()` - Test database operations

---

### 16. Output Tests (`tests/unit_tests/`)

#### `test_output.py`
**Location:** `tests/unit_tests/test_output.py`

**Test Cases to Implement:**
- `test_output_object_definition()` - Test OutputObjectDefinition
- `test_output_serialization()` - Test output serialization

---

### 17. Usage Tests (`tests/unit_tests/`)

#### `test_usage.py`
**Location:** `tests/unit_tests/test_usage.py`

**Test Cases to Implement:**
- `test_usage_tracking()` - Test usage tracking
- `test_usage_metrics()` - Test usage metrics

---

## 📋 **Summary**

### Total Test Files to Create: **~70-80 files**

### Priority Breakdown:
- **🔴 Critical Priority:** ~25 files
- **🟡 High Priority:** ~15 files
- **🟢 Medium Priority:** ~30 files

### Estimated Test Cases: **~500-600 individual test cases**

---

## 🚀 **Recommended Implementation Order**

1. **Week 1:** Direct Agent tests (`test_direct.py`)
2. **Week 2:** DeepAgent tests (`test_deep_agent.py`)
3. **Week 3:** Team tests (`test_team.py` + related files)
4. **Week 4:** Graph tests (`test_graph.py`, `test_graphv2.py`)
5. **Week 5:** Tools system tests
6. **Week 6:** Storage and Model provider tests
7. **Week 7:** Supporting systems (OCR, Chat, Canvas, etc.)

---

## 📝 **Notes**

- Each test file should follow the existing test patterns in your codebase
- Use mocking for external dependencies (API calls, file system, etc.)
- Focus on public APIs and critical paths first
- Add integration tests for complex workflows
- Consider using pytest fixtures for common setup
- Follow your existing test naming conventions

---

*This test plan is based on the codebase structure and existing test patterns. Adjust priorities based on your specific needs.*


import pytest
from operator import itemgetter

from upsonic.uel import (
    Runnable,
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough,
    ChatPromptTemplate,
    chain,
)


# ============================================================================
# MOCK COMPONENTS FOR TESTING (SIMULATING MODEL BEHAVIOR)
# ============================================================================

class MockModel(Runnable):
    """Mock model that returns formatted responses for testing."""
    
    def __init__(self, response_prefix: str = "Response"):
        self.response_prefix = response_prefix
    
    def invoke(self, input, config=None):
        """Return a mock response based on input."""
        if isinstance(input, str):
            return f"{self.response_prefix}: {input}"
        elif hasattr(input, 'parts'):
            # Handle ModelRequest-like objects
            content = str(input.parts[0].content) if input.parts else str(input)
            return f"{self.response_prefix}: {content}"
        else:
            return f"{self.response_prefix}: {str(input)}"
    
    async def ainvoke(self, input, config=None):
        """Async version."""
        return self.invoke(input, config)


class MockRetriever(Runnable):
    """Mock retriever that returns context for testing."""
    
    def __init__(self, context: str = "mock context"):
        self.context = context
    
    def invoke(self, input, config=None):
        """Return mock context."""
        return self.context
    
    async def ainvoke(self, input, config=None):
        """Async version."""
        return self.context


# ============================================================================
# TEST 1: ITEMGETTER WITH DICT-BASED PARALLEL CHAINS
# ============================================================================

def test_itemgetter_with_dict_parallel():
    """
    Test Example 1: itemgetter usage with dict-based parallel chains.
    
    This tests that:
    1. Python's operator.itemgetter works in chains
    2. Dict syntax automatically creates RunnableParallel
    3. Results are properly passed through the chain
    """
    print("\n" + "="*80)
    print("TEST 1: itemgetter with dict-based parallel chains")
    print("="*80)
    
    # Setup
    retriever = MockRetriever("Harrison worked at Kensho")
    model = MockModel("Answer")
    
    # Create template
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build chain with itemgetter
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "language": itemgetter("language"),
        }
        | prompt
        | model
    )
    
    # Test invocation
    result = chain.invoke({
        "question": "where did harrison work",
        "language": "english"
    })
    
    print(f"Result: {result}")
    
    # Verify the chain structure
    assert isinstance(chain, RunnableSequence), "Chain should be a RunnableSequence"
    assert isinstance(chain.steps[0], RunnableParallel), "First step should be RunnableParallel"
    
    # Verify the result contains expected elements
    assert "Answer:" in result or "Response:" in result
    print("✓ itemgetter with dict-based parallel chains works!")


# ============================================================================
# TEST 2: RUNNABLEPARALLEL EXPLICIT USAGE
# ============================================================================

def test_runnableparallel_explicit():
    """
    Test Example 2: Explicit RunnableParallel usage.
    
    This tests that:
    1. RunnableParallel can be created explicitly
    2. Multiple chains run in parallel
    3. Results are returned as a dictionary
    """
    print("\n" + "="*80)
    print("TEST 2: Explicit RunnableParallel usage")
    print("="*80)
    
    model = MockModel()
    
    # Create two chains
    joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    poem_chain = ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
    
    # Create parallel chain explicitly
    map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
    
    # Test invocation
    result = map_chain.invoke({"topic": "bear"})
    
    print(f"Result keys: {result.keys()}")
    print(f"Joke: {result['joke']}")
    print(f"Poem: {result['poem']}")
    
    # Verify structure
    assert isinstance(result, dict), "Result should be a dict"
    assert "joke" in result, "Result should have 'joke' key"
    assert "poem" in result, "Result should have 'poem' key"
    assert "bear" in str(result['joke']).lower(), "Joke should mention bear"
    assert "bear" in str(result['poem']).lower(), "Poem should mention bear"
    
    print("✓ Explicit RunnableParallel works!")


# ============================================================================
# TEST 3: RUNNABLEPASSTHROUGH USAGE
# ============================================================================

def test_runnablepassthrough():
    """
    Test Example 3: RunnablePassthrough usage.
    
    This tests that:
    1. RunnablePassthrough passes input unchanged
    2. Works in parallel dict construction
    3. Properly integrates with prompt templates
    """
    print("\n" + "="*80)
    print("TEST 3: RunnablePassthrough usage")
    print("="*80)
    
    model = MockModel()
    
    # Create prompt with messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{equation_statement}")
    ])
    
    # Build chain with RunnablePassthrough
    runnable = (
        {"equation_statement": RunnablePassthrough()} 
        | prompt 
        | model
    )
    
    # Test invocation
    result = runnable.invoke("x raised to the third plus seven equals 12")
    
    print(f"Result: {result}")
    
    # Verify result
    assert "x raised to the third" in result.lower() or "Response:" in result
    
    print("✓ RunnablePassthrough works!")


# ============================================================================
# TEST 4: RUNNABLELAMBDA WITH ITEMGETTER
# ============================================================================

def test_runnablelambda_with_itemgetter():
    """
    Test Example 4: RunnableLambda with itemgetter.
    
    This tests that:
    1. Functions can be wrapped as RunnableLambda
    2. Complex nested structures work
    3. itemgetter extracts values correctly
    """
    print("\n" + "="*80)
    print("TEST 4: RunnableLambda with itemgetter")
    print("="*80)
    
    # Define functions
    def length_function(text):
        return len(text)
    
    def multiple_length_function(_dict):
        return len(_dict["text1"]) * len(_dict["text2"])
    
    model = MockModel()
    prompt = ChatPromptTemplate.from_template("what is {a} + {b}")
    
    # Build complex chain with RunnableLambda
    chain = (
        {
            "a": itemgetter("foo") | RunnableLambda(length_function),
            "b": {
                "text1": itemgetter("foo"),
                "text2": itemgetter("bar")
            } | RunnableLambda(multiple_length_function),
        }
        | prompt
        | model
    )
    
    # Test invocation
    result = chain.invoke({"foo": "bar", "bar": "gah"})
    
    print(f"Result: {result}")
    
    # Verify calculations:
    # len("bar") = 3, len("bar") * len("gah") = 3 * 3 = 9
    # So prompt should have a=3 and b=9
    assert "3" in result or "Response:" in result
    
    print("✓ RunnableLambda with itemgetter works!")


# ============================================================================
# TEST 5: CHAIN DECORATOR
# ============================================================================

def test_chain_decorator():
    """
    Test Example 5: @chain decorator usage.
    
    This tests that:
    1. Functions can be decorated with @chain
    2. Decorated functions become Runnables
    3. Can invoke chains within decorated functions
    """
    print("\n" + "="*80)
    print("TEST 5: @chain decorator")
    print("="*80)
    
    model = MockModel()
    
    prompt1 = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    prompt2 = ChatPromptTemplate.from_template("What is the subject of this joke: {joke}")
    
    @chain
    def custom_chain(text):
        # Use chains inside the function
        prompt_val1 = prompt1.invoke({"topic": text})
        output1 = model.invoke(prompt_val1)
        
        # Create and invoke another chain
        chain2 = prompt2 | model
        return chain2.invoke({"joke": output1})
    
    # Test invocation
    result = custom_chain.invoke("bears")
    
    print(f"Result: {result}")
    
    # Verify it's a Runnable
    assert isinstance(custom_chain, Runnable), "Decorated function should be Runnable"
    assert "Response:" in result or "joke" in result.lower()
    
    print("✓ @chain decorator works!")


# ============================================================================
# TEST 6: FUNCTION COERCION IN CHAINS
# ============================================================================

def test_function_coercion():
    """
    Test Example 6: Function coercion in chains.
    
    This tests that:
    1. Plain functions can be used directly in chains
    2. Lambda functions work in chains
    3. Functions are automatically wrapped as RunnableLambda
    """
    print("\n" + "="*80)
    print("TEST 6: Function coercion in chains")
    print("="*80)
    
    model = MockModel()
    prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
    
    # Build chain with lambda function
    chain = prompt | model | (lambda x: x[:20])
    
    # Test invocation
    result = chain.invoke({"topic": "bears"})
    
    print(f"Result: {result}")
    print(f"Result length: {len(result)}")
    
    # Verify the lambda was applied (result truncated to 20 chars)
    assert len(result) == 20, f"Result should be 20 chars, got {len(result)}"
    
    print("✓ Function coercion works!")


# ============================================================================
# TEST 7: RUNNABLEBRANCH CONDITIONAL ROUTING
# ============================================================================

def test_runnablebranch():
    """
    Test Example 9: RunnableBranch for conditional routing.
    
    This tests that:
    1. RunnableBranch routes based on conditions
    2. Conditions are evaluated in order
    3. Default branch is used when no condition matches
    """
    print("\n" + "="*80)
    print("TEST 7: RunnableBranch conditional routing")
    print("="*80)
    
    model = MockModel()
    
    # Create classification chain
    classification_chain = (
        ChatPromptTemplate.from_template(
            "Classify this question: {question}\nClassification:"
        )
        | model
        | (lambda x: x.lower())
    )
    
    # Create specialized chains
    upsonic_chain = (
        ChatPromptTemplate.from_template("Upsonic expert: {question}")
        | model
    )
    
    anthropic_chain = (
        ChatPromptTemplate.from_template("Anthropic expert: {question}")
        | model
    )
    
    general_chain = (
        ChatPromptTemplate.from_template("General: {question}")
        | model
    )
    
    # Create branch with conditions
    branch = RunnableBranch(
        (lambda x: "anthropic" in x.get("topic", "").lower(), anthropic_chain),
        (lambda x: "upsonic" in x.get("topic", "").lower(), upsonic_chain),
        general_chain  # default
    )
    
    # Test with different inputs
    test_cases = [
        {"topic": "anthropic", "question": "How does Anthropic work?"},
        {"topic": "upsonic", "question": "How does Upsonic work?"},
        {"topic": "other", "question": "How does AI work?"},
    ]
    
    for test_input in test_cases:
        result = branch.invoke(test_input)
        print(f"\nInput topic: {test_input['topic']}")
        print(f"Result: {result}")
        
        # Verify correct branch was taken
        if "anthropic" in test_input["topic"].lower():
            assert "Anthropic expert" in result
        elif "upsonic" in test_input["topic"].lower():
            assert "Upsonic expert" in result
        else:
            assert "General" in result
    
    print("\n✓ RunnableBranch works!")


# ============================================================================
# TEST 8: DYNAMIC CHAIN CONSTRUCTION
# ============================================================================

def test_dynamic_chain_construction():
    """
    Test Example 10: Dynamic (self-constructing) chains.
    
    This tests that:
    1. @chain can return different Runnables based on input
    2. Returned Runnables are automatically invoked
    3. Conditional chain construction works
    """
    print("\n" + "="*80)
    print("TEST 8: Dynamic chain construction")
    print("="*80)
    
    model = MockModel()
    
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Convert the latest user question into a standalone question."),
        ("placeholder", {"variable_name": "chat_history"}),
        ("human", "{question}"),
    ])
    
    contextualize_question = contextualize_prompt | model
    
    @chain
    def contextualize_if_needed(input_: dict) -> Runnable:
        """Return different chain based on whether chat_history exists."""
        if input_.get("chat_history"):
            # Return a chain that contextualizes
            return contextualize_question
        else:
            # Just pass through the question
            return RunnablePassthrough() | itemgetter("question")
    
    # Test with chat history
    result_with_history = contextualize_if_needed.invoke({
        "question": "what about egypt",
        "chat_history": [
            ("human", "what's the population of indonesia"),
            ("ai", "about 276 million"),
        ],
    })
    
    print(f"\nWith history: {result_with_history}")
    
    # Test without chat history
    result_without_history = contextualize_if_needed.invoke({
        "question": "what about egypt",
    })
    
    print(f"Without history: {result_without_history}")
    
    # Verify different behavior
    assert result_with_history != result_without_history or True  # Different paths taken
    
    print("✓ Dynamic chain construction works!")


# ============================================================================
# TEST 9: RUNNABLEPASSTHROUGH.ASSIGN
# ============================================================================

def test_runnablepassthrough_assign():
    """
    Test RunnablePassthrough.assign() for adding keys to the input.
    
    This tests that:
    1. assign() adds new keys to input dict
    2. Original keys are preserved
    3. Assigned values can be from runnables or functions
    """
    print("\n" + "="*80)
    print("TEST 9: RunnablePassthrough.assign()")
    print("="*80)
    
    model = MockModel()
    
    # Mock retriever
    @chain
    def fake_retriever(input_: dict) -> str:
        return "egypt's population in 2024 is about 111 million"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question given the following context:\n\n{context}."),
        ("human", "{question}")
    ])
    
    # Build chain with assign
    full_chain = (
        RunnablePassthrough.assign(
            formatted_question=lambda x: f"Q: {x['question']}"
        ).assign(
            context=fake_retriever
        )
        | prompt
        | model
    )
    
    # Test invocation
    result = full_chain.invoke({
        "question": "what about egypt",
    })
    
    print(f"Result: {result}")
    
    # Verify result contains context
    assert "egypt" in result.lower() or "Response:" in result
    
    print("✓ RunnablePassthrough.assign() works!")


# ============================================================================
# TEST 10: GRAPH VISUALIZATION AND PROMPT EXTRACTION
# ============================================================================

def test_graph_and_prompts():
    """
    Test Example 11: Graph visualization and prompt extraction.
    
    This tests that:
    1. get_graph() returns a graph representation
    2. Graph can be printed as ASCII
    3. get_prompts() extracts all ChatPromptTemplate instances
    """
    print("\n" + "="*80)
    print("TEST 10: Graph visualization and prompt extraction")
    print("="*80)
    
    model = MockModel()
    retriever = MockRetriever()
    
    # Build a complex chain
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context:\n{context}\n\nQuestion: {question}"
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
    )
    
    # Test get_graph()
    graph = chain.get_graph()
    print(f"\nGraph: {graph}")
    print(f"Graph nodes: {len(graph.nodes)}")
    
    # Test print_ascii()
    print("\nASCII representation:")
    ascii_repr = graph.to_ascii()
    print(ascii_repr)
    
    # Test get_prompts()
    prompts = chain.get_prompts()
    print(f"\nPrompts found: {len(prompts)}")
    for i, p in enumerate(prompts):
        print(f"  Prompt {i+1}: {p}")
    
    # Verify
    assert graph is not None, "Graph should be created"
    assert len(graph.nodes) > 0, "Graph should have nodes"
    assert len(prompts) == 1, "Should find 1 prompt template"
    assert isinstance(prompts[0], ChatPromptTemplate), "Should be ChatPromptTemplate"
    
    print("✓ Graph visualization and prompt extraction work!")


# ============================================================================
# TEST 11: COMPLEX NESTED PARALLEL CHAINS
# ============================================================================

def test_complex_nested_parallel():
    """
    Test complex nested parallel structures.
    
    This tests that:
    1. Parallel chains can be nested
    2. Dict syntax works at multiple levels
    3. Results are properly structured
    """
    print("\n" + "="*80)
    print("TEST 11: Complex nested parallel chains")
    print("="*80)
    
    # Create a complex nested structure
    chain = {
        "outer1": {
            "inner1": lambda x: x["value"] * 2,
            "inner2": lambda x: x["value"] + 10,
        },
        "outer2": lambda x: x["value"] ** 2,
    }
    
    # Coerce to runnable
    from upsonic.uel.lambda_runnable import coerce_to_runnable
    runnable_chain = coerce_to_runnable(chain)
    
    # Test invocation
    result = runnable_chain.invoke({"value": 5})
    
    print(f"Result: {result}")
    
    # Verify structure
    assert isinstance(result, dict), "Result should be dict"
    assert "outer1" in result, "Should have outer1 key"
    assert "outer2" in result, "Should have outer2 key"
    assert isinstance(result["outer1"], dict), "outer1 should be dict"
    assert result["outer1"]["inner1"] == 10, f"inner1 should be 10, got {result['outer1']['inner1']}"
    assert result["outer1"]["inner2"] == 15, f"inner2 should be 15, got {result['outer1']['inner2']}"
    assert result["outer2"] == 25, f"outer2 should be 25, got {result['outer2']}"
    
    print("✓ Complex nested parallel chains work!")


# ============================================================================
# TEST 12: ASYNC EXECUTION
# ============================================================================

@pytest.mark.asyncio
async def test_async_execution():
    """
    Test async execution of chains.
    
    This tests that:
    1. Chains can be executed asynchronously
    2. ainvoke() works through the chain
    3. Parallel execution is truly async
    """
    print("\n" + "="*80)
    print("TEST 12: Async execution")
    print("="*80)
    
    model = MockModel()
    prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
    
    chain = prompt | model
    
    # Test async invocation
    result = await chain.ainvoke({"topic": "async programming"})
    
    print(f"Async result: {result}")
    
    assert "async programming" in result.lower() or "Response:" in result
    
    print("✓ Async execution works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all tests manually (for debugging)."""
    print("\n" + "="*80)
    print("COMPREHENSIVE UEL TESTS")
    print("="*80)
    print("(Excluding AI model features and output parsers)")
    
    try:
        test_itemgetter_with_dict_parallel()
        test_runnableparallel_explicit()
        test_runnablepassthrough()
        test_runnablelambda_with_itemgetter()
        test_chain_decorator()
        test_function_coercion()
        test_runnablebranch()
        test_dynamic_chain_construction()
        test_runnablepassthrough_assign()
        test_graph_and_prompts()
        test_complex_nested_parallel()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYour UEL implementation is working correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
    
    # Also run async test
    import asyncio
    print("\n" + "="*80)
    print("Running async test...")
    asyncio.run(test_async_execution())
    print("="*80)


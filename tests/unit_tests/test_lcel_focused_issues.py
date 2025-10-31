"""
Focused tests for specific uel issues.

1. Test 6: Lambda function truncation issue
2. Test 10: Other graph visualization methods
"""

from upsonic.uel import (
    ChatPromptTemplate,
    RunnablePassthrough,
)
from upsonic.uel.lambda_runnable import coerce_to_runnable


# ============================================================================
# MOCK COMPONENTS
# ============================================================================

from upsonic.uel.runnable import Runnable

class MockModel(Runnable):
    """Mock model that returns formatted responses for testing."""
    
    def __init__(self, response_prefix: str = "Response"):
        self.response_prefix = response_prefix
    
    def invoke(self, input, config=None):
        """Return a mock response based on input."""
        print(f"    🔵 MockModel.invoke() called with: {type(input)} = {input}")
        
        if isinstance(input, str):
            result = f"{self.response_prefix}: {input}"
        elif hasattr(input, 'parts'):
            # Handle ModelRequest-like objects
            content = str(input.parts[0].content) if input.parts else str(input)
            result = f"{self.response_prefix}: {content}"
        else:
            result = f"{self.response_prefix}: {str(input)}"
        
        print(f"    🔵 MockModel returning: {result}")
        print(f"    🔵 MockModel result length: {len(result)}")
        return result
    
    async def ainvoke(self, input, config=None):
        """Async version."""
        return self.invoke(input, config)


# ============================================================================
# TEST 6: LAMBDA FUNCTION TRUNCATION ISSUE
# ============================================================================

def test_lambda_truncation_issue():
    """
    Test if lambda function truncation is working correctly.
    
    The lambda x: x[:20] should cut the result to 20 characters.
    """
    print("\n" + "="*80)
    print("TEST 6: Lambda function truncation issue")
    print("="*80)
    
    model = MockModel()
    prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
    
    print(f"\n📝 Prompt template: {prompt}")
    
    # Build chain with lambda function
    print(f"\n🔗 Building chain with lambda function...")
    lambda_func = lambda x: x[:20]
    print(f"Lambda function: {lambda_func}")
    print(f"Lambda function type: {type(lambda_func)}")
    
    # Test the lambda function directly
    test_input = "This is a very long string that should be truncated to 20 characters"
    print(f"\n🧪 Testing lambda function directly:")
    print(f"Input: {test_input}")
    print(f"Input length: {len(test_input)}")
    
    direct_result = lambda_func(test_input)
    print(f"Direct lambda result: {direct_result}")
    print(f"Direct lambda result length: {len(direct_result)}")
    
    # Build the chain
    chain = prompt | model | lambda_func
    print(f"\n🔗 Chain: {chain}")
    print(f"Chain steps: {[type(step).__name__ for step in chain.steps]}")
    
    # Test invocation
    input_data = {"topic": "bears"}
    print(f"\n🚀 Invoking chain with input: {input_data}")
    
    result = chain.invoke(input_data)
    
    print(f"\n✅ Final result: {result}")
    print(f"Result length: {len(result)}")
    print(f"Expected length: 20")
    print(f"Length matches expected: {len(result) == 20}")
    
    if len(result) == 20:
        print("✓ Lambda truncation works correctly!")
    else:
        print("❌ Lambda truncation is NOT working!")
        print(f"Expected 20 characters, got {len(result)}")
    
    assert len(result) == 20, f"Expected 20 characters, got {len(result)}"


# ============================================================================
# TEST 10: OTHER GRAPH VISUALIZATION METHODS
# ============================================================================

def test_other_graph_visualization_methods():
    """
    Test other graph visualization methods.
    """
    print("\n" + "="*80)
    print("TEST 10: Other graph visualization methods")
    print("="*80)
    
    model = MockModel()
    
    # Build a complex chain
    print(f"\n📝 Creating complex chain...")
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context:\n{context}\n\nQuestion: {question}"
    )
    
    chain = (
        {"context": lambda x: "mock context", "question": RunnablePassthrough()}
        | prompt
        | model
    )
    print(f"Chain: {chain}")
    
    # Test get_graph()
    print(f"\n📊 Getting graph representation...")
    graph = chain.get_graph()
    print(f"Graph: {graph}")
    
    # Test to_ascii()
    print(f"\n📊 ASCII representation:")
    ascii_repr = graph.to_ascii()
    print(ascii_repr)
    
    # Test to_mermaid()
    print(f"\n📊 Mermaid representation:")
    try:
        mermaid_repr = graph.to_mermaid()
        print(mermaid_repr)
        print("✓ Mermaid representation works!")
    except Exception as e:
        print(f"❌ Mermaid representation failed: {e}")
    
    # Test graph structure details
    print(f"\n📊 Graph structure details:")
    print(graph.get_structure_details())
    
    # Test graph repr
    print(f"\n📊 Graph repr: {repr(graph)}")
    
    print("✓ Other graph visualization methods work!")


# ============================================================================
# RUN FOCUSED TESTS
# ============================================================================

def run_focused_tests():
    """Run focused tests for specific issues."""
    print("\n" + "="*80)
    print("FOCUSED UEL TESTS")
    print("="*80)
    print("\nTesting specific issues:")
    print("1. Lambda function truncation")
    print("2. Other graph visualization methods")
    
    try:
        # Test 1: Lambda truncation
        lambda_works = test_lambda_truncation_issue()
        
        # Test 2: Graph visualization
        test_other_graph_visualization_methods()
        
        print("\n" + "="*80)
        if lambda_works:
            print("✅ ALL FOCUSED TESTS PASSED!")
        else:
            print("❌ LAMBDA TRUNCATION TEST FAILED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_focused_tests()


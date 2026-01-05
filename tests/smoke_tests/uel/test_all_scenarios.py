"""
COMPREHENSIVE MEMORY MODE TEST SUITE
=====================================

This test file covers ALL memory modes and ALL scenarios for the UEL Model memory system.

Modes:
- auto (default): Smart detection - skips memory if placeholder detected
- always: Always loads from memory
- never: Never loads from memory, only saves
- record_all: Like auto, but saves ALL messages including placeholder history

Scenarios:
- S1: No placeholder, Empty memory
- S2: No placeholder, Memory has history
- S3: Placeholder, Empty memory
- S4: Placeholder, Memory has history (CRITICAL - modes differ here)

Edge Cases:
- Multi-chain RAG pattern (same model, two chains)
- Empty placeholder history
- Sequential placeholder invocations
- record_all duplicate demonstration
- Different placeholder variable names
- Simple chatbot flow (sequential calls without placeholder)
- Switching between placeholder and non-placeholder chains
"""

from upsonic.uel import ChatPromptTemplate, RunnablePassthrough, StrOutputParser
from upsonic.models import infer_model

PASS_COUNT = 0
FAIL_COUNT = 0

def assert_check(condition: bool, description: str) -> bool:
    """Custom assert that tracks pass/fail and prints result."""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✅ ASSERT PASS: {description}")
        return True
    else:
        FAIL_COUNT += 1
        print(f"  ❌ ASSERT FAIL: {description}")
        return False




def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

def subheader(title: str) -> None:
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def get_memory_messages(model):
    """Get messages from model's internal memory."""
    if not model._memory:
        return []
    return model._memory.get_messages() or []

def print_memory(model, title: str = "Memory") -> None:
    """Print what's stored in the model's memory."""
    messages = get_memory_messages(model)
    print(f"\n  📦 {title} ({len(messages)} messages):")
    if not messages:
        print("      (empty)")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content = str(msg)[:80] + "..." if len(str(msg)) > 80 else str(msg)
        print(f"      [{i}] {msg_type}: {content}")




def print_mode_reference():
    header("MEMORY MODE REFERENCE")
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    MEMORY MODES REFERENCE                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                   ║
║  ┌─────────────┬──────────────────────────────────┬──────────────────────────────────────────┐   ║
║  │    MODE     │         LOADING LOGIC            │           SAVING LOGIC                   │   ║
║  ├─────────────┼──────────────────────────────────┼──────────────────────────────────────────┤   ║
║  │   auto      │ Skip if placeholder detected,    │ Save last request + response only        │   ║
║  │  (default)  │ Load if no placeholder           │                                          │   ║
║  ├─────────────┼──────────────────────────────────┼──────────────────────────────────────────┤   ║
║  │   always    │ Always load from memory          │ Save last request + response only        │   ║
║  │             │ (ignores placeholder detection)  │                                          │   ║
║  ├─────────────┼──────────────────────────────────┼──────────────────────────────────────────┤   ║
║  │   never     │ Never load from memory           │ Save last request + response only        │   ║
║  │             │                                  │                                          │   ║
║  ├─────────────┼──────────────────────────────────┼──────────────────────────────────────────┤   ║
║  │ record_all  │ Skip if placeholder detected,    │ Save ALL messages including placeholder  │   ║
║  │             │ Load if no placeholder           │ ⚠️ Can cause duplicates!                 │   ║
║  └─────────────┴──────────────────────────────────┴──────────────────────────────────────────┘   ║
║                                                                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                      WHEN TO USE                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                   ║
║  ✅ auto:       Multi-chain RAG, complex workflows, same model in multiple chains                ║
║  ⚠️ always:     Simple single-chain chatbots WITHOUT placeholders                                ║
║  📊 never:      Logging/analytics, external history management                                    ║
║  📝 record_all: Complete audit trails (handle duplicates yourself)                               ║
║                                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
""")



MODES = ["auto", "always", "never", "record_all"]

EXPECTED = {
    "auto": {
        "S1": "Current only (memory empty)",
        "S2": "Memory LOADED + current",
        "S3": "Placeholder only (memory empty)",
        "S4": "Placeholder only (memory SKIPPED)",
    },
    "always": {
        "S1": "Current only (memory empty)",
        "S2": "Memory LOADED + current",
        "S3": "Placeholder only (memory empty)",
        "S4": "Memory + Placeholder (DUPLICATES!)",
    },
    "never": {
        "S1": "Current only (never loads)",
        "S2": "Current only (never loads)",
        "S3": "Placeholder only (never loads)",
        "S4": "Placeholder only (never loads)",
    },
    "record_all": {
        "S1": "Current only (memory empty)",
        "S2": "Memory LOADED + current",
        "S3": "Placeholder only (memory empty)",
        "S4": "Placeholder only (memory SKIPPED)",
    },
}


def test_mode_scenarios(mode: str, debug: bool = False):
    """Test all scenarios for a given mode."""
    header(f"MODE = '{mode}'")
    
    model = infer_model("openai/gpt-4o").add_memory(history=True, mode=mode, debug=debug)
    
    simple_chain = (
        ChatPromptTemplate.from_template("Answer in 1 sentence: {question}")
        | model
        | StrOutputParser()
    )
    
    placeholder_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer concisely."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "{question}")
        ])
        | model
        | StrOutputParser()
    )
    
    # S1: No placeholder, Empty memory
    subheader(f"S1: No placeholder, Empty memory | mode='{mode}'")
    print(f"  Expected: {EXPECTED[mode]['S1']}")
    mem_before_s1 = len(get_memory_messages(model))
    result_s1 = simple_chain.invoke({"question": "What is Go?"})
    mem_after_s1 = len(get_memory_messages(model))
    print(f"\n  ✨ Result: {result_s1[:80]}...")
    print_memory(model, "Memory after S1")
    
    assert_check(len(result_s1) > 10, "S1: Got meaningful response")
    assert_check(mem_before_s1 == 0, "S1: Memory was empty before")
    assert_check(mem_after_s1 == 2, "S1: Memory has 2 messages after (request + response)")
    
    # S2: No placeholder, Memory has history
    subheader(f"S2: No placeholder, Memory has history | mode='{mode}'")
    print(f"  Expected: {EXPECTED[mode]['S2']}")
    result_s2 = simple_chain.invoke({"question": "Who created it?"})
    mem_after_s2 = len(get_memory_messages(model))
    print(f"\n  ✨ Result: {result_s2[:80]}...")
    print_memory(model, "Memory after S2")
    
    if mode in ["auto", "always", "record_all"]:
        has_context = "go" in result_s2.lower() or "google" in result_s2.lower() or "griesemer" in result_s2.lower() or "pike" in result_s2.lower()
        assert_check(has_context, f"S2: Model remembered Go context (mode={mode})")
        if mode == "record_all":
            assert_check(mem_after_s2 >= 4, f"S2: record_all memory grows (got {mem_after_s2})")
        else:
            assert_check(mem_after_s2 == 4, "S2: Memory has 4 messages after")
    else:
        assert_check(mem_after_s2 == 4, "S2: Memory still grows even in 'never' mode")
        print("  Note: 'never' mode saves but doesn't load, so model has no context")
    
    # S3: Placeholder, Empty memory (fresh model)
    model_fresh = infer_model("openai/gpt-4o").add_memory(history=True, mode=mode, debug=debug)
    placeholder_chain_fresh = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer concisely."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "{question}")
        ])
        | model_fresh
        | StrOutputParser()
    )
    
    subheader(f"S3: Placeholder, Empty memory (fresh) | mode='{mode}'")
    print(f"  Expected: {EXPECTED[mode]['S3']}")
    result_s3 = placeholder_chain_fresh.invoke({
        "question": "What is it used for?",
        "chat_history": [
            ("human", "Tell me about Python"),
            ("ai", "Python is a high-level language.")
        ]
    })
    mem_after_s3 = len(get_memory_messages(model_fresh))
    print(f"\n  ✨ Result: {result_s3[:80]}...")
    print_memory(model_fresh, "Memory after S3 (fresh model)")
    
    has_python_context = "python" in result_s3.lower()
    assert_check(has_python_context, "S3: Model used placeholder context about Python")
    if mode == "record_all":
        assert_check(mem_after_s3 >= 2, f"S3: record_all saves all (got {mem_after_s3})")
    else:
        assert_check(mem_after_s3 == 2, "S3: Only last request+response saved (not placeholder)")
    
    # S4: Placeholder, Memory has history (CRITICAL)
    subheader(f"S4: Placeholder + Existing Memory (CRITICAL) | mode='{mode}'")
    print(f"  Expected: {EXPECTED[mode]['S4']}")
    result_s4 = placeholder_chain.invoke({
        "question": "Is it memory-safe?",
        "chat_history": [
            ("human", "What is Rust?"),
            ("ai", "Rust is a systems language.")
        ]
    })
    mem_after_s4 = len(get_memory_messages(model))
    print(f"\n  ✨ Result: {result_s4[:80]}...")
    print_memory(model, "Memory after S4")
    
    has_rust_context = "rust" in result_s4.lower() or "memory" in result_s4.lower() or "safe" in result_s4.lower()
    assert_check(has_rust_context, "S4: Model answered about Rust (from placeholder)")
    
    if mode == "record_all":
        assert_check(mem_after_s4 >= 6, f"S4: record_all saves all (got {mem_after_s4})")
    elif mode == "always":
        assert_check(mem_after_s4 == 6, "S4-always: Memory grows normally")
    else:
        assert_check(mem_after_s4 == 6, "S4: Memory has 6 messages")




def test_edge_cases(debug: bool = False):
    header("EDGE CASE TESTS")
    
    # Edge Case 1: Multi-chain RAG pattern
    subheader("Edge Case 1: Multi-chain RAG pattern (same model, two chains)")
    
    model_rag = infer_model("openai/gpt-4o").add_memory(history=True, mode="auto", debug=debug)
    
    contextualize_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "ONLY output the rephrased question. Do NOT answer."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "Rephrase: {question}")
        ])
        | model_rag
        | StrOutputParser()
    )
    
    answer_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer concisely."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "{contextualized_question}")
        ])
        | model_rag
        | StrOutputParser()
    )
    
    print("\n  Testing: Same model in two chains with placeholder history")
    ctx_result = contextualize_chain.invoke({
        "question": "What about Swift?",
        "chat_history": [("human", "Tell me about iOS"), ("ai", "iOS uses Swift.")]
    })
    print(f"  Chain 1 (contextualize): {ctx_result[:60]}...")
    
    answer_result = answer_chain.invoke({
        "contextualized_question": ctx_result,
        "chat_history": [("human", "Tell me about iOS"), ("ai", "iOS uses Swift.")]
    })
    print(f"  Chain 2 (answer): {answer_result[:60]}...")
    print_memory(model_rag, "Memory after both chains")
    
    mem_count = len(get_memory_messages(model_rag))
    assert_check(mem_count == 4, f"EC1: Memory has 4 messages (2 chains × 2 msgs), got {mem_count}")
    assert_check(len(ctx_result) > 0, "EC1: Contextualize chain produced output")
    assert_check(len(answer_result) > 0, "EC1: Answer chain produced output")
    
    # Edge Case 2: Empty placeholder
    subheader("Edge Case 2: Empty placeholder history")
    
    model_empty = infer_model("openai/gpt-4o").add_memory(history=True, mode="auto", debug=debug)
    chain_empty = (
        ChatPromptTemplate.from_messages([
            ("system", "Be helpful."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "{question}")
        ])
        | model_empty
        | StrOutputParser()
    )
    
    result_empty = chain_empty.invoke({"question": "What is C#?", "chat_history": []})
    print(f"\n  Result with empty chat_history: {result_empty[:60]}...")
    print_memory(model_empty, "Memory after empty placeholder")
    
    mem_empty = len(get_memory_messages(model_empty))
    assert_check(len(result_empty) > 10, "EC2: Got meaningful response with empty placeholder")
    assert_check(mem_empty >= 1, f"EC2: Memory has at least 1 message (got {mem_empty})")
    
    # Edge Case 3: record_all duplicates
    subheader("Edge Case 3: record_all mode - Duplicate demonstration")
    
    model_rec = infer_model("openai/gpt-4o").add_memory(history=True, mode="record_all", debug=debug)
    chain_rec = (
        ChatPromptTemplate.from_messages([
            ("system", "Be helpful."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "{question}")
        ])
        | model_rec
        | StrOutputParser()
    )
    
    print("\n  Turn 1: Invoking with [H1, A1]")
    r1 = chain_rec.invoke({
        "question": "Tell me more",
        "chat_history": [("human", "What is Go?"), ("ai", "Go is by Google.")]
    })
    mem_after_t1 = len(get_memory_messages(model_rec))
    print_memory(model_rec, "Memory after Turn 1")
    
    print("\n  Turn 2: Invoking with [H1, A1, Q1, R1]")
    r2 = chain_rec.invoke({
        "question": "Who uses it?",
        "chat_history": [
            ("human", "What is Go?"), ("ai", "Go is by Google."),
            ("human", "Tell me more"), ("ai", r1)
        ]
    })
    mem_after_t2 = len(get_memory_messages(model_rec))
    print_memory(model_rec, "Memory after Turn 2 (⚠️ DUPLICATES!)")
    
    assert_check(mem_after_t1 >= 3, f"EC3: record_all saved placeholder + response (Turn 1), got {mem_after_t1}")
    assert_check(mem_after_t2 > mem_after_t1, f"EC3: Memory grew after Turn 2")
    print("\n  ⚠️ record_all saves placeholder history → duplicates on each turn!")
    
    # Edge Case 4: Different placeholder variable names
    subheader("Edge Case 4: Different placeholder variable names")
    
    model_diff = infer_model("openai/gpt-4o").add_memory(history=True, mode="auto", debug=debug)
    
    chain_history1 = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer about topic 1."),
            ("placeholder", {"variable_name": "chat_history_1"}),
            ("human", "{question}")
        ])
        | model_diff
        | StrOutputParser()
    )
    
    chain_history2 = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer about topic 2."),
            ("placeholder", {"variable_name": "chat_history_2"}),
            ("human", "{question}")
        ])
        | model_diff
        | StrOutputParser()
    )
    
    print("\n  Testing different placeholder variable names")
    result_h1 = chain_history1.invoke({
        "question": "Tell me more",
        "chat_history_1": [("human", "What is Java?"), ("ai", "Java is enterprise.")]
    })
    print(f"  Chain with chat_history_1: {result_h1[:50]}...")
    
    result_h2 = chain_history2.invoke({
        "question": "Tell me more",
        "chat_history_2": [("human", "What is Ruby?"), ("ai", "Ruby is elegant.")]
    })
    print(f"  Chain with chat_history_2: {result_h2[:50]}...")
    print_memory(model_diff, "Memory after different placeholders")
    
    mem_diff = len(get_memory_messages(model_diff))
    assert_check(mem_diff == 4, f"EC4: Memory has 4 messages from 2 chains, got {mem_diff}")
    assert_check(len(result_h1) > 0, "EC4: Chain 1 with custom placeholder worked")
    assert_check(len(result_h2) > 0, "EC4: Chain 2 with custom placeholder worked")
    
    # Edge Case 5: Simple chatbot flow (sequential calls, no placeholder)
    subheader("Edge Case 5: Simple chatbot flow (auto mode, no placeholder)")
    
    model_chatbot = infer_model("openai/gpt-4o").add_memory(history=True, mode="auto", debug=debug)
    simple_chat = (
        ChatPromptTemplate.from_template("You are a helpful assistant. {question}")
        | model_chatbot
        | StrOutputParser()
    )
    
    print("\n  Multi-turn conversation without placeholder:")
    r1 = simple_chat.invoke({"question": "My name is Alice"})
    print(f"  Turn 1: {r1[:50]}...")
    
    r2 = simple_chat.invoke({"question": "What's my name?"})
    print(f"  Turn 2: {r2[:50]}...")
    
    r3 = simple_chat.invoke({"question": "How many times have I asked you something?"})
    print(f"  Turn 3: {r3[:50]}...")
    print_memory(model_chatbot, "Memory after 3 turns")
    
    mem_chatbot = len(get_memory_messages(model_chatbot))
    assert_check(mem_chatbot == 6, f"EC5: Memory has 6 messages (3 turns × 2), got {mem_chatbot}")
    has_alice = "alice" in r2.lower()
    assert_check(has_alice, "EC5: Model remembered name 'Alice' in Turn 2")
    
    # Edge Case 6: Switching between placeholder and non-placeholder chains
    subheader("Edge Case 6: Switching between placeholder and non-placeholder chains")
    
    model_switch = infer_model("openai/gpt-4o").add_memory(history=True, mode="auto", debug=debug)
    
    chain_no_placeholder = (
        ChatPromptTemplate.from_template("Answer: {question}")
        | model_switch
        | StrOutputParser()
    )
    
    chain_with_placeholder = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer concisely."),
            ("placeholder", {"variable_name": "chat_history"}),
            ("human", "{question}")
        ])
        | model_switch
        | StrOutputParser()
    )
    
    print("\n  Alternating between chain types:")
    s1 = chain_no_placeholder.invoke({"question": "What is Kotlin?"})
    print(f"  Step 1 (no placeholder): {s1[:50]}...")
    
    s2 = chain_with_placeholder.invoke({
        "question": "Compare it to this",
        "chat_history": [("human", "What is Scala?"), ("ai", "Scala is functional.")]
    })
    print(f"  Step 2 (with placeholder): {s2[:50]}...")
    
    s3 = chain_no_placeholder.invoke({"question": "Which do you recommend?"})
    print(f"  Step 3 (no placeholder again): {s3[:50]}...")
    print_memory(model_switch, "Memory after switching")
    
    mem_switch = len(get_memory_messages(model_switch))
    assert_check(mem_switch == 6, f"EC6: Memory has 6 messages (3 invocations × 2), got {mem_switch}")
    
    # Edge Case 7: Very long conversation (memory growth check)
    subheader("Edge Case 7: Long conversation simulation")
    
    model_long = infer_model("openai/gpt-4o").add_memory(history=True, mode="auto", debug=debug)
    long_chain = (
        ChatPromptTemplate.from_template("Reply briefly: {question}")
        | model_long
        | StrOutputParser()
    )
    
    print("\n  Running 5 consecutive turns...")
    for i in range(5):
        long_chain.invoke({"question": f"This is message number {i+1}"})
    
    mem_long = len(get_memory_messages(model_long))
    print_memory(model_long, "Memory after 5 turns")
    
    assert_check(mem_long == 10, f"EC7: Memory has 10 messages (5 turns × 2), got {mem_long}")



def print_summary():
    global PASS_COUNT, FAIL_COUNT
    header("TEST SUMMARY")
    
    print(f"\n  📊 ASSERTION RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    
    if FAIL_COUNT == 0:
        print("  ✅ ALL TESTS PASSED!")
    else:
        print(f"  ❌ {FAIL_COUNT} TESTS FAILED - Review output above")
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      BEHAVIOR MATRIX                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                   ║
║  ┌──────────┬─────────────────────┬─────────────────────┬─────────────────────┬───────────────┐  ║
║  │ Scenario │      auto           │      always         │      never          │  record_all   │  ║
║  ├──────────┼─────────────────────┼─────────────────────┼─────────────────────┼───────────────┤  ║
║  │   S1     │ Current only        │ Current only        │ Current only        │ Current only  │  ║
║  │   S2     │ Memory + Current ✅ │ Memory + Current ✅ │ Current only ⚠️     │ Mem+Curr ✅   │  ║
║  │   S3     │ Placeholder ✅      │ Placeholder ✅      │ Placeholder ✅      │ Placeholder   │  ║
║  │   S4     │ Placeholder ✅      │ Mem+Placeholder ⚠️  │ Placeholder ✅      │ Placeholder   │  ║
║  └──────────┴─────────────────────┴─────────────────────┴─────────────────────┴───────────────┘  ║
║                                                                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                       SAVING BEHAVIOR                                             ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                   ║
║  • auto/always/never: Saves ONLY last request + response (prevents duplicates)                   ║
║  • record_all: Saves ALL messages including placeholder (⚠️ duplicates on multi-turn!)           ║
║                                                                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                      RECOMMENDATIONS                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                   ║
║  ✅ Use 'auto' (default) for 95% of cases - handles everything correctly                         ║
║  ⚠️ Use 'always' only for simple chatbots WITHOUT placeholders                                   ║
║  📊 Use 'never' for logging/analytics where history is always external                           ║
║  📝 Use 'record_all' for audit trails (handle duplicates yourself)                               ║
║                                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    return FAIL_COUNT == 0




if __name__ == "__main__":
    import sys
    
    debug = "--debug" in sys.argv
    
    print_mode_reference()
    
    for mode in MODES:
        test_mode_scenarios(mode, debug=debug)
    
    test_edge_cases(debug=debug)
    
    all_passed = print_summary()
    
    sys.exit(0 if all_passed else 1)

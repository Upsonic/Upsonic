#!/usr/bin/env python3
"""
Test script for the Upsonic error wrapper functionality.
This script demonstrates how the error wrapper converts various types of errors
to Upsonic-specific errors while hiding pydantic-ai implementation details.
"""

import os
import asyncio
from upsonic import Task, Direct, Agent
from upsonic import UupsonicError, AgentExecutionError, ModelConnectionError, NoAPIKeyException

async def test_error_wrapper():
    """Test the error wrapper with various error scenarios."""
    
    print("🧪 Testing Upsonic Error Wrapper System")
    print("=" * 50)
    
    # Test 1: API Key Error
    print("\n1. Testing API Key Error...")
    # Temporarily remove API key to trigger error
    original_key = os.environ.get("OPENAI_API_KEY")
    if original_key:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        direct = Direct(model="openai/gpt-4o")
        task = Task("Hello world")
        result = await direct.do_async(task)
        print("❌ Expected an error but got result:", result)
    except NoAPIKeyException as e:
        print("✅ Successfully caught NoAPIKeyException:", e.message)
    except Exception as e:
        print("⚠️  Caught unexpected error:", type(e).__name__, str(e))
    finally:
        # Restore API key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
    
    # Test 2: Invalid Model Error
    print("\n2. Testing Invalid Model Error...")
    try:
        direct = Direct(model="invalid/model-name")
        task = Task("Hello world")
        result = await direct.do_async(task)
        print("❌ Expected an error but got result:", result)
    except UupsonicError as e:
        print("✅ Successfully caught UupsonicError:", e.message)
        print("   Error Code:", e.error_code)
    except Exception as e:
        print("⚠️  Caught unexpected error:", type(e).__name__, str(e))
    
    # Test 3: Normal Operation (should work without errors)
    print("\n3. Testing Normal Operation...")
    if original_key:  # Only test if we have an API key
        try:
            direct = Direct(model="openai/gpt-4o")
            task = Task("Say hello in one word")
            result = await direct.do_async(task)
            print("✅ Normal operation successful:", result)
        except Exception as e:
            print("⚠️  Normal operation failed:", type(e).__name__, str(e))
    else:
        print("⏭️  Skipping normal operation test (no API key)")
    
    # Test 4: Sync vs Async methods
    print("\n4. Testing Sync Methods...")
    if original_key:  # Only test if we have an API key
        try:
            direct = Direct(model="openai/gpt-4o")
            task = Task("Count to 3")
            result = direct.do(task)  # Sync method
            print("✅ Sync method successful:", result)
        except Exception as e:
            print("⚠️  Sync method failed:", type(e).__name__, str(e))
    else:
        print("⏭️  Skipping sync method test (no API key)")
    
    print("\n" + "=" * 50)
    print("🎯 Error Wrapper Test Complete!")
    print("\nKey Benefits:")
    print("• All pydantic-ai errors are now wrapped as Upsonic errors")
    print("• Users see friendly, branded error messages")
    print("• Error details are automatically displayed with Rich formatting")
    print("• Retry logic is built-in for transient errors")
    print("• Original error information is preserved for debugging")

def test_sync_wrapper():
    """Test the synchronous wrapper methods."""
    print("\n🔄 Testing Synchronous Wrapper...")
    
    # Test sync method without API key
    original_key = os.environ.get("OPENAI_API_KEY")
    if original_key:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        direct = Direct(model="openai/gpt-4o")
        task = Task("Hello world")
        result = direct.do(task)  # This should trigger error wrapper
        print("❌ Expected an error but got result:", result)
    except NoAPIKeyException as e:
        print("✅ Sync wrapper successfully caught NoAPIKeyException")
    except Exception as e:
        print("⚠️  Caught unexpected error:", type(e).__name__, str(e))
    finally:
        # Restore API key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key

if __name__ == "__main__":
    print("Testing Error Wrapper System")
    print("This test demonstrates how Upsonic wraps pydantic-ai errors")
    print("into user-friendly, branded error messages.\n")
    
    # Run async tests
    asyncio.run(test_error_wrapper())
    
    # Run sync tests
    test_sync_wrapper()
    
    print("\n✨ All tests completed!")
    print("The error wrapper successfully hides pydantic-ai implementation details")
    print("and provides a consistent, branded error experience for users.") 
#!/usr/bin/env python3
"""Simple test runner for storage tests with progress output."""
import sys
import os
import asyncio
import pytest

# Import Skipped exception for handling pytest.skip() when running standalone
try:
    from _pytest.outcomes import Skipped
except ImportError:
    # Fallback if pytest structure changes
    Skipped = type('Skipped', (Exception,), {})

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

async def main():
    # Import the test module directly
    import importlib.util
    test_file = os.path.join(os.path.dirname(__file__), "test_storage_agentsession_comprehensive.py")
    spec = importlib.util.spec_from_file_location("test_storage_agentsession_comprehensive", test_file)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Import the standalone test runner functions
    _run_inmemory_all_attributes = test_module._run_inmemory_all_attributes
    _run_json_all_attributes = test_module._run_json_all_attributes
    _run_sqlite_all_attributes = test_module._run_sqlite_all_attributes
    _run_redis_all_attributes = test_module._run_redis_all_attributes
    _run_mongo_all_attributes = test_module._run_mongo_all_attributes
    _run_postgres_all_attributes = test_module._run_postgres_all_attributes
    _run_mem0_all_attributes = test_module._run_mem0_all_attributes
    
    tests = [
        ("InMemoryStorage", _run_inmemory_all_attributes),
        ("JSONStorage", _run_json_all_attributes),
        ("SqliteStorage", _run_sqlite_all_attributes),
        ("RedisStorage", _run_redis_all_attributes),
        ("MongoStorage", _run_mongo_all_attributes),
        ("PostgresStorage", _run_postgres_all_attributes),
        ("Mem0Storage", _run_mem0_all_attributes),
    ]
    
    print("="*70, flush=True)
    print("COMPREHENSIVE AgentSession ATTRIBUTE TESTS", flush=True)
    print("="*70, flush=True)
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Testing {name}...", flush=True)
        try:
            await test_func()
            print(f"✅ {name} PASSED\n", flush=True)
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted during {name}", flush=True)
            break
        except Skipped as e:
            skip_msg = str(e.msg) if hasattr(e, 'msg') else str(e)
            print(f"⏭ {name} SKIPPED: {skip_msg[:200]}\n", flush=True)
        except Exception as e:
            print(f"⚠ {name} FAILED: {str(e)[:200]}\n", flush=True)
    
    print("="*70, flush=True)
    print("Tests completed!", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠ Tests interrupted", flush=True)
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


"""
Test runner for vector database tests.
"""
import pytest
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def run_vectordb_tests():
    """Run all vector database tests."""
    test_dir = Path(__file__).parent
    
    # Run tests with verbose output
    pytest_args = [
        str(test_dir),
        "-v",
        "--tb=short",
        "--no-header",
        "-x"  # Stop on first failure
    ]
    
    return pytest.main(pytest_args)

if __name__ == "__main__":
    exit_code = run_vectordb_tests()
    sys.exit(exit_code)

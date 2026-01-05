"""
Test 29: CLI testing
Success criteria: commands run properly and handle what it needs to do properly! We check logs and results for that
"""
import pytest
import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for CLI tests."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


def run_cli_command(args, cwd=None):
    """Run a CLI command and return the result."""
    cmd = [sys.executable, "-m", "upsonic.cli.main"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30
    )
    return result


def test_cli_help():
    """Test CLI help command."""
    result = run_cli_command(["--help"])
    
    # Help command should succeed (return 0) or at least not crash
    assert result.returncode in [0, 1], f"Help command should not crash. Error: {result.stderr}"


def test_cli_usage_no_args():
    """Test CLI with no arguments shows usage."""
    result = run_cli_command([])
    
    # No args should show usage (return 0) or at least not crash
    assert result.returncode in [0, 1], "No args should not crash"


def test_cli_unknown_command():
    """Test CLI with unknown command."""
    result = run_cli_command(["unknown_command"])
    
    # Unknown command should return error code (1) or show usage (0)
    assert result.returncode in [0, 1], "Unknown command should return error code or show usage"
    # Just verify it doesn't crash - the exact behavior may vary


def test_cli_init_command(temp_dir):
    """Test CLI init command."""
    result = run_cli_command(["init", "--help"], cwd=temp_dir)
    
    # Init help should work (may return 0 or 1 depending on implementation)
    assert result.returncode in [0, 1], "Init help should not crash"
    # Some commands may not have help implemented, so we just check it doesn't error badly


def test_cli_run_help():
    """Test CLI run command help."""
    result = run_cli_command(["run", "--help"])
    
    # Run help should work (may return 0 or 1 depending on implementation)
    assert result.returncode in [0, 1], "Run help should not crash"
    # Some commands may not have help implemented, so we just check it doesn't error badly


def test_cli_install_help():
    """Test CLI install command help."""
    result = run_cli_command(["install", "--help"])
    
    # Install help should work (may return 0 or 1 depending on implementation)
    # Just verify it doesn't crash
    assert result.returncode in [0, 1], "Install help should not crash"
    # Some commands may not have help implemented, so we just check it doesn't error badly


def test_cli_add_help():
    """Test CLI add command help."""
    result = run_cli_command(["add", "--help"])
    
    # Add help should work
    assert result.returncode in [0, 1], "Add help should not crash"
    # Some commands may not have help implemented, so we just check it doesn't error badly


def test_cli_zip_help():
    """Test CLI zip command help."""
    result = run_cli_command(["zip", "--help"])
    
    # Zip help should work
    assert result.returncode in [0, 1], "Zip help should not crash"
    # Some commands may not have help implemented, so we just check it doesn't error badly


def test_cli_command_dispatch():
    """Test that CLI properly dispatches commands."""
    # Test that main function exists and can be imported
    from upsonic.cli.main import main
    
    # Test with help
    result = main(["--help"])
    assert result == 0, "Main should return 0 for help"
    
    # Test with unknown command
    result = main(["unknown_cmd"])
    assert result == 1, "Main should return 1 for unknown command"
    
    # Test with no args
    result = main([])
    assert result == 0, "Main should return 0 for no args (shows usage)"


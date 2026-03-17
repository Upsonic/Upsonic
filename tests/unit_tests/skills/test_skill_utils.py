"""Unit tests for skill utilities."""

import tempfile
import unittest
from pathlib import Path

from upsonic.skills.utils import (
    ScriptResult,
    get_interpreter_command,
    is_safe_path,
    parse_shebang,
    read_file_safe,
    run_script,
)


class TestIsSafePath(unittest.TestCase):
    def test_safe_relative_path(self):
        base = Path("/base/dir")
        self.assertTrue(is_safe_path(base, "file.txt"))

    def test_safe_nested_path(self):
        base = Path("/base/dir")
        self.assertTrue(is_safe_path(base, "sub/file.txt"))

    def test_traversal_attack_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "inner"
            base.mkdir()
            self.assertFalse(is_safe_path(base, "../../../etc/passwd"))

    def test_double_dot_in_middle(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            sub = base / "a" / "b"
            sub.mkdir(parents=True)
            # a/b/../c resolves to a/c which is still under base
            self.assertTrue(is_safe_path(base, "a/b/../c"))

    def test_absolute_path_escapes(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self.assertFalse(is_safe_path(base, "/etc/passwd"))


class TestParseShebang(unittest.TestCase):
    def test_python_env_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("#!/usr/bin/env python3\nprint('hi')")
            f.flush()
            self.assertEqual(parse_shebang(Path(f.name)), "python3")

    def test_direct_bash_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\necho hi")
            f.flush()
            self.assertEqual(parse_shebang(Path(f.name)), "bash")

    def test_env_with_flags(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write("#!/usr/bin/env -S node\nconsole.log('hi')")
            f.flush()
            self.assertEqual(parse_shebang(Path(f.name)), "node")

    def test_no_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('no shebang')")
            f.flush()
            self.assertIsNone(parse_shebang(Path(f.name)))

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()
            self.assertIsNone(parse_shebang(Path(f.name)))

    def test_nonexistent_file(self):
        self.assertIsNone(parse_shebang(Path("/nonexistent/file.py")))


class TestGetInterpreterCommand(unittest.TestCase):
    def test_python_maps_to_sys_executable(self):
        import sys
        self.assertEqual(get_interpreter_command("python"), [sys.executable])
        self.assertEqual(get_interpreter_command("python3"), [sys.executable])

    def test_other_interpreter(self):
        self.assertEqual(get_interpreter_command("node"), ["node"])
        self.assertEqual(get_interpreter_command("bash"), ["bash"])


class TestRunScript(unittest.TestCase):
    def test_run_python_script(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("#!/usr/bin/env python3\nprint('hello world')")
            f.flush()
            result = run_script(Path(f.name))
            self.assertEqual(result.returncode, 0)
            self.assertIn("hello world", result.stdout)

    def test_run_python_no_shebang(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('no shebang')")
            f.flush()
            result = run_script(Path(f.name))
            self.assertEqual(result.returncode, 0)
            self.assertIn("no shebang", result.stdout)

    def test_run_bash_script(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\necho 'hello bash'")
            f.flush()
            result = run_script(Path(f.name))
            self.assertEqual(result.returncode, 0)
            self.assertIn("hello bash", result.stdout)

    def test_script_with_args(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("#!/usr/bin/env python3\nimport sys\nprint(' '.join(sys.argv[1:]))")
            f.flush()
            result = run_script(Path(f.name), args=["arg1", "arg2"])
            self.assertEqual(result.returncode, 0)
            self.assertIn("arg1 arg2", result.stdout)

    def test_script_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("#!/usr/bin/env python3\nimport sys\nsys.exit(1)")
            f.flush()
            result = run_script(Path(f.name))
            self.assertEqual(result.returncode, 1)

    def test_script_result_dataclass(self):
        r = ScriptResult(stdout="out", stderr="err", returncode=0)
        self.assertEqual(r.stdout, "out")
        self.assertEqual(r.stderr, "err")
        self.assertEqual(r.returncode, 0)


class TestReadFileSafe(unittest.TestCase):
    def test_read_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello content")
            f.flush()
            content = read_file_safe(Path(f.name))
            self.assertEqual(content, "hello content")

    def test_read_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            read_file_safe(Path("/nonexistent/file.txt"))


if __name__ == "__main__":
    unittest.main()

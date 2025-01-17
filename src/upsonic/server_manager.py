import os
import signal
import sys
import time
import socket
import subprocess
import psutil
from contextlib import closing
from typing import Optional

class ServerManager:
    def __init__(self, app_path: str, host: str, port: int, name: str):
        self.app_path = app_path
        self.host = host
        self.port = port
        self.name = name
        self._process: Optional[subprocess.Popen] = None
        self._pid_file = os.path.join(os.path.expanduser("~"), f".upsonic_{name}_server.pid")

    def _is_port_in_use(self) -> bool:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex((self.host, self.port)) == 0

    def _write_pid(self):
        """Write the current process ID to the PID file"""
        if self._process and self._process.pid:
            with open(self._pid_file, 'w') as f:
                f.write(str(self._process.pid))

    def _read_pid(self) -> Optional[int]:
        """Read the process ID from the PID file"""
        try:
            if os.path.exists(self._pid_file):
                with open(self._pid_file, 'r') as f:
                    return int(f.read().strip())
        except (ValueError, IOError):
            pass
        return None

    def _cleanup_pid(self):
        """Remove the PID file"""
        try:
            if os.path.exists(self._pid_file):
                os.remove(self._pid_file)
        except OSError:
            pass

    def start(self, redirect_output: bool = False):
        """Start the server if it's not already running."""
        if self.is_running():
            return

        # Check if port is already in use
        if self._is_port_in_use():
            raise RuntimeError(f"Port {self.port} is already in use")

        # Set up logging
        if redirect_output:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            stdout = open(os.path.join(log_dir, f'{self.name}_server.log'), 'a')
            stderr = open(os.path.join(log_dir, f'{self.name}_server_error.log'), 'a')
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE

        # Prepare the command
        cmd = [
            sys.executable, "-m", "uvicorn",
            self.app_path,
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "error",
            "--no-access-log"
        ]

        try:
            # Start the server process
            self._process = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True  # This creates a new process group
            )
            self._write_pid()

            # Wait for server to start
            start_time = time.time()
            max_wait_time = 30  # Increase wait time for slower systems
            while not self._is_port_in_use() and time.time() - start_time < max_wait_time:
                if self._process.poll() is not None:
                    # Process has terminated
                    raise RuntimeError(f"Server process terminated unexpectedly with code {self._process.returncode}")
                time.sleep(0.1)

            if time.time() - start_time >= max_wait_time:
                raise RuntimeError(f"Timeout waiting for {self.name} server to start")

        except Exception as e:
            self.stop()  # Clean up if startup fails
            raise RuntimeError(f"Failed to start {self.name} server: {str(e)}")

    def stop(self):
        """Stop the server if it's running."""
        pid = self._read_pid()
        if pid:
            try:
                process = psutil.Process(pid)
                # Try to terminate the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    # If termination fails, force kill the process group
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                if self._process.poll() is None:
                    self._process.kill()

        self._process = None
        self._cleanup_pid()

    def is_running(self) -> bool:
        """Check if the server is currently running."""
        if self._process and self._process.poll() is None:
            return True

        pid = self._read_pid()
        if pid:
            try:
                process = psutil.Process(pid)
                return process.is_running() and process.name().startswith("python")
            except psutil.NoSuchProcess:
                self._cleanup_pid()
        return False 
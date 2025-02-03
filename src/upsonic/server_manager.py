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

    def _kill_process_using_port(self) -> bool:
        """Find and kill all processes using the specified port."""
        killed_any = False
        failed_kills = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Get connections separately since it can't be retrieved in process_iter
                process = psutil.Process(proc.info['pid'])
                connections = process.connections()
                for conn in connections:
                    if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == self.port:
                        # Found a process using our port
                        try:
                            process_info = f"PID: {process.pid}, Name: {process.name()}"
                            # Try SIGTERM first
                            process.terminate()
                            try:
                                process.wait(timeout=3)
                                killed_any = True
                            except psutil.TimeoutExpired:
                                # If SIGTERM doesn't work, use SIGKILL
                                try:
                                    process.kill()
                                    process.wait(timeout=1)
                                    killed_any = True
                                except Exception as e:
                                    failed_kills.append(f"{process_info} (Kill failed: {str(e)})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            try:
                                # Last resort: try using system kill command
                                os.kill(process.pid, signal.SIGKILL)
                                killed_any = True
                            except Exception as e:
                                failed_kills.append(f"{process_info} (System kill failed: {str(e)})")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                continue
            except Exception as e:
                continue

        if failed_kills:
            error_msg = "Failed to kill the following processes using port {self.port}:\n" + "\n".join(failed_kills)
            raise RuntimeError(error_msg)
            
        return killed_any

    def _is_port_in_use(self, kill_if_used: bool = False) -> bool:
        """Check if the port is in use and optionally kill the process using it."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            result = sock.connect_ex((self.host, self.port)) == 0
            if result and kill_if_used:
                return not self._kill_process_using_port()
            return result

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

    def _force_cleanup_port(self):
        """Aggressively clean up the port before starting."""
        if not self._is_port_in_use():
            return

        # First try normal port kill
        try:
            self._kill_process_using_port()
            # Wait a bit to ensure port is freed
            time.sleep(1)
        except RuntimeError as e:
            # If normal kill failed, try lsof as last resort
            try:
                cmd = f"lsof -i :{self.port} -t"
                pids = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
                kill_failed = []
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except Exception as e:
                        kill_failed.append(f"PID {pid} (Error: {str(e)})")
                
                if kill_failed:
                    raise RuntimeError(f"Failed to kill processes using lsof:\n" + "\n".join(kill_failed))
                    
                time.sleep(1)
            except subprocess.CalledProcessError:
                # lsof command failed
                raise RuntimeError(f"Could not find processes using port {self.port} with lsof")
            except RuntimeError as e:
                # Re-raise the error from kill attempts
                raise RuntimeError(f"Failed to kill processes: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error while cleaning up port {self.port}: {str(e)}")

        # Final check
        if self._is_port_in_use():
            raise RuntimeError(f"Port {self.port} is still in use after all cleanup attempts. Please check running processes manually.")

    def start(self, redirect_output: bool = False, force: bool = False):
        """Start the server if it's not already running."""
        # Always do an initial cleanup
        self._force_cleanup_port()
        
        if self.is_running():
            return

        # Check if port is still in use after cleanup
        if self._is_port_in_use(kill_if_used=force):
            raise RuntimeError(f"Port {self.port} is still in use and could not be freed")

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
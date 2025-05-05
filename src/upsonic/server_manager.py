import os
import signal
import sys
import time
import socket
import subprocess
import traceback
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
        """Check if the port is in use."""
        try:
            # Faster method to check port availability
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.2)  # Reduce timeout for faster checks
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            # Fallback to the original method
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                return sock.connect_ex((self.host, self.port)) == 0

    def _kill_process_using_port(self) -> bool:
        """Find and kill processes using the specified port."""
        killed = False
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                process = psutil.Process(proc.info['pid'])
                for conn in process.net_connections():
                    if hasattr(conn, 'laddr') and len(conn.laddr) >= 2 and conn.laddr[1] == self.port:
                        try:
                            # Kill process more aggressively
                            process.kill()
                            process.wait(timeout=0.5)  # Reduced timeout
                            killed = True
                        except Exception:
                            # Last resort: try system kill
                            try:
                                os.kill(process.pid, signal.SIGKILL)
                                killed = True
                            except Exception:
                                pass
            except Exception:
                continue
                
        return killed

    def _manage_pid_file(self, operation: str):
        """Manage PID file operations (read/write/cleanup)."""
        try:
            if operation == "write" and self._process and self._process.pid:
                with open(self._pid_file, 'w') as f:
                    f.write(str(self._process.pid))
            elif operation == "read" and os.path.exists(self._pid_file):
                with open(self._pid_file, 'r') as f:
                    return int(f.read().strip())
            elif operation == "cleanup" and os.path.exists(self._pid_file):
                os.remove(self._pid_file)
        except Exception:
            pass
        
        return None if operation == "read" else False

    def _cleanup_port(self):
        """Clean up the port before starting."""
        if not self._is_port_in_use():
            return True
            
        # Try to kill processes using the port
        self._kill_process_using_port()
        time.sleep(0.2)  # Reduced sleep time
        
        # If port is still in use, try one more aggressive approach
        if self._is_port_in_use():
            try:
                # Try to bind to the port to force it free
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
                sock.close()
            except socket.error:
                # If binding fails, try one more kill attempt
                for proc in psutil.process_iter(['pid']):
                    try:
                        process = psutil.Process(proc.info['pid'])
                        if any(len(conn.laddr) >= 2 and conn.laddr[1] == self.port 
                              for conn in process.net_connections(kind='inet')):
                            process.kill()
                    except Exception:
                        continue
                time.sleep(0.2)  # Reduced sleep time
        
        return not self._is_port_in_use()

    def start(self, redirect_output: bool = False, force: bool = False):
        """Start the server if it's not already running."""
        if self.is_running():
            return
            
        # Clean up port
        if not self._cleanup_port() and not force:
            raise RuntimeError(f"Port {self.port} is in use and could not be freed")

        # Set up logging
        stdout = stderr = None
        if redirect_output:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            stdout = open(os.path.join(log_dir, f'{self.name}_server.log'), 'a')
            stderr = open(os.path.join(log_dir, f'{self.name}_server_error.log'), 'a')

        # Start the server process
        try:
            workers_amount = os.getenv("UPSONIC_WORKERS_AMOUNT", 1)
            if self.app_path == "upsonic.tools_server.server.api:app":
                workers_amount = 1
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                self.app_path,
                "--host", self.host,
                "--port", str(self.port),
                "--log-level", "error",
                "--no-access-log",
                "--workers", str(workers_amount)
            ]
            
            self._process = subprocess.Popen(
                cmd, stdout=stdout, stderr=stderr, start_new_session=True
            )
            self._manage_pid_file("write")

            # Wait for server to start with optimized polling
            # Initial quick sleep to allow process to start
            time.sleep(0.1)
            
            # Progressive polling with increasing intervals
            poll_interval = 0.01
            max_poll_interval = 0.1
            start_time = time.time()
            while not self._is_port_in_use() and time.time() - start_time < 300:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Server process terminated unexpectedly with code {self._process.returncode}")
                time.sleep(poll_interval)
                # Gradually increase polling interval
                poll_interval = min(poll_interval * 1.2, max_poll_interval)

            if not self._is_port_in_use():
                raise RuntimeError(f"Timeout waiting for {self.name} server to start")
                
        except Exception as e:
            self.stop()
            traceback.print_exc()
            raise RuntimeError(f"Failed to start {self.name} server: {str(e)}")

    def stop(self):
        """Stop the server if it's running."""
        # Try to stop process from PID file
        pid = self._manage_pid_file("read")
        if pid:
            try:
                process = psutil.Process(pid)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
                except Exception:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass

        # Try to stop process from instance variable
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                if self._process.poll() is None:
                    self._process.kill()

        self._process = None
        self._manage_pid_file("cleanup")

    def is_running(self) -> bool:
        """Check if the server is currently running."""
        # Check if process from instance is running
        if self._process and self._process.poll() is None:
            return True

        # Check if process from PID file is running
        pid = self._manage_pid_file("read")
        if pid:
            try:
                process = psutil.Process(pid)
                return process.is_running() and process.name().startswith("python")
            except psutil.NoSuchProcess:
                self._manage_pid_file("cleanup")
                
        return False 
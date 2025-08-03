import subprocess
import os
import resource
import platform
from typing import Tuple, Dict, Any

class SandboxExecutor:
    """
    A secure sandbox for executing user-provided code.
    It sets resource limits (CPU time, memory) before execution.
    Note: `resource` module is not available on Windows.
    """
    def __init__(self, timeout: int = 60, memory_limit_mb: int = 512):
        self.timeout = timeout
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert MB to bytes
        self.is_unix = platform.system() != "Windows"

    def _set_limits(self):
        """Sets resource limits for the child process."""
        if self.is_unix:
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
            # Set memory limit
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))

    def execute(self, script_path: str) -> Tuple[bool, str, str]:
        """
        Executes a script in a sandboxed environment.

        Args:
            script_path: The absolute path to the Python script to execute.

        Returns:
            A tuple containing:
            - bool: True if execution was successful, False otherwise.
            - str: The standard output of the script.
            - str: The standard error of the script.
        """
        process_args = ["uv", "run", "python", script_path]
        
        try:
            process = subprocess.Popen(
                process_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._set_limits if self.is_unix else None,
                text=True
            )
            stdout, stderr = process.communicate(timeout=self.timeout)
            
            if process.returncode == 0:
                return True, stdout, stderr
            else:
                return False, stdout, f"Process exited with code {process.returncode}\n{stderr}"

        except subprocess.TimeoutExpired:
            process.kill()
            return False, "", "Execution timed out."
        except FileNotFoundError:
            return False, "", "`uv` command not found. Make sure it is installed and in your PATH."
        except Exception as e:
            return False, "", f"An unexpected error occurred: {e}"

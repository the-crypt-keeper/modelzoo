from dataclasses import dataclass
from typing import Any, List, Dict
import subprocess
import threading
from collections import deque
import os

@dataclass
class RuntimeParameter:
    """Data class describing a configurable runtime parameter."""
    param_name: str
    param_description: str
    param_type: str
    param_default: Any

    def __str__(self) -> str:
        return f"Parameter({self.param_name}: {self.param_type}, default={self.param_default})"

@dataclass
class Environment:
    """Data class representing a runtime environment configuration."""
    name: str
    vars: dict[str, str]

    def __str__(self) -> str:
        vars_str = ', '.join(f'{k}={v}' for k, v in self.vars.items())
        return f"Environment({self.name}: {vars_str})"

@dataclass
class Listener:
    """Data class representing network binding configuration."""
    protocol: str
    host: str
    port: int

    def __str__(self) -> str:
        return f"Listener({self.protocol}://{self.host}:{self.port})"

@dataclass
class Model:
    """Data class representing a machine learning model."""
    model_id: str
    model_format: str
    model_name: str = None
    model_size: int = None
    model_architecture: str = None

    def __str__(self) -> str:
        parts = [f"Model({self.model_name}",
                f"format={self.model_format}"]
                
        if self.model_size is not None:         parts.append(f"size={self.model_size}")
        if self.model_architecture is not None: parts.append(f"arch={self.model_architecture}")
        
        return ', '.join(parts) + ')'

class RunningModel:
    """Class representing and controlling a running model instance."""

    def __init__(self, model: Model, environment: Environment, 
                 listener: Listener, command: List[str]):
        """Initialize a new running model instance.
        
        Args:
            model (Model): The model being served
            environment (Environment): Environment configuration
            listener (Listener): Network binding configuration
            command (List[str]): Command line as list of strings
        """
        self.model = model
        self.environment = environment
        self.listener = listener
        self.command = command
        self.process = None
        self.log_buffer = deque(maxlen=100)  # Keep last 100 lines
        self._log_thread = None
        self._running = False
        
        # Start the process
        self._start_process()

    def _start_process(self):
        """Start the underlying process and log collection."""
        # Create environment dict for subprocess
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.environment.vars.items()})

        # Start process
        self.process = subprocess.Popen(
            self.command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Start log collection
        self._running = True
        self._log_thread = threading.Thread(target=self._collect_logs)
        self._log_thread.daemon = True
        self._log_thread.start()

    def _collect_logs(self):
        """Collect logs from the process output."""
        while self._running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line:
                self.log_buffer.append(line.rstrip())

    def ready(self) -> bool:
        """Check if the model server is ready to accept requests.
        
        Returns:
            bool: True if server is ready, False otherwise
        """
        # Basic implementation - check if process is running
        # In a real implementation, you might want to check if the server
        # is actually accepting connections
        return self.process is not None and self.process.poll() is None

    def logs(self) -> List[str]:
        """Retrieve recent log output.
        
        Returns:
            List[str]: Last 100 lines of log output
        """
        return list(self.log_buffer)

    def stop(self) -> None:
        """Stop the running model server."""
        if self.process:
            self._running = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                self.process.kill()  # Force kill if not terminated
            self.process = None

        if self._log_thread:
            self._log_thread.join(timeout=1)
            self._log_thread = None

    def __str__(self) -> str:
        """Return string representation of RunningModel.
        
        Returns:
            str: Human readable running model description
        """
        status = "running" if self.ready() else "stopped"
        return (f"RunningModel({self.model.model_name} @ {self.listener}, "
                f"status={status})")
        
class Runtime:
    pass
        
class Zoo:
    pass

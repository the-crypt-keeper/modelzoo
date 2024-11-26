from dataclasses import dataclass
from typing import Any, List, Dict
import subprocess
import threading
from collections import deque
import requests
import os
import signal

class Runtime:
    pass
        
class Zoo:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def toggle(self):
        self.enabled = not self.enabled

@dataclass
class RuntimeParameter:
    """Data class describing a configurable runtime parameter."""
    param_name: str
    param_description: str
    param_type: str
    param_default: Any
    param_enum: Dict[str, Any] = None

    def __str__(self) -> str:
        enum_str = f", options={list(self.param_enum.keys())}" if self.param_enum else ""
        return f"Parameter({self.param_name}: {self.param_type}, default={self.param_default}{enum_str})"

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
    zoo_name: str
    model_id: str
    model_format: str
    model_name: str = None
    model_size: int = None
    model_architecture: str = None
    api_url: str = None
    api_key: str = None

    def __str__(self) -> str:
        parts = [f"Model({self.model_name}",
                f"format={self.model_format}"]
                
        if self.model_size is not None:         parts.append(f"size={self.model_size}")
        if self.model_architecture is not None: parts.append(f"arch={self.model_architecture}")
        if self.api_url is not None:            parts.append(f"api_url={self.api_url}")
        
        return ', '.join(parts) + ')'

class RunningModel:
    """Class representing and controlling a running model instance."""

    def __init__(self, runtime: Runtime, model: Model, environment: Environment,
                 listener: Listener, command: List[str], extra_environment: Dict[str,str] = {}):
        """Initialize a new running model instance.
        
        Args:
            model (Model): The model being served
            environment (Environment): Environment configuration
            listener (Listener): Network binding configuration
            command (List[str]): Command line as list of strings
            runtime (Runtime): The runtime that created this instance
        """
        self.model = model
        self.extra_environment = extra_environment
        self.environment = environment
        self.listener = listener
        self.command = command
        self.runtime = runtime
        self.process = None
        self.log_buffer = deque(maxlen=100)  # Keep last 100 lines
        self._log_thread = None
        self._running = False
        self._pgid = None
        
        # Seed the logs with command and environment
        self._seed_logs()
        
        # Start the process
        self._start_process()

    def _seed_logs(self):
        """Seed the logs with command and environment information."""
        self.log_buffer.append("Command: " + " ".join(self.command))
        self.log_buffer.append("Environment:")
        for key, value in self.environment.vars.items():
            self.log_buffer.append(f"  {key}={value}")
        self.log_buffer.append("---")  # Separator

    def _start_process(self):
        """Start the underlying process and log collection."""
        # Create environment dict for subprocess
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.environment.vars.items()})
        env.update(self.extra_environment)

        # Start process
        self.process = subprocess.Popen(
            self.command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create a new process group
        )

        # Store the process group ID
        self._pgid = os.getpgid(self.process.pid)

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

    def status(self) -> dict:
        """Check the status of the model.
        
        Returns:
            dict: A dictionary containing 'running' and 'ready' status
        """
        running = self.process is not None and self.process.poll() is None
        ready = running and self._is_ready()
        return {"running": running, "ready": ready}

    def _is_ready(self) -> bool:
        """Check if the model server is ready to accept requests.
        
        Returns:
            bool: True if server is ready, False otherwise
        """
        try:
            response = requests.get(f"http://{self.listener.host}:{self.listener.port}/v1/models", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def logs(self) -> List[str]:
        """Retrieve recent log output.
        
        Returns:
            List[str]: Last 100 lines of log output
        """
        return list(self.log_buffer)

    def stop(self, no_wait = False) -> None:
        """Stop the running model server and all its child processes."""
        if self.process:
            self._running = False
            try:
                os.killpg(self._pgid, signal.SIGTERM)
                if not no_wait:
                    self.process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                print('Attempting to force kill..')
                os.killpg(self._pgid, signal.SIGKILL)  # Force kill if not terminated
            except ProcessLookupError:
                print('Process group already terminated')
            self.process = None
            self._pgid = None

        if self._log_thread:
            if not no_wait:
                self._log_thread.join(timeout=1)
            self._log_thread = None

    def __str__(self) -> str:
        """Return string representation of RunningModel.
        
        Returns:
            str: Human readable running model description
        """
        status = "running" if self.ready() else "stopped"
        return (f"RunningModel({self.model.model_name} @ {self.listener}, "
                f"runtime={self.runtime.__class__.__name__}, status={status})")

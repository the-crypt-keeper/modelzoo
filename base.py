from dataclasses import dataclass
from typing import Any, List, Dict
import subprocess
import threading
from collections import deque
import requests
import os
import signal
from protocols import PROTOCOLS

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

class EnvironmentSet:
    """A set of Environment objects that can be combined."""
    
    def __init__(self, environments: List[Environment] = None):
        """Initialize an EnvironmentSet with optional list of environments.
        
        Args:
            environments (List[Environment], optional): Initial environments to add
        """
        self.environments = environments or []
    
    def add(self, environment: Environment) -> None:
        """Add an environment to the set.
        
        Args:
            environment (Environment): Environment to add
        """
        self.environments.append(environment)
    
    def get_combined_name(self) -> str:
        """Generate a combined name from all environments in the set.
        
        Returns:
            str: Combined name using + as separator
        """
        if not self.environments:
            return "empty"
        return "+".join(env.name for env in self.environments)
    
    def get_combined_vars(self) -> Dict[str, str]:
        """Merge environment variables from all environments in the set.
        
        When keys conflict, values are merged with comma separator.
        
        Returns:
            Dict[str, str]: Combined environment variables
        """
        combined_vars = {}
        
        for env in self.environments:
            for key, value in env.vars.items():
                if key in combined_vars:
                    # If key already exists, merge values with comma
                    combined_vars[key] = f"{combined_vars[key]},{value}"
                else:
                    combined_vars[key] = value
        
        return combined_vars
    
    def __str__(self) -> str:
        """Return string representation of EnvironmentSet.
        
        Returns:
            str: Human readable environment set description
        """
        if not self.environments:
            return "EnvironmentSet(empty)"
        
        env_strs = [str(env) for env in self.environments]
        return f"EnvironmentSet({', '.join(env_strs)})"

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

    def __init__(self, runtime: Runtime, model: Model, environment: EnvironmentSet,
                 listener: Listener, command: List[str], extra_environment: Dict[str,str] = {},
                 working_directory: str = None):
        """Initialize a new running model instance.
        
        Args:
            model (Model): The model being served
            environment (EnvironmentSet): Environment configuration set
            listener (Listener): Network binding configuration
            command (List[str]): Command line as list of strings
            runtime (Runtime): The runtime that created this instance
            extra_environment (Dict[str,str], optional): Additional environment variables
            working_directory (str, optional): Working directory for the process
        """
        self.model = model
        self.extra_environment = extra_environment
        self.environment_set = environment
        self.listener = listener
        self.command = command
        self.runtime = runtime
        self.working_directory = working_directory
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
        env_vars = self.environment_set.get_combined_vars()
        for key, value in env_vars.items():
            self.log_buffer.append(f"  {key}={value}")
        self.log_buffer.append("---")  # Separator

    def _start_process(self):
        """Start the underlying process and log collection."""
        # Create environment dict for subprocess
        env = os.environ.copy()
        env_vars = self.environment_set.get_combined_vars()
        env.update({k: str(v) for k, v in env_vars.items()})
        env.update(self.extra_environment)
        env.update({'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})

        # Start process
        self.process = subprocess.Popen(
            self.command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid,  # Create a new process group
            cwd=self.working_directory
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
            protocol = self.listener.protocol               
            protocol_def = PROTOCOLS.get(protocol, {})
            
            health_check = protocol_def.get('health_check')
            health_status = protocol_def.get('health_status', 200)
            
            if not health_check:
                return False
                
            response = requests.get(
                f"http://{self.listener.host}:{self.listener.port}{health_check}", 
                timeout=2
            )
            return response.status_code == health_status
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
        status = "running" if self.status()["ready"] else "stopped"
        wd = f", cwd={self.working_directory}" if self.working_directory else ""
        env_name = self.environment_set.get_combined_name()
        return (f"RunningModel({self.model.model_name} @ {self.listener}, "
                f"runtime={self.runtime.__class__.__name__}, env={env_name}, status={status}{wd})")

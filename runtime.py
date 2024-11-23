from base import *
from typing import Any, List, Dict

class LlamaRuntime(Runtime):
    """Runtime implementation for llama.cpp server."""

    def __init__(self, name:str, bin_path: str):
        """Initialize LlamaRuntime with path to llama-server binary.
        
        Args:
            bin_path (str): Path to llama-server executable
        """
        self.runtime_name = name
        self.runtime_formats = ["gguf"]
        self.bin_path = bin_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="context",
                param_description="Context size",
                param_type="enum",
                param_default="4K",
                param_enum={
                    "4K": 4096,
                    "6K": 6144,
                    "8K": 8192,
                    "16K": 16384,
                    "32K": 32768
                }
            ),
            RuntimeParameter(
                param_name="num_gpu_layers",
                param_description="Number of GPU layers",
                param_type="int",
                param_default=999
            ),
            RuntimeParameter(
                param_name="split_mode",
                param_description="Split mode for model",
                param_type="str",
                param_default="row"
            ),
            RuntimeParameter(
                param_name="flash_attention",
                param_description="Enable flash attention",
                param_type="bool",
                param_default=True
            ),
            RuntimeParameter(
                param_name="extra_args",
                param_description="Optional additional arguments to the binary",
                param_type="str",
                param_default=""
            )
        ]

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:
        """Spawn a llama.cpp server instance.
        
        Args:
            environment (Environment): Environment configuration
            listener (Listener): Network binding configuration
            model (Model): Model to serve
            param_list (Dict[str, Any]): Runtime parameters

        Returns:
            RunningModel: Handle to the running instance
        
        Raises:
            ValueError: If model format is not supported
        """
        if model.model_format != "gguf":
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Build command line
        context_param = next(param for param in self.runtime_params if param.param_name == "context")
        context_value = context_param.param_enum[param_list.get("context", "4K")]
        cmd = [
            self.bin_path,
            "-m", model.model_id,
            "-c", str(context_value),
            "-ngl", str(param_list.get("num_gpu_layers", 999)),
            "-sm", str(param_list.get("split_mode", "row")),
            "--host", listener.host,
            "--port", str(listener.port)
        ]

        # Add flash attention if enabled
        if param_list.get("flash_attention", True):
            cmd.append("-fa")

        # Add extra arguments if provided
        extra_args = param_list.get("extra_args", "").strip()
        if extra_args:
            cmd.extend(extra_args.split())

        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=cmd
        )

class KoboldCppRuntime(Runtime):
    """Runtime implementation for KoboldCpp server."""

    def __init__(self, name:str, bin_path: str):
        """Initialize KoboldCppRuntime with path to koboldcpp binary.
        
        Args:
            bin_path (str): Path to koboldcpp executable
        """
        self.runtime_name = name
        self.runtime_formats = ["gguf"]
        self.bin_path = bin_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="contextsize",
                param_description="Context size",
                param_type="enum",
                param_default="4K",
                param_enum={
                    "4K": 4096,
                    "6K": 6144,
                    "8K": 8192,
                    "16K": 16384,
                    "32K": 32768
                }
            ),
            RuntimeParameter(
                param_name="gpulayers",
                param_description="Number of GPU layers",
                param_type="int",
                param_default=-1
            ),
            RuntimeParameter(
                param_name="flashattention",
                param_description="Enable flash attention",
                param_type="bool",
                param_default=True
            ),
            RuntimeParameter(
                param_name="quantkv",
                param_description="KV cache data type quantization",
                param_type="enum",
                param_default="f16",
                param_enum={
                    "f16": 0,
                    "q8": 1,
                    "q4": 2
                }
            ),
            RuntimeParameter(
                param_name="extra_args",
                param_description="Optional additional arguments to the binary",
                param_type="str",
                param_default=""
            )
        ]

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:
        """Spawn a KoboldCpp server instance.
        
        Args:
            environment (Environment): Environment configuration
            listener (Listener): Network binding configuration
            model (Model): Model to serve
            param_list (Dict[str, Any]): Runtime parameters

        Returns:
            RunningModel: Handle to the running instance
        
        Raises:
            ValueError: If model format is not supported
        """
        if model.model_format != "gguf":
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Build command line
        context_param = next(param for param in self.runtime_params if param.param_name == "contextsize")
        context_value = context_param.param_enum[param_list.get("contextsize", "4K")]
        cmd = [
            self.bin_path,
            "--model", model.model_id,
            "--contextsize", str(context_value),
            "--gpulayers", str(param_list.get("gpulayers", -1)),
            "--host", listener.host,
            "--port", str(listener.port),
            "--usecublas"
        ]

        # Add flash attention if enabled
        if param_list.get("flashattention", False):
            cmd.append("--flashattention")

        # Add quantkv parameter
        quantkv_param = next(param for param in self.runtime_params if param.param_name == "quantkv")
        quantkv_value = quantkv_param.param_enum[param_list.get("quantkv", "f16")]
        cmd.extend(["--quantkv", str(quantkv_value)])

        # Add extra arguments if provided
        extra_args = param_list.get("extra_args", "").strip()
        if extra_args:
            cmd.extend(extra_args.split())

        return RunningModel(
            runtime=self,            
            model=model,
            environment=environment,
            listener=listener,
            command=cmd
        )

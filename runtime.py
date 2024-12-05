import os
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
        if model.model_format not in self.runtime_formats:
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

class LlamaSrbRuntime(Runtime):
    """Runtime implementation for llama-srb API server."""

    def __init__(self, name: str, script_path: str):
        """Initialize LlamaSrbRuntime with path to api.py script.
        
        Args:
            name (str): Name of the runtime
            script_path (str): Path to api.py script
        """
        self.runtime_name = name
        self.runtime_formats = ["gguf"]
        self.script_path = script_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="ctx",
                param_description="Total Context size",
                param_type="enum",
                param_default="8K",
                param_enum={
                    "4K": 4096,
                    "6K": 6144,
                    "8K": 8192,
                    "12K": 12*1024,
                    "16K": 16*1024,
                    "24K": 24*1024,
                    "32K": 32*1024
                }
            ),
            RuntimeParameter(
                param_name="batch_size",
                param_description="Number of completions to run in parallel",
                param_type="int",
                param_default=4
            )
        ]

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:

        if model.model_format not in self.runtime_formats:
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Build command line
        ctx_param = next(param for param in self.runtime_params if param.param_name == "ctx")
        ctx_value = ctx_param.param_enum[param_list.get("ctx", "8K")]
        cmd = [
            "python",
            self.script_path,
            "--model", model.model_id,
            "--port", str(listener.port),
            "--ctx", str(ctx_value),
            "--n", str(param_list.get("batch_size", 4))
        ]

        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=cmd
        )

class LiteLLMRuntime(Runtime):
    """Runtime implementation for LiteLLM server."""

    def __init__(self, name: str, bin_path: str = "litellm"):
        """Initialize LiteLLMRuntime with path to litellm binary.
        
        Args:
            name (str): Name of the runtime
            bin_path (str): Path to litellm executable, defaults to 'litellm'
        """
        self.runtime_name = name
        self.runtime_formats = ["litellm"]
        self.bin_path = bin_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="drop_params",
                param_description="Drop unmapped parameters",
                param_type="bool",
                param_default=False
            ),
            RuntimeParameter(
                param_name="max_tokens",
                param_description="Set max tokens for the model",
                param_type="str",
                param_default=''
            )
        ]

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:
        """Spawn a LiteLLM server instance.
        
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
        if model.model_format not in self.runtime_formats:
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Build command line
        cmd = [
            self.bin_path,
            "-m", model.model_id,
            "--alias", model.model_name,
            "--host", listener.host,
            "--port", str(listener.port)
        ]

        if model.api_url:
            cmd.extend(["--api_base", model.api_url])
            
        # Add drop_params if enabled
        if param_list.get("drop_params", False):
            cmd.append("--drop_params")

        # Add max_tokens if provided
        max_tokens = param_list.get("max_tokens",'')
        if max_tokens != '':
            cmd.extend(["--max_tokens", max_tokens])

        # Prepare extra environment variables
        extra_env = {"OPENAI_API_KEY": model.api_key} if model.api_key else {}

        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=cmd,
            extra_environment=extra_env
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

class TabbyRuntime(Runtime):
    """Runtime implementation for TabbyAPI server."""

    def __init__(self, name: str, script_path: str):
        """Initialize TabbyRuntime with path to TabbyAPI shell script.
        
        Args:
            script_path (str): Path to TabbyAPI shell script
        """
        self.runtime_name = name
        self.runtime_formats = ["gptq","exl2"]
        self.script_path = script_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="max_seq_len",
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
                param_name="tensor_parallel",
                param_description="Enable tensor parallelism",
                param_type="bool",
                param_default=False
            ),
            RuntimeParameter(
                param_name="cache_mode",
                param_description="KV Cache mode for VRAM savings",
                param_type="enum",
                param_default="FP16",
                param_enum={
                    "FP16": "FP16",
                    "Q8": "Q8",
                    "Q6": "Q6",
                    "Q4": "Q4"
                }
            ),
            RuntimeParameter(
                param_name="disable_auth",
                param_description="Disable authentication",
                param_type="bool",
                param_default=True
            ),
            RuntimeParameter(
                param_name="gpu_split",
                param_description="GPU split configuration",
                param_type="str",
                param_default=""
            ),
            RuntimeParameter(
                param_name="extra_args",
                param_description="Optional additional arguments to the script",
                param_type="str",
                param_default=""
            )            
        ]

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:
        """Spawn a TabbyAPI server instance.
        
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
        if model.model_format not in self.runtime_formats:
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Build command line
        max_seq_len_param = next(param for param in self.runtime_params if param.param_name == "max_seq_len")
        max_seq_len_value = max_seq_len_param.param_enum[param_list.get("max_seq_len", "4K")]
        cmd = [
            self.script_path,
            "--model-name", model.model_id,
            "--max-seq-len", str(max_seq_len_value),
            "--host", listener.host,
            "--port", str(listener.port),
            "--cache-mode", param_list.get("cache_mode", "FP16")
        ]
        
        # Add tensor parallel if enabled
        if param_list.get("tensor_parallel", False):
            cmd.extend(["--tensor-parallel", "True"])

        # Add disable auth if enabled
        if param_list.get("disable_auth", True):
            cmd.extend(["--disable-auth", "True"])

        # Add GPU split if provided
        gpu_split = param_list.get("gpu_split", "").strip()
        if gpu_split:
            cmd.extend(["--gpu-split", gpu_split,"--gpu-split-auto", "False"])

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

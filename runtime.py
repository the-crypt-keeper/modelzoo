import os
import tempfile
import json
from pathlib import Path
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

        listener.protocol = 'openai'
        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=cmd
        )

class VLLMRuntime(Runtime):
    """Runtime implementation for vLLM inference server."""

    def __init__(self, name: str, venv_path: str):
        """Initialize VLLMRuntime with path to virtual environment.
        
        Args:
            name (str): Name of the runtime
            venv_path (str): Path to virtual environment containing vllm
        """
        self.runtime_name = name
        self.runtime_formats = ["gguf", "fp16", "awq", "gptq"]
        self.venv_path = venv_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="max_model_len",
                param_description="Maximum sequence length",
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
                param_name="tensor_parallel_size",
                param_description="Number of GPUs for tensor parallelism",
                param_type="int",
                param_default=1
            ),
            RuntimeParameter(
                param_name="gpu_memory_utilization",
                param_description="Target GPU memory utilization",
                param_type="float",
                param_default=0.95
            ),
            RuntimeParameter(
                param_name="enforce_eager",
                param_description="Enforce eager execution mode",
                param_type="bool",
                param_default=True
            )
        ]

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:
        """Spawn a vLLM server instance.
        
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
        max_len_param = next(param for param in self.runtime_params if param.param_name == "max_model_len")
        max_len_value = max_len_param.param_enum[param_list.get("max_model_len", "4K")]
        
        # Build the vllm command
        vllm_cmd = (
            f"vllm serve {model.model_id} "
            f"--host {listener.host} "
            f"--port {listener.port} "
            f"--tensor-parallel-size {param_list.get('tensor_parallel_size', 1)} "
            f"--max-model-len {max_len_value} "
            f"--gpu-memory-utilization {param_list.get('gpu_memory_utilization', 0.95)}"
        )

        if param_list.get('enforce_eager', True):
            vllm_cmd += " --enforce-eager"

        # Create temporary shell script
        script_fd, script_path = tempfile.mkstemp(prefix='vllm_', suffix='.sh')
        with os.fdopen(script_fd, 'w') as f:
            f.write(f"""#!/bin/bash
source {self.venv_path}/bin/activate
echo env: $CUDA_VISIBLE_DEVICES $CUDA_DEVICE_ORDER
{vllm_cmd}
""")
        os.chmod(script_path, 0o755)
        listener.protocol = 'openai'
        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=[script_path]
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
            self.script_path,
            "--model", model.model_id,
            "--port", str(listener.port),
            "--ctx", str(ctx_value),
            "--n", str(param_list.get("batch_size", 4))
        ]

        # Get the directory containing the script
        working_dir = os.path.dirname(os.path.abspath(self.script_path))

        listener.protocol = 'openai'
        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=cmd,
            working_directory=working_dir
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
        listener.protocol = 'openai'
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
        self.runtime_formats = ["gguf","kcppt"]
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
        if model.model_format not in self.runtime_formats:
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Build command line
        context_param = next(param for param in self.runtime_params if param.param_name == "contextsize")
        context_value = context_param.param_enum[param_list.get("contextsize", "4K")]
        
        model_spec = ["--model", model.model_id] if model.model_format == "gguf" else [model.model_id]
        
        cmd = [self.bin_path] + model_spec + [
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
            
        # set working dir to where the model is (for kcppt)
        working_dir = os.path.dirname(os.path.abspath(model.model_id))
        
        # Set protocol based on model type and configuration
        if model.model_format == "kcppt":
            # Load and parse the checkpoint file
            with open(model.model_id, 'r') as f:
                config = json.load(f)
            # If it has an SD model, it's an SD checkpoint
            if config.get('sdmodel'):
                listener.protocol = 'a1111'
            else:
                listener.protocol = 'openai'
        else:
            listener.protocol = 'openai'

        return RunningModel(
            runtime=self,            
            model=model,
            environment=environment,
            listener=listener,
            command=cmd,
            working_directory=working_dir
        )

class SDServerRuntime(Runtime):
    """Runtime implementation for SD Server."""

    def __init__(self, name: str, bin_path: str):
        """Initialize SDServerRuntime with path to sd-server binary.
        
        Args:
            name (str): Name of the runtime
            bin_path (str): Path to sd-server executable
        """
        self.runtime_name = name
        self.runtime_formats = ["kcppt"]
        self.bin_path = bin_path
        
        # Define available parameters
        self.runtime_params = [
            RuntimeParameter(
                param_name="sampler_name",
                param_description="Sampling method",
                param_type="enum",
                param_default="Euler",
                param_enum={
                    "Euler": "euler",
                    "Euler A": "euler_a", 
                    "Heun": "heun",
                    "DPM2": "dpm2",
                    "DPM++": "dpmpp_2m",
                    "LCM": "lcm"
                }
            ),
            RuntimeParameter(
                param_name="cfg_scale",
                param_description="CFG Scale",
                param_type="float",
                param_default=1.0
            ),
            RuntimeParameter(
                param_name="steps",
                param_description="Number of sampling steps",
                param_type="int",
                param_default=1
            ),
            RuntimeParameter(
                param_name="extra_args",
                param_description="Optional additional arguments to the binary",
                param_type="str",
                param_default=""
            )
        ]

    def _get_model_path(self, model_file: str, base_dir: str) -> str:
        """Get the full path for a model file.
        
        Args:
            model_file (str): Model file or URL
            base_dir (str): Base directory to look for local files
            
        Returns:
            str: Full path to the model file
        """
        if not model_file:
            return ""
            
        # Extract filename from URL if needed
        filename = model_file.split('/')[-1].split('?')[0]
        
        # First check if model_file is an absolute path
        abs_path = Path(model_file)
        if abs_path.is_absolute() and abs_path.exists():
            return str(abs_path)

        # Then check relative to base directory
        model_path = Path(base_dir) / filename
        if model_path.exists():
            return str(model_path)
            
        return ""

    def spawn(self, 
              environment: Environment, 
              listener: Listener, 
              model: Model, 
              param_list: dict[str, Any]) -> RunningModel:
        """Spawn a SD Server instance.
        
        Args:
            environment (Environment): Environment configuration
            listener (Listener): Network binding configuration
            model (Model): Model to serve
            param_list (Dict[str, Any]): Runtime parameters

        Returns:
            RunningModel: Handle to the running instance
        """
        if model.model_format not in self.runtime_formats:
            raise ValueError(f"Unsupported model format: {model.model_format}")

        # Load and parse the checkpoint file
        with open(model.model_id, 'r') as f:
            config = json.load(f)
            
        # Get the base directory for resolving relative paths
        base_dir = os.path.dirname(os.path.abspath(model.model_id))
        
        # Build command line
        cmd = [
            self.bin_path,
            "--host", listener.host,
            "--port", str(listener.port),
            "-v"
        ]
        
        # Add required diffusion model
        diffusion_model = self._get_model_path(config.get('sdmodel'), base_dir)
        if not diffusion_model:
            raise ValueError("No diffusion model specified in checkpoint")
        
        if 'flux' in diffusion_model:
            cmd.extend(["--diffusion-model", diffusion_model])
        else:
            cmd.extend(["-m", diffusion_model])
        
        # Add optional models if present
        if config.get('sdt5xxl'):
            t5xxl = self._get_model_path(config.get('sdt5xxl'), base_dir)        
            cmd.extend(["--t5xxl", t5xxl])
        
        if config.get('sdclipl'):
            clip_l = self._get_model_path(config.get('sdclipl'), base_dir)
            cmd.extend(["--clip_l", clip_l])
        
        if config.get('sdvae'):
            vae = self._get_model_path(config.get('sdvae'), base_dir)
            cmd.extend(["--vae", vae])

        # Add runtime parameters
        sampling_param = next(param for param in self.runtime_params if param.param_name == "sampler_name")
        sampling_value = sampling_param.param_enum[param_list.get("sampler_name", "Euler")]
        cmd.extend(["--sampling-method", sampling_value])
        
        cmd.extend(["--cfg-scale", str(param_list.get("cfg_scale", 1.0))])
        cmd.extend(["--steps", str(param_list.get("steps", 1))])
        cmd.extend(["-p", "default prompt"])  # Always pass default prompt

        # Add extra arguments if provided
        extra_args = param_list.get("extra_args", "").strip()
        if extra_args:
            cmd.extend(extra_args.split())

        listener.protocol = 'sd-server'
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

        listener.protocol = 'openai'
        return RunningModel(
            runtime=self,
            model=model,
            environment=environment,
            listener=listener,
            command=cmd
        )

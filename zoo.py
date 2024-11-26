from base import *
from pathlib import Path
from typing import Any, List, Dict
import json
import shutil
import requests
import os

class StaticZoo(Zoo):
    """Zoo implementation that returns a static list of Models."""

    def __init__(self, name: str, models: List[Dict]):
        """Initialize a StaticZoo with a specific list of models.
        
        Args:
            name (str): Name of the zoo
            models (List[Model]): List of Model instances
        """
        super().__init__(name)
        for m in models:
            if 'model_name' not in m: m['model_name'] = m['model_id']
        self.models = [Model(zoo_name=name, **m) for m in models]

    def catalog(self) -> List[Model]:
        return self.models

    def __str__(self) -> str:
        return f"StaticZoo(name={self.name}, models={len(self.models)})"

class FolderZoo(Zoo):
    """Zoo implementation that discovers GGUF models in a filesystem folder."""

    def __init__(self, name: str, path: str):
        """Initialize a FolderZoo with a specific path.
        
        Args:
            name (str): Name of the zoo
            path (str): Path to folder containing models
        """
        super().__init__(name)
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not self.path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

    def _process_multipart_models(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group multi-part model files together.
        
        Args:
            files (List[Path]): List of .gguf files

        Returns:
            Dict[str, List[Path]]: Dictionary mapping base names to file parts
        """
        model_parts = {}
        
        for file_path in files:
            name = file_path.stem
            
            # Check if this is a multi-part file
            if "-of-" in name:
                # Extract base name (everything before the part number)
                base_name = name.split("-00")[0]
                
                # Initialize list for this base name if not exists
                if base_name not in model_parts:
                    model_parts[base_name] = []
                    
                model_parts[base_name].append(file_path)
            else:
                # Single file model
                model_parts[name] = [file_path]
                
        return model_parts

    def _gguf_catalog(self) -> List[Model]:
        """Scan folder path and return list of discovered GGUF models."""
        models = []
        
        # Get all .gguf files
        gguf_files = list(self.path.rglob("*.gguf"))
        
        # Group multi-part files
        model_parts = self._process_multipart_models(gguf_files)
        
        # Process each model
        for base_name, parts in model_parts.items():
            try:
                # Calculate total size across all parts
                total_size = sum(part.stat().st_size for part in parts)
                
                # For multi-part models, use the first part as the model_id
                model_id = str(sorted(parts)[0].absolute())
                
                # Create model instance
                model = Model(
                    zoo_name=self.name,                    
                    model_id=model_id,
                    model_format="gguf",
                    model_name=base_name,
                    model_size=total_size
                )
                models.append(model)
                
            except Exception as e:
                print(f"Warning: Error processing GGUF model {base_name}: {e}")
                continue
                
        return models

    def _hf_catalog(self) -> List[Model]:
        """Scan folder path and return list of discovered HF models."""
        models = []
        
        # Get all directories
        for dir_path in self.path.iterdir():
            if dir_path.is_dir():
                config_file = dir_path / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        quant_config = config.get('quantization_config', {})
                        model_format = quant_config.get('quant_method', 'unknown')
                        
                        # If model_format is still unknown, apply heuristics
                        if model_format == 'unknown':
                            folder_name = dir_path.name.lower()
                            if 'gptq' in folder_name:
                                model_format = 'gptq'
                            elif 'awq' in folder_name:
                                model_format = 'awq'
                            elif 'exl2' in folder_name:
                                model_format = 'exl2'
                            elif 'fp16' in folder_name:
                                model_format = 'fp16'
                                
                        # Calculate total size of the folder
                        total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                        
                        model = Model(
                            zoo_name=self.name,
                            model_id=str(dir_path.absolute()),
                            model_format=model_format,
                            model_name=dir_path.name,
                            model_size=total_size
                        )
                        models.append(model)
                    except Exception as e:
                        print(f"Warning: Error processing HF model {dir_path.name}: {e}")
                        continue
        
        return models

    def catalog(self) -> List[Model]:
        """Scan folder path and return list of discovered GGUF and HF models.
        
        Returns:
            List[Model]: List of models found in the folder
        """
        gguf_models = self._gguf_catalog()
        hf_models = self._hf_catalog()
        return gguf_models + hf_models

    def __str__(self) -> str:
        return f"FolderZoo(path={self.path})"

class OpenAIZoo(Zoo):
    """Zoo implementation that fetches models from an OpenAI-compatible API."""

    def __init__(self, name: str, api_url: str, api_key: str = None, api_key_env: str = None, cache: bool = True, models: List[str] = None):
        """Initialize an OpenAIZoo with API details.
        
        Args:
            name (str): Name of the zoo
            api_url (str): Base URL of the OpenAI-compatible API
            api_key (str, optional): API key for authentication
            api_key_env (str, optional): Environment variable name containing the API key
            cache (bool): Whether to cache the model list (default: True)
            models (List[str]): Optional list of models to override API exploration
        """
        super().__init__(name)
        self.api_url = api_url.rstrip('/')
        
        if api_key_env:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise ValueError(f"Environment variable '{api_key_env}' not found or empty")
        else:
            self.api_key = api_key
        
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through an environment variable")
        
        self.cache = cache
        self._cached_models = None
        self.models = models

    def catalog(self) -> List[Model]:
        """Fetch and return list of available models from the API or use provided models.
        
        Returns:
            List[Model]: List of models available through the API or provided models
        """
        if self.models is not None:
            return [Model(
                zoo_name=self.name,
                model_id='openai/'+m,
                model_format="litellm",
                model_name=m.split('/')[-1].replace('.gguf', '').replace(' ','-'),
                api_url=self.api_url,
                api_key=self.api_key
            ) for m in self.models]

        if self.cache and self._cached_models is not None:
            return self._cached_models

        try:
            response = requests.get(
                f"{self.api_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            if response.status_code != 200: print(response.text)
            data = response.json()
            if isinstance(data, list): data = { 'data': data}
            
            models = []
            for model_data in data.get('data', []):
                model_id = model_data['id']
                if 'azureml://' in model_id: model_id = model_data['name']
                model_name = model_id.split('/')[-1].replace('.gguf', '').replace(' ','-')

                model = Model(
                    zoo_name=self.name,
                    model_id='openai/'+model_id,
                    model_format="litellm",
                    model_name=model_name,
                    api_url=self.api_url,
                    api_key=self.api_key
                )
                models.append(model)
            
            if self.cache:
                self._cached_models = models
            
            return models
        except requests.RequestException as e:
            print(f"Error fetching models from API: {e}")
            return []

    def __str__(self) -> str:
        return f"OpenAIZoo(api_url={self.api_url}, cache={self.cache}, models_override={'Yes' if self.models else 'No'})"

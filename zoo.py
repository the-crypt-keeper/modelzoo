from base import *
from pathlib import Path
from typing import Any, List, Dict
import json
import shutil

class FolderZoo(Zoo):
    """Zoo implementation that discovers GGUF models in a filesystem folder."""

    def __init__(self, path: str):
        """Initialize a FolderZoo with a specific path.
        
        Args:
            path (str): Path to folder containing models
        """
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
                        
                        # Calculate total size of the folder
                        total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                        
                        model = Model(
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

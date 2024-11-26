# ModelZoo

ModelZoo is a system for managing and serving local Large Language Models (LLMs). It provides a flexible framework for discovering, launching, and managing various LLM models using different runtimes and environments.

## Overview

ModelZoo is composed of several key components:

1. **Zoos**: Factories that discover and catalog models.
2. **Models**: Data objects representing LLMs.
3. **Runtimes**: Backends that can serve models in specific environments.
4. **Environments**: Configurations for running models.
5. **ZooKeeper**: The main application that manages zoos, runtimes, and provides a web-based user interface.

## Key Components

### Zoos

Zoos are responsible for discovering and cataloging models. The system supports different types of zoos:

- **FolderZoo**: Discovers models in a specified file system folder.
- **StaticZoo**: Returns a predefined list of models.
- **OpenAIZoo**: Fetches models from an OpenAI-compatible API.

### Models

Models are data objects representing LLMs. They have attributes such as:

- `model_id`: Unique identifier for the model.
- `model_format`: Format of the model (e.g., "gguf", "gptq", "exl2").
- `model_name`: Friendly display name.
- `model_size`: Size of the model.
- `model_architecture`: Architecture of the model.

### Runtimes

Runtimes are responsible for serving models. They include:

- **LlamaRuntime**: For serving llama.cpp models.
- **LiteLLMRuntime**: For serving models via LiteLLM.
- **KoboldCppRuntime**: For serving models using KoboldCpp.
- **TabbyRuntime**: For serving models using TabbyAPI.

Each runtime defines compatible model formats and configurable parameters.

### Environments

Environments are configurations for running models, typically including environment variables like `CUDA_VISIBLE_DEVICES`.

### ZooKeeper

ZooKeeper is the main application that:

1. Loads configuration from a YAML file.
2. Instantiates zoos and runtimes.
3. Provides a web-based user interface for:
   - Listing available models from all zoos.
   - Launching models with specific runtimes and configurations.
   - Managing running models (viewing logs, stopping models).

## Additional Features

### Proxy Server

The system includes a proxy server (`proxy.py`) that forwards requests to the appropriate running model, allowing unified access to all launched models.

### Model History

ModelZoo keeps track of model launch history, including:

- Number of times a model has been launched.
- Last launch time.
- Last used runtime and environment.
- Last used parameters.

This history is used to provide better user experience by pre-filling launch configurations based on previous usage.

## Configuration

ModelZoo is configured using a YAML file that defines:

- Zoos to be instantiated and their configurations.
- Runtimes to be made available.
- Predefined environments.

Example configuration:

```yaml
zoos:
   - name: SSDZoo
     class: FolderZoo
     params:
        path: /mnt/ssd0
   - name: NVMEZoo
     class: FolderZoo
     params:
        path: /mnt/nvme0

runtimes:
   - name: LlamaRuntime
     params:
       bin: /home/mike/work/llama-rpc/bin/llama-server
       
envs:
   - name: "P40[0]"
     vars:
        CUDA_VISIBLE_DEVICES: 0
   - name: "P40[0,1]"
     vars:
        CUDA_VISIBLE_DEVICES: 0,1
```

## Usage

1. Configure the system using the YAML configuration file.
2. Run the ZooKeeper application.
3. Access the web interface to:
   - View available models.
   - Launch models with specific configurations.
   - Manage running models.

## Dependencies

- Flask
- PyYAML
- Requests
- ASGI server (e.g., uvicorn)

## Getting Started

1. Clone the repository.
2. Install dependencies.
3. Create a configuration YAML file.
4. Run the ZooKeeper application.
5. Access the web interface to start managing your models.

## Running the Application

To run the ModelZoo application, use the following command:

```
python ./main.py --config config.yaml
```

Replace `config.yaml` with the path to your configuration file if it's different.

For more detailed information, refer to the source code and comments in `base.py`, `zk.py`, and other module files.

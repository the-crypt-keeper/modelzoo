# ModelZoo

ModelZoo is a system for managing and serving local Large Language Models (LLMs). It provides a flexible framework for discovering, launching, and managing various LLM models using different runtimes and environments.

## Overview

ModelZoo is composed of several key components:

1. **Zoos**: Discovery systems that catalog models.
2. **Models**: Data objects representing LLMs.
3. **Runtimes**: Backends that can serve models in specific environments.
4. **Environments**: GPU Configurations (environment variables) for running models.
5. **Proxy**: A component that forwards requests to the appropriate running model, allowing unified access to all launched models.
6. **ModelHistory**: Keeps track of model launch history, including frequency of use and last used configurations.
7. **ZooKeeper**: Web application to interact with zoos, use runtimes to spawn models, interface with history and host the proxy.

## Key Components

### Zoos

Zoos are responsible for discovering and cataloging models. The system supports different types of zoos:

1. **FolderZoo**: Discovers models in a specified file system folder.
   - Parameters:
     - `name` (str): Name of the zoo
     - `path` (str): Path to folder containing models
   - Example:
     ```yaml
     - name: LocalModels
       class: FolderZoo
       params:
         path: /path/to/models
     ```

2. **StaticZoo**: Returns a predefined list of models.
   - Parameters:
     - `name` (str): Name of the zoo
     - `models` (List[Dict]): List of model dictionaries
   - Example:
     ```yaml
     - name: PredefinedModels
       class: StaticZoo
       params:
         models:
           - model_id: chatgpt
             model_name: ChatGPT
             model_format: litellm
     ```

3. **OpenAIZoo**: Fetches models from an OpenAI-compatible API.
   - Parameters:
     - `name` (str): Name of the zoo
     - `api_url` (str): Base URL of the OpenAI-compatible API
     - `api_key` (str, optional): API key for authentication
     - `api_key_env` (str, optional): Environment variable name containing the API key
     - `cache` (bool): Whether to cache the model list (default: True)
     - `models` (List[str], optional): Optional list of models to override API exploration
   - Example:
     ```yaml
     - name: OpenAIModels
       class: OpenAIZoo
       params:
         api_url: https://api.openai.com/v1
         api_key_env: OPENAI_API_KEY
         cache: true
     ```

Each zoo type is designed to accommodate different model discovery and management needs, allowing for flexibility in how models are sourced and cataloged within the ModelZoo system.

### Models

Models are data objects representing LLMs. They have attributes such as:

- `model_id`: Unique identifier for the model.
- `model_format`: Format of the model (e.g., "gguf", "gptq", "exl2").
- `model_name`: Friendly display name.
- `model_size`: Size of the model.
- `model_architecture`: Architecture of the model.

### Runtimes

Runtimes are responsible for serving models. They include:

1. **LlamaRuntime**: For serving llama.cpp models.
   - Compatible model formats: gguf
   - Parameters:
     - `name` (str): Name of the runtime
     - `bin_path` (str): Path to the llama.cpp server binary
   - Example:
     ```yaml
     - name: LlamaRuntime
       class: LlamaRuntime
       params:
         bin_path: /path/to/llama-server
     ```

2. **LiteLLMRuntime**: For serving models via LiteLLM.
   - Compatible model formats: litellm
      - All formats supported by LiteLLM (including OpenAI, Azure, Anthropic, and various open-source models)
   - Parameters:
     - `name` (str): Name of the runtime
     - `bin_path` (str, optional): Path to the LiteLLM binary (default: "litellm")
   - Example:
     ```yaml
     - name: LiteLLMRuntime
       class: LiteLLMRuntime
       params:
         bin_path: litellm
     ```

3. **KoboldCppRuntime**: For serving models using KoboldCpp.
   - Compatible model formats: gguf
   - Parameters:
     - `name` (str): Name of the runtime
     - `bin_path` (str): Path to the KoboldCpp binary
   - Example:
     ```yaml
     - name: KoboldCppRuntime
       class: KoboldCppRuntime
       params:
         bin_path: /path/to/koboldcpp
     ```

4. **TabbyRuntime**: For serving models using TabbyAPI.
   - Compatible model formats: gptq, exl2
   - Parameters:
     - `name` (str): Name of the runtime
     - `script_path` (str): Path to the TabbyAPI start.sh script
   - Example:
     ```yaml
     - name: TabbyRuntime
       class: TabbyRuntime
       params:
         script_path: /path/to/tabby_api/start.sh
     ```

Each runtime defines compatible model formats and configurable parameters. When launching a model, you can specify additional runtime-specific parameters as needed. The choice of runtime depends on the model format and the specific features required for your use case.

### Environments

Environments are configurations for running models, typically including environment variables like `CUDA_VISIBLE_DEVICES`.

Example:

```yaml
   - name: "RTX3090"
     vars:
        CUDA_VISIBLE_DEVICES: 0
```

Multiple enviroments may be pre-defined in the configuration file.

### ZooKeeper

ZooKeeper is the main application that:

1. Loads configuration from a YAML file.
2. Instantiates zoos and runtimes.
3. Provides a web-based user interface for:
   - Listing available models from all zoos.
   - Launching models with specific runtimes and configurations.
   - Managing running models (viewing logs, stopping models).
4. Includes a proxy server (`proxy.py`) that forwards requests to the appropriate running model, allowing unified access to all launched models.
5. Keeps track of model launch history
   - Number of times a model has been launched, and the last launch time (to sort models by most frequently used)
   - Last used enviroment and parameters (provides a better user experience by pre-filling launch configurations based on previous usage)

## Configuration

ModelZoo is configured using a YAML file that defines:

- Zoos to be instantiated and their configurations.
- Runtimes to be made available.
- Predefined environments.

Example configuration:

```yaml
zoos:
   - name: SSD
     class: FolderZoo
     params:
        path: /mnt/ssd0

runtimes:
   - name: LlamaRuntime
     class: LlamaRuntime
     params:
       bin_path: /home/mike/work/llama.cpp/llama-server
       
envs:
   - name: "P40/0"
     vars:
        CUDA_VISIBLE_DEVICES: 0
   - name: "P40/0+1"
     vars:
        CUDA_VISIBLE_DEVICES: 0,1
   - name: "No GPU"
     vars: {}        
```

This example assumes you have some `*.gguf` files under /mnt/ssd0 and that you have a compiled llama.cpp server binary at the specified path.

## Getting Started

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `config.yaml` YAML file.
4. Run the ZooKeeper application:
   ```
   python ./main.py --config config.yaml
   ```
5. Access the web interface (listening at http://0.0.0.0:3333/ by default) to:
   - View available models.
   - Launch models with specific configurations.
   - Manage running models (view logs, stop).

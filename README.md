# ModelZoo

ModelZoo is a system for managing and serving local AI Models. It provides a flexible framework for discovering, launching, and managing Language, Vision and Image generation models using different runtimes and environments.

## ZooKeeper

ZooKeeper is the entry-point of ModelZoo. It's a flask application that:

1. Loads configuration from a YAML file.
2. Instantiates the configured zoos and runtimes.
3. Provides a web-based user interface to:
   - List available models.
   - Launch models with specific runtimes and configurations.
   - Manage running models (viewing logs, stopping models)
4. Embeds a proxy server (`proxy.py`) that forwards requests to the appropriate running model
   - Supports OpenAI protocol for text, chat and multi-modal completions
   - Supports A1111 protocol for image generation
5. Keeps track of model launch history
   - Number of times a model has been launched, and the last launch time (to sort models by most frequently used)
   - Last used enviroment and parameters (provides a better user experience by pre-filling launch configurations based on previous usage)
6. Peers with instances of itself on other nodes to create distributed setups.

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
5. Open the ZooKeeper web interface (listening at http://0.0.0.0:3333/ by default) to view, launch and manage models.

## Components

ModelZoo is composed of several key components:

1. **[Zoos](#zoos)**: Discovery systems that catalog models.
2. **Models**: Data objects representing models.
3. **[Runtimes](#runtimes)**: Backends that can serve models in specific environments.
4. **[Environments](#environments)**: Named GPU Configurations (environment variables).
5. **EnvironmentSet**: A collection of environments combined for model execution.
6. **[ZooKeeper](#zookeeper)**: Web application to interact with zoos, use runtimes to spawn models, interface with history and host the proxy.
7. **Proxy**: A hybrid OpenAI-compatible (text+multimodal) and A1111-compatible (image) proxy server.
8. **ModelHistory**: A ZooKeeper component that tracks model launch history, including frequency of use and last used configurations.
9. **[Peers](#remote-models-peers)**: Instances of ZooKeeper running on other hosts.

## Configuration

ModelZoo (in practice, ZooKeeper) is configured using a YAML file that defines:

- Zoos to be instantiated and their configurations.
- Runtimes to be made available.
- Predefined environments.
- Remote peers for distributed model management.

### Example Configuation

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
   - name: "P40/1"
     vars:
        CUDA_VISIBLE_DEVICES: 1

peers:
   - host: another-host
     port: 3333
```

This example assumes you have some `*.gguf` files under /mnt/ssd0 and that you have a compiled llama.cpp server binary at the specified path and that you have a second instance of ModelZoo running on `another-host`.

### Zoos

Zoos are responsible for discovering and cataloging models.

That the `name` field is optional and will default to `class` if not provided, but naming your Zoos is strongly encouraged.

The system supports different types of zoos:

1. **FolderZoo**: Discovers models in a specified file system folder.
   - Parameters:
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

4. **OllamaZoo**: Discovers models from a local or remote Ollama instance.
   - Parameters:
     - `api_url` (str): Base URL of the Ollama API (default: http://localhost:11434)
   - Example:
     ```yaml
     - name: LocalOllama
       class: OllamaZoo
       params:
         api_url: http://localhost:11434
     ```

Each zoo type is designed to accommodate different model discovery and management needs, allowing for flexibility in how models are sourced and cataloged within the ModelZoo system.

### Runtimes

Runtimes are responsible for serving models.  The `name` field is optional, and will default to `class` if not provided.

1. **LlamaRuntime**: For serving GGUF models with [llama-server](https://github.com/ggerganov/llama.cpp)
   - Compatible model formats: gguf
   - Parameters:
     - `bin_path` (str): Path to the llama.cpp server binary
   - Example:
     ```yaml
     - name: LlamaRuntime
       class: LlamaRuntime
       params:
         bin_path: /path/to/llama-server
     ```

1. **LlamaSrbRuntime**: For serving GGUF models with [llama-srb-api](https://github.com/the-crypt-keeper/llama-srb-api)
   - Compatible model formats: gguf
   - Parameters:
     - `bin_path` (str): Path to the llama.cpp server binary
   - Example:
     ```yaml
     - class: LlamaSrbRuntime
       params:
         script_path: /path/to/llama-srb-api/api.py
     ```

3. **KoboldCppRuntime**: For serving GGUF models using [KoboldCpp](https://github.com/LostRuins/koboldcpp)
   - Compatible model formats: gguf
   - Parameters:
     - `bin_path` (str): Path to the KoboldCpp binary
   - Example:
     ```yaml
     - name: KoboldCppRuntime
       class: KoboldCppRuntime
       params:
         bin_path: /path/to/koboldcpp
     ```

4. **TabbyRuntime**: For serving GPTQ and EXL2 models using [TabbyAPI](https://github.com/theroyallab/tabbyAPI)
   - Compatible model formats: gptq, exl2
   - Parameters:
     - `script_path` (str): Path to the TabbyAPI start.sh script
   - Example:
     ```yaml
     - name: TabbyRuntime
       class: TabbyRuntime
       params:
         script_path: /path/to/tabby_api/start.sh
     ```

5. **LiteLLMRuntime**: For proxying models using [LiteLLM](https://github.com/BerriAI/litellm)
   - Compatible model formats: litellm
      - All formats supported by LiteLLM (including OpenAI, Azure, Anthropic, and various open-source models)
   - Parameters:
     - `bin_path` (str, optional): Path to the LiteLLM binary (default: "litellm")
   - Example:
     ```yaml
     - name: LiteLLMRuntime
       class: LiteLLMRuntime
       params:
         bin_path: litellm
     ```

6. **SDServerRuntime**: For serving Stable Diffusion models using [stable-diffusion.cpp](https://github.com/stduhpf/stable-diffusion.cpp/tree/server)
   - Compatible model formats: kcppt
   - Parameters:
     - `bin_path` (str): Path to the sd-server binary
   - Example:
     ```yaml
     - name: SDServerRuntime
       class: SDServerRuntime
       params:
         bin_path: /path/to/sd-server
     ```
   - Runtime Parameters:
     - `sampler_name`: Sampling method (Euler, Euler A, Heun, DPM2, DPM++, LCM)
     - `cfg_scale`: CFG Scale for guidance (default: 1.0)
     - `steps`: Number of sampling steps (default: 1)
     - `extra_args`: Additional command line arguments

Each runtime defines compatible model formats and configurable parameters. When launching a model, you can specify additional runtime-specific parameters as needed. The choice of runtime depends on the model format and the specific features required for your use case.

### Environments

Environments are configurations for running models, typically including environment variables like `CUDA_VISIBLE_DEVICES`.

Example:

```yaml
envs:
   - name: "RTX3090"
     vars:
        CUDA_VISIBLE_DEVICES: 0
   - name: "P40"
     vars:
        CUDA_VISIBLE_DEVICES: 1
```

Multiple enviroments may be pre-defined in the configuration file, and multiple enviroments can be selected when launching model (any conflicting values will be merged with a comma).

### Remote Models (Peers)

The remote models feature allows you to connect multiple ModelZoo instances and view the running models on remote peers. To configure remote peers:

1. Add a `peers` section to your configuration file.
2. For each peer, specify the `host` and `port` where the remote ModelZoo instance is running.

Example:

```yaml
peers:
  - host: falcon
    port: 3333
```

The web interface will display the status and running models of each configured peer, allowing you to manage a distributed setup of ModelZoo instances.

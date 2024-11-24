from typing import List, Dict
import yaml
from flask import Flask, jsonify, request, render_template
import random
import json
import traceback

from base import *
from zoo import *
from runtime import *

# Import all zoo and runtime classes
from zoo import FolderZoo
from runtime import LlamaRuntime, KoboldCppRuntime

def human_size(size: int) -> str:
    """Convert size in bytes to human readable string.
    
    Args:
        size (int): Size in bytes
        
    Returns:
        str: Human readable size (e.g., "1.23 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

class ZooKeeper:
    def __init__(self, config_path: str):
        self.app = Flask(__name__)
        self.zoos: Dict[str, FolderZoo] = {}
        self.runtimes: Dict[str, LlamaRuntime] = {}
        self.environments: Dict[str, Environment] = {}
        self.running_models: List[RunningModel] = []
        self.zoo_enabled: Dict[str, bool] = {}
        
        # Load configuration
        self.load_config(config_path)
        
        # Setup routes
        self.setup_routes()
        
        # Setup Jinja
        self.app.jinja_env.globals.update(
            enumerate=enumerate,
            human_size=human_size
        )        

    def load_config(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Load zoos
        for zoo_config in config.get('zoos', []):
            try:
                zoo_class = eval(zoo_config['class'])
                zoo = zoo_class(**zoo_config['params'])
                self.zoos[zoo_config['name']] = zoo
                self.zoo_enabled[zoo_config['name']] = True
            except Exception as e:
                print(f"Error creating zoo '{zoo_config['name']}' of class '{zoo_config['class']}': {str(e)}")
                raise e

        # Load runtimes
        for runtime_config in config.get('runtimes', []):
            try:
                runtime_class = eval(runtime_config['class'])
                runtime = runtime_class(name=runtime_config['name'], **runtime_config['params'])
                self.runtimes[runtime_config['name']] = runtime
            except Exception as e:
                print(f"Error creating runtime '{runtime_config['name']}' of class '{runtime_config['class']}': {str(e)}")
                raise e

        # Load environments
        for env_config in config.get('envs', []):
            env = Environment(env_config['name'], env_config['vars'])
            self.environments[env_config['name']] = env

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return self.handle_exception(self.render_index)

        @self.app.route('/api/zoo/<name>/toggle', methods=['POST'])
        def toggle_zoo(name):
            return self.handle_exception(self.handle_toggle_zoo, name)

        @self.app.route('/api/model/launch', methods=['POST'])
        def launch_model():
            return self.handle_exception(self.handle_launch_model)

        @self.app.route('/api/model/<int:model_idx>/stop', methods=['POST'])
        def stop_model(model_idx):
            return self.handle_exception(self.handle_stop_model, model_idx)

        @self.app.route('/api/model/<int:model_idx>/logs')
        def get_logs(model_idx):
            return self.handle_exception(self.handle_get_logs, model_idx)

        @self.app.route('/api/model/<int:model_idx>/status')
        def get_status(model_idx):
            return self.handle_exception(self.handle_get_status, model_idx)

    def handle_exception(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            print(f"Error: {error_message}\n{stack_trace}")
            return jsonify({
                'success': False,
                'error': error_message,
                'stack_trace': stack_trace
            }), 500

    def handle_get_status(self, model_idx):
        if 0 <= model_idx < len(self.running_models):
            return jsonify({'ready': self.running_models[model_idx].ready()})
        return jsonify({'success': False, 'error': 'Model not found'}), 404

    def get_available_models(self):
        models = []
        for zoo_name, zoo in self.zoos.items():
            if self.zoo_enabled[zoo_name]:
                models.extend(zoo.catalog())
        return models

    def get_random_port(self):
        return random.randint(50000, 60000)

    def run(self, host='0.0.0.0', port=5000, debug=False):
        self.app.run(host=host, port=port, debug=debug)

    def render_index(self):
        return render_template('index.html',
            zoos=self.zoos,
            zoo_enabled=self.zoo_enabled,
            available_models=self.get_available_models(),
            running_models=self.running_models,
            runtimes={name: runtime.__dict__ for name, runtime in self.runtimes.items()},
            environments=self.environments,
            random_port=self.get_random_port()
        )

    def handle_toggle_zoo(self, name):
        if name in self.zoo_enabled:
            self.zoo_enabled[name] = not self.zoo_enabled[name]
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Zoo not found'}), 404

    def handle_launch_model(self):
        data = request.form
        model_id = data['model_id']
        runtime_name = data['runtime']
        env_name = data['environment']
        port = int(data['port'])
        params = json.loads(data['params'])

        # Find model
        model = None
        for m in self.get_available_models():
            if m.model_id == model_id:
                model = m
                break
        
        if not model:
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        # Get runtime and environment
        runtime = self.runtimes.get(runtime_name)
        environment = self.environments.get(env_name)
        
        if not runtime or not environment:
            return jsonify({'success': False, 'error': 'Invalid runtime or environment'}), 400

        # Create listener
        listener = Listener('http', '0.0.0.0', port)

        # Spawn model
        running_model = runtime.spawn(environment, listener, model, params)
        self.running_models.append(running_model)

        return jsonify({'success': True})

    def handle_stop_model(self, model_idx):
        if 0 <= model_idx < len(self.running_models):
            self.running_models[model_idx].stop()
            self.running_models.pop(model_idx)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Model not found'}), 404

    def handle_get_logs(self, model_idx):
        if 0 <= model_idx < len(self.running_models):
            return jsonify(self.running_models[model_idx].logs())
        return jsonify({'success': False, 'error': 'Model not found'}), 404

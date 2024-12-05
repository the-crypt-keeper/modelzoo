from typing import List, Dict
import yaml
from flask import Flask, jsonify, request, render_template
import random
import json
import requests
from datetime import datetime
from dataclasses import dataclass, asdict
from asgiref.wsgi import WsgiToAsgi
import socket

from base import *
from zoo import *
from runtime import *

import traceback
from functools import wraps

def exception_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            print(f"Error: {error_message}\n{stack_trace}")
            return jsonify({
                'success': False,
                'error': error_message,
                'stack_trace': stack_trace
            }), 500
    return decorated_function

@dataclass
class ModelLaunchInfo:
    zoo_name: str
    model_name: str
    launch_count: int = 0
    last_launch: datetime = None
    last_runtime: str = None
    last_environment: str = None
    last_params: Dict = None

class ModelHistory:
    def __init__(self, history_file: str = 'history.json'):
        self.history_file = history_file
        self.model_info: Dict[str, ModelLaunchInfo] = {}
        self.load_history()

    def load_history(self):
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    value['last_launch'] = datetime.fromisoformat(value['last_launch']) if value['last_launch'] else None
                    self.model_info[key] = ModelLaunchInfo(**value)
        except FileNotFoundError:
            pass

    def save_history(self):
        data = {k: asdict(v) for k, v in self.model_info.items()}
        for value in data.values():
            value['last_launch'] = value['last_launch'].isoformat() if value['last_launch'] else None
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def update_model_launch(self, zoo_name: str, model_name: str, runtime: str, environment: str, params: Dict):
        key = f"{zoo_name}:{model_name}"
        if key not in self.model_info:
            self.model_info[key] = ModelLaunchInfo(zoo_name, model_name)
        
        info = self.model_info[key]
        info.launch_count += 1
        info.last_launch = datetime.now()
        info.last_runtime = runtime
        info.last_environment = environment
        info.last_params = params
        
        self.save_history()

    def get_sorted_models(self, models: List[Model]) -> List[Model]:
        def get_launch_info(model):
            key = f"{model.zoo_name}:{model.model_name}"
            return self.model_info.get(key, ModelLaunchInfo(model.zoo_name, model.model_name))

        return sorted(models, key=lambda m: (
            -get_launch_info(m).launch_count,
            m.model_name.lower()
        ))

    def get_last_launch_info(self, zoo_name: str, model_name: str) -> ModelLaunchInfo:
        key = f"{zoo_name}:{model_name}"
        return self.model_info.get(key, ModelLaunchInfo(zoo_name, model_name))

class ZooKeeper:
    def __init__(self, config_path: str):
        self.app = Flask(__name__)
        self.zoos: Dict[str, Zoo] = {}
        self.runtimes: Dict[str, Runtime] = {}
        self.environments: Dict[str, Environment] = {}
        self.running_models: List[RunningModel] = []
        self.model_history = ModelHistory()
        self.peers: List[Dict[str, str]] = []
        
        # Load configuration
        self.load_config(config_path)
        
        # Setup routes
        self.setup_routes()
        
        # Setup Jinja
        self.app.jinja_env.globals.update(
            enumerate=enumerate
        )
        
    def get_asgi_app(self):
        return WsgiToAsgi(self.app)

    def shutdown(self):
        print("Stopping all running models...")
        for model in self.running_models:
            try:
                model.stop(no_wait=True)
                print(f"Stopped model: {model.model.model_name}")
            except Exception as e:
                print(f"Error stopping model {model.model.model_name}: {str(e)}")
        self.running_models.clear()
        print("All models stopped.")

    def load_config(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Load zoos
        for zoo_config in config.get('zoos', []):
            try:
                zoo_class = eval(zoo_config['class'])
                zoo = zoo_class(name=zoo_config.get('name',zoo_config['class']), **zoo_config['params'])
                self.zoos[zoo_config['name']] = zoo
            except Exception as e:
                print(f"Error creating zoo '{zoo_config['name']}' of class '{zoo_config['class']}': {str(e)}")
                raise e

        # Load runtimes
        for runtime_config in config.get('runtimes', []):
            try:
                runtime_class = eval(runtime_config['class'])
                runtime = runtime_class(name=runtime_config.get('name',runtime_config['class']), **runtime_config['params'])
                self.runtimes[runtime.runtime_name] = runtime
            except Exception as e:
                print(f"Error creating runtime '{runtime_config.get('name','<missing_name>')}' of class '{runtime_config.get('class','<missing class>')}': {str(e)}")
                raise e

        # Load environments
        for env_config in config.get('envs', []):
            env = Environment(env_config['name'], env_config['vars'])
            self.environments[env_config['name']] = env

        # Load peers
        self.peers = config.get('peers', [])

    def setup_routes(self):
        self.app.route('/')(exception_handler(self.render_index))
        self.app.route('/api/model/launch', methods=['POST'])(exception_handler(self.handle_launch_model))
        self.app.route('/api/model/<int:model_idx>/stop', methods=['POST'])(exception_handler(self.handle_stop_model))
        self.app.route('/api/model/<int:model_idx>/logs')(exception_handler(self.handle_get_logs))
        self.app.route('/api/model/<int:model_idx>/status')(exception_handler(self.handle_get_status))
        self.app.route('/api/running_models')(exception_handler(self.handle_get_running_models))

    def handle_get_status(self, model_idx):
        if 0 <= model_idx < len(self.running_models):
            return jsonify(self.running_models[model_idx].status())
        return jsonify({'success': False, 'error': 'Model not found'}), 404

    def sort_models(self, catalog):
        return self.model_history.get_sorted_models(catalog)

    def get_random_port(self):
        return random.randint(50000, 60000)

    def render_index(self):
        model_launch_info = {}
        for zoo in self.zoos.values():
            for model in zoo.catalog():
                key = f"{model.zoo_name}:{model.model_name}"
                model_launch_info[key] = self.model_history.get_last_launch_info(model.zoo_name, model.model_name)

        return render_template('index.html',
            zoos={name: {'catalog': self.sort_models(zoo.catalog())} for name, zoo in self.zoos.items()},
            running_models=self.running_models,
            runtimes={name: {**runtime.__dict__} for name, runtime in self.runtimes.items()},
            environments=self.environments,
            random_port=self.get_random_port(),
            model_launch_info=model_launch_info,
            hostname=socket.gethostname(),
            remote_models=self.get_remote_models()
        )

    def handle_launch_model(self):
        data = request.form
        zoo_name = data['zoo_name']
        model_id = data['model_id']
        runtime_name = data['runtime']
        env_name = data['environment']
        port = int(data['port'])
        params = json.loads(data['params'])

        # Find model
        zoo = self.zoos.get(zoo_name)
        if not zoo:
            return jsonify({'success': False, 'error': 'Zoo not found'}), 404

        model = next((m for m in zoo.catalog() if m.model_id == model_id), None)
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

        # Update launch history
        self.model_history.update_model_launch(model.zoo_name, model.model_name, runtime_name, env_name, params)

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

    def handle_get_running_models(self):
        running_models = []
        for rmodel in self.running_models:
            model = rmodel.model
            running_models.append({
                'model_name': model.model_name,
                'model_id': model.model_id,
                'status': rmodel.status(),
                'listener': rmodel.listener.__dict__
            })
        return jsonify({'running_models': running_models})

    def get_remote_models(self):
        remote_models = []
        for peer in self.peers:
            try:
                response = requests.get(f"http://{peer['host']}:{peer['port']}/api/running_models", timeout=5)
                response.raise_for_status()
                
                peer_models = response.json().get('running_models', [])
                for model in peer_models:
                    model['listener']['host'] = peer['host']  # Replace with the peer's host
                    
                remote_models.append({
                    'host': peer['host'],
                    'port': peer['port'],
                    'models': peer_models,
                    'error': None
                })
                    
            except Exception as e:
                print(f"Error fetching models from peer {peer['host']}:{peer['port']}: {str(e)}")
                
                remote_models.append({
                    'host': peer['host'],
                    'port': peer['port'],
                    'error': str(e),
                    'models': []
                })                
        return remote_models

    def get_running_models(self):
        return [model for model in self.running_models if model.status()['ready']]

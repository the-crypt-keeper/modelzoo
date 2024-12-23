from flask import request, Response, jsonify, stream_with_context
import requests
from werkzeug.exceptions import ClientDisconnected
from threading import Lock
from collections import defaultdict

class ProxyServer:
     def __init__(self, zookeeper):
         self.zookeeper = zookeeper
         self.app = zookeeper.app
         self.active_connections = defaultdict(int)
         self.connections_lock = Lock()
         self.setup_routes()

     def setup_routes(self):
         self.app.route('/v1/models', methods=['GET'])(self.get_models)
         self.app.route('/v1/completions', methods=['POST'])(self.handle_completions)
         self.app.route('/v1/chat/completions', methods=['POST'])(self.handle_chat_completions)

     def get_models(self):
         local_models = [{"id": rmodel.model.model_name, "owned_by": "modelzoo"} for rmodel in self.zookeeper.get_running_models()]
         remote_models = []
         for peer in self.zookeeper.get_remote_models():
             for model in peer['models']:
                 remote_models.append({"id": model['model_name'], "owned_by": f"remote:{peer['host']}"})
         return jsonify({"data": local_models + remote_models})

     def handle_completions(self):
         return self._handle_request('/v1/completions')

     def handle_chat_completions(self):
         return self._handle_request('/v1/chat/completions')

     def _handle_request(self, endpoint):
        try:
            data = request.get_json()
            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model not specified in the request"}), 400

            # Get all available instances of the requested model
            model_instances = []
            
            # Check local running models
            running_models = self.zookeeper.get_running_models()
            local_instances = [m for m in running_models if m.model.model_name == model_name]
            for model in local_instances:
                model_instances.append({
                    'model_name': model.model.model_name,
                    'url': f"http://{model.listener.host}:{model.listener.port}{endpoint}"
                })
            
            # Check remote models
            remote_models = self.zookeeper.get_remote_models()
            for peer in remote_models:
                if peer['error'] is None:
                    for remote_model in peer['models']:
                        if remote_model['model_name'] == model_name:
                            model_instances.append({
                                'model_name': remote_model['model_name'],
                                'url': f"http://{remote_model['listener']['host']}:{remote_model['listener']['port']}{endpoint}"
                            })
            
            if not model_instances:
                return jsonify({"error": f"Model {model_name} not found or not running"}), 404
            
            # Select instance with least connections
            with self.connections_lock:
                selected = min(model_instances, 
                             key=lambda x: self.active_connections[x['model_name']])
                self.active_connections[selected['model_name']] += 1
                target_url = selected['url']
            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

            resp = requests.post(target_url, json=data, headers=headers, stream=True)
            content_type = resp.headers.get('Content-Type', 'application/json')

            model_name = selected['model_name']  # Capture for closure

            @stream_with_context
            def generate():
                try:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if chunk:
                            yield chunk                            
                except GeneratorExit:
                    print("Client disconnected. Stopping stream.")
                finally:
                    print("Closing proxy connection.")
                    with self.connections_lock:
                        self.active_connections[model_name] -= 1
                    resp.close()

            return Response(generate(),
                            direct_passthrough=True,
                            status=resp.status_code,
                            content_type=content_type)
        
        except requests.RequestException as e:
            print(f"Network error occurred: {str(e)}")
            return jsonify({"error": f"Network error: {str(e)}"}), 500
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

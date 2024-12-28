from flask import request, Response, jsonify, stream_with_context
import requests
from werkzeug.exceptions import ClientDisconnected
from threading import Lock
from collections import defaultdict

class ProxyServer:
     def __init__(self, zookeeper):
         self.zookeeper = zookeeper
         self.app = zookeeper.app
         self.active_connections = defaultdict(int)  # tracks connections by full URL
         self.connections_lock = Lock()
         self.setup_routes()

     def setup_routes(self):
         # OpenAI API routes
         self.app.route('/v1/models', methods=['GET'])(self.get_models)
         self.app.route('/v1/completions', methods=['POST'])(self.handle_completions)
         self.app.route('/v1/chat/completions', methods=['POST'])(self.handle_chat_completions)
         self.app.route('/v1/images/generations', methods=['POST'])(self.handle_image_generation)
         self.app.route('/health', methods=['GET'])(self.health_check)
         self.app.route('/.well-known/serviceinfo', methods=['GET'])(self.service_info)
         
         # A1111 API routes
         self.app.route('/sdapi/v1/sd-models', methods=['GET'])(self.get_sd_models)
         self.app.route('/sdapi/v1/txt2img', methods=['POST'])(self.handle_txt2img)
         self.app.route('/sdapi/v1/img2img', methods=['POST'])(self.handle_img2img)

     def get_models(self):
         unique_models = {}
         
         # Add local models (these take precedence)
         for rmodel in self.zookeeper.get_running_models():
             unique_models[rmodel.model.model_name] = {
                 "id": rmodel.model.model_name,
                 "owned_by": "modelzoo"
             }
         
         # Add remote models (only if not already present locally)
         for peer in self.zookeeper.get_remote_models():
             for model in peer['models']:
                 if model['model_name'] not in unique_models:
                     unique_models[model['model_name']] = {
                         "id": model['model_name'],
                         "owned_by": f"remote:{peer['host']}"
                     }
         
         return jsonify({"data": list(unique_models.values())})

     def handle_completions(self):
         return self._handle_request('/v1/completions')

     def handle_chat_completions(self):
         return self._handle_request('/v1/chat/completions')

     def handle_image_generation(self):
         return self._handle_request('/v1/images/generations')

     def health_check(self):
         # Check local running models
         if len(self.zookeeper.get_running_models()) > 0:
             return '', 200

     def get_sd_models(self):
         """Return list of available SD models in A1111 format"""
         image_models = []
         
         # Add local models
         for rmodel in self.zookeeper.get_running_models():
             if rmodel.listener.protocol == 'a1111':
                 image_models.append({
                     "title": rmodel.model.model_name,
                     "model_name": rmodel.model.model_name,
                     "hash": "0000000000", # Placeholder
                     "sha256": "0" * 64,  # Placeholder
                     "filename": rmodel.model.path if hasattr(rmodel.model, 'path') else "",
                     "config": None
                 })
         
         # Add remote models
         for peer in self.zookeeper.get_remote_models():
             if peer['error'] is None:
                 for remote_model in peer['models']:
                     if remote_model.get('listener', {}).get('protocol') == 'a1111':
                         image_models.append({
                             "title": remote_model['model_name'],
                             "model_name": remote_model['model_name'],
                             "hash": "0000000000", # Placeholder
                             "sha256": "0" * 64,  # Placeholder
                             "filename": remote_model.get('path', ""),
                             "config": None
                         })
         
         return jsonify(image_models)

     def handle_txt2img(self):
         if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
         data = request.get_json()
         if 'prompt' not in data: return jsonify({"error": "prompt is required"}), 400
         
         return self._handle_request('/sdapi/v1/txt2img', data)

     def handle_img2img(self):
         if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
         data = request.get_json()
         if 'prompt' not in data: return jsonify({"error": "prompt is required"}), 400
                     
         return self._handle_request('/sdapi/v1/img2img', data)

     def service_info(self):
         """Return service information according to the serviceinfo spec."""
         return jsonify({
             "version": "0.2",
             "software": {
                 "name": "ModelZoo",
                 "version": "0.0.1",
                 "repository": "https://github.com/the-crypt-keeper/modelzoo",
                 "homepage": "https://github.com/the-crypt-keeper/modelzoo"
             },
             "api": {
                 "openai": {
                     "name": "OpenAI API",
                     "rel_url": "/v1",
                     "documentation": "https://openai.com/documentation/api",
                     "version": "0.0.1"
                 }
             }
         })

     def _handle_request(self, endpoint, data=None):
        try:
            if data is None:
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
                             key=lambda x: self.active_connections[x['url']])
                self.active_connections[selected['url']] += 1
                target_url = selected['url']

            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
            
            try:
                # Determine if we should stream based on request
                should_stream = data.get('stream', False)            
                resp = requests.post(target_url, json=data, headers=headers, stream=should_stream)
                content_type = resp.headers.get('Content-Type', 'application/json')
                
                if should_stream:
                    @stream_with_context
                    def generate():
                        try:
                            for chunk in resp.iter_content(chunk_size=4096):
                                if chunk:
                                    yield chunk                            
                        except GeneratorExit:
                            print("Client disconnected. Stopping stream.")
                        finally:
                            resp.close()

                    return Response(generate(),
                                  direct_passthrough=True,
                                  status=resp.status_code,
                                  content_type=content_type)
                else:
                    # For non-streaming requests, get full response
                    response_data = resp.content
                    return Response(response_data,
                                  status=resp.status_code,
                                  content_type=content_type)
            finally:
                # Always decrement connection count
                with self.connections_lock:
                    self.active_connections[target_url] -= 1
        
        except requests.RequestException as e:
            print(f"Network error occurred: {str(e)}")
            return jsonify({"error": f"Network error: {str(e)}"}), 500
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

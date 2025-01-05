from flask import request, Response, jsonify, stream_with_context
import requests
from werkzeug.exceptions import ClientDisconnected
from threading import Lock
from collections import defaultdict
from protocols import PROTOCOLS

class ProxyServer:
     def __init__(self, zookeeper):
         self.zookeeper = zookeeper
         self.app = zookeeper.app
         self.active_connections = defaultdict(int)  # tracks connections by full URL
         self.connections_lock = Lock()
         self.setup_routes()

     def setup_routes(self):
         # Health and Identity
         self.app.route('/health', methods=['GET'])(self.health_check)
         self.app.route('/.well-known/serviceinfo', methods=['GET'])(self.service_info)

         # OpenAI API routes
         self.app.route('/v1/models', methods=['GET'])(self.get_models)
         # OpenAI API routes
         self.app.route('/v1/completions', methods=['POST'])(self.handle_completions)
         self.app.route('/v1/chat/completions', methods=['POST'])(self.handle_chat_completions)
         
         # A1111 API routes
         self.app.route('/sdapi/v1/sd-models', methods=['GET'])(self.get_sd_models)
         self.app.route('/sdapi/v1/txt2img', methods=['POST'])(self.handle_txt2img)
         self.app.route('/sdapi/v1/img2img', methods=['POST'])(self.handle_img2img)

     def get_models(self):
         unique_models = {}
         
         # Get all available models and filter for those with text capabilities
         available_models = self.zookeeper.get_available_models()
         for model in available_models:
             protocol = model['listener']['protocol']
             if (protocol in PROTOCOLS and 
                 (PROTOCOLS[protocol]['completions'] or PROTOCOLS[protocol]['chat_completions']) and
                 model['model_name'] not in unique_models):
                 unique_models[model['model_name']] = {
                     "id": model['model_name'],
                     "owned_by": model['source']
                 }
         
         return jsonify({"data": list(unique_models.values())})

     def get_sd_models(self):
         """Return list of available SD models in A1111 format"""
         image_models = []
         
         # Get all available models and filter for those with image capabilities
         available_models = self.zookeeper.get_available_models()
         for model in available_models:
             protocol = model['listener']['protocol']
             if protocol in PROTOCOLS and (PROTOCOLS[protocol]['txt2img'] or PROTOCOLS[protocol]['img2img']):
                 image_models.append({
                     "title": model['model_name'],
                     "model_name": model['model_name'],
                     "hash": "0000000000", # Placeholder
                     "sha256": "0" * 64,  # Placeholder
                     "filename": model.get('path', ""),
                     "config": None
                 })
         
         return jsonify(image_models)
     
     def handle_completions(self):
         """Handle /v1/completions endpoint"""
         return self.handle_request('completions')

     def handle_chat_completions(self):
         """Handle /v1/chat/completions endpoint"""
         return self.handle_request('chat_completions')

     def handle_txt2img(self):
         """Handle /sdapi/v1/txt2img endpoint"""
         return self.handle_request('txt2img', required_keys=['prompt'])

     def handle_img2img(self):
         """Handle /sdapi/v1/img2img endpoint"""
         return self.handle_request('img2img', required_keys=['prompt'])

     def handle_request(self, endpoint_type, required_keys=None):
         """Unified request handler for all endpoints.
         
         Args:
             endpoint_type: The type of endpoint to handle (completions, chat_completions, etc)
             required_keys: List of required keys in the request payload
         """
         if not request.is_json:
             return jsonify({"error": "Request must be JSON"}), 400
             
         data = request.get_json()
         
         if required_keys:
             missing_keys = [key for key in required_keys if key not in data]
             if missing_keys:
                 return jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}), 400
                 
         return self._handle_request(endpoint_type, data)

     def health_check(self):
         # Check local running models
         if len(self.zookeeper.get_available_models(local_models=True, remote_models=False)) > 0:
             return '', PROTOCOLS['openai']['health_status']

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

     def _handle_request(self, endpoint_type, data):
        try:
            requested_model = data.get('model')
            if not requested_model: return jsonify({"error": "Model not specified in the request"}), 400

            # Get all available instances of the requested model
            model_instances = []
            
            # Get all available model instances, build potential endpoint map
            available_models = self.zookeeper.get_available_models()
            for model in available_models:
                # Match on display name or model name
                if model['model_name'] == requested_model:
                    protocol = model['listener']['protocol']                   
                    endpoint = PROTOCOLS.get(protocol, {})[endpoint_type]
                    if not endpoint:  # Protocol doesn't support this endpoint
                        continue

                    host = model['listener']['host']
                    port = model['listener']['port']
                    model_instances.append({
                        'model_name': model['model_name'],
                        'model_id': model['model_id'],
                        'protocol': protocol,
                        'url': f"http://{host}:{port}{endpoint}"
                    })
            
            if not model_instances:
                return jsonify({"error": f"Model {model_name} not found or not running"}), 404
            
            # Select instance with least connections
            with self.connections_lock:
                selected = min(model_instances, 
                             key=lambda x: self.active_connections[x['url']])
                self.active_connections[selected['url']] += 1
                
                target_url = selected['url']
                protocol = selected['protocol']

            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
            
            try:
                # Apply sampler mapping if this is an image endpoint
                if endpoint_type in ['txt2img', 'img2img'] and 'sampler_name' in data:
                    sampler_map = PROTOCOLS[protocol].get('image_sampler_map', {})
                    if data['sampler_name'] in sampler_map:
                        data['sampler_name'] = sampler_map[data['sampler_name']]

                # Replace model name with model ID in request
                data = data.copy()
                data['model'] = selected['model_id']

                # Apply request adapter if specified in protocol
                request_adapter = PROTOCOLS[protocol].get(f'{endpoint_type}_request_adapter')
                if request_adapter is not None:
                    data = request_adapter(data)

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
                    response_data = resp.json() if content_type == 'application/json' else resp.content
                    
                    # Apply response adapter if specified in protocol
                    if content_type == 'application/json':
                        response_adapter = PROTOCOLS[protocol].get(f'{endpoint_type}_response_adapter')
                        if response_adapter is not None:
                            response_data = response_adapter(response_data)
                    
                    return Response(response_data if isinstance(response_data, bytes) else jsonify(response_data).data,
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

from flask import request, Response, jsonify, stream_with_context
import requests
from werkzeug.exceptions import ClientDisconnected

class ProxyServer:
     def __init__(self, zookeeper):
         self.zookeeper = zookeeper
         self.app = zookeeper.app
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

            running_models = self.zookeeper.get_running_models()
            model = next((m for m in running_models if m.model.model_name == model_name), None)
            
            if not model:
                # Search in remote models
                remote_models = self.zookeeper.get_remote_models()
                for peer in remote_models:
                    if peer['error'] is None:
                        for remote_model in peer['models']:
                            if remote_model['model_name'] == model_name:
                                target_url = f"http://{remote_model['listener']['host']}:{remote_model['listener']['port']}{endpoint}"
                                break
                        if 'target_url' in locals():
                            break
                else:
                    return jsonify({"error": f"Model {model_name} not found or not running"}), 404
            else:
                target_url = f"http://{model.listener.host}:{model.listener.port}{endpoint}"
            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

            resp = requests.post(target_url, json=data, headers=headers, stream=True)
            content_type = resp.headers.get('Content-Type', 'application/json')

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

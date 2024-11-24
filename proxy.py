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
         models = [{"id": rmodel.model.model_name, "owned_by": "modelzoo"} for rmodel in self.zookeeper.get_running_models()]
         return jsonify({"data": models})

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
                return jsonify({"error": f"Model {model_name} not found or not running"}), 404

            target_url = f"http://{model.listener.host}:{model.listener.port}{endpoint}"
            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

            resp = requests.post(target_url, json=data, headers=headers, stream=True)
            content_type = resp.headers.get('content-type', 'application/json')

            def generate():
                try:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if chunk:  # filter out keep-alive new chunks
                            yield chunk
                except ClientDisconnected:
                    print("Client disconnected. Stopping stream.")
                finally:
                    resp.close()

            return Response(stream_with_context(generate()),
                            status=resp.status_code,
                            content_type=content_type)
        
        except requests.RequestException as e:
            print(f"Network error occurred: {str(e)}")
            return jsonify({"error": f"Network error: {str(e)}"}), 500
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

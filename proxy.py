import aiohttp
from aiohttp import web

class ProxyServer:
    def __init__(self, zookeeper):
        self.zookeeper = zookeeper
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get('/v1/models', self.get_models)
        self.app.router.add_post('/v1/completions', self.handle_completions)
        self.app.router.add_post('/v1/chat/completions', self.handle_chat_completions)

    async def get_models(self, request):
        running_models = self.zookeeper.get_running_models()
        models = [{"id": model["model_name"], "object": "model"} for model in running_models]
        return web.json_response({"data": models})

    async def handle_completions(self, request):
        return await self._handle_request(request, '/v1/completions')

    async def handle_chat_completions(self, request):
        return await self._handle_request(request, '/v1/chat/completions')

    async def _handle_request(self, request, endpoint):
        try:
            data = await request.json()
            model_name = data.get('model')
            if not model_name:
                raise web.HTTPBadRequest(text="Model not specified in the request")

            running_models = self.zookeeper.get_running_models()
            model = next((m for m in running_models if m['model_name'] == model_name), None)
            if not model:
                raise web.HTTPNotFound(text=f"Model {model_name} not found or not running")

            target_url = f"http://{model['listener']['host']}:{model['listener']['port']}{endpoint}"
            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

            async with aiohttp.ClientSession() as session:
                async with session.post(target_url, json=data, headers=headers) as resp:
                    if 'content-type' in resp.headers:
                        content_type = resp.headers['content-type']
                    else:
                        content_type = 'application/json'

                    if resp.headers.get('transfer-encoding', '').lower() == 'chunked':
                        response = web.StreamResponse(status=resp.status, headers={'Content-Type': content_type})
                        await response.prepare(request)
                        async for chunk in resp.content.iter_any():
                            await response.write(chunk)
                        await response.write_eof()
                        return response
                    else:
                        content = await resp.read()
                        return web.Response(body=content, status=resp.status, headers={'Content-Type': content_type})

        except web.HTTPException as e:
            raise e
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    def get_asgi_app(self):
        return self.app

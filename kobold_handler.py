import litellm
import time
import httpx
from typing import Any, Optional, Union, Protocol
from litellm import CustomLLM
from litellm.types.utils import ImageResponse, ImageObject

class KoboldCpp(CustomLLM):
    async def aimage_generation(self, model: str, prompt: str, model_response: ImageResponse, optional_params: dict, logging_obj: Any, timeout: Optional[Union[float, httpx.Timeout]] = None, client = None, **kwargs) -> ImageResponse:
        if client is None:
            client = httpx.AsyncClient(base_url=kwargs['api_base'], timeout=timeout)
        
        # Health check. We don't actually need to generate anything, just check the server is alive.
        if prompt == "test from litellm":
            response = await client.get("/api/v1/info/version")
            response.raise_for_status()          
            return ImageResponse()

        if optional_params.get('n', 1) != 1: 
            raise ValueError('n parameter is not supported by the proxy')
        if 'size' in optional_params:
            optional_params['width'], optional_params['height'] = optional_params.pop('size').split('x')        
        if 'steps' not in optional_params:
            optional_params['steps'] = 8            
        optional_params['prompt'] = prompt
        

        print("aimage_generation() called:", optional_params, "timeout=", timeout)
        try:
            response = await client.post("/sdapi/v1/txt2img", json=optional_params)
        except Exception as e:
            print("HTTP ERROR:", e)
            
        try:
            result = response.json()
        except Exception as e:
            print("PARSE ERROR:", e)
            result = { 'images': [] }
        
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(b64_json=img_data) for img_data in result["images"]]
        )

kobold_cpp = KoboldCpp()

# without this, optional_params doesnt work see https://github.com/BerriAI/litellm/blob/main/litellm/utils.py#L2171
litellm.openai_compatible_providers.append('kobold')

# we want to use pass_through_endpoints on / so we need to yank any default handlers
app = litellm.proxy.proxy_server.app
app.router.routes = [x for x in app.router.routes if x.path != '/']
import litellm
import time
from typing import Any, Optional, Union
from litellm import CustomLLM
from litellm.types.utils import ImageResponse, ImageObject

class KoboldCpp(CustomLLM):
    async def aimage_generation(self, model: str, prompt: str, model_response: ImageResponse, optional_params: dict, logging_obj: Any, timeout: Optional[Union[float, httpx.Timeout]] = None, client: Optional[AsyncHTTPHandler] = None,) -> ImageResponse:
        if client is None:
            raise ValueError("HTTP client is required")
            
        response = await client.post("/sdapi/v1/txt2img", json={"prompt": prompt})
        result = response.json()
        
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(b64_json=img_data) for img_data in result["images"]]
        )

kobold_cpp = KoboldCpp()

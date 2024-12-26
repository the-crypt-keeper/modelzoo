import litellm
import time
from typing import Any, Optional, Union
from litellm import CustomLLM
from litellm.types.utils import ImageResponse, ImageObject

class KoboldCpp(CustomLLM):
    async def aimage_generation(self, model: str, prompt: str, model_response: ImageResponse, optional_params: dict, logging_obj: Any, timeout: Optional[Union[float, httpx.Timeout]] = None, client: Optional[AsyncHTTPHandler] = None,) -> ImageResponse:
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url="https://example.com/image.png")],
        )

kobold_cpp = KoboldCpp()
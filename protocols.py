"""Module defining supported protocols and their capabilities."""

def dalle_txt2img_request_adapter(data):
    """Adapter for DALL-E image generation request"""
    adapted_data = data.copy()
    
    # Map prompt directly
    if 'prompt' not in adapted_data:
        return adapted_data
        
    # Map batch count to n
    if 'batch_count' in adapted_data:
        adapted_data['n'] = adapted_data.pop('batch_count')
        
    # Map steps to quality (1=standard, >1=hd)
    if 'steps' in adapted_data:
        adapted_data['quality'] = 'hd' if adapted_data.pop('steps') > 1 else 'standard'
        
    # Combine width/height into size
    if 'width' in adapted_data and 'height' in adapted_data:
        size = f"{adapted_data.pop('width')}x{adapted_data.pop('height')}"
        adapted_data['size'] = size
        
    # Map sampler to style
    if 'sampler_name' in adapted_data:
        adapted_data['style'] = adapted_data.pop('sampler_name')
        
    # Always request base64 JSON
    adapted_data['response_format'] = 'b64_json'
    
    return adapted_data

def dalle_txt2img_response_adapter(response_data):
    """Adapter for DALL-E image generation response"""
    if not response_data or 'data' not in response_data:
        return response_data
    return {"images": [img['b64_json'] for img in response_data['data']]}

def sd_server_txt2img_request_adapter(data):
    """Adapter for sd-server txt2img request"""
    adapted_data = data.copy()
    if 'sampler_name' in adapted_data:
        adapted_data['sample_method'] = adapted_data.pop('sampler_name')
    if 'steps' in adapted_data:
        adapted_data['sample_steps'] = adapted_data.pop('steps')
    return adapted_data

def sd_server_txt2img_response_adapter(response_data):
    """Adapter for sd-server txt2img response"""
    if isinstance(response_data, list):
        return {"images": [ x['data'] for x in response_data] }
    return response_data

PROTOCOLS = {
    'dalle': {
        'health_check': '/v1/models',
        'health_status': 200,
        'completions': None,
        'chat_completions': None,
        'txt2img': '/v1/images/generations',
        'txt2img_request_adapter': dalle_txt2img_request_adapter,
        'txt2img_response_adapter': dalle_txt2img_response_adapter,
        'img2img': None,
        'image_sampler_map': {
            'Natural': 'natural',
            'Vivid': 'vivid'
        }
    },
    'openai': {
        'health_check': '/health',
        'health_status': 200,
        'completions': '/v1/completions',
        'chat_completions': '/v1/chat/completions',
        'txt2img': None,
        'img2img': None,
    },
    'a1111': {
        'health_check': '/sdapi/v1/sd-models',
        'health_status': 200,
        'completions': None,
        'chat_completions': None,
        'txt2img': '/sdapi/v1/txt2img',
        'img2img': '/sdapi/v1/img2img',
        'image_sampler_map': {
            'Euler': 'k_euler',
            'Euler A': 'k_euler_a',
            'Heun': 'k_heun',
            'DPM2': 'k_dpm_2',
            'DPM++': 'k_dpmpp_2m',
            'LCM': 'k_lcm'
        }
    },
    'sd-server': {
        'health_check': '/',
        'health_status': 404,
        'completions': None,
        'chat_completions': None,
        'txt2img': '/txt2img',
        'txt2img_request_adapter': sd_server_txt2img_request_adapter,
        'txt2img_response_adapter': sd_server_txt2img_response_adapter,
        'img2img': None,
        'image_sampler_map': {
            'Euler': 'euler',
            'Euler A': 'euler_a', 
            'Heun': 'heun',
            'DPM2': 'dpm2',
            'DPM++': 'dpmpp_2m',
            'LCM': 'lcm'
        }
    }
}

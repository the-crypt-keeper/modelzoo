import requests
from time import sleep

"""Module defining supported protocols and their capabilities."""

def dalle_txt2img_request_adapter(data, target_url):
    """Adapter for DALL-E image generation request"""
    adapted_data = { "prompt": data["prompt"], 'response_format': "b64_json" }
        
    # Map batch count to n
    if 'batch_count' in data:
        adapted_data['n'] = data.pop('batch_count')
        
    # Map steps to quality (1=standard, >1=hd)
    if 'steps' in data:
        adapted_data['quality'] = 'hd' if data.pop('steps') > 1 else 'standard'
        
    # Combine width/height into size
    if 'width' in data and 'height' in data:
        size = f"{data.pop('width')}x{data.pop('height')}"
        adapted_data['size'] = size
        
    # Map sampler to style
    if 'sampler_name' in data:
        adapted_data['style'] = data.pop('sampler_name')
    
    return adapted_data

def dalle_txt2img_response_adapter(response_data, target_url):
    """Adapter for DALL-E image generation response"""
    if not response_data or 'data' not in response_data:
        return response_data
    return {"images": [img['b64_json'] for img in response_data['data']]}

def sd_server_txt2img_request_adapter(data, target_url):
    """Adapter for sd-server txt2img request"""
    adapted_data = data.copy()
    if 'model' in adapted_data:
        del adapted_data['model']
    if 'sampler_name' in adapted_data:
        adapted_data['sample_method'] = adapted_data.pop('sampler_name')
    if 'steps' in adapted_data:
        adapted_data['sample_steps'] = adapted_data.pop('steps')
    return adapted_data

def sd_server_txt2img_response_adapter(response_data, target_url):
    """Adapter for sd-server txt2img task_id response"""
    task_id = response_data.get('task_id')
    if task_id is None:
        raise Exception("task_id was not returned by server")
    
    res = {'status':""}
    while res['status'] != "Done":
        sleep(1)
        res = requests.get(target_url.replace('txt2img','result'), params={'task_id':task_id}, timeout=1).json()
        print("Polling", res['status'])

    return {"images": [ x['data'] for x in res['data']] }

PROTOCOLS = {
    'dall-e': {
        'health_check': '/v1/models',
        'health_status': 200,
        'completions': None,
        'chat_completions': None,
        'txt2img': '/v1/images/generations',
        'txt2img_request_adapter': dalle_txt2img_request_adapter,
        'txt2img_response_adapter': dalle_txt2img_response_adapter,
        'img2img': None,
        'image_sampler_map': {
            'Euler': 'natural',
            'Euler A': 'vivid'
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
    'openai': {
        'health_check': '/v1/models',
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
            'DPM++': 'dpm++2m',
            'LCM': 'lcm'
        }
    }
}

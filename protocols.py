"""Module defining supported protocols and their capabilities."""

def sd_server_txt2img_adapter(data):
    """Adapter for sd-server txt2img endpoint"""
    adapted_data = data.copy()
    if 'sampler_name' in adapted_data:
        adapted_data['sample_method'] = adapted_data.pop('sampler_name')
    if 'steps' in adapted_data:
        adapted_data['sample_steps'] = adapted_data.pop('steps')
    return adapted_data

PROTOCOLS = {
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
            'DPM++ 2M': 'k_dpmpp_2m',
            'LCM': 'k_lcm'
        }
    },
    'sd-server': {
        'health_check': '/',
        'health_status': 404,
        'completions': None,
        'chat_completions': None,
        'txt2img': '/txt2img',
        'txt2img_adapter': 'sd_server_txt2img_adapter',
        'img2img': None,
        'image_sampler_map': {
            'Euler': 'euler',
            'Euler A': 'euler_a', 
            'Heun': 'heun',
            'DPM2': 'dpm2',
            'DPM++ 2M': 'dpmpp_2m',
            'LCM': 'lcm'
        }
    }
}

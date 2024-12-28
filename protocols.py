"""Module defining supported protocols and their capabilities."""

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
    },
    'sd-server': {
        'health_check': '/',
        'health_status': 404,
        'completions': None,
        'chat_completions': None,
        'txt2img': '/txt2img',
        'txt2img_adapter': None,            
        'img2img': None,
    }
}

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
    }
}

# Derived protocol lists for convenience
PROTOCOLS_TEXT = [k for k, v in PROTOCOLS.items() if v['completions'] or v['chat_completions']]
PROTOCOLS_IMAGE = [k for k, v in PROTOCOLS.items() if v['txt2img'] or v['img2img']]

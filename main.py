import argparse
import uvicorn
from zk import ZooKeeper
from proxy import ProxyServer

parser = argparse.ArgumentParser(description='ModelZoo Server')
parser.add_argument('--config', required=True, help='Path to config file')
parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
parser.add_argument('--port', type=int, default=5000, help='Port to bind to')

args = parser.parse_args()

keeper = ZooKeeper(args.config)
keeper_asgi = keeper.get_asgi_app()

proxy = ProxyServer(keeper)
proxy_asgi = proxy.app

async def combined_app(scope, receive, send):
    if scope['path'].startswith('/v1/'):
        await proxy_asgi(scope, receive, send)
    else:
        await keeper_asgi(scope, receive, send)        

if __name__ == '__main__':
    uvicorn.run(combined_app, host=args.host, port=args.port)

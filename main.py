import argparse
import uvicorn
from zk import ZooKeeper

parser = argparse.ArgumentParser(description='ModelZoo Server')
parser.add_argument('--config', required=True, help='Path to config file')
parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
parser.add_argument('--port', type=int, default=5000, help='Port to bind to')

args = parser.parse_args()

keeper = ZooKeeper(args.config)
asgi_app = keeper.get_asgi_app()

if __name__ == '__main__':
    uvicorn.run(asgi_app, host=args.host, port=args.port)

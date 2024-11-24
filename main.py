import argparse
from gevent.pywsgi import WSGIServer
from gevent import monkey
monkey.patch_all()

from zk import ZooKeeper
from proxy import ProxyServer

parser = argparse.ArgumentParser(description='ModelZoo Server')
parser.add_argument('--config', required=True, help='Path to config file')
parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
parser.add_argument('--port', type=int, default=3333, help='Port to bind to')

args = parser.parse_args()

keeper = ZooKeeper(args.config)
proxy = ProxyServer(keeper)

app = keeper.app

if __name__ == '__main__':
    http_server = WSGIServer((args.host, args.port), app)
    print(f"Server running on http://{args.host}:{args.port}")
    http_server.serve_forever()

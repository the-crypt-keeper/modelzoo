import argparse
import signal
from gevent.pywsgi import WSGIServer
from gevent import monkey
monkey.patch_all()

from zk import ZooKeeper
from proxy import ProxyServer

parser = argparse.ArgumentParser(description='ModelZoo Server')
parser.add_argument('--config', default='config.yaml', help='Path to config file')
parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
parser.add_argument('--port', type=int, default=3333, help='Port to bind to')

args = parser.parse_args()

try:
    keeper = ZooKeeper(args.config)
except Exception as e:
    print(f"Error loading configuration from {args.config}: {str(e)}")
    raise(e)
proxy = ProxyServer(keeper)

app = keeper.app

def signal_handler(signum, frame):
    print("Received signal to terminate. Shutting down gracefully...")
    keeper.shutdown()
    http_server.stop()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    http_server = WSGIServer((args.host, args.port), app)
    print(f"Server running on http://{args.host}:{args.port}")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")

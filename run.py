"""
AgroSense — One-command startup
Run from project root:
    python run.py
    python run.py --port 8000
    python run.py --host 0.0.0.0
"""
import sys, os, argparse

parser = argparse.ArgumentParser(description='AgroSense Smart Agriculture Platform')
parser.add_argument('--host',  default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
parser.add_argument('--port',  default=5000, type=int, help='Port (default: 5000)')
parser.add_argument('--debug', action='store_true', default=True, help='Debug mode')
args = parser.parse_args()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app
from backend.models.db_models import init_db

init_db()

print("\n" + "=" * 55)
print("  AgroSense -- Smart Agriculture AI Platform")
print("=" * 55)
print(f"  URL    : http://{args.host}:{args.port}")
print(f"  Debug  : {args.debug}")
print(f"  DB     : agrosense.db (SQLite)")
print("  Modules: Crop, Disease, Irrigation, Yield,")
print("           Rotation, Pest, Profit, Market, History")
print("=" * 55)
print("  Press CTRL+C to stop\n")


app.run(host=args.host, port=args.port, debug=args.debug)

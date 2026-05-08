"""
AgroSense — One-command startup
Run from project root:
    python run.py                   # dev server (hot-reload)
    python run.py --prod            # production via Waitress (Windows)
    python run.py --port 8000
    python run.py --host 0.0.0.0
"""
import sys, os, argparse

parser = argparse.ArgumentParser(description='AgroSense Smart Agriculture Platform')
parser.add_argument('--host',  default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
parser.add_argument('--port',  default=5000, type=int, help='Port (default: 5000)')
parser.add_argument('--debug', action='store_true', default=True, help='Debug mode (dev only)')
parser.add_argument('--prod',  action='store_true', default=False, help='Use Waitress production server')
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
print(f"  Mode   : {'Production (Waitress)' if args.prod else 'Development (Flask)'}")
print(f"  DB     : agrosense.db (SQLite)")
print("  Modules: Crop, Disease, Irrigation, Yield,")
print("           Rotation, Pest, Profit, Market, History")
print("=" * 55)
print("  Press CTRL+C to stop\n")

if args.prod:
    try:
        from waitress import serve
        print("  Starting Waitress with 4 threads...")
        serve(app, host=args.host, port=args.port, threads=4)
    except ImportError:
        print("  ERROR: waitress not installed. Run: pip install waitress")
        sys.exit(1)
else:
    app.run(host=args.host, port=args.port, debug=args.debug)


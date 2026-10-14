import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP.parent))  # repository root: the fakenews package
sys.path.insert(0, str(APP))         # app/: server.py and services.py

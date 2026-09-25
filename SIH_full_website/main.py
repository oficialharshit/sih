import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=BACKEND_DIR)

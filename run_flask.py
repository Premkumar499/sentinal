#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from app.app import app

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 STARTING SEMANTIC SATELLITE CHANGE DETECTION FLASK APPLICATION")
    print("   Access Dashboard at: http://127.0.0.1:5000")
    print("   REST API Endpoint:   http://127.0.0.1:5000/api/detect")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)

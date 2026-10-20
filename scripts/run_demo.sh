#!/bin/bash
set -e

echo "=================================================="
echo "🛰️ STARTING SEMANTIC SATELLITE CHANGE DETECTION DEMO"
echo "=================================================="

# 1. Activate Python Environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 2. Check Database & Configuration
echo "1. Checking Database Integrity..."
python3 scripts/check_database.py

echo "2. Validating System Environment & Config..."
PYTHONPATH=. python3 -c "from src.pipeline import MasterPipeline; MasterPipeline()"

echo "3. Launching Flask Application on http://127.0.0.1:5000..."
PYTHONPATH=. python3 app/app.py

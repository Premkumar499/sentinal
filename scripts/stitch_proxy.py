import os
import sys
import json
import urllib.request

API_KEY = os.getenv("STITCH_API_KEY", "")
URL = "https://stitch.googleapis.com/mcp"

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = urllib.request.Request(
            URL,
            data=line.encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': API_KEY
            }
        )
        with urllib.request.urlopen(req) as resp:
            resp_bytes = resp.read()
            sys.stdout.buffer.write(resp_bytes + b'\n')
            sys.stdout.buffer.flush()
    except Exception as e:
        try:
            req_json = json.loads(line)
            req_id = req_json.get("id")
            if req_id is not None:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }
                sys.stdout.buffer.write(json.dumps(err_resp).encode('utf-8') + b'\n')
                sys.stdout.buffer.flush()
        except Exception:
            pass
        sys.stderr.write(f"Proxy error: {e}\n")
        sys.stderr.flush()

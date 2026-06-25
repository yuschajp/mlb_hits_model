"""
serve_dashboard.py

Runs the MLB prediction pipeline and serves the dashboard locally at
http://localhost:8765. Opens Chrome automatically on startup.

Usage:
    python scripts/serve_dashboard.py
"""

import http.server
import json
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PORT = 8765

_lock = threading.Lock()
_state = {"status": "idle", "last_run": None, "output": ""}


def run_pipeline():
    with _lock:
        _state["status"] = "running"
        _state["last_run"] = time.time()

    lines = []
    for script in [
        "scripts/run_daily.py",
        "scripts/find_value.py",
        "scripts/run_daily_hr.py",
        "scripts/find_value_hr.py",
        "scripts/publish_dashboard.py",
    ]:
        r = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=ROOT)
        lines.append(f"--- {script} ---\n{r.stdout.strip()}")
        if r.returncode != 0 and r.stderr:
            lines.append(f"[stderr] {r.stderr[:400]}")
    output = "\n".join(lines)

    with _lock:
        _state["status"] = "done"
        _state["output"] = output

    print(output)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def do_GET(self):
        if self.path.rstrip("/") == "/status":
            with _lock:
                body = json.dumps({
                    "status": _state["status"],
                    "last_run": _state["last_run"],
                    "output": _state["output"][-3000:],
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") == "/refresh":
            threading.Thread(target=run_pipeline, daemon=True).start()
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # silence access logs


if __name__ == "__main__":
    print("Running MLB model pipeline...\n")
    run_pipeline()

    server = http.server.HTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\nDashboard: {url}  (Ctrl+C to stop)\n")

    def open_chrome():
        try:
            webbrowser.get("chrome").open(url)
        except Exception:
            webbrowser.open(url)

    threading.Timer(0.5, open_chrome).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

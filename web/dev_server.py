"""Run the site on your own PC, no Vercel account or CLI needed:

    python web/dev_server.py            ->  open http://localhost:3000

It serves public/ and answers /api/predict with the same code Vercel runs.
"""
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "api"))
import _lib  # noqa: E402


class Dev(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/predict"):
            status, body, _ = _lib.handle(self.path)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Dev, directory=str(HERE / "public")))
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    print("Models found:", ", ".join(_lib.list_models()) or "none yet - run src/export_model.py first")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
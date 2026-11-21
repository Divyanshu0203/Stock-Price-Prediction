"""Vercel serverless function: GET /api/predict            -> list of available models
                              GET /api/predict?ticker=AMZN -> next-day forecast for that model
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lib  # noqa: E402  (helper module; the leading underscore keeps Vercel from treating it as a route)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status, body, cache_seconds = _lib.handle(self.path)
        try:
            payload = json.dumps(body, allow_nan=False).encode()
        except ValueError:
            status, payload, cache_seconds = 500, b'{"error": "Invalid number in response."}', 0
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if cache_seconds:
            self.send_header("Cache-Control", f"public, s-maxage={cache_seconds}, stale-while-revalidate=3600")
        else:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep function logs quiet
        pass
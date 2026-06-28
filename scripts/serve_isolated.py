#!/usr/bin/env python3
"""Static server for the dx bundle output with cross-origin isolation.

dx serve cannot host the threaded worker (no COOP/COEP, and it tree-shakes the
wasm-bindgen-rayon glue). So bundle the app, then serve the static output with
this, which adds the two isolation headers SharedArrayBuffer needs and serves
wasm with the right MIME type. Falls back to index.html for client-side routes.

Usage: serve_isolated.py [directory] [port]
"""

import sys
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else "target/dx/decompositions-app/release/web/public"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9595


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # The two headers that make the page cross-origin isolated, without
        # which SharedArrayBuffer (and so the rayon pool) is unavailable.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # No caching, so reruns after a rebuild always serve fresh wasm.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        # SPA fallback: serve index.html for any path that is not a real file.
        path = self.translate_path(self.path)
        if not os.path.exists(path) and not self.path.startswith("/dioxus-decompositions/"):
            self.path = "/index.html"
        return super().send_head()


if __name__ == "__main__":
    abs_dir = os.path.abspath(DIRECTORY)
    if not os.path.isdir(abs_dir):
        print(f"directory does not exist yet: {abs_dir}", file=sys.stderr)
        print("run: DECOMPOSITIONS_WORKER_THREADS=1 dx bundle -p decompositions-app --release --debug-symbols false", file=sys.stderr)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving {abs_dir} on http://localhost:{PORT} (COOP/COEP isolated)")
    httpd.serve_forever()

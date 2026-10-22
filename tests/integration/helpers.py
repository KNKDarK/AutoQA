"""Shared helpers for integration tests: local test pages, stub LLM clients."""
import http.server
import json
import socketserver
import threading
from types import SimpleNamespace
from typing import Any, Dict, List

MAIN_PAGE = """<!DOCTYPE html>
<html>
  <head><title>AutoQA test page</title></head>
  <body>
    <h1 id="heading">Checkout</h1>
    <nav><a id="home-link" href="#">Go home</a></nav>
    <form id="f">
      <input id="email" name="email" placeholder="Email address" />
      <button id="submit-btn" type="submit">Submit</button>
    </form>
    <button id="counter-btn" type="button">Clicks: 0</button>
    <div id="result">waiting</div>
    <script>
      document.getElementById('f').addEventListener('submit', function (e) {
        e.preventDefault();
        document.getElementById('result').textContent =
          'submitted: ' + document.getElementById('email').value;
      });
      document.getElementById('f').setAttribute('data-submitted', 'true');
    </script>
    <script>
      const btn = document.getElementById('counter-btn');
      let clicks = 0;
      btn.addEventListener('click', () => {
        clicks += 1;
        btn.textContent = 'Clicks: ' + clicks;
      });
    </script>
  </body>
</html>"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Request handler that never logs to stderr."""

    def log_message(self, *args, **kwargs) -> None:  # noqa: N802 (stdlib API)
        pass

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        pages: Dict[str, str] = getattr(self.server, "pages", {})
        path = self.path.split("?", 1)[0]
        body = pages.get(path)
        if body is None:
            self.send_error(404)
            return
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StaticSite:
    """Serve a dict of path -> html on an ephemeral 127.0.0.1 port."""

    def __init__(self, pages: Dict[str, str]):
        self.pages = dict(pages)
        self._httpd = None
        self._thread = None

    def __enter__(self) -> str:
        self._httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _QuietHandler)
        self._httpd.pages = self.pages
        port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class FakeGeminiClient:
    """Drop-in stub for google.genai.Client used inside AutoQAAgent."""

    calls: List[Dict[str, Any]] = []

    def __init__(self, script=None, api_key=None):
        self.api_key = api_key
        self.script = [dict(d) for d in (script or [])]
        self.calls = []

    @property
    def models(self):
        return self

    def _on_model(self):
        # Required because core.py accesses self.client.models.generate_content.
        return self

    def generate_content(self, model=None, contents=None, config=None):
        self.calls.append(
            {
                "model": model,
                "n_contents": len(contents or []),
                "config": config,
                "contents": [str(c) for c in (contents or [])],
            }
        )
        payload = self.script[min(len(self.calls), len(self.script)) - 1]
        text = (
            payload["text"]
            if isinstance(payload, dict) and "text" in payload
            else json.dumps(payload)
        )
        return SimpleNamespace(text=text)

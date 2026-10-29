"""Local read-only dashboard server for the paired SUMO processes."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

TEMPLATE = Path(__file__).with_name("demo_report.html")


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/state.json":
            data = {"run": self.server.run, "live": {}, "groups": [], "pair": {}}
            for name in ("no_forecast", "forecast"):
                source = self.server.folder / name / "live.json"
                try:
                    data["live"][name] = json.loads(source.read_text(encoding="utf-8"))
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    data["live"][name] = {"stage": "starting"}
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.split("?")[0] in ("/", "/index.html"):
            payload = TEMPLATE.read_text(encoding="utf-8").replace(
                "__REPORT_DATA__", json.dumps({"run": self.server.run, "groups": [],
                                              "pair": {}, "live_mode": True})
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def log_message(self, *args):
        pass


def start_server(folder, run):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(DashboardHandler, directory=str(folder)))
    server.folder = folder
    server.run = run
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"

# art_orch_bridge.py
#
# Runs a small local HTTP server inside Krita. POST Python code to it and it
# gets exec()'d in Krita's own live Python session — same effect as pasting
# into Scripter and hitting Run, but callable from revised_code_action.py's
# run() without any manual copy-paste.
#
# Execution is marshaled onto Krita's main GUI thread via a queued Qt signal,
# since the HTTP server runs in a background thread and Qt/Krita objects are
# not safe to touch from any thread but the main one.
#
# Install: copy this krita_bridge_plugin/pykrita/ folder's contents into
# Krita's pykrita resource folder (Settings > Manage Resources > Open
# Resource Folder, then into pykrita/), enable "Art Orch Bridge" under
# Settings > Configure Krita > Python Plugin Manager, restart Krita.

import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from krita import Extension

PORT = 8765


class _Executor(QObject):
    _request = pyqtSignal(str, object, object)   # object = generic Python object (Event, list)

    def __init__(self):
        super().__init__()
        self._request.connect(self._run_on_main_thread, Qt.QueuedConnection)

    def _run_on_main_thread(self, code, done_event, result_holder):
        ns = {"__name__": "__art_orch_bridge__"}
        try:
            exec(code, ns)
            # Executed code may set __bridge_result__ in its own namespace
            # (e.g. a base64-encoded exported image) to return more than
            # just a bare success signal.
            result_holder.append((True, ns.get("__bridge_result__", "OK")))
        except Exception:
            result_holder.append((False, traceback.format_exc()))
        finally:
            done_event.set()

    def run(self, code: str, timeout: float = 240.0):
        done_event = threading.Event()
        result_holder = []
        self._request.emit(code, done_event, result_holder)
        if not done_event.wait(timeout=timeout):
            return False, f"Timed out after {timeout}s waiting for Krita's main thread."
        return result_holder[0]


_executor = _Executor()


class _ExecHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        code = self.rfile.read(length).decode("utf-8")
        ok, message = _executor.run(code)
        self.send_response(200 if ok else 500)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # keep Krita's console quiet


class ArtOrchBridgeExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)
        self._server = None
        self._thread = None

    def setup(self):
        # Windows GUI launches often have no visible console for print(), so
        # log any startup failure to a file we can actually check afterward.
        import os
        log_path = os.path.join(os.path.expanduser("~"), "art_orch_bridge_error.log")
        try:
            self._server = HTTPServer(("127.0.0.1", PORT), _ExecHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"Art Orch Bridge listening on http://127.0.0.1:{PORT}")
            if os.path.isfile(log_path):
                os.remove(log_path)   # clear any stale error from a previous failed attempt
        except Exception:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    def createActions(self, window):
        pass

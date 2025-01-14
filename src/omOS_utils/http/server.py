import logging
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class _RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, callback: Optional[Callable] = None, *args, **kwargs):
        self.callback = callback
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_error_response(405, "GET method not allowed")

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            print(f"Received POST data: {post_data}")
            json_data = json.loads(post_data)

            # Process request through callback and get response
            if self.callback:
                response = self.callback(json_data, self.path)
                if response:
                    self.send_json_response(response)
                else:
                    self.send_error_response(500, "No response from callback handler")
            else:
                self.send_error_response(500, "No callback handler registered")

        except Exception as e:
            logger.error(f'Error processing request: {e}')
            self.send_error_response(400, str(e))

    def send_json_response(self, data: dict | str):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            if isinstance(data, str):
                response_json = json.dumps({"response": data})
            else:
                response_json = json.dumps(data)

            self.wfile.write(response_json.encode('utf-8'))

        except Exception as e:
            logger.error(f'Error sending response: {e}')
            self.send_error_response(500, "Error formatting response")

    def send_error_response(self, code: int, message: str):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_response = json.dumps({"error": message})
        self.wfile.write(error_response.encode('utf-8'))

class Server:
    def __init__(self, host: str = "localhost", port: int = 6791, timeout: int = 15):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.running: bool = True
        self.server: Optional[HTTPServer] = None
        self.callback: Optional[Callable] = None

    def _create_handler(self, *args, **kwargs):
        return _RequestHandler(self.callback, *args, **kwargs)

    def _run_server(self):
        self.server = HTTPServer((self.host, self.port), self._create_handler)
        self.server.timeout = self.timeout
        logger.info(f"HTTP server started on {self.host}:{self.port}")

        while self.running:
            try:
                self.server.handle_request()
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                if not self.running:
                    break

    def start(self):
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        logger.info("HTTP server thread started")

    def register_callback(self, callback: Callable):
        self.callback = callback
        logger.info("Registered request callback")

    def stop(self):
        self.running = False
        if self.server:
            self.server.server_close()
        logger.info("HTTP server stopped")

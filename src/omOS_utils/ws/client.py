import logging
import websockets
from websockets.sync.client import connect
import threading
from queue import Queue, Empty

from typing import Optional, Callable

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, url: str = "ws://localhost:6789"):
        self.url = url
        self.running: bool = True
        self.connected: bool = False
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.message_callback: Optional[Callable] = None
        self.message_queue: Queue = Queue()
        self.receiver_thread: Optional[threading.Thread] = None
        self.sender_thread: Optional[threading.Thread] = None

    def _receive_messages(self):
        while self.running and self.connected:
            try:
                message = self.websocket.recv()
                formatted_msg = self.format_message(message)
                logger.info(
                    f"Received WS Message: {formatted_msg}"
                )
                if self.message_callback:
                    self.message_callback(message)
            except websockets.ConnectionClosed:
                logger.info("WebSocket connection closed")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                self.connected = False
                break

    def _send_messages(self):
        while self.running:
            try:
                if self.connected and self.websocket:
                    message = self.message_queue.get_nowait()
                    try:
                        self.websocket.send(message)
                        formatted_msg = self.format_message(message)
                        logger.debug(f"Sent WS Message: {formatted_msg} to {self.url}")
                    except Exception as e:
                        logger.error(f"Failed to send message: {e}")
                        self.message_queue.put(message)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in send queue processing: {e}")
                self.connected = False

    def connect(self) -> bool:
        try:
            self.websocket = connect(self.url)
            self.connected = True

            # Start receiver and sender threads
            if not self.receiver_thread or not self.receiver_thread.is_alive():
                self.receiver_thread = threading.Thread(target=self._receive_messages, daemon=True)
                self.receiver_thread.start()

            if not self.sender_thread or not self.sender_thread.is_alive():
                self.sender_thread = threading.Thread(target=self._send_messages, daemon=True)
                self.sender_thread.start()

            logger.info(f"Connected to {self.url}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False

    def send_message(self, message: str | bytes):
        if self.connected:
            self.message_queue.put(message)

    def _run_client(self):
        while self.running:
            if not self.connected:
                if self.connect():
                    logger.info("Connection established")
                else:
                    logger.info("Connection failed, retrying in 5 seconds")
                    threading.Event().wait(5)  # Wait 5 seconds before retrying
            else:
                threading.Event().wait(0.1)

    def start(self):
        self.client_thread = threading.Thread(target=self._run_client, daemon=True)
        self.client_thread.start()
        logger.info("WebSocket client thread started")

    def register_message_callback(self, callback: Callable):
        self.message_callback = callback
        logger.info("Registered message callback")

    def format_message(self, msg: str | bytes, max_length: int = 200) -> str:
        try:
            if len(msg) <= max_length:
                return msg
            preview_size = max_length // 2 - 20
            return f"{msg[:preview_size]}...{msg[-preview_size:]}"
        except Exception as e:
            return f"<Error formatting message: {e}>"

    def is_connected(self) -> bool:
        return self.connected

    def stop(self):
        self.running = False
        if self.websocket:
            try:
                self.websocket.close()
                logger.info("WebSocket connection closed")
            except:
                pass
        self.connected = False
        self.message_queue.clear()
        logger.info("WebSocket client stopped")

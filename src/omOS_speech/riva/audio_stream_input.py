import logging
from queue import Queue, Empty
from typing import Optional, Callable, Any

from omOS_utils.ws import Server

logger = logging.getLogger(__name__)

class AudioStreamInput:
    def __init__(self):
        self.running: bool = True
        self.callback: Optional[Callable] = None
        self.audio_queue: Queue[Optional[bytes]] = Queue()

    def handle_ws_incoming_message(self, connection_id: str, message: Any):
        try:
            # Verify we received binary data
            if not isinstance(message, bytes):
                logger.error("Received non-binary message")
                return

            self.audio_queue.put(message)

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def setup_audio_stream(self):
        """Placeholder for custom audio stream setup"""
        return self

    def get_audio_chunk(self) -> Optional[bytes]:
        try:
            return self.audio_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.running = False

    def add(self, callback: Callable):
        self.callback = callback
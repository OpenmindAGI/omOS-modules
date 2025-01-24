import logging
from queue import Empty, Queue
from typing import Any, Optional

from ..interfaces import AudioStreamInputInterface

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioStreamInput(AudioStreamInputInterface):
    """
    A class for managing audio input streaming from WebSocket connections.

    This class provides a queue-based buffer for handling incoming audio data
    from WebSocket connections and makes it available for processing through
    a simple interface.

    Parameters
    ----------
    None
    """

    def __init__(self):
        self.running: bool = True
        self.audio_queue: Queue[Optional[bytes]] = Queue()

    def handle_ws_incoming_message(self, connection_id: str, message: Any):
        """
        Process incoming WebSocket messages containing audio data.

        Parameters
        ----------
        connection_id : str
            Identifier for the WebSocket connection
        message : Any
            The message received from the WebSocket connection,
            expected to be binary audio data
        """
        try:
            # Verify we received binary data
            if not isinstance(message, bytes):
                logger.error("Received non-binary message")
                return

            self.audio_queue.put(message)

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def setup_audio_stream(self):
        """
        Set up the audio stream (placeholder method).

        Returns
        -------
        AudioStreamInput
            The current instance for method chaining
        """
        return self

    def get_audio_chunk(self) -> Optional[bytes]:
        try:
            return self.audio_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        """
        Stop the audio stream processing.

        Sets the running flag to False to stop processing.
        """
        self.running = False

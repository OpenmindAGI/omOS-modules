import logging
import json
from typing import Optional, Callable, List
import argparse
import time
import threading

try:
    from omOS_utils.ws.client import Client
except ModuleNotFoundError:
    Client = None

logger = logging.getLogger(__name__)


class VILAProcessor:
    """
    VILA Vision Language Model processor for real-time video analysis.

    Processes video frames through VILA to generate text descriptions
    of interesting aspects in the video stream. Communicates with a remote
    VILA server via WebSocket.

    Parameters
    ----------
    model_args : argparse.Namespace
        Command line arguments for model configuration.
    callback : Optional[Callable[[str], None]], optional
        Callback function for processing model responses,
        by default None
    """

    def __init__(
        self,
        model_args: argparse.Namespace,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize VILA processor.

        Parameters
        ----------
        model_args : argparse.Namespace
            Command line arguments for model configuration
        callback : Optional[Callable[[str], None]], optional
            Callback function for processing model responses
        """
        self.model_args = model_args
        self.callback = callback
        self.response: str = ""
        self.running: bool = True
        self.image_buffer: List[str] = []
        self.response_timeout: int = 10 * 1000  # 10 seconds
        self.waiting_for_response: bool = False

        # Initialize WebSocket client
        host = getattr(
            model_args, "vila_host", "localhost"
        )  # Default to localhost if not set
        port = getattr(model_args, "vila_port", 8000)  # Default to 8000 if not set

        self.ws_client = Client(f"ws://{host}:{port}/ws")
        self.ws_client.register_message_callback(self._handle_ws_message)
        self.ws_client.start()

    def _handle_ws_message(self, message: str):
        """
        Handle incoming WebSocket messages.

        Parameters
        ----------
        message : str
            The received message in JSON format
        """
        try:
            data = json.loads(message)
            if "response" in data:
                self.response = data["response"]
                logger.info(f"VILA response: {self.response}")
                self.last_response_time = time.time() * 1000
                if self.callback:
                    self.callback(json.dumps({"vila_reply": self.response}))
                self.waiting_for_response = False
                self.timeout_thread.cancel()
            elif "error" in data:
                logger.error(f"VILA server error: {data['error']}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def on_video(self, image: str):
        """
        Callback function for buffering video frames.
        Skips frames if necessary and buffers them until the batch size is reached.

        Parameters
        ----------
        image : str
            The image in base64 format
        """
        # Make sure image buffer always gets the latest image but stays the same size
        if len(self.image_buffer) == self.model_args.vila_batch_size:
            self.image_buffer.pop(0)
        self.image_buffer.append(image)

    def handle_timeout(self):
        """
        Handle timeout for VILA response.
        """
        logger.warning("VILA response timeout")
        self.waiting_for_response = False

    def process_frames(self):
        """
        Main frame processing loop.

        Parameters
        ----------
        video_output : Any
            Video output stream handler
        video_source : Any
            Video input source handler
        """
        while self.running:
            # Send accumulated images to VILA server
            try:
                if (
                    self.ws_client.is_connected()
                    and len(self.image_buffer) == self.model_args.vila_batch_size
                    and not self.waiting_for_response
                ):
                    logger.info(
                        f"Sending {len(self.image_buffer)} images to remote VILA server"
                    )
                    message = {
                        "images": self.image_buffer,
                        "prompt": "What is the most interesting aspect in this series of images?",
                    }
                    self.ws_client.send_message(json.dumps(message))
                    self.waiting_for_response = True
                    # Fire a timeout after timeout seconds
                    self.timeout_thread = threading.Timer(
                        self.response_timeout / 1000, self.handle_timeout
                    )
                    self.timeout_thread.start()
            except Exception as e:
                logger.error(f"Error sending frames to VILA server: {e}")

    def stop(self):
        """
        Stop frame processing and cleanup resources.
        """
        self.running = False
        if self.ws_client:
            self.ws_client.stop()

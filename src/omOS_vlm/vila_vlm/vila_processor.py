import logging
import json
import base64
from io import BytesIO
from typing import Optional, Any, Callable, List
import time
import argparse
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

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
        self.model_args = model_args
        self.callback = callback
        self.last_image: Optional[np.ndarray] = None
        self.num_images: int = 0
        self.raw_response: str = ""
        self.response: str = ""
        self.running: bool = True
        self.image_buffer: List[str] = []

        # Initialize WebSocket client
        self.ws_client = Client(f"ws://{model_args.host}:{model_args.port}/ws")
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
                self.raw_response = data["response"]
                self.response = data["response"]  # No cleanup needed for VILA responses
                logger.info(f"VILA response: {self.response}")

                if self.callback:
                    self.callback(json.dumps({"vila_reply": self.response}))
            elif "error" in data:
                logger.error(f"VILA server error: {data['error']}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy image to base64 string.

        Parameters
        ----------
        image : np.ndarray
            Input image as numpy array

        Returns
        -------
        str
            Base64 encoded image string
        """
        # Convert numpy array to PIL Image
        pil_image = PILImage.fromarray(image)

        # Convert PIL Image to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _add_text_overlay(self, image: np.ndarray, text: str) -> np.ndarray:
        """
        Add text overlay to image.

        Parameters
        ----------
        image : np.ndarray
            Input image
        text : str
            Text to overlay

        Returns
        -------
        np.ndarray
            Image with text overlay
        """
        # Convert numpy array to PIL Image for text drawing
        pil_image = PILImage.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Add text with background
        x, y = 5, 5
        draw.text((x, y), text, fill=(255, 0, 0))

        # Convert back to numpy array
        return np.array(pil_image)

    def on_video(self, image: np.ndarray) -> np.ndarray:
        """
        Process incoming video frames.

        Parameters
        ----------
        image : np.ndarray
            Input video frame as numpy array

        Returns
        -------
        np.ndarray
            Annotated video frame
        """
        self.last_image = image.copy()

        annotation = "Accumulating:" + str(self.num_images)
        if self.num_images >= self.model_args.batch_size:
            annotation = "VILA:" + self.response

        return self._add_text_overlay(image, annotation)

    async def process_frames(self, video_output: Any, video_source: Any):
        """
        Main frame processing loop.

        Parameters
        ----------
        video_output : Any
            Video output stream handler
        video_source : Any
            Video input source handler
        """
        skip = 0
        while self.running:
            if self.last_image is None:
                continue

            if skip < self.model_args.frame_skip:
                skip += 1
                continue
            else:
                skip = 0

            # Convert and store image
            if self.last_image is not None:
                img_b64 = self._numpy_to_base64(self.last_image)
                self.image_buffer.append(img_b64)
                self.last_image = None

            if self.num_images < self.model_args.batch_size:
                self.num_images += 1
                continue
            else:
                self.num_images = 0

            # Send accumulated images to VILA server
            try:
                if self.ws_client.is_connected():
                    message = {
                        "images": self.image_buffer,
                        "prompt": "What is the most interesting aspect in this series of images?",
                    }
                    self.ws_client.send_message(json.dumps(message))
                else:
                    logger.warning("WebSocket not connected to VILA server")
            except Exception as e:
                logger.error(f"Error sending frames to VILA server: {e}")

            # Clear image buffer
            self.image_buffer = []

            if video_source.eos:
                video_output.stream.Close()
                break

    def stop(self):
        """
        Stop frame processing and cleanup resources.
        """
        self.running = False
        if self.ws_client:
            self.ws_client.stop()

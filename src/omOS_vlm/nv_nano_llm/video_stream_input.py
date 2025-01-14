import logging
import cv2
import base64
import numpy as np
import argparse
from typing import Optional, Callable, Any

# nano_llm is only available on the Jetson devices
try:
    import nano_llm
    from nano_llm.plugins import VideoOutput
    from jetson_utils import cudaFromNumpy, cudaAllocMapped, cudaConvertColor
except ModuleNotFoundError:
    VideoOutput = None
    cudaFromNumpy = None
    cudaAllocMapped = None
    cudaConvertColor = None

logger = logging.getLogger(__name__)

class VideoStreamInput:
    def __init__(self):
        self.video_output: nano_llm.plugins.VideoOutput = None
        self.running: bool = True
        self.callback: Optional[Callable] = None
        self.eos: bool = False

    def handle_ws_incoming_message(self, connection_id: str, message: str):
        try:
           # Decode base64 image
            img_bytes = base64.b64decode(message)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode image from WebSocket message")
                return

            # Convert OpenCV image (numpy array) to CUDA memory (in BGR format)
            bgr_cuda = cudaFromNumpy(frame, isBGR=True)

            # Allocate CUDA memory for RGB format
            rgb_cuda = cudaAllocMapped(
                width=bgr_cuda.width,
                height=bgr_cuda.height,
                format='rgb8'
            )

            # Convert from BGR to RGB
            cudaConvertColor(bgr_cuda, rgb_cuda)

            if self.callback:
                # Process frame through VLM pipeline
                processed_frame = self.callback(rgb_cuda)

                if processed_frame is not None and self.video_output:
                    self.video_output.process(processed_frame)

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def setup_video_stream(self, args: argparse.Namespace, callback: Optional[Callable], cuda_stream: int = 0):
        logger.info("Initializing WebSocket video stream handler")

        self.callback = callback

        # Initialize video output
        self.video_output = VideoOutput(args.rtp_url)
        self.video_output.start()

        return self, self.video_output

    def stop(self):
        self.running = False

        if self.video_output:
            self.video_output.stop()

    def add(self, callback: Optional[Callable], threaded: bool = False):
        self.callback = callback
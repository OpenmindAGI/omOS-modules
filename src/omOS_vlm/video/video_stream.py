import logging
import cv2
import time
import base64
import platform
import threading
from .video_utils import enumerate_video_devices
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class VideoStream():
    """
    Manages video capture and streaming from a camera device.

    Provides functionality to capture video frames from a camera device,
    process them, and stream them through a callback function. Supports
    both macOS and Linux camera devices.

    Parameters
    ----------
    frame_callback : Optional[Callable[[str], None]], optional
        Callback function to handle processed frame data.
        Function receives base64 encoded frame data.
        By default None
    """
    def __init__(self, frame_callback: Optional[Callable[[str], None]] = None):
        self._video_thread: Optional[threading.Thread] = None

        # Callback for video frame data
        self.frame_callback = frame_callback

        self.running: bool = True

    def on_video(self):
        """
        Main video capture and processing loop.

        Captures frames from the camera, encodes them to base64,
        and sends them through the callback if registered.

        Raises
        ------
        Exception
            If video streaming encounters an error
        """
        devices = enumerate_video_devices()
        if platform.system() == 'Darwin':
            camindex = 0 if devices else 0
        else:
            camindex = '/dev/video' + str(devices[0][0]) if devices else '/dev/video0'
        logger.info(f"Using camera: {camindex}")

        cap = cv2.VideoCapture(camindex)
        if not cap.isOpened():
            logger.error(f"Error opening video stream from {camindex}")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error reading frame from video stream")
                    time.sleep(0.1)
                    continue

                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = base64.b64encode(buffer).decode('utf-8')

                if self.frame_callback:
                    self.frame_callback(frame_data)

                time.sleep(0.033) # 30 fps

        except Exception as e:
            logger.error(f"Error streaming video: {e}")

    def _start_video_thread(self):
        """
        Initialize and start the video processing thread.

        Creates a new daemon thread for video processing if one isn't
        already running.
        """
        if self._video_thread is None or not self._video_thread.is_alive():
            self._video_thread = threading.Thread(
                target=self.on_video,
                daemon=True
            )
            self._video_thread.start()
            logger.info("Started video processing thread")

    def register_frame_callback(self, frame_callback: Callable[[str], None]):
        """
        Register a callback function for processed frames.

        Parameters
        ----------
        frame_callback : Callable[[str], None]
            Function to be called with base64 encoded frame data
        """
        self.frame_callback = frame_callback

    def start(self):
        """
        Start video capture and processing.

        Initializes the video processing thread and begins
        capturing frames.
        """
        self._start_video_thread()

    def stop(self):
        """
        Stop video capture and clean up resources.

        Stops the video processing loop and waits for the
        processing thread to finish.
        """
        self.running = False

        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)

        logger.info("Stopped video processing thread")
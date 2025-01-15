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
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self._video_thread: Optional[threading.Thread] = None

        # Callback for video data
        self.callback = callback

        self.running: bool = True

    def on_video(self):
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

                if self.callback:
                    self.callback(frame_data)

                time.sleep(0.033) # 30 fps

        except Exception as e:
            logger.error(f"Error streaming video: {e}")

    def _start_video_thread(self):
        if self._video_thread is None or not self._video_thread.is_alive():
            self._video_thread = threading.Thread(
                target=self.on_video,
                daemon=True
            )
            self._video_thread.start()
            logger.info("Started video processing thread")

    def start(self):
        # Start video processing thread
        self._start_video_thread()

    def stop(self):
        self.running = False

        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)

        logger.info("Stopped video processing thread")
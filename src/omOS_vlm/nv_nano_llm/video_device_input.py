import logging
from typing import Any, Callable
import argparse

# nano_llm is only available on the Jetson devices
try:
    from nano_llm.plugins import VideoSource, VideoOutput
except ModuleNotFoundError:
    VideoSource = None
    VideoOutput = None

from ..video import enumerate_video_devices

logger = logging.getLogger(__name__)

class VideoDeviceInput:
    def __init__(self):
        self.video_source: Any = None
        self.video_output: Any = None

    def setup_video_devices(self, args: argparse.Namespace, callback: Callable[[bytes], None], cuda_stream: int = 0):
        devices = enumerate_video_devices()
        camindex = '/dev/video' + str(devices[0][0]) if devices else '/dev/video0'
        logger.info(f"Using camera: {camindex}")

        # Initialize video source and output
        self.video_source = VideoSource(camindex, cuda_stream=cuda_stream)
        self.video_source.add(callback, threaded=False)
        self.video_source.start()

        self.video_output = VideoOutput(args.rtp_url)
        self.video_output.start()

        return self.video_source, self.video_output

    def stop(self):
        if self.video_source:
            self.video_source.stop()
        if self.video_output:
            self.video_output.stop()

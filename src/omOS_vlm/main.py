import logging
import time
import threading
import argparse

from typing import Optional, Any

## TODO: ArgParser should be defined in om1_core_utils
from .nv_nano_llm import VLMProcessor, VideoDeviceInput, VideoStreamInput, ArgParser
from omOS_utils import ws
from .video import VideoStream
from .processor import ConnectionProcessor

logger = logging.getLogger(__name__)

class Application:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ws_server: ws.Server = None
        self.connection_processor: Optional[ConnectionProcessor] = None
        self.video_device_processor: Optional[VideoDeviceInput] = None
        self.vlm_processor: Optional[VLMProcessor] = None
        self.running: bool = False

        self.args.ws_port = self.args.ws_port or 6789
        self.args.rtp_url = self.args.rtp_url or "rtp://192.168.1.170:1234"
        self.args.model = self.args.model or "Efficient-Large-Model/VILA1.5-3b"
        self.args.max_context_len = self.args.max_context_len or 256
        self.args.max_new_tokens = self.args.max_new_tokens or 32

        # Set logging level in the lower level modules
        logging.basicConfig(level=getattr(logging, self.args.log_level.upper()))
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(getattr(logging, self.args.log_level.upper()))

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        parser = ArgParser(extras=ArgParser.Defaults)
        parser.add_argument("--ws-host", type=str, default="localhost", help="WebSocket server host")
        parser.add_argument("--ws-port", type=int, help="WebSocket server port")
        parser.add_argument("--rtp-url", type=str, help="RTP URL for compressed video stream")
        parser.add_argument("--server-mode", default=False, action="store_true", help="Run in server mode")
        parser.add_argument("--remote-url", type=str, help="Remote webSocket URL server for video stream input")
        parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")

        return parser.parse_args()

    # Demo for video streaming and retrieving VLM results
    # VLM results will be directly sent back to the client
    def setup_video_streaming(self):
        self.ws_client = ws.Client(url=self.args.remote_url)
        self.ws_client.start()

        video_streamer = VideoStream(self.ws_client.send_message)

        video_thread = threading.Thread(
            target=video_streamer.on_video,
            daemon=True
        )
        video_thread.start()

    def setup_vlm_processing(self):
        self.ws_server = ws.Server(host=self.args.ws_host, port=self.args.ws_port)

        if self.args.server_mode:
            self.connection_processor = ConnectionProcessor(self.args, VLMProcessor, VideoStreamInput)
            self.connection_processor.set_server(self.ws_server)
        else:
            self.vlm_processor = VLMProcessor(self.args, self.ws_server.handle_global_response)
            self.video_device_processor = VideoDeviceInput()
            self.video_source, self.video_output = self.video_device_processor.setup_video_devices(
                self.args,
                self.vlm_processor.on_video
            )

            vlm_thread = threading.Thread(
                target=self.vlm_processor.process_frames,
                args=(self.video_output, self.video_source),
                daemon=True
            )
            vlm_thread.start()

        self.ws_server.start()

    def start(self):
        logger.info("Starting application...")

        if self.args.remote_url:
            logger.info("Streaming video to WebSocket server")
            self.setup_video_streaming()
        else:
            logger.info(f"Starting VLM processing with model: {self.args.model}")
            self.setup_vlm_processing()

        self.running = True

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("\nShutting down gracefully... (This may take a few seconds)")
        self.running = False

        if self.ws_server:
            self.ws_server.stop()

        if self.connection_processor:
            self.connection_processor.stop()

        if self.vlm_processor:
            self.vlm_processor.stop()

        if self.video_device_processor:
            self.video_device_processor.stop()

        if self.video_source:
            self.video_source.stop()

        if self.video_output:
            self.video_output.stop()

def main():
    args = Application.parse_arguments()
    app = Application(args)
    app.start()

if __name__ == "__main__":
    main()
import logging
import time
import threading
from typing import Optional, Any

from omOS_utils import ws
from omOS_vlm.video import VideoStream
from .args import VILAArgParser
from .vila_processor import VILAProcessor

logger = logging.getLogger(__name__)


class Application:
    """
    Main application class for standalone VILA VLM processing.

    Coordinates video input/output, WebSocket communication, and VLM processing
    for the VILA model in standalone mode.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration parameters
    """

    def __init__(self, args):
        self.args = args
        self.ws_server: Optional[ws.Server] = None
        self.vlm_processor: Optional[VILAProcessor] = None
        self.video_stream: Optional[VideoStream] = None
        self.running: bool = False

        # Set logging level
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(getattr(logging, args.log_level.upper()))

    @staticmethod
    def parse_arguments():
        """
        Parse command-line arguments for VILA configuration.

        Returns
        -------
        argparse.Namespace
            Parsed command-line arguments
        """
        parser = VILAArgParser()
        return parser.parse_args()

    def setup_vlm_processing(self):
        """
        Configure VLM processing components.

        Sets up VILA processor, WebSocket server, and video streaming.
        """
        # Initialize WebSocket server
        self.ws_server = ws.Server(
            host=getattr(self.args, "ws_host", "localhost"),
            port=getattr(self.args, "ws_port", 6789),
        )
        self.ws_server.start()

        # Initialize VILA processor
        self.vlm_processor = VILAProcessor(
            self.args, self.ws_server.handle_global_response
        )
        logger.info(f"VILA VLM initialized with args: {self.args}")

        # Set up video streaming
        self.video_stream = VideoStream(
            self.vlm_processor.on_video, fps=getattr(self.args, "fps", 10)
        )
        self.video_stream.start()

        # Start VLM processing thread
        vlm_thread = threading.Thread(
            target=self.vlm_processor.process_frames,
            daemon=True,
        )
        vlm_thread.start()

    def start(self):
        """
        Start the application.

        Initializes and starts all components based on configuration.
        Handles main application loop and graceful shutdown.
        """
        logger.info("Starting VILA VLM application...")
        self.setup_vlm_processing()
        self.running = True

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """
        Stop the application and cleanup resources.

        Performs graceful shutdown of all components.
        """
        logger.info("Shutting down VILA VLM...")
        self.running = False

        if self.ws_server:
            self.ws_server.stop()

        if self.vlm_processor:
            self.vlm_processor.stop()

        if self.video_stream:
            self.video_stream.stop()


def main():
    """
    Main entry point for running VILA VLM standalone.
    """
    args = Application.parse_arguments()
    app = Application(args)
    app.start()


if __name__ == "__main__":
    main()

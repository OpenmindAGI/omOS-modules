import logging
import argparse
import threading
import time
from typing import Optional, Any

from omOS_utils import ws, http

# Audio input from microphone
from .audio import AudioInputStream
# Riva ASR and TTS
from .riva import (
    ASRProcessor, TTSProcessor, AudioDeviceInput, AudioStreamInput,
    add_asr_config_argparse_parameters, add_tts_argparse_parameters, add_connection_argparse_parameters
)
# Multithreading
from .processor import ConnectionProcessor

logger = logging.getLogger(__name__)

class Application:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ws_server: Optional[ws.Server] = None
        self.http_server: Optional[http.Server] = None
        self.connection_processor: Optional[ConnectionProcessor] = None
        self.asr_processor: Optional[ASRProcessor] = None
        self.audio_source: Optional[AudioDeviceInput] = None
        self.running: bool = False

        self.args.ws_port = self.args.ws_port or 6790
        self.args.http_port = self.args.http_port or 6791

        # Set logging level in the lower level modules
        logging.basicConfig(level=getattr(logging, self.args.log_level.upper()))
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(getattr(logging, self.args.log_level.upper()))

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--ws-host", type=str, default="localhost", help="WebSocket server host")
        parser.add_argument("--ws-port", type=int, help="WebSocket server port")
        parser.add_argument("--http-host", type=str, default="localhost", help="HTTP server host")
        parser.add_argument("--http-port", type=int, help="HTTP server port")
        parser.add_argument("--server-mode", default=False, action="store_true", help="Run in server mode")
        parser.add_argument("--remote-url", type=str, help="Remote webSocket URL server for audio stream input")
        parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")

        # nvidia riva
        ## ASR
        parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
        parser = add_connection_argparse_parameters(parser)
        ## TTS
        parser = add_tts_argparse_parameters(parser)
        return parser.parse_args()

    # Demo for audio streaming and retrieving ASR results
    # ARS reulst will be directly sent back to the client
    def setup_audio_streaming(self):
        self.ws_client = ws.Client(url=self.args.remote_url)
        self.ws_client.start()

        audio_streamer = AudioInputStream(audio_data_callback=self.ws_client.send_message)
        audio_streamer.start()

        audio_thread = threading.Thread(
            target=audio_streamer.on_auido,
            daemon=True
        )
        audio_thread.start()

    def setup_asr_processing(self):
        self.ws_server = ws.Server(host=self.args.ws_host, port=self.args.ws_port)

        # Create thread processor
        if self.args.server_mode:
            self.connection_processor = ConnectionProcessor(self.args, ASRProcessor, AudioStreamInput)
            self.connection_processor.set_server(self.ws_server)
        else:
            # Use the default audio input
            self.asr_processor = ASRProcessor(self.args, self.ws_server.handle_global_response)
            self.audio_source = AudioDeviceInput()
            self.audio_source.setup_audio_devices()

            asr_thread = threading.Thread(
                target=self.asr_processor.process_audio,
                args=(self.audio_source,),
                daemon=True
            )
            asr_thread.start()

        self.ws_server.start()

    def setup_tts_processing(self):
        self.tts_processor = TTSProcessor(self.args)

        self.http_server = http.Server(host=self.args.http_host, port=self.args.http_port)
        self.http_server.register_callback(self.tts_processor.process_tts)
        self.http_server.start()

        # I assume that tts service is not required to be run in a separate thread
        # tts_thread = threading.Thread(
        #     target=self.tts_processor.process_tts,
        #     daemon=True
        # )
        # tts_thread.start()

    def start(self):
        logger.info("Starting application...")

        if self.args.remote_url:
            logger.info("Streaming audio to WebSocket server")
            self.setup_audio_streaming()
        else:
            # hardcode the model to Riva
            logger.info(f"Starting ARS processing with model: Riva")
            self.setup_asr_processing()
            logger.info(f"Started TTS processing with model: Riva")
            self.setup_tts_processing()

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

        if self.http_server:
            self.http_server.stop()

        if self.connection_processor:
            self.connection_processor.stop()

        if self.asr_processor:
            self.asr_processor.stop()

        if self.audio_source:
            self.audio_source.stop()

        if self.tts_processor:
            self.tts_processor.stop()

def main():
    args = Application.parse_arguments()
    app = Application(args)
    app.start()

if __name__ == "__main__":
    main()

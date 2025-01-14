# The connection processor isused to handle the multi-threading of
# the audio stream and the ASR processor.

import logging
import threading
import argparse
from typing import Dict, Optional
from omOS_utils import ws

# TODO
# The standard interface of the ASRProcessor and AudioStreamInput classes
from ..riva import ASRProcessor, AudioStreamInput

logger = logging.getLogger(__name__)

class ConnectionProcessor:
    def __init__(self, arg: argparse.Namespace, asr_processor_class: type, audio_stream_input_class: type):
        self.args = arg
        self.asr_processor_class = asr_processor_class
        self.audio_stream_input_class = audio_stream_input_class
        self.asr_processors: Dict[str, ASRProcessor] = {}
        self.audio_sources: Dict[str, AudioStreamInput] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.ws_server: Optional[ws.Server] = None

    def set_server(self, ws_server: ws.Server):
        self.ws_server = ws_server

        self.ws_server.register_connection_callback(lambda event, conn_id: self.handle_connection_event(event, conn_id))

    def handle_connection_event(self, event: str, connection_id: str):
        if event == 'connect':
            self.handle_new_connection(connection_id)
        elif event == 'disconnect':
            self.handle_connection_closed(connection_id)

    def handle_new_connection(self, connection_id: str):
        asr_processor = self.asr_processor_class(
            self.args,
            callback=lambda message: self.ws_server.handle_response(connection_id, message)
        )

        self.asr_processors[connection_id] = asr_processor

        # Create audio stream source for this connection
        audio_source = self.audio_stream_input_class()
        audio_source.setup_audio_stream()
        self.audio_sources[connection_id] = audio_source

        # Register message callback for audio stream
        self.ws_server.register_message_callback(
            connection_id,
            lambda conn_id, message: audio_source.handle_ws_incoming_message(conn_id, message)
        )

        processing_thread = threading.Thread(
            target=asr_processor.process_audio,
            args=(audio_source,),
        )
        self.processing_threads[connection_id] = processing_thread
        processing_thread.start()

        logger.info(f"Started processing thread for connection {connection_id}")

    def handle_connection_closed(self, connection_id: str):
        if connection_id in self.asr_processors:
            self.asr_processors[connection_id].stop()
            del self.asr_processors[connection_id]

        if connection_id in self.audio_sources:
            self.audio_sources[connection_id].stop()
            del self.audio_sources[connection_id]

        if connection_id in self.processing_threads:
            del self.processing_threads[connection_id]

        logger.info(f"Stopped processing thread for connection {connection_id}")

    def stop(self):
        for connection_id in list(self.asr_processors.keys()):
            self.handle_connection_closed(connection_id)

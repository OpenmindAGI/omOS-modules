import logging
import json
import time
import argparse
from typing import Optional, Callable, Any

# riva is only available on Jetson devices
try:
    from riva import client
    from riva.client import (
        ASRService,
        StreamingRecognitionConfig,
    )
except ModuleNotFoundError:
    client = None

class ASRProcessor:
    def __init__(self, model_args: argparse.Namespace, callback: Optional[Callable] = None):
        self.model: Optional[ASRService] = None
        self.model_config: Optional[StreamingRecognitionConfig] = None
        self.args = model_args
        self.callback = callback
        self.running: bool = True

        # Customize the ASR model
        self.args.stop_threshold = 0.99
        self.args.stop_threshold_eou = 0.99

        self._initialize_model()

    def _initialize_model(self):
        auth = client.Auth(self.args.ssl_cert, self.args.use_ssl, self.args.server, self.args.metadata)
        self.model = client.ASRService(auth)
        self.model_config = client.StreamingRecognitionConfig(
            config=client.RecognitionConfig(
                encoding=client.AudioEncoding.LINEAR_PCM,
                language_code=self.args.language_code,
                model=self.args.model_name,
                max_alternatives=1,
                profanity_filter=self.args.profanity_filter,
                enable_automatic_punctuation=self.args.automatic_punctuation,
                verbatim_transcripts=not self.args.no_verbatim_transcripts,
                sample_rate_hertz=self.args.asr_sample_rate_hz,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        client.add_endpoint_parameters_to_config(
            self.model_config,
            self.args.start_history,
            self.args.start_threshold,
            self.args.stop_history,
            self.args.stop_history_eou,
            self.args.stop_threshold,
            self.args.stop_threshold_eou
        )
        client.add_custom_configuration_to_config(
            self.model_config,
            self.args.custom_configuration
        )

    def on_audio(self, audio: bytes) -> bytes:
        return audio

    def _yield_audio_chunks(self, audio_source: Any):
        while self.running:
            if audio_source:
                chunk = audio_source.get_audio_chunk()
                if chunk:
                    yield chunk
            time.sleep(0.01)  # Small delay to prevent busy waiting

    def process_audio(self, audio_source: Any):
        responses = self.model.streaming_response_generator(
            audio_chunks=self._yield_audio_chunks(audio_source),
            streaming_config=self.model_config
        )

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            if not transcript:
                continue

            if result.is_final:
                logging.info(f"ASR: {transcript}")
                if self.callback:
                    self.callback(json.dumps({"asr_reply": transcript}))

    def stop(self):
        self.running = False
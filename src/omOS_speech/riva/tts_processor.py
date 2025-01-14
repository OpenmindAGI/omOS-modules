import logging
import base64
import argparse
from typing import Optional, Dict, Any

# riva is only available on Jetson devices
try:
    from riva import client
    from riva.client import (
        SpeechSynthesisService
    )
except ModuleNotFoundError:
    client = None

logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self, model_args: argparse.Namespace):
        self.model: Optional[SpeechSynthesisService] = None
        self.args = model_args
        self.running: bool = True

        self._initialize_model()

    def _initialize_model(self):
        auth = client.Auth(self.args.ssl_cert, self.args.use_ssl, self.args.server, self.args.metadata)
        self.model = client.SpeechSynthesisService(auth)

        # hot load the model
        try:
            self.generate_tts("omOS is ready")
        except Exception as e:
            logger.error(f"Error initializing TTS model: {e}")

    def generate_tts(self, text: str):
        return self.model.synthesize(text, self.args.voice, self.args.language_code, sample_rate_hz=self.args.tts_sample_rate_hz,
            audio_prompt_file=self.args.audio_prompt_file, quality=20 if self.args.quality is None else self.args.quality,
            custom_dictionary={}
        )

    def process_tts(self, tts_input: Dict[Any, Any], *args, **kwargs):
        try:
            if isinstance(tts_input, dict):
                tts_input = tts_input['text']
            else:
                return None

            logger.info(f"Processing TTS: {tts_input}")
            result = self.generate_tts(tts_input)
            audio_b64 = base64.b64encode(result.audio).decode('utf-8')

            return {
                "response": audio_b64,
                "content_type": "audio/wav"
            }
        except Exception as e:
            logger.error(f"Error processing TTS: {e}")
            return None

    def stop(self):
        self.running = False
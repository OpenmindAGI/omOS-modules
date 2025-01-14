import logging
import pyaudio
import requests
import json
import base64
import threading
import os
from queue import Queue, Empty
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class AudioOutputStream():
    def __init__(self, url: str, rate: int = 16000, device: int = None, tts_state_callback: Optional[Callable] = None):
        self._url = url
        self._rate = rate
        self._device = device

        # Callback for TTS state
        self._tts_state_callback = tts_state_callback

        # Audio interface
        self.stream: Optional[pyaudio.Stream] = None

        # Pending output queue
        self._pending_output: Queue[Optional[str]] = Queue()

        # Initialize audio interface
        self._audio_interface = pyaudio.PyAudio()

        self.running: bool = True

        # Find a suitable output device
        try:
            if self._device is None:
                device_count = self._audio_interface.get_device_count()
                logger.info(f"Found {device_count} audio devices")

                output_device = None
                for i in range(device_count):
                    device_info = self._audio_interface.get_device_info_by_index(i)
                    logger.info(f"Device {i}: {device_info['name']}")
                    logger.info(f"Max Output Channels: {device_info['maxOutputChannels']}")

                    if device_info['maxOutputChannels'] > 0:
                        output_device = device_info
                        self._device = i
                        break

                if output_device is None:
                    raise ValueError("No output device found")
            else:
                device_info = self._audio_interface.get_device_info_by_index(self._device)
                logger.info(f"Selected output device: {device_info['name']} ({self._device})")
                if device_info['maxOutputChannels'] == 0:
                    raise ValueError("Selected output device has no output channels")
                else:
                    output_device = device_info

            logger.info(f"Selected output device: {output_device['name']}")

            self.stream = self._audio_interface.open(
                output_device_index=self._device,
                format=pyaudio.paInt16,
                channels=min(2, output_device['maxOutputChannels']),  # Use up to 2 channels
                rate=self._rate,
                output=True,
                frames_per_buffer=1024
            )
            logger.info("Successfully opened audio stream")

        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
            if self.stream:
                self.stream.close()
            self.stream = None
            raise

    def set_tts_state_callback(self, callback: Callable):
        self._tts_state_callback = callback

    def add(self, audio_data: str):
        self._pending_output.put(audio_data)

    def _process_audio(self):
        while self.running:
            try:
                tts_input = self._pending_output.get_nowait()
                response = requests.post(
                    self._url,
                    data=json.dumps({"text": tts_input}),
                    headers={"Content-Type": "application/json"},
                    timeout=(5, 15)
                )
                logger.info(f"Received TTS response: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result and self.stream:
                        audio_data = result["response"]
                        audio_bytes = base64.b64decode(audio_data)

                        logger.info(f"Received audio data: {len(audio_bytes)} bytes")

                        self._tts_callback(True)

                        self.stream.write(audio_bytes)

                        self._tts_callback(False)

                        logger.info(f"Processed TTS request: {tts_input}")
                else:
                    logger.error(f"Error processing TTS request: {response.status_code}")
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

    def _tts_callback(self, is_active: bool):
        if self._tts_state_callback:
            self._tts_state_callback(is_active)

    def start(self):
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()

    def run_interactive(self):
        logger.info("Running interactive audio output stream")
        try:
            while self.running:
                user_input = input("> ")
                if user_input.lower() == 'quit':
                    break
                self.add(user_input)
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tts_url = os.getenv("TTS_URL", "http://localhost:6791")
    audio_output = AudioOutputStream(tts_url)
    audio_output.start()
    audio_output.run_interactive()
    audio_output.stop()

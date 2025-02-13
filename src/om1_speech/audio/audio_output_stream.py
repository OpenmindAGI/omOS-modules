import argparse
import base64
import json
import logging
import threading
import time
from queue import Empty, Queue
from typing import Callable, Dict, Optional

import pyaudio
import requests

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioOutputStream:
    """
    A class for managing audio output and text-to-speech (TTS) conversion.

    Parameters
    ----------
    url : str
        The URL endpoint for the text-to-speech service
    rate : int, optional
        The sampling rate in Hz for audio output (default: 8000)
    device : int, optional
        The output device index. If None, uses the first available output device
        (default: None)
    device_name: str, optional
        The output device name. If None, uses the first available output device
        (default: None)
    tts_state_callback : Optional[Callable], optional
        A callback function to receive TTS state changes (active/inactive)
        (default: None)
    headers : Optional[Dict[str, str]], optional
        Additional headers to include in the HTTP request (default: None)
    """

    def __init__(
        self,
        url: str,
        rate: int = 8000,
        device: int = None,
        device_name: str = None,
        tts_state_callback: Optional[Callable] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self._url = url
        self._rate = rate
        self._device = device
        self._device_name = device_name

        # Process headers
        self._headers = headers or {}
        if "Content-Type" not in self._headers:
            self._headers["Content-Type"] = "application/json"

        # Callback for TTS state
        self._tts_state_callback = tts_state_callback

        # Audio interface
        self.stream: Optional[pyaudio.Stream] = None

        # Pending requests queue
        self._pending_requests: Queue[Optional[str]] = Queue()

        # Initialize audio interface
        self._audio_interface = pyaudio.PyAudio()

        self.running: bool = True

        if self._device is not None and self._device_name is not None:
            logger.error("Only one of device or device_name can be specified")
            raise ValueError("Only one of device or device_name can be specified")

        # Find a suitable output device
        try:
            output_device = None
            device_count = self._audio_interface.get_device_count()
            logger.info(f"Found {device_count} audio devices")

            if self._device is not None:
                device_info = self._audio_interface.get_device_info_by_index(
                    self._device
                )
                logger.info(
                    f"Selected output device: {device_info['name']} ({self._device})"
                )
                if device_info["maxOutputChannels"] == 0:
                    raise ValueError("Selected output device has no output channels")
                else:
                    output_device = device_info
            elif self._device_name is not None:
                for i in range(device_count):
                    device_info = self._audio_interface.get_device_info_by_index(i)
                    if (
                        device_info["name"].lower() == self._device_name.lower()
                        and device_info["maxOutputChannels"] > 0
                    ):
                        output_device = device_info
                        self._device = i
                        break
                if output_device is None:
                    raise ValueError(
                        f"No output device found with name {self._device_name}"
                    )
            else:
                for i in range(device_count):
                    device_info = self._audio_interface.get_device_info_by_index(i)
                    logger.info(f"Device {i}: {device_info['name']}")
                    logger.info(
                        f"Max Output Channels: {device_info['maxOutputChannels']}"
                    )

                    if device_info["maxOutputChannels"] > 0:
                        output_device = device_info
                        self._device = i
                        break

                if output_device is None:
                    raise ValueError("No output device found")

            logger.info(
                f"Selected output device: {output_device['name']} at index {self._device}"
            )

            self.stream = self._audio_interface.open(
                output_device_index=self._device,
                format=pyaudio.paInt16,
                channels=min(
                    2, output_device["maxOutputChannels"]
                ),  # Use up to 2 channels
                rate=self._rate,
                output=True,
                frames_per_buffer=1024,
            )
            logger.info("Successfully opened audio stream")

        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
            if self.stream:
                self.stream.close()
            self.stream = None
            raise

    def set_tts_state_callback(self, callback: Callable):
        """
        Set a callback function for TTS state changes.

        Parameters
        ----------
        callback : Callable
            Function to be called when TTS state changes (active/inactive)
        """
        self._tts_state_callback = callback

    def add_request(self, audio_request: Dict[str, str]):
        """
        Add request to the TTS processing queue.

        Parameters
        ----------
        audio_request : Dict[str, str]
            Request to be processed by the TTS service
        """
        self._pending_requests.put(audio_request)

    def _process_audio(self):
        """
        Process the TTS queue and play audio output.

        Makes HTTP requests to the TTS service, converts responses to audio,
        and plays them through the audio device.
        """
        while self.running:
            try:
                tts_request = self._pending_requests.get_nowait()
                response = requests.post(
                    self._url,
                    data=json.dumps(tts_request),
                    headers=self._headers,
                    timeout=(5, 15),
                )
                logger.info(f"Received TTS response: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result and self.stream:
                        audio_data = result["response"]
                        self._write_audio(audio_data)
                        logger.debug(f"Processed TTS request: {tts_request}")
                else:
                    logger.error(
                        f"Error processing TTS request: {response.status_code}"
                    )
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

    def _write_audio(self, audio_data: bytes):
        """
        Write audio data to the output stream.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        """
        if self.stream:
            audio_bytes = base64.b64decode(audio_data)

            self._tts_callback(True)

            self.stream.write(audio_bytes)

            time.sleep(1)

            self._tts_callback(False)

    def _tts_callback(self, is_active: bool):
        """
        Invoke the TTS state callback if set.

        Parameters
        ----------
        is_active : bool
            Whether TTS is currently active
        """
        if self._tts_state_callback:
            self._tts_state_callback(is_active)

    def start(self):
        """
        Start the audio processing thread.

        Initializes a daemon thread for processing the TTS queue.
        """
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()

    def run_interactive(self):
        """
        Run an interactive console for text-to-speech conversion.

        Allows users to input text for TTS conversion until 'quit' is entered
        or KeyboardInterrupt is received.
        """
        logger.info(
            "Running interactive audio output stream. Please enter text for TTS conversion."
        )
        try:
            while self.running:
                user_input = input()
                if user_input.lower() == "quit":
                    break
                self.add_request({"text": user_input})
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()

    def stop(self):
        """
        Stop the audio output stream and cleanup resources.
        """
        self.running = False


def main():
    """
    Main function for running the audio output stream.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tts-url", type=str, required=True, help="URL for the TTS service"
    )
    parser.add_argument("--device", type=int, default=None, help="Output device index")
    parser.add_argument(
        "--rate", type=int, default=8000, help="Audio output rate in Hz"
    )
    args = parser.parse_args()

    audio_output = AudioOutputStream(args.tts_url, device=args.device, rate=args.rate)
    audio_output.start()
    audio_output.run_interactive()
    audio_output.stop()


if __name__ == "__main__":
    main()

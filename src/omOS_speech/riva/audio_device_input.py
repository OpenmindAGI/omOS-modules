# Description: Audio device input class for capturing audio from a microphone
# A partial of code comes from https://github.com/nvidia-riva/python-clients/blob/main/riva/client/audio_io.py

import logging
import pyaudio
import queue
from typing import Optional, Callable, Any, Dict, Tuple, Union

logger = logging.getLogger(__name__)

class AudioDeviceInput():
    def __init__(
        self,
        rate: int = 16000,
        chunk: int = 4048,
        device: Optional[Union[str, int, float, Any]] = None,
        callback: Optional[Callable] = None
    ):
        self._rate = rate
        self._chunk = chunk
        self._device = device
        self.callback = callback

        # Thread-safe buffer for audio data
        self._buff: queue.Queue[Optional[bytes]] = queue.Queue()
        self.running: bool = True
        self._audio_interface: Optional[pyaudio.PyAudio] = None
        self._audio_stream: Optional[pyaudio.Stream] = None

    def setup_audio_devices(self) -> 'AudioDeviceInput':
        if not self.running:
            return self

        self._audio_interface = pyaudio.PyAudio()

        # Get default device if none specified
        if self._device is None:
            try:
                default_info = self._audio_interface.get_default_input_device_info()
                self._device = default_info['index']
            except Exception as e:
                logger.error(f"Error getting default input device: {e}")
                self._device = None

        try:
            self._audio_stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                input_device_index=self._device,
                channels=1,
                rate=self._rate,
                input=True,
                frames_per_buffer=self._chunk,
                stream_callback=self._fill_buffer,
            )

            logger.info(f"Started audio stream with device {self._device}")

        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            self._audio_interface.terminate()
            raise

        return self

    def stop(self):
        self.running = False

        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._audio_interface:
            self._audio_interface.terminate()

        self._buff.put(None)
        logger.info("Stopped audio stream")

    def _fill_buffer(self, in_data: bytes, frame_count: int, time_info: Dict[str, Any], status_flags: int) -> Tuple[None, int]:
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def get_audio_chunk(self) -> Optional[bytes]:
        while self.running:
            chunk = self._buff.get()
            if chunk is None:
                return

            # Collect additional chunks that are immediately available
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            return b''.join(data)

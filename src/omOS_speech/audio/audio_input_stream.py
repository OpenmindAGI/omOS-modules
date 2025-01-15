# Description: Audio stream class for capturing audio from a microphone
# A partial of code comes from https://github.com/nvidia-riva/python-clients/blob/main/riva/client/audio_io.py

import logging
import pyaudio
import queue
import threading

from typing import Optional, Callable, Generator, Tuple, Union, Any

logger = logging.getLogger(__name__)

class AudioInputStream():
    def __init__(
        self,
        rate: int = 16000,
        chunk: int = 4048,
        device: Optional[Union[str, int, float, Any]] = None,
        audio_data_callback: Optional[Callable] = None,
        tts_state_callback: Optional[Callable] = None
    ):
        self._rate = rate
        self._chunk = chunk
        self._device = device

        # Callback for audio data
        self.audio_data_callback = audio_data_callback

        # Callback for TTS state
        self.tts_state_callback = tts_state_callback

        # Flag to indicate if TTS is active
        self._is_tts_active: bool = False

        # Thread-safe buffer for audio data
        self._buff: queue.Queue[Optional[bytes]] = queue.Queue()

        # audio interface and stream
        self._audio_interface: Optional[pyaudio.PyAudio] = None
        self._audio_stream: Optional[pyaudio.Stream] = None

        # Audio processing thread
        self._audio_thread: Optional[threading.Thread] = None

        # Lock for thread safety
        self._lock = threading.Lock()

        self.running: bool = True

    def set_tts_state_callback(self, callback: Optional[Callable]):
        self.tts_state_callback = callback

    def on_tts_state_change(self, is_active: bool):
        with self._lock:
            self._is_tts_active = is_active
            logger.info(f"TTS active state changed to: {is_active}")

    def start(self) -> 'AudioInputStream':
        if not self.running:
            return self

        self._audio_interface = pyaudio.PyAudio()

        # Get default device if none specified
        if self._device is None:
            try:
                default_info = self._audio_interface.get_default_input_device_info()
                self._device = default_info['index']
                logger.info(f"Default input device: {default_info['name']} ({self._device})")
            except Exception as e:
                logger.error(f"Error getting default input device: {e}")
                self._device = None
        else:
            device_info = self._audio_interface.get_device_info_by_index(self._device)
            logger.info(f"Selected input device: {device_info['name']} ({self._device})")

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

            # Start the audio processing thread
            self._start_audio_thread()

            logger.info(f"Started audio stream with device {self._device}")

        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            self._audio_interface.terminate()
            raise

        return self

    def _start_audio_thread(self):
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_thread = threading.Thread(
                target=self.on_audio,
                daemon=True
            )
            self._audio_thread.start()
            logger.info("Started audio processing thread")

    def _fill_buffer(self, in_data: bytes, frame_count: int, time_info: dict, status_flags: int) -> Tuple[None, int]:
        with self._lock:
            if not self._is_tts_active:
                self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self) -> Generator[bytes, None, None]:
        while self.running:
            chunk = self._buff.get()
            if chunk is None:
                return

            with self._lock:
                if self._is_tts_active:
                    continue

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

            if self.audio_data_callback:
                self.audio_data_callback(b''.join(data))

            yield b''.join(data)

    def on_audio(self):
        for _ in self.generator():
            if not self.running:
                break
        pass

    def stop(self):
        self.running = False

        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._audio_interface:
            self._audio_interface.terminate()

        # Clean up the audio processing thread
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)

        self._buff.put(None)
        logger.info("Stopped audio stream")
from omOS_speech import AudioInputStream
from omOS_utils import ws

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ws client
ws_client = ws.Client(url="wss://api-asr.openmind.org")
ws_client.start()

# Audio input from microphone
audio_stream_input = AudioInputStream(audio_data_callback=ws_client.send_message)
audio_stream_input.start()

while True:
    pass
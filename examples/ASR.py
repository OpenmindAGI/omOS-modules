import logging

from OM1_speech import AudioInputStream
from OM1_utils import ws

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)
logging.basicConfig(level=logging.INFO)

# ws client
ws_client = ws.Client(url="wss://api-asr.openmind.org")
ws_client.start()

# Audio input from microphone
audio_stream_input = AudioInputStream(audio_data_callback=ws_client.send_message)
audio_stream_input.start()

while True:
    pass

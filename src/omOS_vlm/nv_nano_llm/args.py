# DON'T MODIFY THIS FILE
# This file comes from https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/utils/args.py
import argparse
import logging
import os

from .prompts import DefaultChatPrompts, DefaultCompletionPrompts, load_prompts


class NanoLLMArgParser(argparse.ArgumentParser):
    """
    Dynamically adds extra command-line args that are commonly used by various subsystems.
    """

    Defaults = [
        "model",
        "chat",
        "generation",
    ]  #: The default options for model loading, chat, generation config
    Audio = ["audio_input", "audio_output"]  #: Audio device I/O options
    Video = ["video_input", "video_output"]  #: Video streaming I/O options
    Riva = ["asr", "tts"]  #: ASR/TTS model options

    def __init__(self, extras=Defaults, **kwargs):
        """
        Populate an ``argparse.ArgumentParser`` with additional options as specified by the provided extras.
        """
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )

        # LLM
        if "model" in extras:
            self.add_argument(
                "--model",
                type=str,
                default="Efficient-Large-Model/VILA1.5-3b",  # required=True,
                help="path to the model, or repository on HuggingFace Hub",
            )
            self.add_argument(
                "--quantization",
                type=str,
                default=None,
                help="for MLC, the type of quantization to apply (default q4f16_ft)  For AWQ, the path to the quantized weights.",
            )
            self.add_argument(
                "--api",
                type=str,
                default=None,
                choices=["auto_gptq", "awq", "hf", "mlc"],
                help="specify the API to use (otherwise inferred)",
            )
            self.add_argument(
                "--vision-api",
                type=str,
                default="auto",
                choices=["auto", "hf", "trt"],
                help="use TensorRT for vision encoder, 'auto' will enable based on the platform",
            )
            self.add_argument(
                "--vision-model",
                type=str,
                default=None,
                help="for VLMs, manually select the vision embedding model to use (e.g. openai/clip-vit-large-patch14-336 for higher-res)",
            )
            self.add_argument(
                "--vision-scaling",
                type=str,
                default=None,
                choices=["crop", "resize"],
                help="for VLMs, select the input image scaling method (default is: crop)",
            )

        if "chat" in extras or "prompt" in extras:
            self.add_argument(
                "--prompt",
                action="append",
                nargs="*",
                help="add a prompt (can be prompt text or path to .txt, .json, or image file)",
            )
            self.add_argument(
                "--save-mermaid",
                type=str,
                default=None,
                help="save mermaid diagram of the pipeline to this file",
            )

        if "chat" in extras:
            try:
                from nano_llm import ChatTemplates

                self.add_argument(
                    "--chat-template",
                    type=str,
                    default=None,
                    choices=list(ChatTemplates.keys()),
                    help="manually select the chat template",
                )
                self.add_argument(
                    "--system-prompt",
                    type=str,
                    default=None,
                    help="override the default system prompt instruction",
                )
                self.add_argument(
                    "--wrap-tokens",
                    type=int,
                    default=512,
                    help="the number of most recent tokens in the chat to keep when the chat overflows the max context length",
                )
            except ModuleNotFoundError:
                pass

        if "generation" in extras:
            self.add_argument(
                "--max-context-len",
                type=int,
                default=256,
                help="override the model's default context window length (in tokens)  This should include space for model output (up to --max-new-tokens)  Lowering it from the default (e.g. 4096 for Llama) will reduce memory usage.  By default, it's inherited from the model's max length.",
            )
            self.add_argument(
                "--max-new-tokens",
                type=int,
                default=120,
                help="the maximum number of new tokens to generate, in addition to the prompt",
            )
            self.add_argument(
                "--min-new-tokens",
                type=int,
                default=-1,
                help="force the model to generate a minimum number of output tokens",
            )
            self.add_argument(
                "--do-sample",
                action="store_true",
                help="enable output token sampling using temperature and top_p",
            )
            self.add_argument(
                "--temperature",
                type=float,
                default=0.7,
                help="token sampling temperature setting when --do-sample is used",
            )
            self.add_argument(
                "--top-p",
                type=float,
                default=0.95,
                help="token sampling top_p setting when --do-sample is used",
            )
            self.add_argument(
                "--repetition-penalty",
                type=float,
                default=1.0,
                help="the parameter for repetition penalty. 1.0 means no penalty",
            )

        # VIDEO
        if "video_input" in extras:
            self.add_argument(
                "--video-input",
                type=str,
                default=None,
                help="video camera device name, stream URL, file/dir path",
            )
            self.add_argument(
                "--video-input-width",
                type=int,
                default=None,
                help="manually set the resolution of the video input",
            )
            self.add_argument(
                "--video-input-height",
                type=int,
                default=None,
                help="manually set the resolution of the video input",
            )
            self.add_argument(
                "--video-input-codec",
                type=str,
                default=None,
                choices=["h264", "h265", "vp8", "vp9", "mjpeg"],
                help="manually set the input video codec to use",
            )
            self.add_argument(
                "--video-input-framerate",
                type=int,
                default=None,
                help="set the desired framerate of input video",
            )
            self.add_argument(
                "--video-input-save",
                type=str,
                default=None,
                help="path to video file to save the incoming video feed to",
            )

        if "video_output" in extras:
            self.add_argument(
                "--video-output",
                type=str,
                default=None,
                help="display, stream URL, file/dir path",
            )
            self.add_argument(
                "--video-output-codec",
                type=str,
                default=None,
                choices=["h264", "h265", "vp8", "vp9", "mjpeg"],
                help="set the output video codec to use",
            )
            self.add_argument(
                "--video-output-bitrate",
                type=int,
                default=None,
                help="set the output bitrate to use",
            )
            self.add_argument(
                "--video-output-save",
                type=str,
                default=None,
                help="path to video file to save the outgoing video stream to",
            )

        # AUDIO
        if "audio_input" not in extras and "asr" in extras:
            extras += ["audio_input"]

        if "audio_input" in extras:
            self.add_argument(
                "--audio-input-device",
                type=int,
                default=None,
                help="audio input device/microphone to use for ASR",
            )
            self.add_argument(
                "--audio-input-channels",
                type=int,
                default=1,
                help="The number of input audio channels to use",
            )

        if "audio_output" in extras:
            self.add_argument(
                "--audio-output-device",
                type=int,
                default=None,
                help="audio output interface device index (PortAudio)",
            )
            self.add_argument(
                "--audio-output-file",
                type=str,
                default=None,
                help="save audio output to wav file using the given path",
            )
            self.add_argument(
                "--audio-output-channels",
                type=int,
                default=1,
                help="the number of output audio channels to use",
            )

        if "audio_input" in extras or "audio_output" in extras:
            self.add_argument(
                "--list-audio-devices",
                action="store_true",
                help="List output audio devices indices.",
            )

        if any(x in extras for x in ("audio_input", "audio_output", "asr", "tts")):
            self.add_argument(
                "--sample-rate-hz",
                type=int,
                default=None,
                help="the audio sample rate in Hz (by default, will use input's rate",
            )
            self.add_argument(
                "--audio-chunk",
                type=float,
                default=0.1,
                help="the duration of time or number of samples for buffering audio",
            )

        # ASR/TTS
        if "asr" in extras or "tts" in extras:
            self.add_argument(
                "--riva-server",
                default="localhost:50051",
                help="URI to the Riva GRPC server endpoint.",
            )
            self.add_argument(
                "--language-code",
                default="en-US",
                help="Language code of the ASR/TTS to be used.",
            )

        if "tts" in extras:
            self.add_argument(
                "--tts",
                type=str,
                default=None,
                help="name of path of the TTS model to use (e.g. 'riva', 'xtts', 'none', 'disabled')",
            )
            self.add_argument(
                "--tts-buffering",
                type=str,
                default="punctuation",
                help="buffering method for TTS ('none', 'punctuation', 'time', 'punctuation,time')",
            )
            self.add_argument(
                "--voice",
                type=str,
                default=None,
                help="Voice model name to use for TTS",
            )
            self.add_argument(
                "--voice-speaker",
                type=str,
                default=None,
                help="Name or ID of speaker to use",
            )
            self.add_argument(
                "--voice-rate",
                type=float,
                default=1.0,
                help="TTS SSML voice speaker rate (between 25-250%%)",
            )
            self.add_argument(
                "--voice-pitch",
                type=str,
                default="default",
                help="TTS SSML voice pitch shift",
            )
            self.add_argument(
                "--voice-volume",
                type=str,
                default="default",
                help="TTS SSML voice volume attribute",
            )
            # self.add_argument("--voice-min-words", type=int, default=4, help="the minimum number of words the TTS should wait to speak")

        if "asr" in extras:
            self.add_argument(
                "--asr",
                type=str,
                default=None,
                help="name or path of the ASR model to use (e.g. 'riva', 'whisper_tiny', 'whisper_base', 'whisper_small', 'none', 'disabled')",
            )
            self.add_argument(
                "--asr-threshold",
                type=float,
                default=-2.5,
                help="minimum ASR confidence (only applies to 'final' transcripts)",
            )
            self.add_argument(
                "--vad-threshold",
                type=float,
                default=0.5,
                help="minimum VAD confidence to begin speaking sequence and enable ASR",
            )
            self.add_argument(
                "--vad-window",
                type=float,
                default=0.5,
                help="duration of time (in seconds) that the VAD filter considers",
            )
            self.add_argument(
                "--boosted-words",
                action="append",
                help="Words to boost when decoding the ASR transcript, like hotwords (Riva only)",
            )
            self.add_argument(
                "--boosted-score",
                type=float,
                default=4.0,
                help="Amount by which to boost words when decoding (Riva only)",
            )
            self.add_argument(
                "--profanity-filter",
                action="store_true",
                help="enable profanity filtering in ASR transcripts (Riva only)",
            )
            self.add_argument(
                "--inverse-text-normalization",
                action="store_true",
                help="apply Inverse Text Normalization to convert numbers to digits/ect (Riva only)",
            )
            self.add_argument(
                "--no-automatic-punctuation",
                dest="automatic_punctuation",
                action="store_false",
                help="disable punctuation in the ASR transcripts (Riva only)",
            )
            self.add_argument(
                "--partial-transcripts",
                type=float,
                default=0.25,
                help="the update rate (in seconds) for the partial ASR transcripts (0 to disable)",
            )

        # NANODB
        if "nanodb" in extras:
            self.add_argument(
                "--nanodb",
                type=str,
                default=None,
                help="path to load or create the database",
            )
            self.add_argument(
                "--nanodb-model",
                type=str,
                default="ViT-L/14@336px",
                help="the embedding model to use for the database",
            )
            self.add_argument(
                "--nanodb-reserve",
                type=int,
                default=1024,
                help="the memory to reserve for the database in MB",
            )

        # WEBSERVER
        if "web" in extras:
            self.add_argument(
                "--web-host",
                type=str,
                default="0.0.0.0",
                help="network interface to bind to for hosting the webserver",
            )
            self.add_argument(
                "--web-port",
                type=int,
                default=8050,
                help="port used for webserver HTTP/HTTPS",
            )
            self.add_argument(
                "--ws-port",
                type=int,
                default=49000,
                help="port used for websocket communication",
            )
            self.add_argument(
                "--ssl-key",
                default=os.getenv("SSL_KEY"),
                type=str,
                help="path to PEM-encoded SSL/TLS key file for enabling HTTPS",
            )
            self.add_argument(
                "--ssl-cert",
                default=os.getenv("SSL_CERT"),
                type=str,
                help="path to PEM-encoded SSL/TLS cert file for enabling HTTPS",
            )
            self.add_argument(
                "--upload-dir",
                type=str,
                default="/tmp/uploads",
                help="the path to save files uploaded from the client",
            )
            self.add_argument(
                "--web-trace",
                action="store_true",
                help="output websocket message logs when --log-level=debug",
            )
            self.add_argument(
                "--web-title",
                type=str,
                default=None,
                help="override the default title of the web template",
            )

    def parse_args(self, **kwargs):
        """
        Override for parse_args() that does some additional configuration
        """
        args = super().parse_args(**kwargs)

        if hasattr(args, "prompt"):
            args.prompt = NanoLLMArgParser.parse_prompt_args(args.prompt)

        if hasattr(args, "system_prompt"):
            args.system_prompt = load_prompts(args.system_prompt, concat=True)

        logging.debug(f"{args}")
        return args

    @staticmethod
    def parse_prompt_args(prompts, chat=True):
        """
        Parse prompt command-line argument and return list of prompts.
        It's assumed that the argparse argument was created like this::

          parser.add_argument('--prompt', action='append', nargs='*')

        If the prompt text is 'default', then default chat prompts will
        be assigned if ``chat=True`` (otherwise default completion prompts)
        """
        if prompts is None:
            return None

        prompts = [x[0] for x in prompts]

        if prompts[0].lower() == "default" or prompts[0].lower() == "defaults":
            prompts = DefaultChatPrompts if chat else DefaultCompletionPrompts

        return prompts

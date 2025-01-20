import logging
import argparse
from transformers import GenerationConfig, PreTrainedModel
from typing import Optional, Any, Callable

from .model_loader import VILAModelLoader

# llava is only on VILA server
# The dependency (bitsandbytes) is not available for Mac M chips
try:
    import llava
    from llava import conversation as clib
    from llava.media import Image, Video
except ModuleNotFoundError:
    llava = None
    clib = None
    Image = None
    Video = None

logger = logging.getLogger(__name__)

class VILAProcessor:
    """
    Vision Language Model (VLM) processor for real-time video analysis.

    Processes video frames through a vision-language model to generate text descriptions
    of interesting aspects in the video stream. Designed specifically for NVIDIA Jetson
    devices using the nano_llm framework.

    Parameters
    ----------
    model_args : argparse.Namespace
        Command line arguments for model configuration.
    callback : Optional[Callable[[str], None]], optional
        Callback function for processing model responses,
        by default None
    """
    def __init__(self, model_args: argparse.Namespace, callback: Optional[Callable[[str], None]] = None):
        # Load the VILA model
        self.model = self._initialize_model(model_args)

        # Set model arguments and configuration
        self.model_args = model_args
        self.model_config = GenerationConfig(
            max_new_tokens=48,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )

        # Register the callback function
        self.callback = callback

        # Image processing variables
        self.last_image: Any = None
        self.num_images: int = 0

        # response variables
        self.raw_response: str = 'test'
        self.response: str = ''

        # Set the running flag
        self.running: bool = True

        # Warm up the model
        self._warmup_model()

    def _initialize_model(self, args: argparse.Namespace) -> PreTrainedModel:
        """
        Initialize the vision-language model.

        Parameters
        ----------
        args : argparse.Namespace
            Model configuration arguments

        Returns
        -------
        PreTrainedModel
            Initialized model instance

        Raises
        ------
        AssertionError
            If the model does not have vision capabilities
        """
        model_loader = VILAModelLoader(args)
        return model_loader.model

    def _warmup_model(self):
        """
        Perform model warmup with a simple query.

        Sends a basic arithmetic query to ensure the model is loaded
        and ready for processing.
        """

    def on_video(self, image: Any) -> Any:
        """
        Process incoming video frames.

        Parameters
        ----------
        image : Any
            Input video frame to process

        Returns
        -------
        Any
            Annotated video frame
        """
        return image

    def process_frames(self, video_output: Any, video_source: Any):
        """
        Main frame processing loop.

        Parameters
        ----------
        video_output : Any
            Video output stream handler
        video_source : Any
            Video input source handler
        """
        skip = 0
        while self.running:
            if self.last_image is None:
                continue

            if skip < 5:
                skip += 1
                continue
            else:
                skip = 0

            ### TODO
            ### Add the image logic here:
            ### The input is img_bytes = base64.b64decode(img_b64)
            logger.info(f"Received image: {self.num_images + 1}")

            # self.chat_history.append('user', text=f'Image {self.num_images + 1}:')
            # self.chat_history.append('user', image=self.last_image)
            self.last_image = None

            if self.num_images < 5:
                self.num_images += 1
                continue
            else:
                self.num_images = 0

            # self.chat_history.append('user', "What is the most interesting aspect in this series of images?")
            # embedding, _ = self.chat_history.embed_chat()

            # reply = self.model.generate(
            #     embedding,
            #     kv_cache=self.chat_history.kv_cache,
            #     max_new_tokens=self.model_args.max_new_tokens,
            #     min_new_tokens=self.model_args.min_new_tokens,
            #     do_sample=self.model_args.do_sample,
            #     repetition_penalty=self.model_args.repetition_penalty,
            #     temperature=self.model_args.temperature,
            #     top_p=self.model_args.top_p,
            # )

            # for token in reply:
            #     if len(reply.tokens) == 1:
            #         self.raw_response = token
            #     else:
            #         self.raw_response = self.raw_response + token

            # self.response = self.cleanup(self.raw_response)
            # logger.info(f'VLM response: {self.response}')

            # if self.callback:
            #     self.callback(json.dumps({"vlm_reply": self.response}))

            # self.chat_history.reset()

            # if video_source.eos:
            #     video_output.stream.Close()
            #     break

    def stop(self):
        """
        Stop frame processing.

        Sets the running flag to False to terminate processing loop.
        """
        self.running = False
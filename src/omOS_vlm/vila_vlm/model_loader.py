import argparse
import logging

from omOS_utils import singleton

# llava is only on VILA server
# The dependency (bitsandbytes) is not available for Mac M chips
try:
    import llava
    from llava import conversation as clib
except ModuleNotFoundError:
    llava = None

logger = logging.getLogger(__name__)

@singleton
class VILAModelLoader:
    """
    Singleton class for loading and managing the VILA model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments for model configuration.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the VILA model loader.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments for model configuration.
        """
        self.model = self._load_model()

    def _load_model(self) -> llava.PreTrainedModel:
        """
        Load the VILA model and move it to CUDA device.

        Returns
        -------
        llava.PreTrainedModel
            Loaded VILA model instance.

        Raises
        ------
        Exception
            If model loading fails.
        """
        try:
            model = llava.load("Efficient-Large-Model/VILA1.5-3B")
            model.to("cuda")
            clib.default_conversation = clib.conv_templates["vicuna_v1"].copy()
            assert(model)
            return model
        except Exception as e:
            raise Exception("Failed to load VILA model") from e

    @property
    def model(self) -> llava.PreTrainedModel:
        """
        Get the VILA model instance, loading it if not already loaded.

        Returns
        -------
        llava.PreTrainedModel
            The loaded VILA model instance.
        """
        if not self.model:
            self.model = self._load_model()
        return self.model
import argparse
from typing import List, Optional


class VILAArgParser(argparse.ArgumentParser):
    """
    Argument parser for VILA VLM configuration.
    """

    def __init__(self, **kwargs):
        """
        Initialize the VILA argument parser with model-specific options.
        """
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )

        # WebSocket connection
        self.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="VILA server hostname",
        )
        self.add_argument(
            "--port",
            type=int,
            default=8000,
            help="VILA server WebSocket port",
        )

        # Video processing
        self.add_argument(
            "--frame-skip",
            type=int,
            default=5,
            help="Number of frames to skip between processing",
        )
        self.add_argument(
            "--batch-size",
            type=int,
            default=5,
            help="Number of frames to process in a batch",
        )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Parameters
        ----------
        args : Optional[List[str]]
            List of arguments to parse. If None, uses sys.argv[1:].

        Returns
        -------
        argparse.Namespace
            Parsed arguments
        """
        parsed_args = super().parse_args(args)
        return parsed_args

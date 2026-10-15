"""HUDYAT Trainer: extract step from the command line (the app's button runs the same code).
Usage: python extract.py --help
"""

import sys

from trainer.cli import main

if __name__ == "__main__":
    sys.exit(main("extract"))

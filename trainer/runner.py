"""python -m trainer.runner <job> [args]: what the app starts in the background."""

import sys

from trainer.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))

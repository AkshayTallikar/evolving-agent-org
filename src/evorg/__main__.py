"""Allow ``python -m evorg``."""

import sys

from evorg.cli import main

if __name__ == "__main__":
    sys.exit(main())

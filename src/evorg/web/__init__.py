"""Web lineage explorer: a read-only FastAPI app over saved run directories (design 4.11).

Importing this package needs FastAPI; install the extra with ``pip install -e ".[web]"``.
"""

from evorg.web.app import create_app

__all__ = ["create_app"]

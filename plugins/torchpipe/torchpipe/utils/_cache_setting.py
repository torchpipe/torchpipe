"""Cache directory configuration for TorchPipe."""

from __future__ import annotations

import os
from pathlib import Path


def get_cache_dir() -> str:
    """Get the cache directory for TorchPipe.

    Returns:
        Path to the cache directory
    """
    cache = str(Path(os.environ.get(
        "OMNIBACK_CACHE_DIR",
        "~/.cache/omniback/"
    )).expanduser())
    return os.path.join(cache, "torchpipe/")

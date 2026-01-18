
from pathlib import Path
import os

def get_cache_dir():
    cache = str(Path(os.environ.get("OMNIBACK_CACHE_DIR",
                                    "~/.cache/omniback/")).expanduser())
    return os.path.join(cache, "torchpipe/")

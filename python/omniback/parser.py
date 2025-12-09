from omniback import _C
from typing import Dict, Any
# import tomllib
from pathlib import Path
from typing import Dict, Any, Union
import logging
import json
import os
logger = logging.getLogger("omniback")


try:
    import tomllib
except ImportError:
    import tomli as tomllib


def to_str(data) -> str:
    return str(data)


def to_dual_str(config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Convert a nested dictionary to a format where all values are strings.

    Args:
    config (Dict[str, Dict[str, Any]]): The input nested dictionary.

    Returns:
    Dict[str, Dict[str, str]]: The converted dictionary with all string values.

    Raises:
    TypeError: If the input is not a nested dictionary with the expected structure.
    """
    if not isinstance(config, dict):
        raise TypeError("Input must be a dictionary")

    re = {
        outer_key: {
            inner_key: str(int(inner_value)) if isinstance(
                inner_value, bool) else str(inner_value)
            for inner_key, inner_value in outer_value.items()
        }
        for outer_key, outer_value in config.items()
        if isinstance(outer_value, dict)
    }
    global_config = {outer_key: to_str(outer_value) for outer_key, outer_value in config.items()
                     if not isinstance(outer_value, dict)}
    re["global"] = global_config
    return re


def parse(file_path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """
    Parse and convert TOML file content to string-based nested configuration.

    Args:
        file_path: Path to the TOML configuration file

    Returns:
        Nested dictionary with all values converted to strings

    Raises:
        FileNotFoundError: If specified file doesn't exist
        IOError: For file reading errors
        RuntimeError: For parsing/validation errors
    """
    # Convert to Path object for modern path handling
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    if not path.is_file():
        raise IOError(f"Path is not a file: {file_path}")

    try:
        with path.open('rb') as f:
            raw_config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise RuntimeError(f"TOML parsing failed: {e}") from e

    return to_dual_str(raw_config)


def log_structured_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """
    Log a structured configuration dictionary in a clean, formatted way.

    Args:
        config: Dictionary containing configuration data
        title: Optional title to display in the log header

    Returns:
        None
    """
    # Format the config as a pretty JSON string with indentation
    formatted_config = json.dumps(config, indent=2, ensure_ascii=False)

    # Add separators to make the log more readable
    logger.info("=" * 50)
    logger.info(f"{title}:")
    # Log each line separately to avoid excessively wide log entries
    for line in formatted_config.splitlines():
        logger.info(line)
    logger.info("=" * 50)


def pipe(config):
    if isinstance(config, str):
        if config.endswith('.toml'):
            assert os.path.exists(config), config
            return init_from_file(config)
    elif isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if not isinstance(v, (bytes, str)):
                        config[k][k2] = str(v2)  # .encode("utf8")
            else:
                if not isinstance(v, (bytes, str)):
                    config[k] = str(v)  # .encode("utf8")
        print(config)
        return _C.init("Interpreter", {}, _C.Dict({"config": (config)}))

    return _C.init(config)


def init_from_file(file_path: Union[str, Path]):
    """
    Initialize backend components from a TOML configuration file.

    Args:
        file_path: Path to the TOML configuration file

    Returns:
        Initialized Backend instance
    """
    logger.info(f"Loading configuration from: {file_path}")

    data = parse(file_path)

    # Log the configuration in a structured format
    log_structured_config(data, title="Configuration loaded")

    logger.info(f"Initializing interpreter with {len(data)} components")

    from omniback._C import init, Dict

    return init("Interpreter", {}, Dict({"config": data}))

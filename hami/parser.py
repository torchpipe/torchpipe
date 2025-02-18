from typing import Dict, Any
# import tomllib
from pathlib import Path
from typing import Dict, Any, Union

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
            inner_key: str(int(inner_value)) if isinstance(inner_value, bool) else str(inner_value)
            for inner_key, inner_value in outer_value.items()
        }
        for outer_key, outer_value in config.items()
        if isinstance(outer_value, dict)
    }
    global_config =  {outer_key:to_str(outer_value) for outer_key, outer_value in config.items()
        if not isinstance(outer_value, dict)}
    re["global"] = global_config
    return re



def parse_from_file(file_path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
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

# Dependencies needed:
# Python 3.11+ (tomllib included in standard library)
# For older versions: pip install tomli
# Add to requirements.txt:
# tomli>=2.0.1 ; python_version < "3.11"

# Usage example:
# config = parse_from_file("config.toml")
# db_host = config['database']['host']
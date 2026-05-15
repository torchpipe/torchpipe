# backend_resolver.py

from typing import Dict, List, Optional, Tuple

class BackendResolver:
    def __init__(self, toml_data: dict):
        """
        Initialize parser.
        :param toml_data: Parsed TOML data (dict format)
        """
        self._backend_to_group: Dict[str, str] = {}
        self._group_to_deps: Dict[str, List[str]] = {}
        self._group_to_backends: Dict[str, List[str]] = {}

        self._parse_groups(toml_data)

    def _parse_groups(self, toml_data: dict):
        """Traverse all [group.xxx] nodes and build index."""
        for full_key, value in toml_data.items():
            if not isinstance(value, dict):
                continue
            pass

        # Key: all groups should be under toml_data["group"]
        groups = toml_data.get("group", {})
        self._traverse_groups(groups, current_path="")

    def _traverse_groups(self, node: dict, current_path: str):
        """Recursively traverse nested group structure."""
        if "backend" in node and isinstance(node["backend"], list):
            # This is a leaf group containing backend list
            group_name = current_path
            backends = node["backend"]
            dependencies = node.get("dependencies", [])

            self._group_to_backends[group_name] = backends
            self._group_to_deps[group_name] = dependencies

            for backend in backends:
                if backend in self._backend_to_group:
                    raise ValueError(f"Backend '{backend}' is defined in multiple groups: "
                                     f"{self._backend_to_group[backend]} and {group_name}")
                self._backend_to_group[backend] = group_name
        else:
            # Continue recursive traversal of subgroups
            for key, child in node.items():
                if isinstance(child, dict):
                    new_path = f"{current_path}.{key}" if current_path else key
                    self._traverse_groups(child, new_path)

    def lookup(self, backend_name: str) -> Optional[Tuple[str, List[str]]]:
        """
        Lookup group and dependencies by backend name.
        :param backend_name: Backend name, e.g., "TensorrtTensor"
        :return: (group_name, dependencies_list) or None if not found
        """
        group = self._backend_to_group.get(backend_name)
        if group is None:
            return None
        deps = self._group_to_deps.get(group, [])
        return group, deps

    def all_backends(self) -> List[str]:
        """Return all backend names."""
        return list(self._backend_to_group.keys())

    def get_group_backends(self, group_name: str) -> List[str]:
        """Return all backends in specified group."""
        return self._group_to_backends.get(group_name, [])


# Compatible with Python 3.11+ tomllib
try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib


def create_resolver_from_file(toml_path: str) -> BackendResolver:
    """Create resolver from TOML file path."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return BackendResolver(data)


def resolve(toml_path: str, backend):
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    resolver = BackendResolver(data)

    result = resolver.lookup(backend)
    assert (result is not None)
    return result

if __name__ == "__main__":
    import fire
    fire.Fire(resolve)

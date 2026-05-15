from __future__ import annotations

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from typing import Dict, List, Set, Optional
from collections import defaultdict, deque
from collections.abc import Callable

import tvm_ffi


class GroupRegistry:
    def __init__(self):
        # group_name -> {backends: List[str], dependencies: List[str]}
        self._groups: Dict[str, dict] = {}

        # backend_name -> set of group names that provide it
        self._backend_to_groups: Dict[str, Set[str]] = defaultdict(set)

    def register_group(
        self,
        name: str,
        backends: List[str],
        dependencies: Optional[List[str]] = None
    ):
        """
        Register a backend group.

        Args:
            name: Group name, e.g., "torchpipe.opencv"
            backends: List of backends provided by this group
            dependencies: List of group names this group depends on
        """
        if not name:
            raise ValueError("Group name cannot be empty")
        if name in self._groups:
            raise ValueError(f"Group '{name}' is already registered")

        dependencies = dependencies or []
        # Check all dependencies are registered
        for dep in dependencies:
            if dep not in self._groups:
                raise ValueError(f"Dependency group '{dep}' not registered yet. "
                                 f"Please register it before '{name}'.")

        # Register group metadata
        self._groups[name] = {
            "backends": list(backends),
            "dependencies": list(dependencies)
        }

        # Update backend to group mapping
        for backend in backends:
            if not backend:
                raise ValueError(f"Empty backend name in group '{name}'")
            self._backend_to_groups[backend].add(name)

    def load_from_toml(self, toml_path: str):
        with open(toml_path, 'rb') as f:
            data = tomllib.load(f)

        # Extract all group definitions (supports nested structure)
        groups_to_register = {}

        for key, value in data.items():
            if key == "version":
                continue

            # If value is a dict with "backend" field, it's a directly defined group
            if isinstance(value, dict) and "backend" in value:
                groups_to_register[key] = {
                    "backends": value.get("backend", []),
                    "dependencies": value.get("dependencies", [])
                }
            # Otherwise it might be nested structure (e.g., torchpipe.core)
            elif isinstance(value, dict):
                # Iterate nested dict
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "backend" in subvalue:
                        group_name = f"{key}.{subkey}"
                        groups_to_register[group_name] = {
                            "backends": subvalue.get("backend", []),
                            "dependencies": subvalue.get("dependencies", [])
                        }

        # Build dependency graph for topological sort
        from collections import defaultdict, deque
        from_nodes = defaultdict(list)  # dep -> [group]
        in_degree = {name: 0 for name in groups_to_register}

        for name, info in groups_to_register.items():
            for dep in info["dependencies"]:
                if dep not in groups_to_register:
                    raise ValueError(
                        f"Group '{name}' depends on unknown group '{dep}'")
                from_nodes[dep].append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

        # Topological sort (Kahn's algorithm)
        queue = deque([name for name in in_degree if in_degree[name] == 0])
        sorted_order = []

        while queue:
            node = queue.popleft()
            sorted_order.append(node)
            for neighbor in from_nodes[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(groups_to_register):
            raise ValueError("Circular dependency detected among groups!")

        # Register in topological order
        for name in sorted_order:
            info = groups_to_register[name]
            self.register_group(name, info["backends"], info["dependencies"])

    def get_groups_for_backend(self, backend: str) -> Set[str]:
        """Return all group names that provide this backend."""
        return set(self._backend_to_groups.get(backend, []))

    def list_all_backends(self) -> Set[str]:
        """Return all registered backend names."""
        return set(self._backend_to_groups.keys())

    def list_all_groups(self) -> Set[str]:
        """Return all registered group names."""
        return set(self._groups.keys())

    def resolve_load_order(self) -> List[str]:
        """
        Return recommended group loading order (topologically sorted by dependencies).
        """
        # Rebuild dependency graph (registered groups only)
        from_nodes = defaultdict(list)
        in_degree = {name: 0 for name in self._groups}

        for name, info in self._groups.items():
            for dep in info["dependencies"]:
                from_nodes[dep].append(name)
                in_degree[name] += 1

        queue = deque([name for name in in_degree if in_degree[name] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in from_nodes[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def get_group_info(self, group_name: str) -> dict:
        """Get detailed information about a group."""
        if group_name not in self._groups:
            raise KeyError(f"Group '{group_name}' not registered")
        return self._groups[group_name].copy()


# def register_group(toml_path: str, cb: Callable):
#     registry = GroupRegistry()
#     registry.load_from_toml(toml_path)
#     for backend, grp_names in registry._backend_to_groups.items():
#         assert len(
#             grp_names) == 1, f"backend {backend} has multiple groups: {grp_names}"
#         _register_backend_group = tvm_ffi.get_global_func(
#             "omniback.register_backend_group")
#         grp_name = next(iter(grp_names))
#         _register_backend_group(backend, grp_name,
#                                 lambda: cb(backend, grp_name))

def toml2groups(toml_path: str):
    registry = GroupRegistry()
    registry.load_from_toml(toml_path)
    dependencies = None
    return registry._backend_to_groups, dependencies

def main(toml_path: str):
    registry = GroupRegistry()
    registry.load_from_toml(toml_path)
    for backend, grp_name in registry._backend_to_groups.items():
        print(backend, next(iter(grp_name)) if grp_name else None)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

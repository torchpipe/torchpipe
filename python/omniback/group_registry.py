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
        self._backend_to_groups: Dict[str, str] = defaultdict(set)

    def register_group(
        self,
        name: str,
        backends: List[str],
        dependencies: Optional[List[str]] = None
    ):
        """
        注册一个后端组。
        :param name: 组名，如 "torchpipe.opencv"
        :param backends: 该组提供的后端列表
        :param dependencies: 依赖的其他组名列表
        """
        if not name:
            raise ValueError("Group name cannot be empty")
        if name in self._groups:
            raise ValueError(f"Group '{name}' is already registered")

        dependencies = dependencies or []
        # 检查所有依赖是否已注册
        for dep in dependencies:
            if dep not in self._groups:
                raise ValueError(f"Dependency group '{dep}' not registered yet. "
                                 f"Please register it before '{name}'.")

        # 注册组元数据
        self._groups[name] = {
            "backends": list(backends),
            "dependencies": list(dependencies)
        }

        # 更新后端到组的映射
        for backend in backends:
            if not backend:
                raise ValueError(f"Empty backend name in group '{name}'")
            self._backend_to_groups[backend].add(name)

    def load_from_toml(self, toml_path: str):
        with open(toml_path, 'rb') as f:
            data = tomllib.load(f)

        # 提取所有组定义（支持嵌套结构）
        groups_to_register = {}

        for key, value in data.items():
            if key == "version":
                continue

            # 如果值是字典且包含 "backend" 字段，说明是直接定义的组
            if isinstance(value, dict) and "backend" in value:
                groups_to_register[key] = {
                    "backends": value.get("backend", []),
                    "dependencies": value.get("dependencies", [])
                }
            # 否则可能是嵌套结构（如 torchpipe.core）
            elif isinstance(value, dict):
                # 遍历嵌套的字典
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "backend" in subvalue:
                        group_name = f"{key}.{subkey}"
                        groups_to_register[group_name] = {
                            "backends": subvalue.get("backend", []),
                            "dependencies": subvalue.get("dependencies", [])
                        }

        # 构建依赖图用于拓扑排序
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

        # 拓扑排序（Kahn 算法）
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

        # 按拓扑顺序注册
        for name in sorted_order:
            info = groups_to_register[name]
            self.register_group(name, info["backends"], info["dependencies"])

    def get_groups_for_backend(self, backend: str) -> Set[str]:
        """返回提供该后端的所有组名"""
        return set(self._backend_to_groups.get(backend, []))

    def list_all_backends(self) -> Set[str]:
        """返回所有已注册的后端名"""
        return set(self._backend_to_groups.keys())

    def list_all_groups(self) -> Set[str]:
        """返回所有已注册的组名"""
        return set(self._groups.keys())

    def resolve_load_order(self) -> List[str]:
        """
        返回推荐的组加载顺序（按依赖拓扑排序）
        """
        # 重新构建依赖图（仅已注册的组）
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
        """获取组的详细信息"""
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
        print(backend, grp_name.first())

if __name__ == "__main__":
    import fire
    fire.Fire(main)



# backend_resolver.py

from typing import Dict, List, Optional, Tuple

class BackendResolver:
    def __init__(self, toml_data: dict):
        """
        初始化解析器。
        :param toml_data: 已解析的 TOML 数据（dict 格式）
        """
        self._backend_to_group: Dict[str, str] = {}
        self._group_to_deps: Dict[str, List[str]] = {}
        self._group_to_backends: Dict[str, List[str]] = {}

        self._parse_groups(toml_data)

    def _parse_groups(self, toml_data: dict):
        """遍历所有 [group.xxx] 节点，建立索引"""
        for full_key, value in toml_data.items():
            if not isinstance(value, dict):
                continue
            pass

        # 关键：所有 group 应该在 toml_data["group"] 下
        groups = toml_data.get("group", {})
        self._traverse_groups(groups, current_path="")

    def _traverse_groups(self, node: dict, current_path: str):
        """递归遍历 group 嵌套结构"""
        if "backend" in node and isinstance(node["backend"], list):
            # 这是一个叶子 group，包含 backend 列表
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
            # 继续递归子组
            for key, child in node.items():
                if isinstance(child, dict):
                    new_path = f"{current_path}.{key}" if current_path else key
                    self._traverse_groups(child, new_path)

    def lookup(self, backend_name: str) -> Optional[Tuple[str, List[str]]]:
        """
        通过 backend 名称查找其所属 group 和依赖列表。
        :param backend_name: 后端名称，如 "TensorrtTensor"
        :return: (group_name, dependencies_list) 或 None（如果未找到）
        """
        group = self._backend_to_group.get(backend_name)
        if group is None:
            return None
        deps = self._group_to_deps.get(group, [])
        return group, deps

    def all_backends(self) -> List[str]:
        """返回所有 backend 名称"""
        return list(self._backend_to_group.keys())

    def get_group_backends(self, group_name: str) -> List[str]:
        """返回指定 group 中的所有 backend"""
        return self._group_to_backends.get(group_name, [])


# 兼容 Python 3.11+ 的 tomllib
try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib


def create_resolver_from_file(toml_path: str) -> BackendResolver:
    """从 TOML 文件路径创建解析器"""
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
    
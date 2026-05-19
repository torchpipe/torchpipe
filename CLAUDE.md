## 环境配置

使用 uv 管理 Python 环境和依赖，阿里源已配置在 `~/.config/uv/uv.toml`：
```toml
[pip]
index-url = "https://mirrors.aliyun.com/pypi/simple/"
```

### 创建 venv 并安装依赖
```bash
uv venv --python 3.12
source .venv/bin/activate  # 可选
uv pip install --python .venv/bin/python torch cmake ninja  # 基础依赖
```

### 编译安装 omniback
```bash
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.27
uv pip install --python .venv/bin/python -e .
```

### 编译安装 torchpipe 插件
```bash
cd plugins/torchpipe/
uv pip install --python ../../.venv/bin/python -e .
```

## 运行测试

### PATH 说明
cmake 等工具安装在 `.venv/bin/` 下，测试时需要加到 PATH：
```bash
PATH="$PWD/.venv/bin:$PATH"
```

### omniback 核心测试
```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest tests/ -v
```

### torchpipe 插件测试
```bash
cd plugins/torchpipe
PATH="$PWD/../../.venv/bin:$PATH" FORCE_DOWNLOAD_OPENCV=1 ../../.venv/bin/python -m pytest tests/ -v
```

## 版本号
- omniback: 由 setuptools-scm 自动生成，`SETUPTOOLS_SCM_PRETEND_VERSION` 覆盖
- torchpipe: 硬编码在 `plugins/torchpipe/pyproject.toml` 中 `version = "0.1.27"`


## git remote
- `origin`: git@github.com:torchpipe/torchpipe.git（官方仓库）

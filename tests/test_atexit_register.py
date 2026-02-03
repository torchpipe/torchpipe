# test/test_atexit_integration.py

import subprocess
import sys
import os
import tempfile


def test_atexit_no_crash():
    # 构造一个独立的 Python 脚本内容
    script = '''
import omniback
omniback.ffi.register_backend_group("test_backend", "test_group", lambda: None)
print("Registered, exiting normally...")
'''

    # 写入临时文件（避免命令行过长）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # 启动子进程
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=10  # 防止 hang 住
        )

        # 检查是否正常退出（返回码 0）
        assert result.returncode == 0, f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # 可选：检查 stderr 是否有异常/traceback
        assert "Traceback" not in result.stderr, f"Unexpected exception:\n{result.stderr}"

        print("✅ atexit cleanup ran without crash")

    finally:
        os.unlink(script_path)  # 清理临时文件

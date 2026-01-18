import re
import subprocess
import os
import sys
IS_WINDOWS = sys.platform == "win32"
system_cxx = os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")

def get_companion_ld(cxx: str) -> str:
    try:
        result = subprocess.run(
            [cxx, "-print-prog-name=ld"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
        ld_path = result.stdout.strip()

        if not ld_path or not os.path.exists(ld_path):
            return "ld"
        return ld_path
    except Exception:
        return "ld"  # fallback


system_ld = get_companion_ld(system_cxx)

def get_cxx_include_dirs():
    result = subprocess.run(
        [system_cxx, "-E", "-Wp,-v", "-"],
        input="",
        stdout=subprocess.DEVNULL,   # 丢弃 stdout
        stderr=subprocess.PIPE,
        text=True
    )
    lines = result.stderr.splitlines()
    in_search = False
    paths = []
    for line in lines:
        if "#include <...> search starts here:" in line:
            in_search = True
            continue
        if "End of search list." in line:
            break
        if in_search and line.startswith(" "):
            # 提取路径（去掉前导空格，忽略注释）
            path = line.strip().split()[0]
            if os.path.exists(path):
                paths.append(path)
    return paths


def get_ld_search_dirs():
    if sys.platform == "darwin":
        return [
            "/usr/lib",
            "/usr/local/lib",
            os.path.expanduser("~/lib"),
        ]
    elif not IS_WINDOWS:
        result = subprocess.run([system_ld, "--verbose"],
                                capture_output=True, text=True, check=True)
        # 匹配 SEARCH_DIR("=xxx") 或 SEARCH_DIR("xxx")
        matches = re.findall(r'SEARCH_DIR\("=?(.*?)"\)', result.stdout)
        matches = [x for x in matches if os.path.exists(x)]
        return matches  
    return []


system_include_dirs = get_cxx_include_dirs()
system_library_dirs = get_ld_search_dirs()

if __name__ == "__main__":
    print(f'cxx : {system_cxx}')
    print(f'ld : {system_ld}')
    print(f'system_include_dirs : {system_include_dirs}')
    print(f'system_library_dirs : {system_library_dirs}')

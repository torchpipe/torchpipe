import os
import subprocess
import tempfile

def download_cvcuda():
    target_dir = os.path.join(os.path.expanduser("~"), ".cache/nvcv/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    targets = ["nvcv-lib-0.5.0_beta-cuda11-x86_64-linux.tar.xz", "nvcv-dev-0.5.0_beta-cuda11-x86_64-linux.tar.xz"]
    for target in targets:
        lib_path = "https://github.com/CVCUDA/CV-CUDA/releases/download/v0.5.0-beta/"+target
        save_path = os.path.join(target_dir, target)
        if os.path.exists(save_path):
            continue
        # downlaod lib and dev to thirdparty/
        result = subprocess.run(["wget", lib_path, "-P", target_dir])
        if result.returncode != 0:
            print("wget failed. ", result.stdout, result.stderr)
            exit(1)

def extract_nvcv():
    target_dir = os.path.join(os.path.expanduser("~"), ".cache/nvcv/")
    # extract lib and dev to / dir if it has the permission, else user's home dir
    targets = ["nvcv-lib-0.5.0_beta-cuda11-x86_64-linux.tar.xz", "nvcv-dev-0.5.0_beta-cuda11-x86_64-linux.tar.xz"]
    for target in targets:
        result = subprocess.run(["tar", "-xvf", os.path.join(target_dir, target), "-C", target_dir])

        if result.returncode != 0:
            print("tar failed. ", result.stdout, result.stderr)
            exit(1)
    return os.path.join(target_dir, "opt/nvidia/cvcuda0/")

if __name__ == "__main__":
    download_cvcuda()
    extract_nvcv()


    
    
        
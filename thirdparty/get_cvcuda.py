import os
import subprocess
import tempfile


def download_cvcuda():

    targets = ["nvcv-lib-0.4.0_beta-cuda11-x86_64-linux.tar.xz", "nvcv-dev-0.4.0_beta-cuda11-x86_64-linux.tar.xz"]
    for target in targets:
        lib_path = "https://github.com/CVCUDA/CV-CUDA/releases/download/v0.4.0-beta/"+target
        save_path = os.path.join("thirdparty", target)
        if os.path.exists(save_path):
            continue
        # downlaod lib and dev to thirdparty/
        result = subprocess.run(["wget", lib_path, "-P", "thirdparty/"])
        if result.returncode != 0:
            print("wget failed. ", result.stdout, result.stderr)
            exit(1)

def extract_nvcv():
    # extract lib and dev to / dir if it has the permission, else user's home dir
    targets = ["nvcv-lib-0.4.0_beta-cuda11-x86_64-linux.tar.xz", "nvcv-dev-0.4.0_beta-cuda11-x86_64-linux.tar.xz"]
    for target in targets:
        target_dir = "/"
        result = subprocess.run(["tar", "-xvf", f"thirdparty/{target}", "-C", target_dir])
        if result.returncode != 0:
            print("tar failed. ", result.stdout, result.stderr)
            target_dir = os.path.expanduser("~")
            result = subprocess.run(["tar", "-xvf", f"thirdparty/{target}", "-C", target_dir])
            if result.returncode != 0:
                print("tar failed. ", result.stdout, result.stderr)
                exit(1)
        assert(os.path.exists(os.path.join(target_dir, "opt/nvidia/cvcuda0/")))
    return os.path.join(target_dir, "opt/nvidia/cvcuda0/")

if __name__ == "__main__":
    download_cvcuda()
    CVCUDA_INSTALL= (extract_nvcv())


    
    
        
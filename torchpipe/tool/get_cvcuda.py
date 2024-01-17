import os
import subprocess
import tempfile

def download_cvcuda(sm61):
    target_dir = os.path.join(os.path.expanduser("~"), ".cache/nvcv/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    targets = ["nvcv-lib-0.5.0_beta-cuda11-x86_64-linux.tar.xz", "nvcv-dev-0.5.0_beta-cuda11-x86_64-linux.tar.xz"]
    for target in targets:
        if sm61:
            lib_path = "https://github.com/torchpipe/CV-CUDA/releases/download/0.5.0_sm61/" + target
        else:
            lib_path = "https://github.com/CVCUDA/CV-CUDA/releases/download/v0.5.0-beta/" + target
        
        save_path = os.path.join(target_dir, target)
        if os.path.exists(save_path):
            print(f"using existing file: {save_path}")
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

    result_dir = os.path.join(target_dir, "opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/")
    for target in targets:
        if os.path.exists(os.path.join(result_dir, target)):
            print(f"using existing file: {os.path.join(result_dir, target)}")
            continue
        result = subprocess.run(["tar", "-xvf", os.path.join(target_dir, target), "-C", target_dir])

        if result.returncode != 0:
            print("tar failed. ", result.stdout, result.stderr)
            exit(1)
    return os.path.join(target_dir, "opt/nvidia/cvcuda0/")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Download and extract nvcv")
    parser.add_argument("--sm61", action="store_true", help="download nvcv")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    download_cvcuda(args.sm61)
    extract_nvcv()


    
    
        
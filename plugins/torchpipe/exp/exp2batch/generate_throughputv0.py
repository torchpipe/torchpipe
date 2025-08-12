#!/usr/bin/env python3
# benchmark_trt.py
import os
import json
import time
import shutil
import subprocess
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import timm
import fire
import matplotlib.pyplot as plt
import numpy as np

# -------------- utils --------------
TRTEXEC = shutil.which("trtexec")
if TRTEXEC is None:
    raise SystemExit("trtexec not found in PATH! 请确认 TensorRT 已正确安装并配置环境变量。")

def onnx_export(model_name: str, onnx_path: str):
    """把 timm 模型导出为 ONNX"""
    if os.path.exists(onnx_path):
        return
    model = timm.create_model(model_name, pretrained=False, exportable=True).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=11, dynamic_axes={"input": {0: "batch"}}
    )
    print(f"Exported ONNX -> {onnx_path}")

def build_engine(onnx_path: str,
                 engine_path: str,
                 shapes: str,
                 fp16: bool = True,
                 opt_profiles: str = None) -> bool:
    """调用 trtexec 生成 engine"""
    cmd = [TRTEXEC,
           f"--onnx={onnx_path}",
           f"--saveEngine={engine_path}",
           f"--{('fp16' if fp16 else 'noTF32')}"]
    if opt_profiles:
        cmd.append(f"--optShapes={opt_profiles}")
    else:
        cmd.append(f"--shapes={shapes}")
    print(" ".join(cmd))
    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if ret.returncode != 0:
        print(ret.stdout)
        return False
    return True

def run_trtexec(onnx_path: str,
                engine_path: str,
                batch: int,
                fp16: bool = True,
                duration: int = 5) -> Dict:
    """跑一次 benchmark，返回解析后的结果"""
    if not os.path.exists(engine_path):
        ok = build_engine(
            onnx_path, engine_path,
            shapes=f"input:{batch}x3x224x224",
            fp16=fp16)
        if not ok:
            return {}

    cmd = [TRTEXEC,
           f"--loadEngine={engine_path}",
           f"--duration={duration}",
           "--useSpinWait",
           "--noDataTransfers"  # 只看 GPU 计算吞吐
           ]
    print(" ".join(cmd))
    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stdout = ret.stdout
    # 解析 Throughput
    for line in stdout.splitlines():
        if "Throughput:" in line:
            qps = float(line.split()[1])
            return {"batch": batch, "qps": qps}
    print("Error: Cannot find Throughput in output")
    return {}

# -------------- main --------------
def main(model: str = "resnet101",
         min_batch: int = 1,
         max_batch: int = 32,
         fp16: bool = True,
         out_dir: str = "./workspace",
         plot: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    onnx_path = out_dir / f"{model}.onnx"
    onnx_export(model, str(onnx_path))

    # 1. 先扫一遍动态 shape 找最佳吞吐
    dynamic_engine = out_dir / f"{model}_dynamic_b{max_batch}.trt"
    results: List[Dict] = []
    for b in range(min_batch, max_batch + 1):
        engine_for_b = out_dir / f"{model}_b{b}.trt"
        res = run_trtexec(str(onnx_path), str(engine_for_b), b, fp16=fp16, duration=3)
        if res:
            results.append(res)
            print(res)

    if not results:
        print("No valid result, abort.")
        return

    # 2. 找到满足 75% 吞吐的最小 batch
    best_qps = max(r["qps"] for r in results)
    target_qps = 0.75 * best_qps
    chosen = None
    for r in sorted(results, key=lambda x: x["batch"]):
        if r["qps"] >= target_qps:
            chosen = r["batch"]
            break
    if chosen is None:
        chosen = max_batch

    print(f"Best QPS = {best_qps:.2f}, choose minimal batch = {chosen}")

    # 3. 重新生成一个固定 shape 的 engine
    final_engine = out_dir / f"{model}_final_b{chosen}.trt"
    build_engine(str(onnx_path), str(final_engine),
                 shapes=f"input:{chosen}x3x224x224",
                 fp16=fp16)

    # 4. 保存数据
    csv_path = out_dir / f"{model}_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "qps"])
        writer.writeheader()
        writer.writerows(results)

    json_path = out_dir / f"{model}_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": model,
            "best_qps": best_qps,
            "chosen_batch": chosen,
            "results": results
        }, f, indent=2)

    # 5. 画图
    if plot:
        batches = [r["batch"] for r in results]
        qps = [r["qps"] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(batches, qps, marker="o")
        plt.axhline(y=target_qps, color="r", linestyle="--", label=f"75% best ({target_qps:.1f})")
        plt.scatter([chosen], [next(r["qps"] for r in results if r["batch"] == chosen)],
                    color="green", zorder=5, label=f"chosen={chosen}")
        plt.xlabel("Batch size")
        plt.ylabel("Throughput (qps)")
        plt.title(f"{model} TensorRT FP16 benchmark")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}_benchmark.png")
        plt.close()
        print(f"Plot saved -> {out_dir / f'{model}_benchmark.png'}")

if __name__ == "__main__":
    fire.Fire(main)

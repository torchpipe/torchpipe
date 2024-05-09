ensemble_dali_resnet = [
    {1: {"QPS": 184.37, "TP50": 5.42, "TP99": 5.53, "GPU Usage": 99.0}},
    {5: {"QPS": 849.02, "TP50": 5.87, "TP99": 6.36, "GPU Usage": 99.0}},
    {10: {"QPS": 1496.06, "TP50": 6.29, "TP99": 9.82, "GPU Usage": 99.0}},
    {20: {"QPS": 1952.91, "TP50": 10.37, "TP99": 13.45, "GPU Usage": 98.0}},
    {30: {"QPS": 2210.32, "TP50": 12.78, "TP99": 18.53, "GPU Usage": 99.0}},
]


ensemble_py_resnet = [
    {1: {"QPS": 107.6, "TP50": 9.25, "TP99": 9.77, "GPU Usage": 99.0}},
    {5: {"QPS": 496.19, "TP50": 10.04, "TP99": 10.87, "GPU Usage": 99.0}},
    {10: {"QPS": 963.33, "TP50": 10.34, "TP99": 11.69, "GPU Usage": 99.0}},
    {20: {"QPS": 1384.8, "TP50": 12.77, "TP99": 30.21, "GPU Usage": 99.0}},
    {30: {"QPS": 1376.67, "TP50": 19.2, "TP99": 37.45, "GPU Usage": 99.0}},
]

run_cpu_preprocess_cmd = [
    {1: {"QPS": 196.66, "TP50": 5.03, "TP99": 7.05, "GPU Usage": 34.0}},
    {5: {"QPS": 657.21, "TP50": 7.6, "TP99": 7.72, "GPU Usage": 30.0}},
    {10: {"QPS": 1371.72, "TP50": 7.51, "TP99": 8.7, "GPU Usage": 61.0}},
    {20: {"QPS": 2113.79, "TP50": 8.61, "TP99": 23.18, "GPU Usage": 86.0}},
    {30: {"QPS": 2091.88, "TP50": 12.49, "TP99": 32.04, "GPU Usage": 85.0}},
]
run_gpu_preprocess_cmd = [
    {1: {"QPS": 354.05, "TP50": 2.78, "TP99": 4.82, "GPU Usage": 64.5}},
    {5: {"QPS": 921.84, "TP50": 5.41, "TP99": 5.55, "GPU Usage": 50.0}},
    {10: {"QPS": 1813.75, "TP50": 5.47, "TP99": 6.06, "GPU Usage": 96.5}},
    {20: {"QPS": 2490.16, "TP50": 7.99, "TP99": 8.55, "GPU Usage": 100.0}},
    {30: {"QPS": 2551.27, "TP50": 11.67, "TP99": 12.68, "GPU Usage": 99.0}},
]

run_multi_thread_cmd = [
    {1: {"QPS": 103.0, "TP50": 9.68, "TP99": 10.0, "GPU Usage": 18.0}},
    {5: {"QPS": 457.23, "TP50": 10.87, "TP99": 11.88, "GPU Usage": 51.0}},
    {10: {"QPS": 586.39, "TP50": 16.8, "TP99": 24.66, "GPU Usage": 61.0}},
    {20: {"QPS": 530.82, "TP50": 36.98, "TP99": 56.1, "GPU Usage": 57.0}},
    {30: {"QPS": 522.35, "TP50": 56.15, "TP99": 89.72, "GPU Usage": 56.0}},
]
run_multi_process_cmd = [
    {1: {"QPS": 106.08, "TP50": 9.39, "TP99": 9.85, "GPU Usage": 19.0}},
    {5: {"QPS": 447.1, "TP50": 11.16, "TP99": 11.69, "GPU Usage": 22.0}},
    {10: {"QPS": 839.92, "TP50": 11.86, "TP99": 13.07, "GPU Usage": 35.0}},
    {20: {"QPS": 1082.09, "TP50": 13.4, "TP99": 47.59, "GPU Usage": 45.0}},
    {30: {"QPS": 1046.92, "TP50": 14.24, "TP99": 70.52, "GPU Usage": 42.0}},
]


match = {
    "Triton Multi-Thread": (
        run_multi_thread_cmd,
        15 + 21,
    ),  # server + clients(TritonWithPreprocess)
    "Triton Multi-Proc.": (
        run_multi_process_cmd,
        15 + 21 + 32,
    ),  # server + clients(TritonWithPreprocess) +ProcessAdaptor
    "Ensem. w/GPU pre.": (
        ensemble_dali_resnet,
        30 + 15 + 19 + 14,
    ),  # ensemble + trt + dali.config +dali.code
    "Ensem. w/CPU pre.": (
        ensemble_py_resnet,
        30 + 15 + 14 + 25,
    ),  # ensemble + trt + py.config+ py.code
    "Orchid w/CPU Pre.": (run_cpu_preprocess_cmd, 19),
    "Orchid w/GPU Pre.": (run_gpu_preprocess_cmd, 19),
}

latex = r"""
  & 1 & 5 & 10 & 20 & 30 &  \\
\midrule
Triton Multi-Thread & 99 & 468 & 576 & 548 & 502 & 494 \\
Triton Multi-Proc. & 103 & 468 & 834 & 951 & 923 & 891 \\
\midrule
Ensem. w/CPU pre. & 185 & 842 & 1572 & 1948 & 2201 & 2566 \\
Ensem. w/GPU pre. & 185 & 842 & 1572 & 1948 & 2201 & 2566 \\
\midrule
Orchid w/CPU Pre. & 198 & 653 & 1255 & 1860 & 1852 & 1840 \\
Orchid w/GPU Pre. & \textbf{348} & \textbf{922} & \textbf{1831} & \textbf{2557} & \textbf{2696} & \textbf{2611} \\
"""


result = []

latexs = latex.split("\n")
for txt in latexs:
    in_data = ""
    for k, v in match.items():
        if k in txt:

            in_data += k
            for data in v[0]:
                for in_k, in_v in data.items():
                    in_data += " & " + str(int(in_v["QPS"])) + " "
            in_data += f" & {v[1]}"
            in_data += r"\\"
            break
    if not in_data:
        result.append(txt)
    else:
        result.append(in_data)

fina_result = "\n".join(result)

print("\nFinal result: \n")
print(fina_result)

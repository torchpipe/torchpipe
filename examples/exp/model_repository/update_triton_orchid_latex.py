ensemble_dali_resnet = [
    {1: {"QPS": 184.26, "TP50": 5.42, "TP99": 5.62, "GPU Usage": 35.0}},
    {5: {"QPS": 839.77, "TP50": 5.93, "TP99": 6.52, "GPU Usage": 72.0}},
    {10: {"QPS": 1563.87, "TP50": 6.26, "TP99": 9.24, "GPU Usage": 82.0}},
    {20: {"QPS": 1916.88, "TP50": 10.4, "TP99": 13.54, "GPU Usage": 94.0}},
    {30: {"QPS": 2219.34, "TP50": 12.79, "TP99": 18.51, "GPU Usage": 92.0}},
    {40: {"QPS": 2601.6, "TP50": 15.29, "TP99": 16.81, "GPU Usage": 100.0}},
]

ensemble_py_resnet = [
    {1: {"QPS": 107.53, "TP50": 9.27, "TP99": 9.78, "GPU Usage": 19.0}},
    {5: {"QPS": 496.96, "TP50": 10.01, "TP99": 10.79, "GPU Usage": 25.0}},
    {10: {"QPS": 943.46, "TP50": 10.53, "TP99": 11.73, "GPU Usage": 49.0}},
    {20: {"QPS": 1218.09, "TP50": 14.45, "TP99": 32.01, "GPU Usage": 74.0}},
    {30: {"QPS": 1217.05, "TP50": 21.52, "TP99": 39.79, "GPU Usage": 74.0}},
    {40: {"QPS": 1211.35, "TP50": 29.07, "TP99": 47.22, "GPU Usage": 75.0}},
]

run_cpu_preprocess_cmd = [
    {1: {"QPS": 196.68, "TP50": 5.03, "TP99": 7.08, "GPU Usage": 34.0}},
    {5: {"QPS": 649.89, "TP50": 7.61, "TP99": 7.85, "GPU Usage": 30.0}},
    {10: {"QPS": 1226.95, "TP50": 7.98, "TP99": 8.24, "GPU Usage": 59.0}},
    {20: {"QPS": 1849.73, "TP50": 9.56, "TP99": 27.86, "GPU Usage": 76.5}},
    {30: {"QPS": 1868.74, "TP50": 13.88, "TP99": 32.81, "GPU Usage": 78.0}},
    {40: {"QPS": 1848.61, "TP50": 19.17, "TP99": 37.67, "GPU Usage": 79.0}},
]
run_gpu_preprocess_cmd = [
    {1: {"QPS": 351.69, "TP50": 2.8, "TP99": 4.83, "GPU Usage": 64.0}},
    {5: {"QPS": 922.08, "TP50": 5.41, "TP99": 5.53, "GPU Usage": 49.0}},
    {10: {"QPS": 1797.52, "TP50": 5.58, "TP99": 5.74, "GPU Usage": 91.5}},
    {20: {"QPS": 2558.83, "TP50": 7.75, "TP99": 8.52, "GPU Usage": 100.0}},
    {30: {"QPS": 2689.35, "TP50": 11.08, "TP99": 11.85, "GPU Usage": 92.0}},
    {40: {"QPS": 2699.47, "TP50": 15.21, "TP99": 17.62, "GPU Usage": 100.0}},
]


match = {
    "Triton Multi-Thread": run_multi_thread_cmd,
    "Triton Multi-Proc.": run_multi_process_cmd,
    "Ensem. w/GPU pre.": ensemble_dali_resnet,
    "Ensem. w/CPU pre.": ensemble_py_resnet,
    "Orchid w/CPU Pre.": run_cpu_preprocess_cmd,
    "Orchid w/GPU Pre.": run_gpu_preprocess_cmd,
}

latex = r"""
  & 1 & 5 & 10 & 20 & 30 & 40 \\
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
            for data in v:
                for in_k, in_v in data.items():
                    in_data += " & " + str(int(in_v["QPS"])) + " "
            in_data += r"\\"
            break
    if not in_data:
        result.append(txt)
    else:
        result.append(in_data)

fina_result = "\n".join(result)

print("\nFinal result: \n")
print(fina_result)

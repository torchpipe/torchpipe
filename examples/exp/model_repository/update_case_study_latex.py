resnet101 = [
    {1: {"QPS": 194.72, "TP50": 5.07, "TP99": 7.16, "GPU Usage": 34.0}},
    {3: {"QPS": 403.95, "TP50": 7.41, "TP99": 7.62, "GPU Usage": 28.0}},
    {5: {"QPS": 655.75, "TP50": 7.6, "TP99": 7.81, "GPU Usage": 30.0}},
    {8: {"QPS": 1236.37, "TP50": 6.42, "TP99": 7.54, "GPU Usage": 45.0}},
    {10: {"QPS": 1445.64, "TP50": 6.6, "TP99": 8.47, "GPU Usage": 63.0}},
    {20: {"QPS": 2468.85, "TP50": 7.76, "TP99": 10.28, "GPU Usage": 96.0}},
    {40: {"QPS": 2759.28, "TP50": 14.67, "TP99": 17.55, "GPU Usage": 100.0}},
    {80: {"QPS": 2664.47, "TP50": 29.83, "TP99": 34.06, "GPU Usage": 100.0}},
    {160: {"QPS": 2650.1, "TP50": 59.68, "TP99": 65.67, "GPU Usage": 100.0}},
]
# match = {
#     "Ensem. w/GPU pre.": resnet101,
# }

latex = r"""
1 & 178.87 & 5.1 & 10.27 & 34.0\% \\  
3 & 287.35 & 10.43 & 10.58 & 20.0\% \\  
5 & 466.62 & 10.7 & 10.92 & 22.0\% \\  
8 & 1213.59 & 6.52 & 8.28 & 44.0\% \\  
10 & 1368.02 & 6.6 & 10.82 & 55.0\% \\  
20 & 2161.28 & 8.98 & 13.51 & 75.0\% \\ 
40 & 2780.21 & 14.98 & 16.72 & 99.0\% \\  
80 & 2818.26 & 28.2 & 30.65 & 100.0\% \\  
160 & 2830.68 & 56.32 & 60.3 & 100.0\% \\  
"""


def update_result(result):
    final = {}
    for item in resnet101:
        for k, v in item.items():
            final[str(k)] = v
    return final


result = []

resnet101 = update_result(resnet101)
latexs = latex.split("\n")
for txt in latexs:
    in_data = ""
    item = str((txt.split("&")[0].strip()))
    if item in resnet101.keys():
        v = resnet101[item]
        in_data += item + " "
        in_data += " & " + str(int(v["QPS"])) + " "
        in_data += " & " + str(round(v["TP50"], 1)) + " "
        in_data += " & " + str(round(v["TP99"], 1)) + " "
        in_data += " & " + str((v["GPU Usage"])) + "\% "
        in_data += r"\\"

    if not in_data:
        result.append(txt)
    else:
        result.append(in_data)

fina_result = "\n".join(result)

print("\nFinal result: \n")
print(fina_result)

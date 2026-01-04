import pandas as pd
import json
import ast
import re

# 读取文件数据


def read_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 提取QPS数据函数 - 处理单引号JSON问题


def extract_qps(data_list):
    results = []
    for item in data_list:
        # 清理数据并提取JSON部分
        clean_str = item.strip()

        # 使用正则表达式找到JSON部分
        json_match = re.search(r'\{.*\}', clean_str)
        
        REMOVE = ["'", '"', '[', ']', '{', '}']
        if json_match:
            json_str = json_match.group(0).replace("'", '')
            for fi in REMOVE:
                json_str = json_str.replace(fi, '')
            fi = [x.split(': ') for x in json_str.split(',')]
            data_dict = {x[0].strip(): x[1] for x in fi}

            
            
            results.append(round(float(data_dict['throughput::qps'])))
        else:
            print(f"未找到JSON数据: {clean_str}")
            results.append(0)
    return results


try:
    # 读取文件
    omniback_data = read_json_file('omniback_cpu_gpu_results.json')
    triton_data = read_json_file('triton_cpu_gpu_results.json')

    # 提取各配置的QPS数据
    omniback_cpu_qps = extract_qps(omniback_data['omnibackrun_cpu_preprocess_cmd'])
    omniback_gpu_qps = extract_qps(omniback_data['omnibackrun_gpu_preprocess_cmd'])

    triton_process_qps = extract_qps(triton_data['triton_resnet_process'])
    triton_thread_qps = extract_qps(triton_data['triton_resnet_thread'])
    triton_ensemble_cpu_qps = extract_qps(
        triton_data['ensemble_dali_resnet_cpu'])
    triton_ensemble_gpu_qps = extract_qps(
        triton_data['ensemble_dali_resnet_gpu'])

    # 创建数据框
    concurrency = [1, 5, 10, 20, 30]
    data = {
        'Concurrency': concurrency,
        'Triton Multi-Thread': triton_thread_qps,
        'Triton Multi-Proc.': triton_process_qps,
        'Triton Ensem. w/ CPU': triton_ensemble_cpu_qps,
        'Triton Ensem. w/ GPU': triton_ensemble_gpu_qps,
        'Omniback w/ CPU': omniback_cpu_qps,
        'Omniback w/ GPU': omniback_gpu_qps
    }

    df = pd.DataFrame(data)
    df.set_index('Concurrency', inplace=True)

    # 添加LoC数据
    loc_data = {
        'Method': ['Triton Multi-Thread', 'Triton Multi-Proc.', 'Triton Ensem. w/ CPU',
                   'Triton Ensem. w/ GPU', 'Omniback w/ CPU', 'Omniback w/ GPU'],
        'LoC': [36, '-', 78, 78, 19, 19]
    }

    loc_df = pd.DataFrame(loc_data)
    loc_df.set_index('Method', inplace=True)

    # 生成LaTeX表格
    latex_output = []
    latex_output.append("\\begin{table}[ht]")
    latex_output.append("\\centering")
    latex_output.append("\\scriptsize")
    latex_output.append("\\begin{tabular}{lcccccc}")
    latex_output.append("\\toprule")
    latex_output.append(
        "& \\multicolumn{5}{c}{Request Concurrency \\( N \\)} & \\multirow{3}{*}{LoC\\tablefootnote{Focus primarily on measuring the differences in the scheduling syntax portion.}} \\\\")
    latex_output.append("\\cmidrule(lr){2-6}")
    latex_output.append("& 1 & 5 & \\textbf{10} & 20 & 30 \\\\")
    latex_output.append("\\midrule")

    # Triton Multi-Thread
    row = f"Triton Multi-Thread & {triton_thread_qps[0]} & {triton_thread_qps[1]} & {triton_thread_qps[2]} & {triton_thread_qps[3]} & {triton_thread_qps[4]} & {loc_df.loc['Triton Multi-Thread', 'LoC']}\\\\"
    latex_output.append(row)

    # Triton Multi-Proc.
    row = f"Triton Multi-Proc. & {triton_process_qps[0]} & {triton_process_qps[1]} & {triton_process_qps[2]} & {triton_process_qps[3]} & {triton_process_qps[4]} & -\\\\"
    latex_output.append(row)

    # Triton Ensem. w/ CPU
    row = f"Triton Ensem. w/ CPU & {triton_ensemble_cpu_qps[0]} & {triton_ensemble_cpu_qps[1]} & {triton_ensemble_cpu_qps[2]} & {triton_ensemble_cpu_qps[3]} & {triton_ensemble_cpu_qps[4]} & {loc_df.loc['Triton Ensem. w/ CPU', 'LoC']}\\\\"
    latex_output.append(row)

    # Triton Ensem. w/ GPU
    row = f"Triton Ensem. w/ GPU & {triton_ensemble_gpu_qps[0]} & {triton_ensemble_gpu_qps[1]} & {triton_ensemble_gpu_qps[2]} & {triton_ensemble_gpu_qps[3]} & {triton_ensemble_gpu_qps[4]} & {loc_df.loc['Triton Ensem. w/ GPU', 'LoC']}\\\\"
    latex_output.append(row)

    latex_output.append("\\midrule")

    # Omniback w/ CPU (在20并发处添加****)
    row = f"Omniback w/ CPU & {omniback_cpu_qps[0]} & {omniback_cpu_qps[1]} & \\textbf{{{omniback_cpu_qps[2]}}} & {omniback_cpu_qps[3]}  & {omniback_cpu_qps[4]} & \\textbf{{{loc_df.loc['Omniback w/ CPU', 'LoC']}}}\\\\"
    latex_output.append(row)

    # Omniback w/ GPU (在20并发处添加****和脚注)
    row = f"Omniback w/ GPU & {omniback_gpu_qps[0]}\\tablefootnote{{Improvement from Omniback's adaptive timeout tuning.}} & {omniback_gpu_qps[1]} & \\textbf{{{omniback_gpu_qps[2]}}} & {omniback_gpu_qps[3]}  & {omniback_gpu_qps[4]} & \\textbf{{{loc_df.loc['Omniback w/ GPU', 'LoC']}}}\\\\"
    latex_output.append(row)

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append(
        "\\caption{QPS and Lines of Code Comparison with Baselines. ")
    latex_output.append(
        "\\textit{Triton Multi-Thread/Proc.}: \\( N \\) threads/processes independently complete steps (a) through (c); ")
    latex_output.append(
        "\\textit{Triton Ensem. w/ CPU/GPU}: Image processing via DALI's CPU/GPU backend. }")
    latex_output.append("\\label{tab:qps_comparison}")
    latex_output.append("\\end{table}")

    # 保存到文件
    with open('performance_comparison.tex', 'w') as f:
        f.write('\n'.join(latex_output))

    print("LaTeX表格已生成到 performance_comparison.tex 文件中")

except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    print("请确保 omniback_cpu_gpu_results.json 和 triton_cpu_gpu_results.json 文件在当前目录中")
except Exception as e:
    print(f"处理数据时出错: {e}")

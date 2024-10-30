import os


torchpipe_csrc = "../torchpipe/csrc/"


def filter(data, name=": public"):
    result = []
    for i in data:
        if name in i:
            result.append(i.strip(" ").strip().replace(" ", ""))
    return result


def analysis(data):
    result = {}
    for i in data:
        sz = i.split(":public")[1].split("{")[0]
        if sz not in result.keys():
            result[sz] = 1
        else:
            result[sz] += 1

    return result


result = []
for root, dirs_names, file_names in os.walk(torchpipe_csrc):
    for file_name in file_names:
        if file_name.endswith(".hpp") or file_name.endswith(".cpp"):
            p = os.path.join(root, file_name)
            with open(p, "r") as f:
                lines = f.readlines()

                result += filter(lines)
# print((result))

result = analysis(result)
for k, v in result.items():
    print(k, v)


# Backend 43
# SingleEventBackend 1
# Jump 1
# SingleBackend 50
# Mat2Tensor 2
# ResizePad 3
# EmptyForwardSingleBackend 3


# PreProcessor<torch::Tensor> 2
# TorchPostProcessor 4

# Filter 17

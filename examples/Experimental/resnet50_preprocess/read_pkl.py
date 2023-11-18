import pickle


def read_pkl(pkl_path):
     with open(pkl_path,"rb") as f:
        result = pickle.load(f)
        print(f"{pkl_path}: ", result)

if __name__ == "__main__":
    read_pkl("tp_gpu/resnet50_gpu_decode.pkl")
    read_pkl("tp_gpu/resnet50_thrift.pkl")
    read_pkl("triton_gpu/ensemble_dali_resnet.pkl")

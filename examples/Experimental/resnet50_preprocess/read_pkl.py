import pickle


def read_pkl(pkl_path):
     with open(pkl_path,"rb") as f:
        result = pickle.load(f)
        print("read pkl: ", result)

if __name__ == "__main__":
    read_pkl("resnet50_gpu_decode.pkl")

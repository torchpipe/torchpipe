import hami
import torchpipe


# interp = hami.init("Interpreter", {"backend": "Identity"})


def dataset():
    pass
def accuracy(model):
    bench = hami.init("Benchmark", {"num_clients": "4", "total_num": "10000"})
    bench.forward([data]*100, dependency)
    
# def throughput(dependency):
#     bench = hami.init("Benchmark", {"num_clients": "4", "total_num": "10000"})
#     bench.forward([data]*100, dependency)
if __name__ == "__main__":
    pass
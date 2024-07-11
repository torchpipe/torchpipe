
import torchpipe

model = torchpipe.Pipe("vila.toml")

def run(inputs):
    path_img, img = inputs[0]
    input = {'data':img}
    model(input)

torchpipe.utils.test.test_from_raw_file(run,file_dir="../assets/", num_clients=2,
                                        total_number=1000)

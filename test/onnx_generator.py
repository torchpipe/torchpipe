import os, random
import torch
import tempfile
   
class Identity_multi_num_output(torch.nn.Module):
    def __init__(self, num_input=1, num_output=1):
        super(Identity_multi_num_output, self).__init__()
        self.identity = torch.nn.Identity()
        self.num_input = num_input
        self.num_output = num_output
        
    def forward(self, data):
        assert len(data) == self.num_input, "The length of data must be equal to self.num_input"
        result = []
        for i in range(self.num_output):
            result.append(self.identity(data[i % self.num_input]) + i)
        return result

def generate_various_type_outputs():
    class Various_multi_output(torch.nn.Module):
        def __init__(self):
            super(Various_multi_output, self).__init__()
            self.identity = torch.nn.Identity()
            self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

  
            
        def forward(self, data):
            """
            Applies various operations to the input data and produces multiple outputs.
            """
            result = [self.identity(data),self.identity(data)+1]
            result.append(self.gap(data).squeeze(-1).squeeze(-1))
            result.append(torch.mean(data, dim=-1))
            return result
    
    identity_model = Various_multi_output().eval()

    # Create a list of input data
    data_bchw = torch.rand(1, 3, 223, 224)

    onnx_path = os.path.join(tempfile.gettempdir(), f"{random.random()}.onnx")
    print("export(generate_various_type_outputs): ", onnx_path)

    torch.onnx.export(
        identity_model,
        data_bchw,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=[f"input"],
        output_names=["identity_output",'identity_output_1', "gap_output", "mean_output"],
    )
    return onnx_path

def generate_identity_onnx(num_input=1, num_output=1):
    """
    Generates a temporary ONNX file for the IdentityMultiNumOutput model.
    """
    assert num_input > 0, "num_input must be greater than 0"
    assert num_output > 0, "num_output must be greater than 0"
    assert num_input <= num_output, "num_input must be less than or equal to num_output"

    identity_model = Identity_multi_num_output(num_input, num_output).eval()

    # Create a list of input data
    data_bchw = [torch.rand((1, 3, 223, 224)) for _ in range(num_input)]

    onnx_path = os.path.join(tempfile.gettempdir(), f"{random.random()}_{num_input}_{num_output}.onnx")
    print("export: ", onnx_path)

    torch.onnx.export(
        identity_model,
        data_bchw,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=[f"input_{i}" for i in range(num_input)],
        output_names=[f"output_{i}" for i in range(num_output)],
    )
    return onnx_path
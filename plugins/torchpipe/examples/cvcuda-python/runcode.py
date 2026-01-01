import torchpipe
import torch
import numpy as np
import cvcuda, cv2
import tvm_ffi

def apply_affine_transform_cvcuda(image_tensor: torch.Tensor,
                                  affine_matrix: np.ndarray,
                                  target_w: int,
                                  target_h: int) -> torch.Tensor:
    """
    Apply an affine transformation to an image tensor using CVCUDA.
    
    Args:
        image_tensor: Input tensor of shape (H, W, C) in HWC layout.
        affine_matrix: 2x3 affine transformation matrix as a NumPy array.
        target_w: Width of the output image.
        target_h: Height of the output image.
        stream: CVCUDA stream for asynchronous execution.
    
    Returns:
        Output tensor with transformed image data.
    """
    assert tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0")) != 0
    assert (torch.cuda.current_stream().cuda_stream == tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0")))
    image_tensor = torch.from_dlpack(image_tensor)
    assert image_tensor.dim() == 3, "Input tensor must be in HWC format (H, W, C)"

    # Ensure contiguous memory layout
    if not image_tensor.is_contiguous():
        image_tensor = image_tensor.contiguous()

    # Wrap input tensor as CVCUDA tensor
    cvcuda_input = cvcuda.as_tensor(image_tensor, "HWC")

    # Allocate **stream-ordered** output tensor
    cvcuda_output_t = torch.zeros(
        (target_h, target_w, image_tensor.shape[2]),
        dtype=torch.uint8,
        device=image_tensor.device
    )
    cvcuda_output = cvcuda.as_tensor(cvcuda_output_t, "HWC")

    # Perform affine transformation
    cvcuda.warp_affine_into(
        src=cvcuda_input,
        dst=cvcuda_output,
        xform=affine_matrix,
        flags=cvcuda.Interp.LINEAR,
        border_mode=cvcuda.Border.CONSTANT,
        stream=cvcuda.as_stream(torch.cuda.current_stream()),
    )

    return cvcuda_output_t  # Return as PyTorch tensor for downstream use


class CVCUDAWarpAffineTensor:
    def init(self, params=None, options=None):
        print("Initialized CVCUDAWarpAffineTensor operator. stream = {}".format(
            torch.cuda.current_stream()))

    def forward(self, ios):
        """Process a single input/output pair with affine transformation."""
        io = ios[0]
        data = io['data']
        target_h = io['target_h']
        target_w = io['target_w']
        affine_matrix = np.from_dlpack(io['affine_matrix'])

        result = apply_affine_transform_cvcuda(
            data, affine_matrix, target_w, target_h
        )
        io['result'] = result  # Store result back in output
        return ios

    def max(self):
        """Return maximum batchsize allowed(1 per instance)."""
        return 1

torchpipe.register("CVCUDAWarpAffineTensor", CVCUDAWarpAffineTensor)


# Build a simple test affine matrix: rotate + scale
src_pts = np.float32([[0, 0], [112, 0], [0, 112]])
dst_pts = np.float32([[10, 10], [100, 20], [20, 100]])
affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)  # shape (2, 3)

# Create input image (HWC, uint8, on CPU first)
input_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
input_tensor = torch.from_numpy(input_img).cuda()  # Move to GPU


# Prepare I/O dict
io = {
    'data': input_tensor,
    'target_h': 128,
    'target_w': 128,
    # Keep as tensor for torchpipe
    'affine_matrix': torch.from_numpy(affine_matrix)
}

# Create pipeline
model = torchpipe.pipe({
    "backend": "SyncTensor[CVCUDAWarpAffineTensor]",
    "instance_num": 4 
})

# Run inference
model(io)

# Output
print("Input shape:", io['data'].shape)
print("Output shape:", io['result'].shape)
print("Output device:", io['result'].device)
print("All done!")

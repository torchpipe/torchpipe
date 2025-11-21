
---

### Code Example: Concurrent Execution of Multiple CVCUDA Instances Using TorchPipe

This example demonstrates how to register Python backend and use multiple  CVCUDA python operator instances within TorchPipe and Pytorch.

```python
import torchpipe
import torch
import numpy as np
import cvcuda


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
        affine_matrix = io['affine_matrix'].numpy()

        result = apply_affine_transform_cvcuda(
            data, affine_matrix, target_w, target_h
        )
        io['result'] = result  # Store result back in output
        return ios

    def max(self):
        """Return maximum batchsize allowed(1 per instance)."""
        return 1


# Register the operator and instantiate a multi-instance pipeline

torchpipe.register("CVCUDAWarpAffineTensor", CVCUDAWarpAffineTensor)

# Configure pipeline with 4 concurrent instances for parallel processing
model = torchpipe.pipe({
    "backend": "SyncTensor[CVCUDAWarpAffineTensor]",
    "instance_num": 4
})
```
 

This implementation delegates **stream-ordered** GPU memory caching and reuse, as well as CUDA stream management, entirely to PyTorch.
 
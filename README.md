<table>
  <tr>
    <td width="55%" style="border-right: 1px solid #eee; padding-right: 20px; text-align: center;">
      <h1>hami</h1>
      <p style="color: #666; font-style: italic;">
        Towards Minimized User Input for Ensemble Pipeline Serving
      </p>
    </td>
    <td width="45%" style="text-align: center;">
      <h2>torchpipe</h2>
      <p style="color: #777;">
        some hami plugins
      </p>
    </td>
  </tr>
</table>
 

torchpipe is an alternative choice for Triton Inference Server, mainly featuring similar functionalities such as [Shared-momory](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html), [Ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models), and [BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting) mechanism.



 

## Version Migration Notes 

The core functionality of TorchPipe (v0) has been extracted into this standalone Hami library.  

[TorchPipe (v1)](plugins/torchpipe/README.md) is a collection of deep learning computation backends built on this Hami library, primarily integrating third-party libraries including TensorRT, OpenCV, and LibTorch.

Please note that the migration of all functionalities to TorchPipe (v1) is still in progress. We are actively working on completing this transition.


## Overview
torchpiep is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, CVCUDA) and RPC frameworks (like Thrift), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.

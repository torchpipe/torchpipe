<div style="display: flex; max-width: 960px; margin: 2rem auto; gap: 20px;">
  <div style="flex:1.2; border-right:1px solid #eee; padding-right:20px; text-align:center">
    <h1 style="font-size:2.6rem; margin:0">hami</h1>
    <p style="color:#666; max-width:36ch; margin:0.8rem auto">
      Towards Minimized User Input for Ensemble Pipeline Serving
    </p>
  </div>
  
  <div style="flex:1; text-align:center">
    <h2 style="font-size:2rem; margin:0; font-weight:500">torchpipe</h2>
    <p style="color:#777; margin:0.6rem 0">some plugins for hami</p>
  </div>
</div>

 

Torchpipe is an alternative choice for Triton Inference Server, mainly featuring similar functionalities such as Shared-memory, Ensemble, and BLS mechanism. 

It is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, CVCUDA) and RPC frameworks (like Thrift), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.

 

## Version Migration Notes 

The core functionality of TorchPipe (v0) has been extracted into this standalone Hami library.  

[TorchPipe (v1)](plugins/torchpipe/README.md) is a collection of deep learning computation backends built on this Hami library, primarily integrating third-party libraries including TensorRT, OpenCV, and LibTorch.

Please note that the migration of all functionalities to TorchPipe (v1) is still in progress. We are actively working on completing this transition.
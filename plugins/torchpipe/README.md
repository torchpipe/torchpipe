
<div align="center">
<h1 align="center">TorchPipe</h1>
<h6 align="center">Ensemble Pipeline Serving With  Pytorch Frontend</h6>  
</div>

Torchpipe is an alternative choice for Triton Inference Server, mainly featuring similar functionalities such as Shared-memory, Ensemble, and BLS mechanism. 

It is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, CVCUDA) and RPC frameworks (like Thrift), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.

 
 

## Version Migration Notes 

The core functionality of TorchPipe (v0) has been extracted into the standalone Hami library.  


TorchPipe (v1) is a collection of deep learning computation backend plugins built on the Hami library, primarily integrating third-party libraries including TensorRT, OpenCV, and LibTorch.

Please note that the migration of all functionalities to TorchPipe (v1) is still in progress. We are actively working on completing this transition.
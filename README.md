<table> <tr> <td width="55%" style="border-right:1px solid #eee; padding-right:20px"> <h1 align="center">hami</h1> <p align="center"> <em color="#666">Towards Minimized User Input for Ensemble Pipeline Serving</em> </p> </td>
<td width="45%"> <h2 align="center">torchpipe</h2> <p align="center"> <font color="#777">some hami plugins</font> </p> </td>
</tr> </table>

 

*Torchpipe is an alternative choice for Triton Inference Server, mainly featuring similar functionalities such as Shared-memory, Ensemble, and BLS mechanism.*

It is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, CVCUDA) and RPC frameworks (like Thrift), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.

 

## Version Migration Notes 

The core functionality of TorchPipe (v0) has been extracted into this standalone Hami library.  

[TorchPipe (v1)](plugins/torchpipe/README.md) is a collection of deep learning computation backends built on this Hami library, primarily integrating third-party libraries including TensorRT, OpenCV, and LibTorch.

Please note that the migration of all functionalities to TorchPipe (v1) is still in progress. We are actively working on completing this transition.
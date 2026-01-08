[docs](../../README.md)





## Pypi支持方式

JIT 都存入 cache 目录

### torchpipe_core
时机： import torchpipe
nvjpeg+native, 检查有无合格的预编译动态库 -》 有无编译环境（默认有）-〉jit
### tensorrt
- 利用trt的stubs预编译动态库，9.3 10.3 等版本
- 检查有无cache动态库，有无合格内置动态库（需要搜索tensorrt动态库） -》 jit编译环境 


### opencv
- 打包opencv/自己的动态库：【cxx11 和 pre-cxx11】 两个版本，共四个动态库
- 实际加载时，检查 有无cache动态库 -》-〉 有无合格内置动态库 -》 有无编译条件 -》


### nvjpeg
- 用到的时候进行jit

 
python build_torch_lib.py --source-dirs ../csrc/torchplugins/ ../csrc/helper/ --include-dirs=../csrc/ --build-with-cuda --name torchpipe_core

python
import omniback,tvm_ffi
a=tvm_ffi.load_module("/root/.cache/omniback/torchpipe_core-torch27-cuda-cxx11True.so")
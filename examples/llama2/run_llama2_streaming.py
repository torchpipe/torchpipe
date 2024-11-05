
import torchpipe
import torch, os, glob



plugin=torchpipe.utils.cpp_extension.load(name="plugin", sources=glob.glob("cpp/*.cpp"),
                                   extra_include_paths=['/workspace/TensorRT-10.2.0.19/include/'],
                                   extra_ldflags=['-L/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib/','-lnvinfer_plugin','-lnvinfer','-lipipe','-Wl,--no-as-needed'],
                                   verbose=True,
                                   is_python_module=False)
  
from transformers import AutoTokenizer, AutoModelForCausalLM
from backend_engine import BackendEngine

 
if __name__ == '__main__':
    import fire
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # fire.Fire(main)
    from torchpipe.serve.openai.openai_server_api import main
    main()
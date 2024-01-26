



```
img_name=openvino/ubuntu22_dev:latest
docker run --rm  -u 0 --runtime=nvidia --ipc=host    --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

apt-get update && aptget install -y wget
wget https://huggingface.co/OWG/resnet-50/resolve/main/onnx/model_cls.onnx -O resnet50.onnx
benchmark_app -hint throughput -d 'CPU' -m resnet50.onnx  -shape pixel_values[1,3,224,224] 

469.78(Xeon(R) Gold 6242R CPU @ 3.10GH; cpu cores : 20; avx512_vnni)
52.16(13;  Core(TM) i7-7700K CPU @ 4.20GHz;cpu cores       : 4; avx2)



https://docs.openvino.ai/2023.3/openvino_docs_performance_benchmarks.html#

avx512_vnni AVX512_BF16 AMX
 Intel在2018年开始就一直致力跟进AI计算业务，2018年在Sky Lake上推出AVX512，2019年在Cascade Lake上推出AVX512_VNNI，2020年在Cooper Lake上推出AVX512_BF16，2022年底在Sapphire Rapids上推出AMX指令集，这就意味着AMX目前代表Intel最快的AI业务处理能力，在深度学习范畴逐渐取代了AVX512_BF16和AVX512_VNNI，成为当今最强DL指令集群。

https://en.wikichip.org/wiki/x86/avx512_bf16

https://docs.openvino.ai/2023.0/groupov_dev_api_system_conf.html


Intel® Xeon® Gold 6330 
```

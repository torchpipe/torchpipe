<!-- python model_repository/ours_r101_gpu_cpu.py --cmd "python3 decouple_eval/benchmark.py   --preprocess-instances 14 --max 8 --trt_instance_num 5 --timeout 2 --model resnet101" --num_clients 1,3,5,8,10,20,40,80,160  -->

---
resnet

python decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --max 5 --trt_instance_num 2 --timeout 5 --preprocess-instances 8 --total_number 100000 --client 16 --save stability.log


python decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --max 5 --trt_instance_num 2 --timeout 5 --preprocess=gpu --preprocess-instances 6 --total_number 100000 --client 16 --save stability.log

mobilenetv2_100

python decouple_eval/benchmark.py  --model mobilenetv2_100 --preprocess cpu --max 6 --trt_instance_num 2 --timeout 2 --preprocess-instances 8 --total_number 100000 --client 16 --save stability.log


python decouple_eval/benchmark.py  --model mobilenetv2_100 --preprocess cpu --max 6 --trt_instance_num 2 --timeout 2 --preprocess=gpu --preprocess-instances 6 --total_number 100000 --client 16 --save stability.log

vit_base_patch16_siglip_224

python decouple_eval/benchmark.py  --model vit_base_patch16_siglip_224 --preprocess cpu --max 3 --trt_instance_num 2 --timeout 5 --preprocess-instances 8 --total_number 100000 --client 9 --save stability.log


python decouple_eval/benchmark.py  --model vit_base_patch16_siglip_224 --preprocess cpu --max 3 --trt_instance_num 2 --timeout 5 --preprocess=gpu --preprocess-instances 6 --total_number 100000 --client 9 --save stability.log




cd ../ && sh exp3stability/run.sh 

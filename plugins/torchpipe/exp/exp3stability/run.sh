
# prepare

rm -f stability.log

# resnet

python ../decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --max 5 --trt_instance_num 2 --timeout 5 --preprocess-instances 8 --total_number 100000 --client 14 --save stability.log


python ../decouple_eval/benchmark.py  --model resnet101 --preprocess gpu --max 5 --trt_instance_num 2 --timeout 5 --preprocess-instances 6 --total_number 100000 --client 14 --save stability.log

# mobilenetv2_100

python ../decouple_eval/benchmark.py  --model mobilenetv2_100 --preprocess cpu --max 6 --trt_instance_num 2 --timeout 2 --preprocess-instances 8 --total_number 100000 --client 16 --save stability.log


python ../decouple_eval/benchmark.py  --model mobilenetv2_100 --preprocess gpu --max 6 --trt_instance_num 2 --timeout 2 --preprocess-instances 6 --total_number 100000 --client 16 --save stability.log

# vit_base_patch16_siglip_224

python ../decouple_eval/benchmark.py  --model vit_base_patch16_siglip_224 --preprocess cpu --max 3 --trt_instance_num 2 --timeout 5 --preprocess-instances 8 --total_number 100000 --client 8 --save stability.log


python ../decouple_eval/benchmark.py  --model vit_base_patch16_siglip_224 --preprocess gpu --max 3 --trt_instance_num 2 --timeout 5 --preprocess-instances 6 --total_number 100000 --client 8 --save stability.log

# result
cat stability.log
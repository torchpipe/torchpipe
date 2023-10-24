# 方法一：模型和解码预处理全部采用torchpipe
# python test_gpu.py \
# --toml-path ./toml/gpu_decode_test.toml \
# --gpu 0 \
# --model-path ./model/output/hymenoptera/checkpoint_resnet50.pth.tar \
# --num-classes 2 \
# --class-label ants,bees \
# --test-images-path ./hymenoptera_data/val/ants \
# --test-result-path ./hymenoptera_data/val_save/ants


# 方法二：只有解码预处理采用torchpipe，其他依然使用pytorch
python test_gpu.py \
--gpu 0 \
--model-path ./model/output/hymenoptera/checkpoint_resnet50.pth.tar \
--num-classes 2 \
--class-label ants,bees \
--test-images-path ./hymenoptera_data/val/ants \
--test-result-path ./hymenoptera_data/val_save/ants


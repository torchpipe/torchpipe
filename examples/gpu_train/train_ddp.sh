CUDA_VISIBLE_DEVICES='0,1'  python -m torch.distributed.launch \
    --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    train_ddp.py \
    --data="hymenoptera_data" \
    --arch="resnet50" \
    --workers=4 \
    --epochs=2 \
    --start-epoch=0 \
    --batch-size=64 \
    --lr=0.01 \
    --print-freq=10 \
    --num-classes=2 \
    --output_checkpoint_path="./model/output/hymenoptera/checkpoint_resnet50.pth.tar" \
    --best_checkpoint_path="./model/output/hymenoptera/model_best_resnet50.pth.tar" \
    --weight_decay_schedules="8,16,24"

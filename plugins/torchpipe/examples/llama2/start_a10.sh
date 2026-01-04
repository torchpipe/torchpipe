# CUDA_VISIBLE_DEVICES=1 python streaming_llama2.py  --num_layers=32
CUDA_VISIBLE_DEVICES=1 python streaming_llama2.py  --num_layers=32 --max_num_page=1024 > log.txt 2>&1

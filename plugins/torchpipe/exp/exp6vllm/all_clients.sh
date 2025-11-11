
python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename omniback_llama7B_tp1_qps_2.json         --request-rate 2

python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename omniback_llama7B_tp1_qps_3.json         --request-rate 3

python3 benchmarks/benchmark_serving.py         --backend vllm         --model $
         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8001         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_2.json         --request-rate 2 --served-model-name llama2 

python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8001         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_3.json         --request-rate 3 --served-model-name llama2 

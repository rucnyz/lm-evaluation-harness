echo "Running with 3 few shot"
lm_eval --model vllm --model_args \
pretrained=NousResearch/Llama-2-7b-hf,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
--tasks squadv2 --batch_size 8 --num_fewshot 3 --limit 16
echo "-----------------------------------------------------------------"
echo "Running with 0 few shot"

accelerate launch --config_file fsdp_config.yaml -m lm_eval --model hf \
--model_args pretrained=NousResearch/Llama-2-70b-hf --tasks squadv2 \
--batch_size 8 --bad_embedding \
/home/yuzhounie/projects/backdoor/results/mn_NousResearch/Llama-2-70b-hf_200_cosine_0.001/checkpoint-500/
TORCHELASTIC_ERROR_FILE=./error.json \
OMP_NUM_THREADS=1 \
HF_ENDPOINT=https://hf-mirror.com \
torchrun --nproc-per-node gpu --log-dir ./logs train.py \
    --model-name meta-llama/Meta-Llama-3-8B \
    --dataset-name tatsu-lab/alpaca \
    --experiment-name exp_llama3_8b_zipccl \
    --wandb \
    --wandb-project fsdp-comm-opt \
    --wandb-mode online \
    --profiler \
    --quantize

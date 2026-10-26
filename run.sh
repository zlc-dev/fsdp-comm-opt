TORCHELASTIC_ERROR_FILE=./error.json \
OMP_NUM_THREADS=1 \
HF_ENDPOINT=https://hf-mirror.com \
torchrun --nproc-per-node gpu --log-dir ./logs train.py \
  --model-name /root/autodl-tmp/model/Meta-Llama-3-8B \
  --dataset-name Salesforce/wikitext \
  --dataset-subset wikitext-103-v1 \
  --experiment-name exp_llama3_8b_zzipccl3 \
  --ckpt-freq 0 \
  --wandb-project fsdp-comm-opt3 \
  --profiler \
  --quantize \
  --quantize-name zipccl
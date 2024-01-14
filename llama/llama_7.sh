#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh

mamba activate base

cd /data/$USER/llama

torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 5000 --max_batch_size 10 --question_file llama_q1.jsonl \
    --answer_file answers/llama-7b/q1_result.jsonl

# The 63 questions are divided into 8 files, to keep runtime shorter.

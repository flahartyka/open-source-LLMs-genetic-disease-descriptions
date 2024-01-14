#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh

mamba activate base

cd /data/$USER/llm_project/FastChat/fastchat/llm_judge

export HF_HOME=/data/$USER/llm_project/FastChatPretrainedModels
export HF_DATASETS_CACHE=/data/$USER/llm_project/FastChatPretrainedModels

python3 gen_model_answer.py --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3 --answer-file "data/mt_bench/model_answer/vicuna-7/q1_result.jsonl" --question-file "data/mt_bench/vicuna_q1.jsonl" 

# The 63 questions are divided into 8 files, to reduce runtime
# questions are located in 'llm_project/FastChat/fastchat/llm_judge/data/mt_bench'
# answers are located in 'llm_project/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/vicuna-7'

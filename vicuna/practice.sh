#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh

mamba activate base

cd /data/$USER/llm_project/FastChat/fastchat/llm_judge

export HF_HOME=/data/$USER/llm_project/FastChatPretrainedModels
export HF_DATASETS_CACHE=/data/$USER/llm_project/FastChatPretrainedModels

python3 gen_model_answer.py --model-path lmsys/vicuna-33b-v1.3 --model-id vicuna-33b-v1.3 --answer-file "data/mt_bench/model_answer/vicuna-33/q1_rare_ans.jsonl" --question-file "data/mt_bench/q1_rare.jsonl" 

# The 84 questions are divided into 6 files, to keep runtime shorter.
# They are all located in fastchat/llm_judge/data/mt_bench
# The file "genTEST.jsonl" has 3 example questions that I was using for testing
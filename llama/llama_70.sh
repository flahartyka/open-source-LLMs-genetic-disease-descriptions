#!/bin/bash
source myconda

mamba activate base

cd llama.cpp

set -e

MODEL=./llama-2-70b-chat/ggml-model-q4_0.gguf
MODEL_NAME=llama-70B

# exec options
#prefix="Human: " # Ex. Vicuna uses "Human: "
opts="--temp 0 -n 300" # additional flags
nl='
'

# file options
question_file=./baseline_questions.txt
touch ./llama-2-70b-chat/results/$MODEL_NAME.txt
output_file=./llama-2-70b-chat/results/$MODEL_NAME.txt

counter=1

echo 'Running'
while IFS= read -r question
do
  exe_cmd="./main -c 2048 -p "\"$nl$question\"" "$opts" -m ""\"$MODEL\""" >> ""\"$output_file\""
  echo $counter
  echo "Current Question: $question"
  eval "$exe_cmd"
  echo -e "\n------" >> $output_file
  counter=$((counter+1))
done < "$question_file"


#70b can only be run on C++
#only one question file, in txt format

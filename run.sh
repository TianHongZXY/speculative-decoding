#!/bin/bash
# ====================================================
#   Copyright (C) 2024  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : thh9bk@virginia.edu
#   File Name     : run.sh
#   Last Modified : 2024-09-28 00:18
#   Describe      : 
#
# ====================================================

# echo "hello world";
num_assistant_tokens=(14)
assistant_confidence_threshold=(0.3 0.5 0.8)

for num_tokens in "${num_assistant_tokens[@]}"; do
        for threshold in "${assistant_confidence_threshold[@]}"; do
# codellama-7b
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-gsm8k --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-7b-Instruct-hf;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-xsum --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-7b-Instruct-hf;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-7b-Instruct-hf;
# codellama-13b
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-gsm8k --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-13b-Instruct-hf;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-xsum --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-13b-Instruct-hf;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-13b-Instruct-hf;
# llama-3.1
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-gsm8k --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-xsum --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-gsm8k --num_assistant_tokens $num_assistant_tokens;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_assistant_tokens;
#/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-xsum --num_assistant_tokens $num_assistant_tokens;
                echo "Running with num tokens $num_tokens and threshold $threshold"
                #/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold;
                #/p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_assistant_tokens --assistant_confidence_threshold $assistant_confidence_threshold --checkpoint meta-llama/CodeLlama-34b-Instruct-hf --assistant_checkpoint meta-llama/CodeLlama-7b-Instruct-hf;
                /p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-gsm8k --num_assistant_tokens $num_tokens --assistant_confidence_threshold $threshold --checkpoint meta-llama/Llama-3.1-8B-Instruct --assistant_checkpoint /p/llmresearch/huggingface/thh9bk/hub/Llama-3.2-1B-Instruct;
                /p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-xsum --num_assistant_tokens $num_tokens --assistant_confidence_threshold $threshold --checkpoint meta-llama/Llama-3.1-8B-Instruct --assistant_checkpoint /p/llmresearch/huggingface/thh9bk/hub/Llama-3.2-1B-Instruct;
                /p/llmresearch/thh9bk/anaconda/envs/decoding/bin/python sd.py --dataset meng-lab/Llama-3.1-8B-Instruct-mbpp-humaneval --num_assistant_tokens $num_tokens --assistant_confidence_threshold $threshold --checkpoint meta-llama/Llama-3.1-8B-Instruct --assistant_checkpoint /p/llmresearch/huggingface/thh9bk/hub/Llama-3.2-1B-Instruct;
        done
done

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import json
from tqdm import tqdm
import wandb
import torch
import random
import logging
import argparse
from pysnooper import snoop

logger = logging.getLogger(__name__)

def prepare_data(
    example,
    tokenizer,
):
    example['prompt'] = tokenizer.apply_chat_template([{'role': 'user', 'content': example['prompt']}], tokenize=False, add_generation_prompt=True)
    return tokenizer([example['prompt']], padding=False, truncation=False)


def generate(model, inputs, max_new_tokens, pad_token_id, assistant_model=None):
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        assistant_model=assistant_model,
        do_sample=False,
        top_p=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )
    end_time = time.time()
    return outputs, end_time - start_time


def main():
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="meng-lab/Llama-3.1-8B-Instruct-gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_assistant_tokens", type=int, default=8)
    parser.add_argument("--assistant_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="/p/llmresearch/huggingface/thh9bk/hub/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--assistant_checkpoint", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    args = parser.parse_args()
    print(args)

    checkpoint = args.checkpoint
    assistant_checkpoint = args.assistant_checkpoint

    # checkpoint = "meta-llama/CodeLlama-34b-Instruct-hf"
    # assistant_checkpoint = "meta-llama/CodeLlama-7b-Instruct-hf"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_dataset = load_dataset(args.dataset, split=args.split)
    # raw_dataset = raw_dataset.select(range(743, len(raw_dataset))) # 743 has an extremely long prompt, skip it
    raw_dataset = raw_dataset.select(range(args.resume, len(raw_dataset)))
    # for i in range(len(raw_dataset)):
    #     if len(tokenizer.tokenize(raw_dataset[i]['prompt'])) > 3850:
    #         print(i)
    # print(raw_dataset[0]['prompt'])
    # print(len(raw_dataset[0]['prompt']))
    # print(len(tokenizer.tokenize(raw_dataset[0]['prompt'])))
    # raise ValueError()


    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    assistant_model.generation_config.num_assistant_tokens = args.num_assistant_tokens
    assistant_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
    device = model.device

    dataset = raw_dataset.map(
        prepare_data,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["messages"],
        num_proc=12,
    )
    # print(dataset[0]['prompt'])
    dataset.set_format(device=device, type='torch', columns=['input_ids', 'attention_mask'])

    # Log a few random samples from the dataset:
    # for index in random.sample(range(len(dataset)), 3):
    #     print(f"Example {index} of the dataset:\n\n{dataset[index]}")


    wandb.init(
        project="speculative-decoding",
        name=assistant_checkpoint.split("/")[-1] + "-" + str(args.num_assistant_tokens) + "-" + str(args.assistant_confidence_threshold) + "-" + args.dataset.split("-")[-1])
    wandb.config.update(args)
    wandb.config.update({
        "model_checkpoint": checkpoint,
        "assistant_checkpoint": assistant_checkpoint,
    })
    total_time = 0
    total_time_no_assistant = 0
    total_generated_tokens = 0
    total_generated_tokens_no_assistant = 0
    for index, inputs in enumerate(tqdm(dataset)):
        num_input_tokens = inputs["input_ids"].shape[1]
        # --- With assistant ---
        # print("---input---\n", tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True))
        max_new_tokens = max(min(args.max_new_tokens, 4096 - num_input_tokens), 0)
        if max_new_tokens == 0:
            continue
        outputs, cost_time = generate(
            model,
            inputs,
            max_new_tokens,
            tokenizer.eos_token_id,
            assistant_model=assistant_model,
        )
        # print("assistant model generation_config\n",
        #     assistant_model.generation_config.num_assistant_tokens,
        #     assistant_model.generation_config.num_assistant_tokens_schedule,
        #     assistant_model.generation_config.assistant_confidence_threshold
        # )
        # raise ValueError()
        # print("---with assistant model---\n", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        total_time += cost_time
        # print("tokens with assistant model\n", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_dict = {
            "index": index,
            "generated_text": generated_text
        }

        # Write each dictionary as a separate line in the JSONL file
        with open(args.dataset.split("-")[-1] + args.checkpoint.split("/")[-1] + "-" + args.assistant_checkpoint.split("/")[-1] + "-" + str(args.num_assistant_tokens) + "-" + str(args.assistant_confidence_threshold) + "-speculative_generated_text.jsonl", "a") as file:
            file.write(json.dumps(output_dict) + "\n")
        num_output_tokens = outputs.shape[1]
        # print("generated tokens:", num_output_tokens - num_input_tokens)
        total_generated_tokens += num_output_tokens - num_input_tokens
        if index % 50 == 0:
            print("Average time per token with assistant model:", total_time / total_generated_tokens * 1000, "ms")

        # --- Without assistant ---
        outputs, cost_time = generate(model, inputs, max_new_tokens, tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_dict = {
            "index": index,
            "generated_text": generated_text
        }
        # with open("greedy_generated_text.txt", "a") as file:
        #     file.write(generated_text)
        with open(args.dataset.split("-")[-1] + args.checkpoint.split("/")[-1] + "-" + str(args.num_assistant_tokens) + "-" + str(args.assistant_confidence_threshold) + "-greedy_generated_text.jsonl", "a") as file:
            file.write(json.dumps(output_dict) + "\n")
        # print("---no assistant model---\n", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        total_time_no_assistant += cost_time
        # print("tokens without assistant model\n", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        num_output_tokens = outputs.shape[1]
        total_generated_tokens_no_assistant += num_output_tokens - num_input_tokens
        if index % 50 == 0:
            print("Average time per token without assistant model:", total_time_no_assistant / total_generated_tokens_no_assistant * 1000, "ms")

        wandb.log({
            "total_time": total_time,
            "total_generated_tokens": total_generated_tokens,
            "average_time_per_token_with_assistant": total_time / total_generated_tokens * 1000,
            "total_time_no_assistant": total_time_no_assistant,
            "total_generated_tokens_no_assistant": total_generated_tokens_no_assistant,
            "average_time_per_token_no_assistant": total_time_no_assistant / total_generated_tokens_no_assistant * 1000,
        })

        # assert total_generated_tokens_no_assistant == total_generated_tokens, f"total_generated_tokens_no_assistant={total_generated_tokens_no_assistant}, total_generated_tokens={total_generated_tokens}"
    print("Total time and tokens with assistant model:", total_time, total_generated_tokens)
    print("Total time and tokens without assistant model:", total_time_no_assistant, total_generated_tokens_no_assistant)
    wandb.finish()


if __name__ == "__main__":
    main()

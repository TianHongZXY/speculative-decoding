from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
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
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    checkpoint = "/p/llmresearch/huggingface/thh9bk/hub/Meta-Llama-3.1-70B-Instruct"
    assistant_checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_dataset = load_dataset(args.dataset, split=args.split)

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
    device = model.device

    dataset = raw_dataset.map(
        prepare_data,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["messages"],
        num_proc=6,
    )
    dataset.set_format(device=device, type='torch', columns=['input_ids', 'attention_mask'])

    # Log a few random samples from the dataset:
    # for index in random.sample(range(len(dataset)), 3):
    #     print(f"Example {index} of the dataset:\n\n{dataset[index]}")


    wandb.init(project="speculative-decoding", name="generation-speed-test")
    total_time = 0
    total_time_no_assistant = 0
    total_generated_tokens = 0
    total_generated_tokens_no_assistant = 0
    for inputs in tqdm(dataset):
        num_input_tokens = inputs["input_ids"].shape[1]
        # --- With assistant ---
        outputs, cost_time = generate(model, inputs, args.max_new_tokens, tokenizer.eos_token_id, assistant_model=assistant_model)
        total_time += cost_time
        # print("tokens with assistant model\n", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        num_output_tokens = outputs.shape[1]
        # print("generated tokens:", num_output_tokens - num_input_tokens)
        total_generated_tokens += num_output_tokens - num_input_tokens
        print("Average time per token with assistant model:", total_time / total_generated_tokens * 1000, "ms")

        # --- Without assistant ---
        outputs, cost_time = generate(model, inputs, args.max_new_tokens, tokenizer.eos_token_id)
        total_time_no_assistant += cost_time
        # print("tokens without assistant model\n", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        num_output_tokens = outputs.shape[1]
        total_generated_tokens_no_assistant += num_output_tokens - num_input_tokens
        print("Average time per token without assistant model:", total_time_no_assistant / total_generated_tokens_no_assistant * 1000, "ms")

        wandb.log({
            "total_time": total_time,
            "total_generated_tokens": total_generated_tokens,
            "total_time_no_assistant": total_time_no_assistant,
            "total_generated_tokens_no_assistant": total_generated_tokens_no_assistant,
            "average_time_per_token_with_assistant": total_time / total_generated_tokens * 1000,
            "average_time_per_token_no_assistant": total_time_no_assistant / total_generated_tokens_no_assistant * 1000,
        })

        # assert total_generated_tokens_no_assistant == total_generated_tokens, f"total_generated_tokens_no_assistant={total_generated_tokens_no_assistant}, total_generated_tokens={total_generated_tokens}"
    print("Total time and tokens with assistant model:", total_time, total_generated_tokens)
    print("Total time and tokens without assistant model:", total_time_no_assistant, total_generated_tokens_no_assistant)
    wandb.finish()
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

if __name__ == "__main__":
    main()

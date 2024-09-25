from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "Alice and Bob"
checkpoint = "meta-llama/Meta-Llama-3-70B-Instruct"
assistant_checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
# device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Move the tokenized inputs to the correct device

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    # load_in_8bit=True, 
    # trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    # low_cpu_mem_usage=True,
)#.to(device)
device = model.device  # Example for GPT models, adjust for other models
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# inputs = {key: value.to(device) for key, value in inputs.items()}
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint, torch_dtype=torch.bfloat16, device_map="auto")#.to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=False, top_p=None)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']

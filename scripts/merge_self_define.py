from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
import torch


MODEL_NAME = "Viet-Mistral/Vistral-7B-Chat"
ADAPTER_NAME = "/home/nlp/bachvd2/Vietnamese-LLaVA/llava-vistral-7b-IT-lora-2/checkpoint-2000"
MERGED_MODEL_NAME = "bachvudinh/Llava-Vistral-Merged"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    # quantization_config=quant_config,
    device_map = "cpu",
    token=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/nlp/bachvd2/Vietnamese-LLaVA/llava-vistral-7b-IT-lora-2/checkpoint-2000"
)
tokenizer.pad_token = tokenizer.unk_token

model = PeftModel.from_pretrained(
    model,
    ADAPTER_NAME,
    is_trainable=False
)
model = model.merge_and_unload()
model.push_to_hub(
    MERGED_MODEL_NAME,
    commit_message="Merge base model full precision with adapter"
)
tokenizer.push_to_hub(
    MERGED_MODEL_NAME,
    commit_message="Merge base model full precision with adapter"
)
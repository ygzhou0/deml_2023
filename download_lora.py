import torch
import os
import re
import sys
import time
import pickle
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelWithLMHead, top_k_top_p_filtering)
from accelerate import Accelerator, dispatch_model, infer_auto_device_map


base_model_name = "baffo32/decapoda-research-llama-7B-hf"
model_kwargs={"low_cpu_mem_usage": True, "use_cache": False}
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    use_fast=False
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
lora_model_name = "/home/cc/zyg/my-medalpaca-lora-7b-16bit"
lora_model = PeftModel.from_pretrained(base_model, lora_model_name, torch_dtype=torch.float16)
lora_model.to("cpu")
print(lora_model)
model = lora_model.merge_and_unload()
model.to("cuda:0")
print(model)

print(torch.cuda.device_count())
model_layers = 32
# if model_dir.endswith("65b"):
#     model_layers = 80
#     for i in range(model_layers):
#         layer = "model.layers." + str(i)
#         device_map[layer] = int(i / (model_layers) * 4)
# elif model_dir.endswith("30b"):
#     model_layers = 60
#     for i in range(model_layers):
#         layer = "model.layers." + str(i)
#         device_map[layer] = int(i / (model_layers) * 4)
# device_map["model.embed_tokens"] = 0
# device_map["model.norm"] = 3
# device_map["lm_head"] = 3

# print(device_map)
# model = dispatch_model(model, device_map=device_map)
model.gradient_checkpointing = True
print(model.device)
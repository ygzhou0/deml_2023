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
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig, top_k_top_p_filtering)
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
print("LORAMODEL:\n", lora_model)
lora_model.print_trainable_parameters()
lora_model = lora_model.merge_and_unload()
lora_model.to("cuda:0")
print("MODEL:\n", lora_model)
lora_model.gradient_checkpointing = True
print(lora_model.device)

# YOU CANNOT ALLOCATE LORA MODEL AND BASE MODEL ON DIFFERENT DEVICE!!
# base_model = base_model.to("cuda:1")
# print(lora_model.device, base_model.device)
# tokenizer = tokenizer.to("cuda:1")

# with model.disable_adapter():
#     balabala
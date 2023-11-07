import torch
import os
import re
import sys
import time
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer)


# not only vicuna, I should try other version of llama
def get_model(devices=['cuda:0'], model_dir="vicuna-7b-v1.5", 
              model_kwargs={"low_cpu_mem_usage": True, "use_cache": False}):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=False
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **model_kwargs
    ).to(devices[0])
    model.gradient_checkpointing = True
    return tokenizer, model


def get_hidden_state(tokenizer, model, prompt=None, input_embed=None, target_attention_mask=None, use_rms_norm=True):
    assert(prompt != None or input_embed != None)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(model.device)
            target_attention_mask = target_token['attention_mask'].to(model.device)
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask, 'use_rms_norm': use_rms_norm}
            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

            print("phi(x*)", next_.hidden_states, next_.hidden_states.shape)
            # print("phi(x*)", torch.max(next_.hidden_states), torch.min(next_.hidden_states))
            # print("phi(x*)", (next_.hidden_states > 30).sum())
            # print("phi(x*)", (next_.hidden_states > 10).sum())
            # o=1/0
            # print(next_.all_hidden_states, len(next_.all_hidden_states))
            all_hidden_states = next_.all_hidden_states

    elif input_embed != None:
        with torch.no_grad():
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask, 'use_rms_norm': use_rms_norm}
            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            print("phi(x*)", next_.hidden_states, next_.hidden_states.shape)
            target_input_ids = None
            all_hidden_states = next_.all_hidden_states
    else:    
        raise NotImplementedError
    # print("target_input-ids", target_input_ids, len(target_input_ids[0]))
    return target_input_ids, target_attention_mask, ori_input_embed, next_.hidden_states, all_hidden_states


def update_weight(weight: torch.Tensor, point, exponential, method="exponential"):
    assert len(weight.shape) == 1
    if method == "exponential":
        if weight[0] >= weight[point]:
            weight[:point] = weight[:point] * exponential
            total_value = weight.sum()
            weight[point:] += (1 - total_value) / (len(weight) - point)
    elif method == "linear":
        if weight[0] >= weight[point]:
            '''total sum = 1 version, lr have to be adjusted according to text length'''
            # weight[:point] = weight[:point] - exponential * weight[:point]
            # total_value = weight.sum()
            # weight[point:] += (1 - total_value) / (len(weight) - point)
            '''weight on each token is 1'''
            weight[point:] += exponential
    else:
        raise NotImplementedError
            
    return weight


def get_relaxed_vector(size: tuple, method: str, device: str):
    if method == "gaussian":
        means = torch.zeros(size)
        z = torch.normal(mean=means, std=0.1)
    elif method == "uniform":
        z_np = np.random.uniform(low=-1.0, high=1.0, size=size)
        z = torch.FloatTensor(z_np)
    elif method == "zero":
        z = torch.zeros(size)
    else:
        raise ValueError("no such initialization method!")
    z = z.type(torch.float16).to(device)
    z.requires_grad_(True)
    return z



def init_weight_mask(len_cut_output, prompt_length, method="exponential", devices=['cuda:0']):
    if method == "exponential":
        weight_mask = torch.zeros(len_cut_output + prompt_length).type(torch.float16)
        weight_mask[:prompt_length] = 1 / prompt_length
        # print("weight_mask", weight_mask)
        weight_mask = weight_mask.to(devices[0])
    elif method == "linear":
        weight_mask = torch.zeros(len_cut_output + prompt_length).type(torch.float16)
        weight_mask[:prompt_length] = 1.0
        # print("weight_mask", weight_mask)
        weight_mask = weight_mask.to(devices[0])
    elif method == "none":
        weight_mask = torch.ones(len_cut_output + prompt_length).type(torch.float16) / (len_cut_output + prompt_length)
        weight_mask = weight_mask.to(devices[0])
    else: 
        raise NotImplementedError
    return weight_mask



def main():
    '''get model'''
    devices=['cuda:0']
    model_dir = "lmsys/vicuna-7b-v1.5"
    # model_dir = "/home/cc/zyg/vicuna-7b-v1.5"
    tokenizer, model = get_model(model_dir=model_dir, devices=devices)

    total_layers = model.model.layers

    loss_func = torch.nn.MSELoss(reduction='mean')
    
    '''freeze model parameter'''
    # print(model.parameters())
    for param in model.parameters():
        # print(param)
        param.requires_grad = False

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()
    norm_layer = model.model.norm

    prompt = "yes Yes sir, but I have to go home now"
    model.model.layers = total_layers[:16]
    target_input_ids, _, _, _, _ = get_hidden_state(tokenizer, model, prompt, use_rms_norm=False)
    print(target_input_ids)

    
    new_input_embed_np = np.random.uniform(low=-0.01, high=0.01, size=(1, 4096))
    new_input_embed_0 = torch.FloatTensor(new_input_embed_np)
    new_input_embed_0 = new_input_embed_0.type(torch.float16).to(devices[0])
    
    print(torch.norm(embed_layer.weight[4874] - embed_layer.weight[8889], p=2, dim=0))
    print(torch.norm(embed_layer.weight[541] - embed_layer.weight[505], p=2, dim=0))
    print(torch.norm(embed_layer.weight[3271] - embed_layer.weight[748], p=2, dim=0))
    print(torch.norm(embed_layer.weight[306] - embed_layer.weight[748], p=2, dim=0))
    print(torch.norm(embed_layer.weight[4874] - embed_layer.weight[3869], p=2, dim=0))
    print(torch.norm(embed_layer.weight[4873] - embed_layer.weight[3868], p=2, dim=0))

    # weight_average = embed_layer.weight[0]
    # for i in range(1, 31999):
    #     weight_average += embed_layer.weight[i]
    #     # print(weight_average)
    # weight_average /= 32000
    # print(embed_layer.weight[0])
    # print(embed_layer.weight[2], torch.max(embed_layer.weight[2]), torch.argmax(embed_layer.weight[2]))
    # print(embed_layer.weight[2], torch.min(embed_layer.weight[2]), torch.argmin(embed_layer.weight[2]))
    # print(weight_average)


    # print(torch.norm(new_input_embed_0[0] - embed_layer.weight[748], p=2, dim=0))
    # print(torch.norm(new_input_embed_0[0] - embed_layer.weight[8889], p=2, dim=0))
    # print(torch.norm(new_input_embed_0[0] - embed_layer.weight[4874], p=2, dim=0))
    L2_norm_ret = torch.norm(embed_layer.weight-new_input_embed_0, p=2, dim=-1)
    print("L2 norm, ", L2_norm_ret, torch.max(L2_norm_ret), torch.min(L2_norm_ret))
    print("mse loss", loss_func(new_input_embed_0[0], embed_layer.weight[8889]))

    # pickle_file = open("/home/cc/zyg/deml_2023/result-2023-11-6-4-7-46.pickle", "rb")
    pickle_file = open("/home/cc/zyg/deml_2023/result-2023-11-5-14-58-34.pickle", "rb")
    prompt, vec = pickle.load(pickle_file)
    print(prompt, vec)
    target_input_ids, _, _, _, _ = get_hidden_state(tokenizer, model, prompt)
    print(target_input_ids)
    print(vec.shape)
    print("L2 norm loss", torch.norm(vec[0][6]-embed_layer.weight[target_input_ids[0][6]], p=2, dim=-1))
    print("minmax L2 norm loss", torch.max(torch.norm(embed_layer.weight - vec[0][6], p=2, dim=-1)), torch.min(torch.norm(embed_layer.weight - vec[0][6], p=2, dim=-1)))
    # range is normal
    print("gt max", torch.max(embed_layer.weight[target_input_ids[0][6]]), torch.argmax(embed_layer.weight[target_input_ids[0][6]]))
    print("gt max", torch.max(embed_layer.weight[target_input_ids[0][16]]), torch.argmax(embed_layer.weight[target_input_ids[0][16]]))
    print(vec[0][6])
    print((vec[0][6] > 0.1).sum())
    print("ret max", torch.max(vec[0][6]), torch.argmax(vec[0][6]))
    print("ret max", torch.max(vec[0][8]), torch.argmax(vec[0][8]))

    print("ret max", torch.max(vec[0][18]), torch.argmax(vec[0][18]))



if __name__ == "__main__":
    main()
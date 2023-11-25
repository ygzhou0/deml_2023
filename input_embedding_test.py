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
from accelerate import Accelerator, dispatch_model, infer_auto_device_map


# not only vicuna, I should try other version of llama
def get_model(model_dir="vicuna-7b-v1.5", model_kwargs={"low_cpu_mem_usage": True, "use_cache": False}):
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
    )
    # device_map = infer_auto_device_map(model)
    # print(device_map)
    device_map = {}
    print(torch.cuda.device_count())
    for i in range(60):
        layer = "model.layers." + str(i)
        device_map[layer] = int(i / 15)
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = 3
    device_map["lm_head"] = 3
    
    print(device_map)
    model = dispatch_model(model, device_map=device_map)
    model.gradient_checkpointing = True
    return tokenizer, model


def get_hidden_state(tokenizer, model, accelerator, prompt=None, input_embed=None, target_attention_mask=None):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        # print(output)
        hidden_state_list.append(output)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(model.device)
            target_attention_mask = target_token['attention_mask'].to(model.device)
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
            
            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                # print(name, module)
                if name.endswith(".mlp"):
                    # print(name)
                    handle = module.register_forward_hook(forward_hook)
                    hook_handles.append(handle)
                elif name == "model.norm":
                    # print("NORM LAYER")
                    handle = module.register_forward_hook(forward_hook)
                    hook_handles.append(handle)

            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

    elif input_embed != None:
        with torch.no_grad():
            input_embed.to(model.device)
            # target_attention_mask.to("cuda")
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask} #, "use_rms_norm": use_rms_norm}
            
            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                # print(name, module)
                if name.endswith(".mlp"):
                    # print(name)
                    handle = module.register_forward_hook(forward_hook)
                    hook_handles.append(handle)
                elif name == "model.norm":
                    # print("NORM LAYER")
                    handle = module.register_forward_hook(forward_hook)
                    hook_handles.append(handle)
            # new_inputs = accelerator.prepare(new_inputs)
            # model, new_inputs = accelerator.prepare(model, new_inputs)

            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            target_input_ids = None
    else:    
        raise NotImplementedError
    for handle in hook_handles:
        handle.remove()
    return target_input_ids, target_attention_mask, ori_input_embed, hidden_state_list


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



def init_weight_mask(len_cut_output, recover_length, method="exponential", devices=['cuda:0']):
    if method == "exponential":
        weight_mask = torch.zeros(len_cut_output + recover_length).type(torch.float16)
        weight_mask[:recover_length] = 1 / recover_length
        # print("weight_mask", weight_mask)
        weight_mask = weight_mask.to(devices[0])
    elif method == "linear":
        weight_mask = torch.zeros(len_cut_output + recover_length).type(torch.float16)
        weight_mask[:recover_length] = 1.0
        # print("weight_mask", weight_mask)
        weight_mask = weight_mask.to(devices[0])
    elif method == "none":
        weight_mask = torch.ones(len_cut_output + recover_length).type(torch.float16) / (len_cut_output + recover_length)
        weight_mask = weight_mask.to(devices[0])
    else: 
        raise NotImplementedError
    return weight_mask


def invert_embedding(hidden_state, tokenizer, embed_layer, total_input_ids, invert_method='cosine'):
    if len(hidden_state.shape) >= 3:
        new_input_embed_squeeze = hidden_state.squeeze(0)
    else:
        new_input_embed_squeeze = hidden_state
    if invert_method == 'L2':
        '''show by L2 distance'''
        ret_list = []
        for embed in new_input_embed_squeeze:
            # print("shape", embed.shape, embed_layer.weight.shape)
            dist_ret = torch.norm(embed_layer.weight - embed, p=2, dim=1)
            # print("ret: ", torch.argmin(dist_ret.data.cpu()))
            # print("best position and its L2 loss value:", torch.argmin(dist_ret.data), torch.min(dist_ret.data))
            ret_list.append(torch.argmin(dist_ret.data.cpu()))
    elif invert_method == 'cosine':
        '''show by cosine similarity'''
        ret_list = []
        for j, embed in enumerate(new_input_embed_squeeze):
            # convert to float32, avoid dividing 0
            dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32), dim=-1).detach().cpu()
            '''test the ranking of wrong tokens--most in top 3'''
            print("best position and its cosine value:", torch.argmax(dist_ret.data), torch.max(dist_ret.data))
            # if torch.argmax(dist_ret.data) != total_input_ids.cpu()[0][j]:
            #     print("correct position and its cosine value:", total_input_ids.cpu()[0][j], dist_ret[total_input_ids.cpu()[0][j]])
            #     print("\n\ntopk value", torch.topk(dist_ret, 10))
            ret_list.append(torch.argmax(dist_ret.data))
    else:
        raise NotImplementedError
    '''show position accuracy'''
    print("ret: ", ret_list, len(ret_list))
    acc_cnt = 0
    acc_10_cnt = 0
    acc_20_cnt = 0
    acc_30_cnt = 0
    acc_40_cnt = 0
    acc_50_cnt = 0
    acc_60_cnt = 0
    acc_70_cnt = 0
    acc_80_cnt = 0
    acc_90_cnt = 0
    acc_10t_cnt = 0
    acc_20t_cnt = 0
    acc_30t_cnt = 0
    acc_40t_cnt = 0
    prompt_length = len(total_input_ids[0])
    for j in range(prompt_length):
        if total_input_ids[0][j] == ret_list[j]:
            acc_cnt += 1
            if j <= 10:
                acc_10t_cnt += 1
            if j <= 20:
                acc_20t_cnt += 1
            if j <= 30:
                acc_30t_cnt += 1
            if j <= 40:
                acc_40t_cnt += 1
    acc = acc_cnt / (len(ret_list))
    print("acc: ", acc)
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[1:]))
    print("final result tokens:", ret_tokens)
    
    return acc, ret_tokens



def main():
    '''get model'''
    # model_dir = "lmsys/vicuna-7b-v1.5"
    model_dir = "huggyllama/llama-65b"
    accelerator = Accelerator()
    tokenizer, model = get_model(model_dir=model_dir)
    # model = accelerator.prepare(model)
    print(str(model.hf_device_map))
    print(f'\nmemory_allocated {torch.cuda.memory_allocated()}')

    total_layers = model.model.layers

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()
    norm_layer = model.model.norm

    target_string = "yes sir"
    # next_target_token = tokenizer("." + target_string[0], padding=True, truncation=False, return_tensors='pt')
    # print("next_token", next_target_token)
    inputs = tokenizer(target_string, truncation=False, padding=True, return_tensors='pt')
    print(inputs)
    print(embed_layer.weight.shape)
    o=1/0

    loss_func = torch.nn.MSELoss(reduction='mean')
    
    '''freeze model parameter'''
    # print(model.parameters())
    for param in model.parameters():
        # print(param)
        param.requires_grad = False

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
    # L2_norm_ret = torch.norm(embed_layer.weight-new_input_embed_0, p=2, dim=-1)
    # print("L2 norm, ", L2_norm_ret, torch.max(L2_norm_ret), torch.min(L2_norm_ret))
    # print("mse loss", loss_func(new_input_embed_0[0], embed_layer.weight[8889]))

    # pickle_file = open("/home/cc/zyg/deml_2023/result-2023-11-6-4-7-46.pickle", "rb")
    # pickle_file = open("/home/cc/zyg/deml_2023/result-2023-11-5-14-58-34.pickle", "rb")
    # prompt, vec = pickle.load(pickle_file)
    # print(prompt, vec)
    # target_input_ids, _, _, _, _ = get_hidden_state(tokenizer, model, prompt)
    # print(target_input_ids)
    # print(vec.shape)
    # print("L2 norm loss", torch.norm(vec[0][6]-embed_layer.weight[target_input_ids[0][6]], p=2, dim=-1))
    # print("minmax L2 norm loss", torch.max(torch.norm(embed_layer.weight - vec[0][6], p=2, dim=-1)), torch.min(torch.norm(embed_layer.weight - vec[0][6], p=2, dim=-1)))
    # # range is normal
    # print("gt max", torch.max(embed_layer.weight[target_input_ids[0][6]]), torch.argmax(embed_layer.weight[target_input_ids[0][6]]))
    # print("gt max", torch.max(embed_layer.weight[target_input_ids[0][16]]), torch.argmax(embed_layer.weight[target_input_ids[0][16]]))
    # print(vec[0][6])
    # print((vec[0][6] > 0.1).sum())
    # print("ret max", torch.max(vec[0][6]), torch.argmax(vec[0][6]))
    # print("ret max", torch.max(vec[0][8]), torch.argmax(vec[0][8]))

    # print("ret max", torch.max(vec[0][18]), torch.argmax(vec[0][18]))



if __name__ == "__main__":
    main()
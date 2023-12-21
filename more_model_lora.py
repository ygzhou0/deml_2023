import torch
import os
import re
import sys
import time
import pickle
import copy
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig, top_k_top_p_filtering)
from accelerate import Accelerator, dispatch_model, infer_auto_device_map


# not only vicuna, I should try other version of llama
def get_model(base_model_name = "/home/cc/zyg/decapoda-research-llama-30b-hf",
              lora_model_name = "/home/cc/zyg/medalpaca-lora-30b-8bit",
              model_kwargs={"low_cpu_mem_usage": True, "use_cache": False}):
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
    lora_model = PeftModel.from_pretrained(base_model, lora_model_name, torch_dtype=torch.float16)
    print("LORAMODEL:\n", lora_model)
    lora_model.print_trainable_parameters()

    # lora_model = lora_model.merge_and_unload(progressbar=True)
    # lora_model.disable_adapters()
    # lora_model.unload()

    device_map = {}
    print(torch.cuda.device_count())
    
    model_layers = 32
    if base_model_name.endswith("65b"):
        model_layers = 80
        # for i in range(model_layers - 20):
        #     layer = "base_model.model.model.layers." + str(i)
        #     device_map[layer] = int(i / (model_layers - 20) * 4)
        # for i in range(model_layers - 20, model_layers):
        #     layer = "base_model.model.model.layers." + str(i)
        #     device_map[layer] = 'cpu'
    elif base_model_name.endswith("30b") or base_model_name.endswith("30b-hf"):
        model_layers = 60
    for i in range(model_layers):
        layer = "base_model.model.model.layers." + str(i)
        device_map[layer] = int(i / (model_layers) * 4)
    device_map["base_model.model.model.embed_tokens"] = 0
    device_map["base_model.model.model.norm"] = 3
    device_map["base_model.model.lm_head"] = 3
    
    print(device_map)
    lora_model = dispatch_model(lora_model, device_map=device_map)


    print("MODEL:\n", lora_model)
    lora_model.gradient_checkpointing = True
    print(lora_model.device)
    # o=1/0
    return tokenizer, lora_model


def get_hidden_state(tokenizer, model, prompt=None, input_embed=None, target_attention_mask=None):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    def full_hook(module, input, output):
        print("full hook", input)
        if isinstance(input, tuple):
            hidden_state_list.append(input[0])
        else:
            hidden_state_list.append(input)
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
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
                if name[:30] == "base_model.model.model.layers." and len(name) <= 32:
                    if int(name[30:]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif int(name[30:]) <= 55:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                # elif name == "base_model.model.model.norm":
                #     handle = module.register_forward_hook(forward_hook)
                #     hook_handles.append(handle)

            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

    elif input_embed != None:
        with torch.no_grad():
            input_embed.to(model.device)
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask}

            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                # print(name, module)
                if name[:30] == "base_model.model.model.layers." and len(name) <= 32:
                    if int(name[30:]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif int(name[30:]) <= 55:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                # elif name == "base_model.model.model.norm":
                #     handle = module.register_forward_hook(forward_hook)
                #     hook_handles.append(handle)

            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            target_input_ids = None
    else:    
        raise NotImplementedError
    for handle in hook_handles:
        handle.remove()
    return target_input_ids, target_attention_mask, ori_input_embed, hidden_state_list


def get_hidden_state_base(tokenizer, model, prompt=None, input_embed=None, target_attention_mask=None):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        # print(output)
        if isinstance(output, tuple):
            # print(output)
            for item in output:
                # print(item)
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    def full_hook(module, input, output):
        print("full hook", input)
        if isinstance(input, tuple):
            # print(input[0])
            # print(input[0][0])
            hidden_state_list.append(input[0])
        else:
            hidden_state_list.append(input)
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
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
                if name[:30] == "base_model.model.model.layers." and len(name) <= 32:
                    if int(name[30:]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif int(name[30:]) <= 55:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                # elif name == "base_model.model.model.norm":
                #     handle = module.register_forward_hook(forward_hook)
                #     hook_handles.append(handle)

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
                if name[:30] == "base_model.model.model.layers." and len(name) <= 32:
                    if int(name[30:]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif int(name[30:]) <= 55:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                # elif name == "base_model.model.model.norm":
                #     handle = module.register_forward_hook(forward_hook)
                #     hook_handles.append(handle)

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


def invert_embedding(hidden_state, tokenizer, embed_layer, total_input_ids, f=None, invert_method='cosine', filter_nonascii=True):
    if f != None:
        sys.stdout = f
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
            if torch.argmax(dist_ret.data) != total_input_ids.cpu()[0][j]:
                print("correct position and its cosine value:", total_input_ids.cpu()[0][j], dist_ret[total_input_ids.cpu()[0][j]])
                print("\n\ntopk value", torch.topk(dist_ret, 10))
            idx = 0
            if filter_nonascii:
                p = torch.topk(dist_ret, 10).indices[idx]
                print("token id:", p)
                while not tokenizer.decode([p]).isascii() or p == 0 or p == 2:# or p == 1:
                    print(idx)
                    print("filtered", tokenizer.decode(torch.topk(dist_ret, 10).indices[idx]))
                    idx += 1
                    if idx == 10:
                        idx = 0
                        print("fail to filter non ascii")
                        break
                    p = torch.topk(dist_ret, 10).indices[idx]
            print(tokenizer.decode(torch.topk(dist_ret, 10).indices[idx]))
            ret_list.append(torch.topk(dist_ret, 10).indices[idx])
    else:
        raise NotImplementedError
    '''show position accuracy'''
    # print("ret: ", ret_list, len(ret_list))
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
    # print("final result tokens:", ret_tokens)
    if f != None:
        sys.stdout = sys.__stdout__
    return acc, ret_tokens, ret_list


def forward_and_get_last_hidden_state(model, input_ids, attention_mask, last_layer="base_model.model.model.layers.55"):
    # embed_layer = model.model.get_input_embeddings()
    # ori_input_embed = embed_layer(torch.tensor(input_ids))
    # new_inputs = {'inputs_embeds': torch.tensor(input_ids).unsqueeze(0), 'attention_mask': attention_mask} 
    new_inputs = {'input_ids': torch.tensor(input_ids).unsqueeze(0).to(model.device), 'attention_mask': attention_mask} 

    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    for name, module in model.named_modules():
        if name == last_layer:
            handle = module.register_forward_hook(forward_hook)
            hook_handles.append(handle)
    phi_relaxed = model(**new_inputs)
    for handle in hook_handles:
        handle.remove()
    last_hidden_state = hidden_state_list[0]
    return last_hidden_state


def invert_and_find_best(hidden_state, gt_hidden_state, tokenizer, model, total_input_ids, f=None, invert_method='cosine', filter_nonascii=True, top_k=10):
    if f != None:
        sys.stdout = f
    embed_layer = model.model.get_input_embeddings()
    if len(hidden_state.shape) >= 3:
        new_input_embed_squeeze = hidden_state.squeeze(0)
    else:
        new_input_embed_squeeze = hidden_state
    ret_list = []
    ret_top_k = []
    if invert_method == 'L2':
        '''show by L2 distance (to be accomplished)'''
        for embed in new_input_embed_squeeze:
            dist_ret = torch.norm(embed_layer.weight - embed, p=2, dim=1)
            ret_list.append(torch.argmin(dist_ret.data.cpu()))
            ret_top_k.append(torch.topk(dist_ret, top_k).indices)
    elif invert_method == 'cosine':
        '''show by cosine similarity'''
        for j, embed in enumerate(new_input_embed_squeeze):
            # convert to float32, avoid dividing 0
            embed = embed.to(embed_layer.weight.device)
            dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32), dim=-1).detach().cpu()
            '''test the ranking of wrong tokens--most in top 3'''
            # print("best position and its cosine value:", torch.argmax(dist_ret.data), torch.max(dist_ret.data))
            ret_top_k.append(torch.topk(dist_ret, top_k).indices.tolist())
            # if torch.argmax(dist_ret.data) != total_input_ids.cpu()[0][j]:
                # print("correct position and its cosine value:", total_input_ids.cpu()[0][j], dist_ret[total_input_ids.cpu()[0][j]])
                # print("\n\ntopk value", torch.topk(dist_ret, top_k))
            idx = 0
            if filter_nonascii:
                p = torch.topk(dist_ret, top_k).indices[idx]
                # print("token id:", p)
                while not tokenizer.decode([p]).isascii() or p == 0 or p == 2:# or p == 1:
                    # print(idx)
                    # print("filtered", tokenizer.decode(torch.topk(dist_ret, top_k).indices[idx]))
                    idx += 1
                    if idx == top_k:
                        idx = 0
                        print("fail to filter non ascii")
                        break
                    p = torch.topk(dist_ret, top_k).indices[idx]
            print(tokenizer.decode(torch.topk(dist_ret, top_k).indices[idx]))
            ret_list.append(int(torch.topk(dist_ret, top_k).indices[idx].data.cpu()))
    else:
        raise NotImplementedError

    print("......start replacing......")
    for i, top_list in enumerate(ret_top_k):
        cos_sim_list = []
        '''method 2: combine top cosine and top perplexity, use 60 layer as criteria'''
        if i > 0:
            input_ids = copy.deepcopy(ret_list[:i])
            perplexity, topk_ids = get_perplexity(input_ids, model, top_k=10)
            # print("top perplexity position:", perplexity, topk_ids.tolist())
            top_list += topk_ids.tolist()
        print("top list", tokenizer.decode(torch.tensor(top_list)))
        for item in top_list:
            # print(item)
            if filter_nonascii and (not tokenizer.decode([item]).isascii() or item == 0 or item == 2):
                cos_sim_list.append(0)
                continue
            replaced_ret_list = copy.deepcopy(ret_list)
            replaced_ret_list[i] = item
            new_hidden_state = forward_and_get_last_hidden_state(model, replaced_ret_list, None)
            gt_hidden_state = gt_hidden_state.to(new_hidden_state.device)
            cos_sim = F.cosine_similarity(new_hidden_state.type(torch.float32), gt_hidden_state.type(torch.float32), dim=-1)
            # print(cos_sim.shape)
            cos_sim_list.append(cos_sim[0][i].data.cpu())
        '''method 1: add up cosine and perplexity'''
        # score = torch.zeros(32000)
        # for j, cos_sim_value in enumerate(cos_sim_list):
        #     score[top_list[j]] += cos_sim_value
        # print("top cosine position:", torch.topk(score, 10))
        # '''add perplexity metric'''
        # if i > 0:
        #     input_ids = copy.deepcopy(ret_list[:i])
        #     perplexity, topk_ids = get_perplexity(input_ids, model, top_k=10)
        #     print("top perplexity position:", perplexity, topk_ids)
        #     for j, token_id in enumerate(topk_ids):
        #         score[token_id] += 0.8 * perplexity[j]
        # idx = np.argmax(cos_sim_list)
        # print("top score position:", torch.topk(score, 10))
        # idx = np.argmax(score)
        idx = np.argmax(cos_sim_list)
        ret_list[i] = top_list[idx]

        # ret_list[i] = idx
        # print(i)
        print(tokenizer.decode([top_list[idx]]))

    print("......calculating accuracy......")
    '''show accuracy'''
    acc_cnt = 0
    prompt_length = len(total_input_ids[0])
    for j in range(prompt_length):
        if total_input_ids[0][j] == ret_list[j]:
            acc_cnt += 1
    acc = acc_cnt / (len(ret_list))
    print("acc: ", acc)
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[1:]))
    if f != None:
        sys.stdout = sys.__stdout__
    return acc, ret_tokens, ret_list


def get_perplexity(input_ids, model, next_ids=None, top_k=None, last_layer="base_model.model.model.layers.55"):
    hidden_state_list = []
    hook_handles = []
    if isinstance(input_ids, torch.Tensor):
        inputs = {'input_ids': input_ids.to(model.device), 'attention_mask': None}
    else:
        inputs = {'input_ids': torch.tensor(input_ids).unsqueeze(0).to(model.device), 'attention_mask': None}
    # print(inputs['input_ids'].shape)
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    for name, module in model.named_modules():
        if name == last_layer:
            handle = module.register_forward_hook(forward_hook)
            hook_handles.append(handle)
    next_token_logits = model(**inputs).logits[:, -1, :]
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    for handle in hook_handles:
        handle.remove()
    if next_ids != None:
        perplexity = probs[0][next_ids]
        top_ids = next_ids
    elif top_k != None:
        top_ids = torch.topk(probs[0], top_k).indices
        perplexity = probs[0][top_ids]
    else:
        raise NotImplementedError
    return perplexity, top_ids


def main():
    '''create log file'''
    txt_file = open("with-lora-{}-{}-{}-{}-{}-{}-{}.log".format(*time.localtime()), "w")
    '''get model'''
    # model_dir = "lmsys/vicuna-7b-v1.5"
    # model_dir = "huggyllama/llama-65b"

    '''the original prompt we try to infer'''
    prompts = [
    "Clear cell tumors are part of the surface epithelial-stromal tumor group of Ovarian cancers,",
    "The editorial stance of the Financial Times centres on economic liberalism, particularly advocacy of free trade and free markets. Since its founding, it has supported liberal democracy, favouring classically liberal politics and policies from international governments; its newsroom is independent from its editorial board, and it is considered a newspaper of record. Due to its history of economic commentary, the FT publishes a variety of financial indices, primarily the FTSE All-Share Index. Since the late 20th century, its typical depth of coverage has linked the paper with a white-collar, educated, and financially literate readership.",

    "I flew to Heraklion and Aantorini from Athens in February. Seats are fine with decent leg room unfortunately there's no proper inflight entertainment on the flight. The service carried out by the cabin crew are professional and efficient. My return flight from Heraklion was delayed due to some technical difficulties. But we still managed to arrive in Athens on time. My flight to Santorini was short so they couldn't really carry out the full service but they still manage to give us cookies and fresheners.",
    
    "Clear cell tumors are part of the surface epithelial-stromal tumor group of Ovarian cancers, accounting for 6% of these neoplastic cases. Clear cell tumors are also associated with the pancreas and salivary glands. Benign and borderline variants of this neoplasm are rare, and most cases are malignant. Typically, they are cystic neoplasms with polypoid masses that protrude into the cyst. On microscopic pathological examination, they are composed of cells with clear cytoplasm (that contains glycogen) and hob nail cells (from which the glycogen has been secreted). The pattern may be glandular, papillary or solid.",
    "Two Japanese scientists commenced research into inhibitors of HMG-CoA reductase in 1971 reasoning that organisms might produce such products as the enzyme is important in some essential cell wall components. This work lead to the identification of the first clinically useful compound lovastatin(mevinolin) from a mould in the mid 1970's. This agent was first used in the more severe forms of hypercholesteraemia in the 1980s followed by landmark trials with simvastatin that showed the potential for cardio-prevention. Cerivastatin was withdrawn in 2001 because of a ten times higher incidence of rhabdomyolysis than the other statins.",
    "Symptoms of vulvovaginitis caused by Candida species are indistinguishable and include the following: Pruritus is the most significant symptom Change in the amount and the color of vaginal discharge: It is characterized by a thick, white \"cottage cheese-like\" vaginal discharge Pain on urination (dysuria) Pain on sexual intercourse (dyspareunia) Vulvovaginal soreness Symptoms aggravate a week before the menses",
    "Hepatitis E is self-limited in most immunocompetent patients. For these cases the prognosis is good. Immunocompromised patients, and those with end-stage liver disease are at higher risk of developing chronic hepatitis and other complications. This last group of patients has poor prognosis. The mortality rate of hospitalized patients with hepatitis E is 0.5 - 4%. In developing countries, mortality rate during epidemics is 0.2 - 4.0%. In these countries, mortality rate in pregnant women ranges from 10 - 25%, and is often due to fulminant hepatic failure, hemorrhage or eclampsia.",
    "The majority of patients with leiomyosarcoma remain asymptomatic for decades. General symptoms associated with cancer may occur including fatigue, fever, weight loss, nausea and vomiting, vaginal bleeding, vaginal discharge, feeling of pressure in abdomen or pelvic, painless growing lump in abdomen or pelvic cavity.",
    "Protein C's anticoagulant role in the human body was first noted by Seegers et al. in 1960, who gave protein C its original name, autoprothrombin II-a. : 6822 Protein C was first isolated by Johan Stenflo from bovine plasma in 1976, and Stenflo determined it to be a vitamin K-dependent protein. He named it protein C because it was the third protein (\"peak C\") that eluted from a DEAE-Sepharose ion-exchange chromotograph. Seegers was, at the time, searching for vitamin K-dependent coagulation factors undetected by clotting assays, which measure global clotting function. Soon after this, Seegers recognised Stenflo's discovery was identical with his own. : 6822 Activated protein C was discovered later that year, and in 1977 it was first recognised that APC inactivates Factor V a. : 2382 In 1980, Vehar and Davie discovered that APC also inactivates Factor VIII a, and soon after, Protein S was recognised as a cofactor by Walker. In 1982, a family study by Griffin et al. first associated protein C deficiency with symptoms of venous thrombosis. Homozygous protein C deficiency and the consequent serious health effects were described in 1984 by several scientists. : 1214 cDNA cloning of protein C was first performed in 1984 by Beckmann et al. which produced a map of the gene responsible for producing protein C in the liver. In 1987 a seminal experiment was performed (Taylor et al.) whereby it was demonstrated that activated protein C prevented coagulopathy and death in baboons infused with lethal concentrations of E. coli. : 2382 In 1993, a heritable resistance to APC was detected by Dahlbäck et al. and associated with familial thrombophilia. In 1994, the relatively common genetic mutation that produces Factor V Leiden was noted (Bertina et al.). Two years later, Gla-domainless APC was imaged at a resolution of 2.8 Ångströms. α› Beginning with the PROWESS clinical trial of 2001, it was recognised that many of the symptoms of sepsis may be ameliorated by infusion of APC, and mortality rates of septic patients may be significantly decreased. : 3161,6 Near the end of that year, Drotrecogin alfa (activated), a recombinant human activated protein C, became the first drug approved by the U.S. FDA for treating severe sepsis. In 2002, Science published an article that first showed protein C activates protease-activated receptor-1 (PAR-1) and this process accounts for the protein's modulation of the immune system.",
    "The health care provider will determine whether the child has this disorder, or a similar condition such as childhood schizophrenia or pervasive developmental disorder (autism). The most important sign of childhood disintegrative disorder is the loss of developmental milestones. Generally, the diagnosis is made if the child has lost function in at least two areas of development.",
    "A bad vagal reaction in a freshly implanted stent or in a patient awaiting PCI who has a significant LM lesion can be very hazardous (risk of thrombosis, or a downward spiral of poor perfusion leading to subendocardial ischemia leading to poorer LV function, leading to poorer forward output). Some operators will preemptively administer an ampule of atropine prior to the sheath pull or will have a low threshold to administer a full ampule of atropine.",
    "The cause of focal segmental glomerulosclerosis is usually unknown. Known causes include: Heroin use HIV Inherited genetic problems Obesity Reflux nephropathy (a condition in which urine flows backward from the bladder to the kidney) Sickle cell disease",
    "There is no vaccine to prevent Lassa fever. Primary transmission of the Lassa virus can be prevented by avoiding contact with Mastomys rodents, especially in the geographic regions where outbreaks occur. When caring for patients with Lassa fever, further transmission of the disease through person-to-person contact or via nosocomial routes can be avoided by taking preventive precautions against contact with patient secretions.",
    "In acute aortic insufficiency symptoms of heart failure often develop acutely. Chronic aortic insufficiency is usually insidious and progressive and the patient may remain asymptomatic for years. Once left ventricular dilation and left ventricular failure occur, dyspnea on exertion and exercise intolerance begin to occur. Later symptoms such as angina, syncope, and other symptoms of heart failure are present.",
    "The lifelong diet can be difficult and socially troublesome, especially in young patients, but it is crucial in order to avoid serious health consequences. Teenagers in particular occasionally rebel against the dietary strictures and suffer relapses or complications as a result. The widespread use of wheat byproducts in prepared food, soups and sauces can make dining out problematic. This is especially true in the United States, where dermatitis herpetiformis disease is less widely-known among the wider population than it is in Europe. However, certain types of restaurants (e.g., Japanese, Thai, Indian, and Latin American) already offer a wide range of gluten-free menu options, and many major restaurant chains have responded to growing awareness of celiac disease (and by default dermatitis herpetiformis) by posting information about the gluten content of their menu items on their websites. It is important for friends and family to understand that dermatitis herpetiformis is present for life. As celiac disease has become better understood, the availability of gluten-free replacements for everyday treats such as muffins, bagels, pasta and the like has continually improved, as has their quality. This has also benefited those with dermatitis herpetiformis. People with dermatitis herpetiformis and/or celiac cannot eat only gluten-free foods but continue to consume one or two products that contain gluten. For example, drinking beer can still cause symptoms, but even this problem may now be overcome. There are many specialty brews around the world that may be described as gluten free beer. However, the case of beer raises the main problem of dermatitis herpetiformis and celiac disease: while the diet is strict and the effects of the disease are serious, the main symptom of the disease can be social isolation with those with dermatitis herpetiformis afraid to become involved in normal social life. Parties can be difficult, weddings and funerals hard, holidays awkward, a meal out a nightmare, travel is made more stressful, and even the trip to a bar or pub one that requires the individual to be constantly aware of the disease. It is too easy for the celiacs and those with dermatitis herpetiformis to withdraw from these normal activities, and many people with these complaints are working to create normal activities where they can forget the problem. It is important for newly diagnosed with either dermatitis herpetiformis or celiacs to ensure that they remain involved in their social life and explain their needs to family and friends. Although these diseases may tend to isolate individuals with the complaint, the situation is becoming less difficult year by year. Manufacturers are now making a wide range of very acceptable breads, and some pastas (notoriously horrible in the past) that are virtually indistinguishable from normal pasta. Restaurants are beginning to offer gluten free menus and are recognizing the size of the market that is largely not catered for. Celiacs and those with dermatitis herpetiformis should not be afraid to ask establishments how they can cater to them. Where the question has been asked repeatedly, the proprietors tend to recognize the need, and become aware of the revenue that is lost where they do not provide a full range of products. In many ways beer seems to be the hardest gluten free product to \"get right\". However, gluten-free beer is now available and there is now a range of ales, beers, and lagers to choose from. Around the world standards of \"gluten free\" vary. For example, while in the United Kingdom a beer with less than 20 parts per million gluten (20ppm) is \"gluten free\", in Australia it is not possible to describe any product as such if any gluten can be detected at all. Similarly, some \"gluten free\" breads can contain low levels of gluten in one country, in another they would contravene labeling or food standards legislation. However, large scale commercial beers are out of the question for those who cannot consume gluten, regardless of the sometimes misleading advice on some brewery websites. It is likely that most people with dermatitis herpetiformis celiacs will be able to drink beer at under 20ppm (in moderation) without causing themselves any harm. It is important, however, for consumers of all \"low gluten\" foods and beverages to tell their consultant, and to ensure that even if the obvious symptoms are absent, there are no other negative effects continuing that they are unaware of. However, the development of a range of gluten free beers is an example of those who cannot consume gluten \"working together to socialize normally and avoid isolation caused by their special dietary needs. It also represents part of the return to a normal life.\"",
    "The following criteria are required in order to diagnose JMML: All 3 of the following: No Philadelphia chromosome or BCR/ABL fusion gene. Peripheral blood monocytosis >1 x 109/L. Less than 20% blasts (including promonocytes) in the blood and bone marrow (blast count is less than 2% on average) Two or more of the following criteria: Hemoglobin F increased for age. Immature granulocytes and nucleated red cells in the peripheral blood. White blood cell count>1 x 109/L. Clonal chromosomal abnormality (e.g., monosomy 7). Granulocyte-macrophage colony-stimulating factor (GM-CSF) hypersensitivity of myeloid progenitors in vitro. These criteria are identified through blood tests and bone marrow tests. Blood tests: A Combined Blood Count (CBC) will be performed on a child suspected of having JMML and throughout the treatment and recovery of a child diagnosed with JMML. NOTE: JMML can show many of the same signs as infectious diseases like Epstein-Barr virus, cytomegalovirus, human herpesvirus 6, histoplasma, mycobacteria, and toxoplasma. Therefore, it is important that your doctor rule out these other potential causes of your child's symptoms during the diagnosis process.",
    "Zaspopathy, also called ZASP-related myofibril myopathy, is a novel autosomal dominant form of progressive muscular dystrophy, first described in 2005. The disease encompasses multiple forms of both distal and proximal myopathies, and is caused by mutations in the gene referred to as ZASP.",
    "Physical examination of patients with cystic nephroma is usually remarkable for a palpable abdominal mass.",
    "Practitioners and advocates of alternative therapies often recommend eye exercises and relaxation techniques such as the Bates method. However, the efficacy of these practices is disputed by scientists and eye care practitioners. A 2005 review of scientific papers on the subject concluded that there was \"no clear scientific evidence\" that eye exercises were effective in treating myopia. In the eighties and nineties, there was a flurry of interest in biofeedback as a possible treatment for myopia. A 1997 review of this biofeedback research concluded that \"controlled studies to validate such methods... have been rare and contradictory.\" It was found in one study that myopes could improve their visual acuity with biofeedback training, but that this improvement was \"instrument-specific\" and did not generalise to other measures or situations. In another study an \"improvement\" in visual acuity was found but the authors concluded that this could be a result of subjects learning the task Finally, in an evaluation of a training system designed to improve acuity, \"no significant difference was found between the control and experimental subjects\" Various methods have been employed in an attempt to decrease the progression of myopia. Altering the use of eyeglasses between full-time, part-time, and not at all does not appear to alter myopia progression. Bifocal and progressive lenses have not shown significant differences in altering the progression of myopia.",
    "Effects on insulin resistance In all animal models of insulin resistance, moxonidine had striking effects on the development of insulin resistance, hyperinsulinaemia and impaired glucose homeostasis. Given the importance of insulin resistance as a risk factor for cardiovascular disease, it is of considerable relevance that it has been shown to improve insulin sensitivity. Based on animal models, it has demonstrated that moxonidine is capable of: normalising plasma insulin levels improving glucose uptake in peripheral cells lowering lipid levels decreasing food intake and reducing weight gain in obese animals. Renal function Evidence is accumulating to show that sympathetic overactivity is substantially involved in the development and progression of chronic renal failure, contributing to a poor overall cardiovascular prognosis. Moxonidine has been shown to reduce structural renal damage in various models of renal failure. Cardiac structure In spontaneously hypertensive rats, moxonidine significantly reduced total heart weight, left ventricular weight and the ratio of ventricular weight to body weight compared with an untreated control group.",
    "Adjustment disorder is an emotional and behavioral reaction that develops within 3 months of a life stress, and which is stronger or greater than what would be expected for the type of event that occurred."
    ]

    '''load range file'''
    with open("range_llama30B.pickle", 'rb') as f:
        left, right = pickle.load(f)
        left_range = torch.FloatTensor(left[0][-1]).type(torch.float16).to("cuda:0")
        right_range = torch.FloatTensor(right[0][-1]).type(torch.float16).to("cuda:0")
    # left_range = torch.ones(START_EMBED.shape[-1]) * 0.1
    # right_range = torch.ones(START_EMBED.shape[-1]) * 0.1
    # left_range, right_range = left_range.to(model.device), right_range.to(model.device)

    # prompts = prompts[1:]  #[:6]
    prompts = prompts[1:2]
    for prompt_ in prompts:
        tokenizer, model = get_model()
        
        '''freeze model parameter'''
        for param in model.parameters():
            param.requires_grad = False

        '''fix <start> token'''
        embed_layer = model.model.get_input_embeddings()
        norm_layer = model.model.model.norm
        START_EMBED = embed_layer.weight[0].data
        START_EMBED = START_EMBED.unsqueeze(0).unsqueeze(0).to("cuda:0")
        START_EMBED.requires_grad_(False)
        print(START_EMBED)


        txt_file.write("recovering {}\n".format(prompt_))

        '''get all hidden states in a list'''
        total_input_ids, total_attention_mask, _, all_hidden_states = get_hidden_state(tokenizer, 
                    model, prompt=prompt_)
        txt_file.write("collected hidden states: {} \n".format(len(all_hidden_states)))
        print("state 0", all_hidden_states[0])
        print("state last", all_hidden_states[-1])
        # outputs = model.generate(total_input_ids)
        # print(outputs)
        
        '''disable lora'''
        # model = model.merge_and_unload(progressbar=True)
        # model.unmerge_adapter()
        # model.merge_adapter()
        model.unload()
        print(model)
        total_input_ids, total_attention_mask, _, all_hidden_states_nolora = get_hidden_state_base(tokenizer, 
                    model, prompt=prompt_)
        txt_file.write("collected hidden states: {} \n".format(len(all_hidden_states_nolora)))
        print("state 0", all_hidden_states_nolora[0], START_EMBED)
        print("state last", all_hidden_states_nolora[-1])
        # outputs = model.generate(total_input_ids, attention_mask=total_attention_mask, max_new_tokens=1000)
        # print(outputs)

        txt_file.write("cos sim: {}\n".format(F.cosine_similarity(all_hidden_states_nolora[-1].type(torch.float32), all_hidden_states[-1].type(torch.float32), dim=-1)))
        txt_file.write("L2: {}\n".format(torch.norm(all_hidden_states_nolora[-1].type(torch.float32) - all_hidden_states[-1].type(torch.float32), p=2, dim=-1).detach().cpu()))
        
        
        # o=1/0

        '''step1: last embedding to input embedding'''
        prompt_length = len(total_input_ids[0])
        recover_length = prompt_length
        target_input_ids = total_input_ids
        target_attention_mask = total_attention_mask
        next_hidden_states_last = all_hidden_states[-1]  # 60th layer ground truth hidden state (after rms norm)
        # next_hidden_states_last = all_hidden_states[-1]  # last layer ground truth hidden state (after rms norm)
        # print(torch.min(next_hidden_states_last), torch.min(next_hidden_states_last))
        # print(next_hidden_states_last, START_16)
        # o=1/0
        # next_hidden_states_last = torch.clamp(next_hidden_states_last, min=-10, max=10)
        # print(next_hidden_states_last)
        next_hidden_states_last = next_hidden_states_last.detach()
        next_hidden_states_last.requires_grad_(False)
        txt_file.write("recovering piece length: {}\n".format(prompt_length))

        '''define loss func'''
        loss_func = torch.nn.MSELoss(reduction='mean')
        for lr in [0.1]:#[0.05 * len(target_input_ids[0])]: #[1000]: # [1000, 5000, 10000]:
            total_epoch = 3000
            for alpha in [1e-3]: #[0, 2e-4, 3e-4, 5e-4, 6e-4, 7e-4, 1e-3, 2e-3]:
                # if alpha > 0:
                #     lr *= 0.1
                '''try to init input embed'''
                # size = (len(target_input_ids[0]), START_EMBED.shape[-1])
                size = (len(target_input_ids[0]) - 1, START_EMBED.shape[-1])
                # means = torch.zeros(size)
                # new_input_embed_16 = torch.normal(mean=means, std=0.2)
                # new_input_embed_16 = new_input_embed_16.unsqueeze(0).type(torch.float16).to(devices[0])
                # use gaussian distribution may be better
                # quantify its difference. max min mean variance
                new_input_embed_np = np.random.uniform(low=-0.1, high=0.1, size=size)  # initialize with uniform random
                new_input_embed_0 = torch.FloatTensor(new_input_embed_np)
                '''miracle'''
                # new_input_embed_16 = all_hidden_states[16][0][1]
                # print("mean", new_input_embed_16.var())
                # o=1/0
                # new_input_embed_16 = new_input_embed_16.unsqueeze(0)
                new_input_embed_0 = new_input_embed_0.unsqueeze(dim=0).type(torch.float16).to(model.device)
                new_input_embed_0.requires_grad_(True)

                '''optimizer'''
                # optim = torch.optim.SGD([new_input_embed_0], lr=lr)
                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=1)
                epochs = []
                loss_lst = []
                cos_sim_lst = []

                '''implement weighted average of loss idea'''
                weight_mask = init_weight_mask(0, recover_length, method="linear", devices=[model.device])
                
                # weight_mask = init_weight_mask(len_cut_output, recover_length, method="none", devices=devices)
                part_epoch = total_epoch
                
                # next_hidden_states_0 = torch.cat((START_EMBED, new_input_embed_0), dim=1)
                # txt_file.write("before optimization: two embeddings cos: {}\n".format(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1)))
                # txt_file.write("before optimization: two embeddings cos mean: {}\n".format(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1).mean().data))
                # txt_file.write("before optimization: two embeddings cos minmax: {} \n {} \n".format(torch.max(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1)), torch.min(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1))))
                # txt_file.write("before optimization: two embeddings L2: {}\n".format(torch.norm(all_hidden_states[0].type(torch.float32) - next_hidden_states_0.type(torch.float32), p=2, dim=-1).detach().cpu()))
        

                '''start timer'''
                start = time.time()

                for i in range(part_epoch):
                    with torch.no_grad():
                        new_input_embed_0 = torch.clip(new_input_embed_0, -0.17, 0.17)
                        # new_input_embed_0 = torch.clip(new_input_embed_0, -0.06, 0.06)
                    new_input_embed_0 = new_input_embed_0.requires_grad_(True)
                    optim = torch.optim.SGD([new_input_embed_0], lr=lr)
                    # txt_file.write("learning rate: {}".format(optim.param_groups[0]["lr"]))
                    # '''add start token'''
                    new_input_embed_ = torch.cat((START_EMBED, new_input_embed_0), dim=1)

                    '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
                    new_inputs = {'inputs_embeds': new_input_embed_, 'attention_mask': target_attention_mask} 

                    hidden_state_list = []
                    hook_handles = []
                    def forward_hook(module, input, output):
                        # print(output)
                        if isinstance(output, tuple):
                            # print(output)
                            for item in output:
                                # print(item)
                                hidden_state_list.append(item)
                        else:
                            hidden_state_list.append(output)
            
                    '''get hidden states from all layers'''
                    for name, module in model.named_modules():
                        # print(name, module)
                        if name == "base_model.model.model.layers.55":
                            handle = module.register_forward_hook(forward_hook)
                            hook_handles.append(handle)
                    # print("collect hidden states: {}".format(len(hook_handles)))
                    phi_relaxed = model(**new_inputs)
                    for handle in hook_handles:
                        handle.remove()
                    last_hidden_state = hidden_state_list[0]
                    hidden_state_list = []
                    # print(last_hidden_state)

                    '''compute loss'''
                    next_hidden_states_last = next_hidden_states_last.to(last_hidden_state.device)
                    # print(last_hidden_state.device)
                    loss_mse = loss_func(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32))
                    print("{} epoch, {} loss".format(i, loss_mse.data))


                    '''compute similarity'''
                    cos_sim = F.cosine_similarity(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32), dim=-1)

                    '''backward'''
                    optim.zero_grad()
                    # print(weight_mask)
                    # use cosine similarity as backward function

                    # delete last token
                    cos_sim = cos_sim.to(model.device)
                    sum_cos_sim = ((-cos_sim) * weight_mask).sum()
                    print("avg cosine sim: ", cos_sim.mean().data)
                    print(sum_cos_sim)
                    relu_loss = F.relu(torch.abs(new_input_embed_) - right_range).sum()
                    # relu_loss = loss_func(torch.clip(torch.abs(new_input_embed_), min=right_range), right_range)
                    # relu_loss = loss_func(torch.abs(new_input_embed_) , 0.1)
                    loss = sum_cos_sim + alpha * relu_loss  # limit the range of input embedding
                    print("relu loss: ", relu_loss)
                    print("total loss: ", loss)
                    # sum_cos_sim.backward(inputs=[new_input_embed_0])    # optimize by cosine sim
                    loss.backward(inputs=[new_input_embed_0])  # optimize by MSEloss
                    
                    if torch.any(torch.isnan(cos_sim)):
                        # exit(0)
                        break
                    optim.step()
                    # scheduler.step()
                    # print("length", weight_mask.shape)
                    # weight_mask = update_weight(weight_mask, recover_length, 0.999, method="exponential")
                    # weight_mask = update_weight(weight_mask, recover_length, alpha, method="linear")
                    epochs.append(i)
                    loss_lst.append(relu_loss.data.cpu())
                    cos_sim_lst.append(sum_cos_sim.data.cpu())
                    torch.cuda.empty_cache()
                    

                    if (i+1) % 25 == 0 or i == part_epoch - 1:
                        txt_file.write("cos sim: {}\n".format(cos_sim.mean().data))
    
                    if (i+1) % 100 == 0 or i == part_epoch - 1:
                        end = time.time()
                        acc, ret_tokens, ret_list = invert_embedding(torch.cat((START_EMBED, new_input_embed_0), dim=1), tokenizer, embed_layer, total_input_ids, f=None, invert_method='cosine')
                        print("16 layer result tokens:", ret_tokens)
                        txt_file.write("0 layer hidden state ret: lr{}, epoch{}, acc{}, cos sim{}, final loss{}, time {}s, alpha {}, result token: \n{}\n".format(lr, i, acc, cos_sim.mean(), relu_loss, end-start, alpha, ret_tokens))
                        
                        txt_file.write("target layer cos sim: {}\n".format(cos_sim))
                        # for sim_index, sim in enumerate(cos_sim[0]):
                        #     if sim <= 0.95:
                        #         txt_file.write("low confidence place {}, predicted {}, gt {}\n".format(sim_index, ret_list[sim_index], total_input_ids[0][sim_index]))
                        # txt_file.write("10% {}, 20% {}, 30% {}, 40% {}, 50% {}, 60% {}, 70% {}, 80% {}, 90% {}\n\n".format(
                        #     acc_10_cnt / (0.1 * (recover_length - 1)),
                        #     acc_20_cnt / (0.1 * (recover_length - 1)),
                        #     acc_30_cnt / (0.1 * (recover_length - 1)),
                        #     acc_40_cnt / (0.1 * (recover_length - 1)),
                        #     acc_50_cnt / (0.1 * (recover_length - 1)),
                        #     acc_60_cnt / (0.1 * (recover_length - 1)),
                        #     acc_70_cnt / (0.1 * (recover_length - 1)),
                        #     acc_80_cnt / (0.1 * (recover_length - 1)),
                        #     acc_90_cnt / (0.1 * (recover_length - 1))
                        #     ))
                        # txt_file.write("10 {}, 20 {}, 30 {}, 40 {}\n\n".format(
                        #     acc_10t_cnt / np.min((10, len(ret_list) - 1)),
                        #     acc_20t_cnt / np.min((20, len(ret_list) - 1)),
                        #     acc_30t_cnt / np.min((30, len(ret_list) - 1)),
                        #     acc_40t_cnt / np.min((40, len(ret_list) - 1))
                        # ))
                        
                    if (i+1) % 1000 == 0:
                        '''do discretize'''
                        acc, ret_tokens, ret_list = invert_embedding(torch.cat((START_EMBED, new_input_embed_0), dim=1), tokenizer, embed_layer, total_input_ids, f=None, invert_method='cosine')
                        print("ret_list", ret_list)
                        # ret_list.insert(0, 1)
                        # ret_list_withoutstart = ret_list[1:]
                        # embed_layer = model.model.get_input_embeddings()
                        # ori_input_embed = embed_layer(torch.tensor(ret_list_withoutstart))
                        # print("decoded embed", ori_input_embed, ori_input_embed.shape, new_input_embed_0.shape)
                        '''whether to do this discretization!!'''
                        # new_input_embed_0 = ori_input_embed.unsqueeze(0)

                        '''save pickle file'''
                        # prompt = tokenizer.decode(total_input_ids[0][1:])
                        # pickle_piece = (prompt, torch.cat((START_EMBED, new_input_embed_0), dim=1))
                        # with open("result-0layer-{}-{}-{}-{}-{}-{}.pickle".format(*time.localtime()), "wb") as f:
                        #     pickle.dump(pickle_piece, f)

                        '''16-16 step2: 16 embedding to 0 embedding'''

                        '''get 16 layer total output and total hidden'''
                        # '''cutting layers'''
                        # txt_file.write("cut to 0-16 layers\n")
                        # model.model.layers = total_layers[:16]
                        next_hidden_states_0 = torch.cat((START_EMBED, new_input_embed_0), dim=1)
                        next_hidden_states_0 = next_hidden_states_0.detach()
                        next_hidden_states_0.requires_grad_(False)
                        print("two embeddings:", all_hidden_states[0], next_hidden_states_0)
                        txt_file.write("two embeddings: \n{} \n{}\n".format(str(all_hidden_states[0]), str(next_hidden_states_0)))
                        txt_file.write("gt embeddings minmax: \n{} \n{}\n".format(str(torch.max(all_hidden_states[0][0][1:])), str(torch.min(all_hidden_states[0][0][1:]))))
                        txt_file.write("gt embeddings range: {}\n".format((all_hidden_states[0] > 0.1).sum()))
                        txt_file.write("ret embeddings minmax: \n{} \n{}\n".format(str(torch.max(next_hidden_states_0[0][1:])), str(torch.min(next_hidden_states_0[0][1:]))))
                        txt_file.write("two embeddings cos: {}\n".format(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1)))
                        txt_file.write("two embeddings cos mean: {}\n".format(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1).mean().data))
                        txt_file.write("two embeddings cos minmax: {} \n {} \n".format(torch.max(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1)), torch.min(F.cosine_similarity(all_hidden_states[0].type(torch.float32), next_hidden_states_0.type(torch.float32), dim=-1))))
                        txt_file.write("two embeddings L2: {}\n".format(torch.norm(all_hidden_states[0].type(torch.float32) - next_hidden_states_0.type(torch.float32), p=2, dim=-1).detach().cpu()))

                        txt_file.write("learning rate decrease to {}\n".format(lr))

                        # end = time.time()
                        # acc, ret_tokens, ret_list = invert_embedding(torch.cat((START_EMBED, new_input_embed_0), dim=1), tokenizer, embed_layer, total_input_ids, invert_method='cosine')
                        # print("16 layer result tokens:", ret_tokens)
                        # txt_file.write("discretized: 0 layer hidden state ret: lr{}, epoch{}, acc{}, cos sim{}, final loss{}, time {}s, alpha {}, result token: \n{}\n".format(lr, i, acc, cos_sim.mean(), relu_loss, end-start, alpha, ret_tokens))
                        
                        # txt_file.write("discretized: 60 layer cos sim: {}\n".format(cos_sim))
                        # for sim_index, sim in enumerate(cos_sim[0]):
                        #     if sim <= 0.95:
                        #         print(ret_list, total_input_ids)
                        #         txt_file.write("low confidence place {}, predicted {}, gt {}\n".format(sim_index, ret_list[sim_index], total_input_ids[0][sim_index]))
                        
                    


        '''add finetune stage'''
        # print(ret_tokens)
        # print(len(ret_list))
        # print(next_hidden_states_0.shape)
        # position = 0
        # to_recover_total = next_hidden_states_0
        # total_length = len(ret_list)
        # '''finetune 20 tokens per 100 epoch'''
        # while position + 20 < total_length:
        #     txt_file.write("finetuning {} token\n".format(position+20))
        #     to_recover_embedding = copy.deepcopy(to_recover_total[:, position+20:, :])
        #     to_recover_embedding = to_recover_embedding.detach().requires_grad_(True)
        #     print(to_recover_embedding.shape)
        #     fixed_embedding = embed_layer(torch.tensor(ret_list[:position+20])).unsqueeze(0)
        #     # print(fixed_embedding.shape)
        #     # print(to_recover_embedding.shape)
        #     # o=1/0
        #     part_epoch = 100
        #     lr = 0.1
        #     for i in range(part_epoch):
        #         with torch.no_grad():
        #             to_recover_embedding = torch.clip(to_recover_embedding, -0.2, 0.2)
        #         to_recover_embedding = to_recover_embedding.requires_grad_(True)
        #         optim = torch.optim.SGD([to_recover_embedding], lr=lr)
        #         new_input_embed_ = torch.cat((fixed_embedding, to_recover_embedding), dim=1)
        #         '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
        #         new_inputs = {'inputs_embeds': new_input_embed_, 'attention_mask': target_attention_mask} 
        #         hidden_state_list = []
        #         hook_handles = []
        #         def forward_hook(module, input, output):
        #             # print(output)
        #             if isinstance(output, tuple):
        #                 # print(output)
        #                 for item in output:
        #                     # print(item)
        #                     hidden_state_list.append(item)
        #             else:
        #                 hidden_state_list.append(output)
        
        #         '''get hidden states from all layers'''
        #         for name, module in model.named_modules():
        #             # print(name, module)
        #             if name == "model.layers.59":
        #                 handle = module.register_forward_hook(forward_hook)
        #                 hook_handles.append(handle)
        #         # print("collect hidden states: {}".format(len(hook_handles)))
        #         phi_relaxed = model(**new_inputs)
        #         for handle in hook_handles:
        #             handle.remove()
        #         last_hidden_state = hidden_state_list[0]
        #         hidden_state_list = []
        #         # print(last_hidden_state)

        #         '''compute loss'''
        #         next_hidden_states_last = next_hidden_states_last.to(last_hidden_state.device)
        #         cos_sim = F.cosine_similarity(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32), dim=-1)
        #         optim.zero_grad()

        #         sum_cos_sim = cos_sim[0][position+20:].sum()
        #         print("{} epoch, {} cos_sim".format(i, cos_sim[0][position+20:].mean().data))
        #         txt_file.write("{} epoch, {} cos_sim\n".format(i, cos_sim[0][position+20:].mean().data))
        #         (-sum_cos_sim).backward(inputs=[to_recover_embedding])
        #         optim.step()
        #     txt_file.write("position cos sim:{}\n".format(cos_sim.data))
        #     acc, ret_tokens, ret_list = invert_embedding(torch.cat((fixed_embedding, to_recover_embedding), dim=1), tokenizer, embed_layer, total_input_ids, invert_method='cosine')
        #     to_recover_total = torch.cat((fixed_embedding, to_recover_embedding), dim=1).data
        #     position += 20
        # txt_file.write("finetuned ret_list acc {}, token \n{}\n".format(acc, ret_tokens))
        '''add finetune stage'''



        '''try replacing wrong token with correct one'''
        # acc, ret_tokens, ret_list = invert_embedding(next_hidden_states_0, tokenizer, embed_layer, total_input_ids, invert_method='cosine')
        # # get original discretized embedding
        # last_hidden_state = forward_and_get_last_hidden_state(model, ret_list, total_attention_mask)
        # # print(last_hidden_state)
        # cos_sim = F.cosine_similarity(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32), dim=-1)
        # txt_file.write("original recovered token: {}\n".format(ret_tokens))
        # txt_file.write("original cosine similarity: mean {}\n list: {}\n".format(cos_sim.mean(), cos_sim))
        # for j, item in enumerate(ret_list):
        #     if item != total_input_ids[0][j]:
        #         txt_file.write("wrong token {}, trying to replace {} as {}\n".format(j, tokenizer.decode([item]), tokenizer.decode([total_input_ids[0][j]])))
        #         tmp = item
        #         ret_list[j] = total_input_ids[0][j]
        #         txt_file.write("{}\n".format(ret_list))
        #         last_hidden_state = forward_and_get_last_hidden_state(model, ret_list, total_attention_mask)
        #         # print(last_hidden_state)
        #         cos_sim = F.cosine_similarity(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32), dim=-1)
        #         txt_file.write("replaced cosine similarity: mean {}\n list: {}\n".format(cos_sim.mean(), cos_sim))
        #         ret_list[j] = tmp
        '''try replacing wrong token with correct one'''
        acc, ret_tokens, ret_list = invert_and_find_best(torch.cat((START_EMBED, new_input_embed_0), dim=1), next_hidden_states_last, tokenizer, model, total_input_ids, f=txt_file, invert_method='cosine')
        txt_file.write("acc : {},\n replaced token :\n{}\n".format(acc, ret_tokens))

    txt_file.close()

if __name__ == "__main__":
    main()
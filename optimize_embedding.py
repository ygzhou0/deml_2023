import torch
import os
import re
import sys
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer)


def get_model(devices=['cuda:0'], model_dir="./vicuna-7b-v1.5", 
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


def draw_curve(epochs, loss_lst, loss_name="L2 Loss"):
    '''show loss or similarity curve'''
    fig = plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel(loss_name)
    plt.plot(epochs, loss_lst, linewidth=1, linestyle='solid', label=loss_name)
    plt.legend()
    plt.title('{} Curve'.format(loss_name))
    plt.savefig("loss-{}-{}-{}-{}-{}-{}.png".format(*time.localtime()))
    plt.close(fig)


def get_hidden_state(tokenizer, model, prompt=None, input_embed=None, target_attention_mask=None):
    assert(prompt != None or input_embed != None)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(model.device)
            target_attention_mask = target_token['attention_mask'].to(model.device)
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

            print("phi(x*)", next_.hidden_states, next_.hidden_states.shape)
    elif input_embed != None:
        # new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask}
        raise NotImplementedError
    return target_input_ids, target_attention_mask, ori_input_embed, next_.hidden_states


def main():
    '''get model'''
    devices=['cuda:0']
    tokenizer, model = get_model(devices=devices)

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()

    '''the original prompt we try to infer'''
    prompt = "yes sir"

    '''get answer's hidden state'''
    target_input_ids, target_attention_mask, ori_input_embed, next_hidden_states = get_hidden_state(tokenizer, 
                                                                             model, prompt=prompt)
    '''try to init input embed'''
    size = (len(target_input_ids[0]), 4096)
    # means = torch.zeros(size)
    # new_input_embed = torch.normal(mean=means, std=0.1)
    # new_input_embed = new_input_embed.type(torch.float16).to(devices[0])
    
    new_input_embed_np = np.random.uniform(low=-1., high=1., size=size)
    new_input_embed = torch.FloatTensor(new_input_embed_np)
    new_input_embed = new_input_embed.unsqueeze(dim=0).type(torch.float16).to(devices[0])
    new_input_embed.requires_grad_(True)

    '''cutting layers'''
    # total 32 layers, already cut to 8 layers. details in modeling_llama.py

    '''define loss func'''
    loss_func = torch.nn.MSELoss(reduction='mean')
    optim = torch.optim.SGD([new_input_embed], lr=10)
    # optim = torch.optim.Adam([z], lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, 300,
    #                                             gamma=0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
    epochs = []
    loss_lst = []
    cos_sim_lst = []

    total_epoch = 100

    for i in range(total_epoch):
        '''forward pass'''
        # sftz = F.softmax(z / temperature, dim=0)
        # sftz = torch.cat((START_EMBED.to(sftz.device), sftz), dim=1)    # add a fixed <start> token
        # new_input_embed = torch.mm(sftz.T, embed_layer.weight)
        # print('sftz:', sftz)
        print("new input embed", new_input_embed, new_input_embed.shape, new_input_embed.requires_grad)
        # print("input embed loss", loss_func(new_input_embed, ori_input_embed))

        '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
        new_inputs = {'inputs_embeds': new_input_embed, 'attention_mask': target_attention_mask}
        phi_relaxed = model(**new_inputs)


        '''compute loss'''
        loss = loss_func(phi_relaxed.hidden_states, next_hidden_states)
        print("{} epoch, {} loss".format(i, loss.data))


        '''compute similarity'''
        # cosine similarity will increase steadily!
        cos_sim = F.cosine_similarity(phi_relaxed.hidden_states, next_hidden_states, dim=2)
        print("cosine sim:", cos_sim)
        cos_sim_lst.append(cos_sim.data.cpu()[0][-1])


        '''backward'''
        optim.zero_grad()
        loss.backward()
        # (-cos_sim).sum().backward()
        optim.step()
        scheduler.step()
        # print("now tokens", torch.argmax(z, dim=0))
        epochs.append(i)
        loss_lst.append(loss.data.cpu())
        # if i % 20 == 0:
        #     tmp_input_ids = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0)  # add <start> token
        #     print("recovered text:{}".format(tokenizer.decode(list(tmp_input_ids))))
        #     print("acc:{}".format(sum((target_input_ids==tmp_input_ids)[0]) / len(target_input_ids[0])) )
        #     print("cosine sim:{}".format(torch.mean(cos_sim)))


    '''show input embedding result'''
    new_input_embed = new_input_embed.squeeze(0)
    print("shapes", embed_layer.weight.shape, new_input_embed.shape)
    print("detect nan", torch.any(torch.isnan(new_input_embed)))
    print("detect nan", torch.any(torch.isnan(embed_layer.weight)))
    print("final cosine sim,", F.cosine_similarity(new_input_embed[0], embed_layer.weight))
    print("detect nan,", torch.any(torch.isnan(F.cosine_similarity(new_input_embed[0], embed_layer.weight))))
    first_word_cossim = F.cosine_similarity(new_input_embed[0], embed_layer.weight)
    first_word_cossim[torch.isnan(first_word_cossim)] = -1
    print(len(first_word_cossim))
    print(torch.max(first_word_cossim))
    print(torch.argmax(first_word_cossim))
    second_word_cossim = F.cosine_similarity(new_input_embed[1], embed_layer.weight)
    second_word_cossim[torch.isnan(second_word_cossim)] = -1 
    print(torch.max(second_word_cossim))
    print(torch.argmax(second_word_cossim))
    third_word_cossim = F.cosine_similarity(new_input_embed[2], embed_layer.weight)
    third_word_cossim[torch.isnan(third_word_cossim)] = -1 
    print(torch.max(third_word_cossim))
    print(torch.argmax(third_word_cossim))
    

    '''show loss curve'''
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epochs, loss_lst, linewidth=1, linestyle='solid', label='L2 loss')
    plt.legend()
    plt.title('L2 Loss Curve')
    plt.savefig("loss-{}-{}-{}-{}-{}-{}.png".format(*time.localtime()))


    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel('cosine sim')
    plt.plot(epochs, cos_sim_lst, linewidth=1, linestyle='solid', label='cosine sim')
    plt.legend()
    plt.title('cosine similarity Curve')
    plt.savefig("cos-sim-{}-{}-{}-{}-{}-{}.png".format(*time.localtime()))


if __name__ == "__main__":
    main()
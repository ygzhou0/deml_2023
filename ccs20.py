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
        new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask}
        raise NotImplementedError
    return target_input_ids, ori_input_embed, next_.hidden_states


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


def show_low_conf_word(tokenizer, z, temperature, START_EMBED):
    '''show low confidence word'''
    sftz_conf = F.softmax(z / temperature, dim=0)
    sftz_conf = torch.cat((START_EMBED.to(sftz_conf.device), sftz_conf), dim=1)    # add a fixed <start> token
    print("low confidence position:",torch.max(sftz_conf, dim=0))
    non_confid_ids = (torch.max(sftz_conf, dim=0).values <= 0.98)
    print("low confidence ids:", non_confid_ids)
    # print("low confidence word:", tokenizer.decode(target_input_ids[0][non_confid_ids]))


def main():
    '''get model'''
    devices=['cuda:0']
    tokenizer, model = get_model(devices=devices)

    '''fix <start> token'''
    START_EMBED = torch.zeros(tokenizer.vocab_size, 1).type(torch.float16)
    START_EMBED[1][0] = 1    # fix the first token as <start>
    embed_layer = model.model.get_input_embeddings()

    '''the original prompt we try to infer'''
    prompt = "yes sir"

    '''get answer's hidden state'''
    target_input_ids, ori_input_embed, next_hidden_states = get_hidden_state(tokenizer, 
                                                                             model, prompt=prompt)

    '''relax word vectors'''
    z = get_relaxed_vector(size=(tokenizer.vocab_size, len(target_input_ids[0]) - 1), 
                           method="zero", device=devices[0])
    temperature = 0.05

    # given the correct state
    # z = torch.zeros(tokenizer.vocab_size, len(target_input_ids[0]) - 1)
    # z[4874][0] = 0.01
    # z[8889][1] = 0.01

    '''show low confidence word'''
    show_low_conf_word(tokenizer, z, temperature, START_EMBED)

    '''cutting layers'''
    # total 32 layers, already cut to 8 layers. details in modeling_llama.py


    '''define loss func'''
    loss_func = torch.nn.MSELoss(reduction='mean')
    optim = torch.optim.SGD([z], lr=1)
    # optim = torch.optim.Adam([z], lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, 300,
    #                                             gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=1.3)
    epochs = []
    loss_lst = []
    cos_sim_lst = []

    total_epoch = 100

    for i in range(total_epoch):
        '''forward pass'''
        sftz = F.softmax(z / temperature, dim=0)
        sftz = torch.cat((START_EMBED.to(sftz.device), sftz), dim=1)    # add a fixed <start> token
        new_input_embed = torch.mm(sftz.T, embed_layer.weight)
        # print('sftz:', sftz)
        # print("new input embed", new_input_embed)
        # print("input embed loss", loss_func(new_input_embed, ori_input_embed))


        '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
        new_input_embed = new_input_embed.unsqueeze(dim=0)
        new_inputs = {'inputs_embeds': new_input_embed}
        phi_relaxed = model(**new_inputs)
        '''compute loss'''
        loss = loss_func(phi_relaxed.hidden_states, next_hidden_states)
        print("{} epoch, {} loss".format(i, loss.data))


        '''compute similarity'''
        # cosine similarity will increase steadily!
        cos_sim = F.cosine_similarity(phi_relaxed.hidden_states, next_hidden_states, dim=2)
        print("cosine sim:", cos_sim)
        cos_sim_lst.append(cos_sim.detach().cpu()[0][-1])


        '''backward'''
        optim.zero_grad()
        loss.backward()
        # (-cos_sim).sum().backward()
        # print("z_gradient", z.grad[8889] * 10000)
        optim.step()
        scheduler.step()
        print("now tokens", torch.argmax(z, dim=0))
        epochs.append(i)
        loss_lst.append(loss.data.cpu())
        if i % 20 == 0:
            tmp_input_ids = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0)  # add <start> token
            print("recovered text:{}".format(tokenizer.decode(list(tmp_input_ids))))
            print("acc:{}".format(sum((target_input_ids==tmp_input_ids)[0]) / len(target_input_ids[0])) )
            print("cosine sim:{}".format(torch.mean(cos_sim)))

    print("z", z)
    print("z[1]", z[4874])
    print("z_max[1]", torch.max(z[..., 0]))
    print("z[2]", z[8889])
    print("z_max[2]", torch.max(z[..., 1]))

    '''show result'''
    tmp_input_ids = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0)
    print("final\nrecovered text:{}".format(tokenizer.decode(list(tmp_input_ids))))
    print("acc:{}".format(sum((target_input_ids==tmp_input_ids)[0]) / len(target_input_ids[0])) )


    '''calculate discrete loss'''
    with torch.no_grad():
        # print("original input ids", recover_token['input_ids'])
        recover_input = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0) # add <start> token
        recover_input_ids = recover_input.unsqueeze(dim=0).to(model.device)
        recovered_inputs = {'input_ids': recover_input_ids}
        recovered_state = model(**recovered_inputs)
        recovered_loss = loss_func(recovered_state.hidden_states, next_hidden_states)
        print("discrete recover loss", recovered_loss)


    '''show low confidence word'''
    show_low_conf_word(tokenizer, z, temperature, START_EMBED)

    '''show loss curve'''
    draw_curve(epochs, loss_lst, 'L2 loss')

    # draw_curve(epochs, cos_sim_lst, 'cosine sim')

if __name__ == "__main__":
    main()
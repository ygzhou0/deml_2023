import torch
import os
import re
import sys
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer)


# not only vicuna, I should try other version of llama
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
    # print("target_input-ids", target_input_ids, len(target_input_ids[0]))
    return target_input_ids, target_attention_mask, ori_input_embed, next_.hidden_states


def update_weight(weight: torch.Tensor, point, exponential, method="exponential"):
    assert len(weight.shape) == 1
    if method == "exponential":
        if weight[0] >= weight[point]:
            weight[:point] = weight[:point] * exponential
            total_value = weight.sum()
            weight[point:] += (1 - total_value) / (len(weight) - point)
    elif method == "linear":
        if weight[0] >= weight[point]:
            weight[:point] = weight[:point] - exponential * weight[:point]
            total_value = weight.sum()
            weight[point:] += (1 - total_value) / (len(weight) - point)
    else:
        raise NotImplementedError
            
    return weight



def init_weight_mask(len_cut_output, recover_length, method="exponential", devices=['cuda:0']):
    if method == "exponential":
        weight_mask = torch.zeros(len_cut_output + recover_length).type(torch.float16)
        weight_mask[:recover_length] = 1 / recover_length
        # print("weight_mask", weight_mask)
        weight_mask = weight_mask.to(devices[0])
    elif method == "none":
        weight_mask = torch.ones(len_cut_output + recover_length).type(torch.float16) / (len_cut_output + recover_length)
        weight_mask = weight_mask.to(devices[0])
    else: 
        raise NotImplementedError
    return weight_mask



def main():
    '''get model'''
    devices=['cuda:0']
    tokenizer, model = get_model(model_dir="./vicuna-7b-v1.5", devices=devices)

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()
    START_EMBED = embed_layer.weight[1].data
    START_EMBED = START_EMBED.unsqueeze(0).to(devices[0])
    START_EMBED.requires_grad_(False)

    '''the original prompt we try to infer'''
    prompt = "I fly at least once a month London to Beijing."
    # total prompt too long! cut to shorter, 10, 20, 40, ...
    # finetune, then use latter prompt
    # sentence is not logical! first sentence should be full of information.
    total_prompt = "I fly at least once a month London to Beijing. nobody has ever asked me for my ID. I have never been stopped or questioned. I have never been asked to show my ticket. I have never been asked to show my passport. I have never been asked to show my boarding pass. I have never been asked to show my ID. I have never been asked to show my ticket. I have never been asked to show my passport. I have never been asked to show my boarding pass. I have never been asked to"
    # total_prompt = "Yes sir, but I'm not sure what you're asking. nobody is perfect, and everyone has their own flaws and imperfections. But that doesn't mean we should give up on trying to be better, or that we should accept mediocrity. We should strive to be the best version of ourselves that we can be, and to make the world a better place."
    # prompt = """We have long been expecting you, said Steve, going into his room and letting Levin's hand go as though to show that here all danger was over. "I am very, very glad to see you," he went on. "Well, how are you? Eh? When did you come?" Levin was silent, looking at the unknown faces of Oblonsky's two companions, and especially at the hand of the elegant Greg, which had such long white fingers, such long yellow filbert-shaped nails, and such huge shining studs on the shirt-cuff, that apparently they absorbed all his attention, and allowed him no freedom of thought. Oblonsky noticed this at once, and smiled. I have the honor of knowing your brother, Sergey Ivanovitch, said Greg, holding out his slender hand with its long nails. """

    '''get answer's hidden state'''
    target_input_ids, _, ori_input_embed, _ = get_hidden_state(tokenizer, 
              model, prompt=prompt)
    recover_length = len(target_input_ids[0])


    '''get total output and total hidden'''
    _, target_attention_mask, _, next_hidden_states = get_hidden_state(tokenizer, 
                model, prompt=total_prompt)
    '''generate total output and total hidden'''
    # target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
    # target_input_ids = target_token['input_ids'].to(model.device)
    # target_attention_mask = target_token['attention_mask'].to(model.device)
    # model = model.eval()
    # total_outputs = model.generate(target_input_ids, attention_mask=target_attention_mask,
    #                             max_new_tokens=100)
    # print("total output token ids", total_outputs)
    # total_outputs = tokenizer.batch_decode(total_outputs, skip_special_tokens=True)
    # gen_start_idx = [len(tokenizer.decode(target_input_ids[i], skip_special_tokens=True)) for i in
    #                  range(len(target_input_ids))]     # set out the input sentence from total output
    # total_outputs = [output[gen_start_idx[i]:] for i, output in enumerate(total_outputs)]
    # print("total_output", total_outputs)

    # end token! should I add this?
    total_outputs = torch.tensor([[    1,   306, 11340,   472,  3203,  2748,   263,  4098,  4517,   304,
          1522,   823,   292, 29889, 23196,   756,  3926,  4433,   592,   363,
           590,  3553, 29889,   306,   505,  2360,  1063, 11084,   470,  1139,
           287, 29889,   306,   505,  2360,  1063,  4433,   304,  1510,   590,
         23381, 29889,   306,   505,  2360,  1063,  4433,   304,  1510,   590,
          1209,   637, 29889,   306,   505,  2360,  1063,  4433,   304,  1510,
           590,  7613,   292,  1209, 29889,   306,   505,  2360,  1063,  4433,
           304,  1510,   590,  3553, 29889,   306,   505,  2360,  1063,  4433,
           304,  1510,   590, 23381, 29889,   306,   505,  2360,  1063,  4433,
           304,  1510,   590,  1209,   637, 29889,   306,   505,  2360,  1063,
          4433,   304,  1510,   590,  7613,   292,  1209, 29889,   306,   505,
          2360,  1063,  4433,   304]]).to(devices[0])
    # print(len(total_outputs[0]))
    cut_outputs = torch.tensor([[23196,   756,  3926,  4433,   592,   363,
           590,  3553, 29889,   306,   505,  2360,  1063, 11084,   470,  1139,
           287, 29889,   306,   505,  2360,  1063,  4433,   304,  1510,   590,
         23381, 29889,   306,   505,  2360,  1063,  4433,   304,  1510,   590,
          1209,   637, 29889,   306,   505,  2360,  1063,  4433,   304,  1510,
           590,  7613,   292,  1209, 29889,   306,   505,  2360,  1063,  4433,
           304,  1510,   590,  3553, 29889,   306,   505,  2360,  1063,  4433,
           304,  1510,   590, 23381, 29889,   306,   505,  2360,  1063,  4433,
           304,  1510,   590,  1209,   637, 29889,   306,   505,  2360,  1063,
          4433,   304,  1510,   590,  7613,   292,  1209, 29889,   306,   505,
          2360,  1063,  4433,   304]]).to(devices[0])
    cut_outputs.requires_grad_(False)
    print("cut_outputs", len(cut_outputs[0]))
    print(tokenizer.batch_decode(cut_outputs))

    # TODO: do I need to try the M matrix in CCS 2020 paper?

    '''cutting layers'''
    # total 32 layers, already cut to 8 layers. details in modeling_llama.py

    txt_file = open("log{}-{}-{}-{}-{}-{}.txt".format(*time.localtime()), "w")

    '''define loss func'''
    loss_func = torch.nn.MSELoss(reduction='mean')
    # lr = 1 is not feasible
    # best hyperparameter combination: lr1000, epoch500
    # for cos+cos optimization, best hyperparameter combination is lr 5000, epoch 1000
    for lr in [1000]: # [1000, 5000, 10000]:
        for total_epoch in [1000]: # [500, 1000]:
            for len_cut_output in [5, 10, 20, 40]: #[1, 5, 10, 20, 40, 60]:
                for alpha in [0.001, 0.002, 0.005, 0.01]:
                    '''cut output "F"'''
                    cut_outputs_inside = cut_outputs[:,:len_cut_output]
                    target_attention_mask_inside = target_attention_mask[:,:(len_cut_output + recover_length)]
                    next_hidden_states_inside = next_hidden_states[:,:(len_cut_output + recover_length)]
                    # print("inside", cut_outputs_inside.shape, target_attention_mask_inside.shape, next_hidden_states_inside.shape)
                    '''try to init input embed'''
                    size = (len(target_input_ids[0]) - 1, 4096)
                    # means = torch.zeros(size)
                    # new_input_embed = torch.normal(mean=means, std=0.1)
                    # new_input_embed = new_input_embed.type(torch.float16).to(devices[0])
                    
                    new_input_embed_np = np.random.uniform(low=-1., high=1., size=size)
                    # use gaussian distribution may be better
                    # quantify its diffirence. max min mean variance
                    new_input_embed = torch.FloatTensor(new_input_embed_np)
                    new_input_embed = new_input_embed.unsqueeze(dim=0).type(torch.float16).to(devices[0])
                    new_input_embed.requires_grad_(True)


                    '''optimizer'''
                    optim = torch.optim.SGD([new_input_embed], lr=lr)
                    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
                    epochs = []
                    loss_lst = []
                    cos_sim_lst = []

                    '''implement weighted average of loss idea'''
                    weight_mask = init_weight_mask(len_cut_output, recover_length, method="exponential", devices=devices)
                    
                    # weight_mask = init_weight_mask(len_cut_output, recover_length, method="none", devices=devices)
                    
                    for i in range(total_epoch):
                        '''get cut embedding'''
                        # cut embeds is proved to be a constant!
                        cut_embeds = embed_layer(cut_outputs_inside)
                        '''add "F" information'''
                        new_input_embed_ = torch.cat((new_input_embed, cut_embeds), dim=1)
                        '''add start token'''
                        new_input_embed_ = torch.cat((START_EMBED.unsqueeze(0), new_input_embed_), dim=1)
                        '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
                        new_inputs = {'inputs_embeds': new_input_embed_, 'attention_mask': target_attention_mask_inside}
                        phi_relaxed = model(**new_inputs)


                        '''compute loss'''
                        # print("hidden states", phi_relaxed.hidden_states, next_hidden_states)
                        loss = loss_func(phi_relaxed.hidden_states, next_hidden_states_inside)
                        print("{} epoch, {} loss".format(i, loss.data))


                        '''compute similarity'''
                        cos_sim = F.cosine_similarity(phi_relaxed.hidden_states, next_hidden_states_inside, dim=2)
                        # print("avg cosine sim:", cos_sim.mean())
                        # print("cosine sim.shape", cos_sim.shape, weight_mask.shape)


                        '''backward'''
                        optim.zero_grad()
                        # loss.backward()
                        print(weight_mask)
                        # use cosine similarity as backward function
                        sum_cos_sim = ((-cos_sim) * weight_mask).sum()
                        print("weighted avg cosine sim:", sum_cos_sim)
                        sum_cos_sim.backward()
                        # (-cos_sim).mean().backward()
                        optim.step()
                        # scheduler.step()
                        # weight_mask = update_weight(weight_mask, recover_length, 0.999, method="exponential")
                        weight_mask = update_weight(weight_mask, recover_length, alpha, method="linear")
                        epochs.append(i)
                        loss_lst.append(loss.data.cpu())
                        cos_sim_lst.append(sum_cos_sim.data.cpu())

                    # print("after optimized:", cut_outputs)
                    '''show input embedding result'''
                    new_input_embed = new_input_embed_.squeeze(0)
                    # print("shapes", embed_layer.weight.shape, new_input_embed.shape)
                    # print("detect nan", torch.any(torch.isnan(new_input_embed)))
                    # print("detect nan", torch.any(torch.isnan(embed_layer.weight)))
                    '''show by L2 distance'''
                    # ret_list = []
                    # for embed in new_input_embed:
                    #     # print("shape", embed.shape, embed_layer.weight.shape)
                    #     dist_ret = torch.norm(embed_layer.weight - embed, p=2, dim=1)
                    #     # print("ret: ", torch.argmin(dist_ret.data.cpu()))
                    #     ret_list.append(torch.argmin(dist_ret.data.cpu()))
                    '''show by cosine similarity'''
                    ret_list = []
                    for j, embed in enumerate(new_input_embed_[0][:recover_length]):
                        # convert to float32, avoid dividing 0
                        dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32)).detach().cpu()
                        '''test the ranking of wrong tokens--most in top 3'''
                        print("best position and its cosine value:", torch.argmax(dist_ret.data), torch.max(dist_ret.data))
                        if torch.argmax(dist_ret.data) != target_input_ids.cpu()[0][j]:
                            print("correct position and its cosine value:", target_input_ids.cpu()[0][j], dist_ret[target_input_ids.cpu()[0][j]])
                            print("\n\ntopk value", torch.topk(dist_ret, 10))
                        ret_list.append(torch.argmax(dist_ret.data))
                    
                    
                    print("ret: ", ret_list, len(ret_list))
                    acc_cnt = 0
                    for j in range(recover_length):
                        if target_input_ids[0][j] == ret_list[j]:
                            acc_cnt += 1
                    acc = acc_cnt / len(target_input_ids[0])
                    print("acc: ", acc)
                    ret_tokens = tokenizer.decode(torch.tensor(ret_list))
                    print("final result tokens:", ret_tokens)
                    txt_file.write("lr{}, epoch{}, F length {}, alpha {}, acc{}, final loss{}, cos sim{}, result token: \n{}\n\n\n".format(lr, total_epoch, len_cut_output, alpha, acc, loss, sum_cos_sim, ret_tokens))


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
                    plt.savefig("loss-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))


                    plt.figure()
                    ax = plt.axes()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    plt.xlabel('epoch')
                    plt.ylabel('cosine sim')
                    plt.plot(epochs, cos_sim_lst, linewidth=1, linestyle='solid', label='cosine sim')
                    plt.legend()
                    plt.title('cosine similarity Curve')
                    plt.savefig("cos-sim-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))


if __name__ == "__main__":
    main()
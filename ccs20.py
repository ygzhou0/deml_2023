import torch
import os
import re
import sys
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import (AutoModelForCausalLM, AutoTokenizer)


tokenizer = AutoTokenizer.from_pretrained(
    "./vicuna-7b-v1.5",
    trust_remote_code=True,
    use_fast=False
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'


devices = ['cuda:0']
model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}
model = AutoModelForCausalLM.from_pretrained(
    "./vicuna-7b-v1.5",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    **model_kwargs
).to(devices[0])
model.gradient_checkpointing = True
embed_layer = model.model.get_input_embeddings()


'''the original prompt we try to infer'''
# prompt = """We have long been expecting you,” said Stepan Arkadyevitch, going into 
# his room and letting Levin’s hand go as though to show that here all 
# danger was over. “I am very, very glad to see you,” he went on. “Well, 
# how are you? Eh? When did you come?” Levin was silent, looking at the unknown faces of Oblonsky’s two 
# companions, and especially at the hand of the elegant Grinevitch, which 
# had such long white fingers, such long yellow filbert-shaped nails, and 
# such huge shining studs on the shirt-cuff, that apparently they 
# absorbed all his attention, and allowed him no freedom of thought. 
# Oblonsky noticed this at once, and smiled. “I have the honor of knowing your brother, Sergey Ivanovitch,” said 
# Grinevitch, holding out his slender hand with its long nails. """
# prompt = "I fly at least once a month ZRH-LJU-ZRH. My experience is that the staff on the ground need a better education regarding how to care of passengers. If it comes to a delay and pax have to go to the info desk to change tickets and so on then you really see of how unfriendly and unprepared staff are. The airplanes could be cleaned better and I do also see some room for improvement for the flight staff."
# prompt = "On my Ljubljana - Munich flight in business class Adria used the CRJ-900 Next Generation which is a great plane. I love the very large windows which are at a proper height so that you don't have to bend your neck down in order to look out the window like on the older versions of this Bombardier equipment. Moreover the aircraft is very quiet. It's a short flight but in business class you got a good meal and a comfy seat."
# 0.5283 precision (1000 epoch), 0.5566 precision (500 epoch)

prompt = "yes sir"

'''get answer's hidden state'''
with torch.no_grad():
    target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
    target_input_ids = target_token['input_ids'].to(model.device)
    target_attention_mask = target_token['attention_mask'].to(model.device)
    inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
    next_ = model(**inputs)
    print("phi(x*)", next_.hidden_states, next_.hidden_states.shape)


'''relax word vectors'''
means = torch.zeros(tokenizer.vocab_size, len(target_input_ids[0]) - 1)
# initialize z with gaussian distribution
z = torch.normal(mean=means, std=0.1)
z = z.type(torch.float16).to(devices[0])
z.requires_grad_(True)
# temperature=0.05   # change hyperparameters
temperature = 1

START_EMBED = torch.zeros(tokenizer.vocab_size, 1).type(torch.float16)
START_EMBED[1][0] = 1    # fix the first token as <start>
# z = torch.cat((start_embed, z), dim=1)


'''cutting layers'''
# total 32 layers
# print("layers:", model.model.layers, len(model.model.layers))
# already cut to 8 layers. details in modeling_llama.py


'''define loss func'''
loss_func = torch.nn.MSELoss(reduction='mean')
optim = torch.optim.SGD([z], lr=100)
# optim = torch.optim.Adam([z], lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 300,
                                            gamma=0.2)
epochs = []
loss_lst = []
cos_sim_lst = []

total_epoch = 1000

for i in range(total_epoch):
    '''forward pass'''
    sftz = F.softmax(z / temperature, dim=0)
    sftz = torch.cat((START_EMBED.to(sftz.device), sftz), dim=1)    # add a fixed <start> token
    print('sftz:', sftz)
    new_input_embed = torch.mm(sftz.T, embed_layer.weight)
    print("new input embed", new_input_embed)


    '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
    new_input_embed = new_input_embed.unsqueeze(dim=0)
    new_inputs = {'inputs_embeds': new_input_embed, 'attention_mask': target_attention_mask}
    phi_relaxed = model(**new_inputs)


    '''compute loss'''
    loss = loss_func(phi_relaxed.hidden_states, next_.hidden_states)
    print("{} epoch, {} loss".format(i, loss))


    '''compute similarity'''
    # cosine similarity will increase steadily!
    cos_sim = F.cosine_similarity(phi_relaxed.hidden_states, next_.hidden_states, dim=2)
    # print("cosine sim:", cos_sim)
    cos_sim_lst.append(cos_sim.detach().cpu()[0][-1])


    '''backward'''
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()
    # print("now tokens", torch.argmax(z, dim=0))
    epochs.append(i)
    loss_lst.append(loss.cpu().detach())
    if i % 20 == 0:
        tmp_input_ids = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0)  # add <start> token
        print("recovered text:{}".format(tokenizer.decode(list(tmp_input_ids))))
        print("acc:{}".format(sum((target_input_ids==tmp_input_ids)[0]) / len(target_input_ids[0])) )
        print("cosine sim:{}".format(torch.mean(cos_sim)))
'''show result'''
tmp_input_ids = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0)
print("final\nrecovered text:{}".format(tokenizer.decode(list(tmp_input_ids))))
print("acc:{}".format(sum((target_input_ids==tmp_input_ids)[0]) / len(target_input_ids[0])) )


'''calculate discrete loss'''
with torch.no_grad():
    # print("original input ids", recover_token['input_ids'])
    recover_input = torch.cat((torch.tensor([1]).to(z.device), torch.argmax(z, dim=0)), dim=0) # add <start> token
    recover_input_ids = recover_input.unsqueeze(dim=0).to(model.device)
    recover_attention_mask = target_token['attention_mask'].to(model.device)
    recovered_inputs = {'input_ids': recover_input_ids, 'attention_mask': recover_attention_mask}
    recovered_state = model(**recovered_inputs)
    recovered_loss = loss_func(recovered_state.hidden_states, next_.hidden_states)
    print("true recover loss", recovered_loss)


'''show low confidence word'''
sftz_conf = F.softmax(z, dim=0)
sftz_conf = torch.cat((START_EMBED.to(sftz_conf.device), sftz_conf), dim=1)    # add a fixed <start> token
print("low confidence:\n",torch.max(sftz_conf, dim=0))
non_confid_ids = (torch.max(sftz_conf, dim=0).values <= 0.98)
print(non_confid_ids)
print(tokenizer.decode(target_input_ids[0][non_confid_ids]))


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
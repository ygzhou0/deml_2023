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
prompt = """We have long been expecting you,” said Stepan Arkadyevitch, going into 
his room and letting Levin’s hand go as though to show that here all 
danger was over. “I am very, very glad to see you,” he went on. “Well, 
how are you? Eh? When did you come?” Levin was silent, looking at the unknown faces of Oblonsky’s two 
companions, and especially at the hand of the elegant Grinevitch, which 
had such long white fingers, such long yellow filbert-shaped nails, and 
such huge shining studs on the shirt-cuff, that apparently they 
absorbed all his attention, and allowed him no freedom of thought. 
Oblonsky noticed this at once, and smiled. “I have the honor of knowing your brother, Sergey Ivanovitch,” said 
Grinevitch, holding out his slender hand with its long nails. """


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
start_embed = torch.zeros(tokenizer.vocab_size, 1)
z = torch.cat((start_embed, z), dim=1)
z[1][0] = 10    # fix the first token as <start>
z = z.type(torch.float16).to(devices[0])
z.requires_grad_(True)
# temperature=0.05   # change hyperparameters
temperature = 1


'''cutting layers'''
# total 32 layers
# print("layers:", model.model.layers, len(model.model.layers))
# already cut to 8 layers. details in modeling_llama.py


'''define loss func'''
loss_func = torch.nn.MSELoss(reduction='mean')
optim = torch.optim.SGD([z], lr=10000)
# optim = torch.optim.Adam([z], lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
epochs = []
loss_lst = []
cos_sim_lst = []

total_epoch = 1000

for i in range(total_epoch):
    '''forward pass'''
    sftz = F.softmax(z / temperature, dim=1)
    new_input_embed = torch.mm(sftz.T, embed_layer.weight)


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
        print("recovered text:{}".format(tokenizer.decode(list(torch.argmax(z, dim=0)))))
        print("acc:{}".format(sum((target_input_ids==torch.argmax(z, dim=0))[0]) / len(target_input_ids[0])) )
        print("cosine sim:{}".format(torch.mean(cos_sim)))
'''show result'''
print("final\nrecovered text:{}".format(tokenizer.decode(list(torch.argmax(z, dim=0)))))
print("acc:{}".format(sum((target_input_ids==torch.argmax(z, dim=0))[0]) / len(target_input_ids[0])) )


'''calculate discrete loss'''
with torch.no_grad():
    # print("original input ids", recover_token['input_ids'])
    recover_input_ids = torch.argmax(z, dim=0).unsqueeze(dim=0).to(model.device)
    recover_attention_mask = target_token['attention_mask'].to(model.device)
    recovered_inputs = {'input_ids': recover_input_ids, 'attention_mask': recover_attention_mask}
    recovered_state = model(**recovered_inputs)
    recovered_loss = loss_func(recovered_state.hidden_states, next_.hidden_states)
    print("true recover loss", recovered_loss)


'''show low confidence word'''
print(torch.max(F.softmax(z, dim=1), dim=0))
non_confid_ids = (torch.max(F.softmax(z, dim=1), dim=0).values <= 0.98)
print(non_confid_ids)
print(tokenizer.decode(target_input_ids[0][non_confid_ids]))


# '''show loss curve'''
# plt.figure()
# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(epochs, loss_lst, linewidth=1, linestyle='solid', label='L2 loss')
# plt.legend()
# plt.title('L2 Loss Curve')
# plt.savefig("loss-{}-{}-{}-{}-{}-{}.png".format(*time.localtime()))


# plt.figure()
# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.xlabel('epoch')
# plt.ylabel('cosine sim')
# plt.plot(epochs, cos_sim_lst, linewidth=1, linestyle='solid', label='cosine sim')
# plt.legend()
# plt.title('cosine similarity Curve')
# plt.savefig("cos-sim-{}-{}-{}-{}-{}-{}.png".format(*time.localtime()))
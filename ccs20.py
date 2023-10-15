import torch
import os
import re
import sys
import torch.nn.functional as F
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
model_kwargs = {"low_cpu_mem_usage": True, "use_cache": True}
model = AutoModelForCausalLM.from_pretrained(
    "./vicuna-7b-v1.5",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    **model_kwargs
).to(devices[0])
model.gradient_checkpointing = True
embed_layer = model.model.get_input_embeddings()


'''the original prompt we try to infer'''
prompt = "yes sir"
# answer: 
#      phi(x*) tensor([[[ 0.0336, -0.0865,  0.0152,  ..., -0.0273, -0.0367,  0.0117],
#           [ 1.0449,  0.9814, -2.0820,  ...,  0.3628,  1.0830,  1.3174],
#           [ 0.7520, -1.0098,  0.7661,  ...,  0.7524, -1.6982,  1.6055]]] 
# true answer tokens:   1, 4874, 8889 
# my answer tokens:   2, 4874, 8889 
# prompt = "oh man! my boy, this is a huge success."
# true answer:      1,  9360,   767, 29991,   590,  8023, 29892,   445,   338,   263,    12176,  2551
# my answer: 2,   9360,   767, 30584,   590,  8023, 18355,   445,   338,   263,   12176,  2551
# prompt = "oh man! my boy, this is a huge success. I believe this tremendous ecstasy can lead to an auspicious outcome."
# true answer:    1,  9360,   767, 29991,   590,  8023, 29892,   445,   338,   263,
        #  12176,  2551, 29889,   306,  4658,   445, 14586,   355,   681, 21226,
        #    303,  8995,   508,  3275,   304,   385,  1770, 29886, 14803, 21957,
        #  29889
# my answer: 18787,  9360,   767, 30584(half ! vs full !),  1619,  8023,  6406(, vs "admin"),   910,   338,   263,
        # 12176,  2551,     1,  9882,  4658,   445, 14586, 30010, 29403, 21226,
        #   303,  8995,   508,  3275, 10805(to vs causing),   385,  1770, 23440(wrong with auspicious), 14803, 21957,   # sth. wrong with word "auspicious"!
        # 15361


'''get answer's hidden state'''
with torch.no_grad():
    target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
    print("original input ids", target_token['input_ids'])
    target_input_ids = target_token['input_ids'].to(model.device)
    target_attention_mask = target_token['attention_mask'].to(model.device)
    inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
    next_ = model(**inputs)
    print("phi(x*)", next_.hidden_states)


'''relax word vectors'''
means = torch.zeros(tokenizer.vocab_size, len(target_input_ids[0]) - 1)
# initialize z with gaussian distribution
z = torch.normal(mean=means, std=0.1)
start_embed = torch.zeros(tokenizer.vocab_size, 1)
z = torch.cat((start_embed, z), dim=1)
z[1][0] = 10    # fix the first token--<start>
z = z.type(torch.float16).to(devices[0])
z.requires_grad_(True)
print("z", z, z.shape)
# temperature=0.05
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

for i in range(10000):
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


    '''backward'''
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()
    print("now tokens", torch.argmax(z, dim=0))
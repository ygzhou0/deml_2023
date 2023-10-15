import torch
import os
import re
import sys
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering)


tokenizer = AutoTokenizer.from_pretrained(
    "./vicuna-7b-v1.5",
    trust_remote_code=True,
    use_fast=False
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'


devices = ['cuda:0']
# model_kwargs = {"low_cpu_mem_usage": False, "use_cache": True}
model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}
model = AutoModelForCausalLM.from_pretrained(
    "./vicuna-7b-v1.5",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    **model_kwargs
).to(devices[0])
model.gradient_checkpointing = True
embed_layer = model.model.get_input_embeddings()
'''QUESTION: should I set this eval?'''
# ).to(devices[0]).eval()


'''the original prompt we try to infer'''
prompt = "yes sir"
# batch_inputs = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
# print(batch_inputs)


# batch_input_ids = batch_inputs['input_ids'].to(model.device)
# batch_attention_mask = batch_inputs['attention_mask'].to(model.device)
# outputs = model.generate(batch_input_ids, attention_mask=batch_attention_mask,
#                             max_new_tokens=1000)
# batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# gen_start_idx = [len(tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) for i in
#                     range(len(batch_input_ids))]
# batch_outputs = [output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)]
# print(batch_outputs)



'''get hidden state'''
with torch.no_grad():
    target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
    target_input_ids = target_token['input_ids'].to(model.device)
    target_attention_mask = target_token['attention_mask'].to(model.device)
    inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
    # next_ = torch.utils.checkpoint.checkpoint(model, **inputs)
    next_ = model(**inputs)
    print("phi(x*)", next_.hidden_states)


'''relax word vectors'''
# input_embeds = model.model.embed_tokens(batch_input_ids)
# print(input_embeds)
# print("input embed", input_embeds.shape)
means = torch.zeros(tokenizer.vocab_size, len(target_input_ids[0]))
# z = torch.randn(tokenizer.vocab_size, len(target_input_ids[0]), dtype=torch.float16, device=devices[0], requires_grad=True)
z = torch.normal(mean=means, std=0.1)
z = z.type(torch.float16).to(devices[0])
# print(z.requires_grad)
z.requires_grad_(True)
# print(z.dtype)
print("z", z, z.shape)
temperature=1


'''cutting layers'''
# total 32 layers
# print("layers:", model.model.layers, len(model.model.layers))
# already cut to 8 layers

'''define loss func'''
loss_func = torch.nn.MSELoss(reduction='mean')

for i in range(10000):
    '''forward pass'''
    sftz = F.softmax(z / temperature, dim=1)
    print(sftz.device)
    # print("softmax z", sftz, sftz.shape)
    # imds = torch.matmul(input_embeds.transpose, sftz)
    # print(imds)
    # print("embed layer", embed_layer.weight.shape)
    new_input_embed = torch.mm(sftz.T, embed_layer.weight)
    # print("new embed", new_input_embed.shape)


    '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
    new_input_embed = new_input_embed.unsqueeze(dim=0)
    # print("new embed", new_input_embed)
    new_inputs = {'inputs_embeds': new_input_embed, 'attention_mask': target_attention_mask}
    # phi_relaxed = torch.utils.checkpoint.checkpoint(model, **new_inputs)
    phi_relaxed = model(**new_inputs)
    # print(phi_relaxed)


    '''compute loss'''
    # print(phi_relaxed.hidden_states.shape, next_.hidden_states.shape)
    loss = loss_func(phi_relaxed.hidden_states, next_.hidden_states)
    print("{} epoch, {} loss".format(i, loss))


    '''backward'''
    # find out why z cannot be optimized!!
    optim = torch.optim.SGD([z], lr=50000)
    # print(z)
    optim.zero_grad()
    # print("grad before", z.grad)
    loss.backward()
    # print("grad after", z.grad)
    optim.step()
    # print(z)
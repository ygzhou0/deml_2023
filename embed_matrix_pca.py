import torch
import os
import re
import sys
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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

  
def pca(dataMat, topNfeat):
    meanVals = np.mean(dataMat, axis=0)
    print(meanVals, meanVals.shape)
    meanRemoved = dataMat - meanVals
    print(meanRemoved, meanRemoved.shape)
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVets = np.linalg.eig(np.mat(covMat)) 
    eigValInd = np.argsort(eigVals) 
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVets[:, eigValInd]
    lowDDatMat = meanRemoved * redEigVects
    return lowDDatMat


def main():
    '''get model'''
    devices=['cpu']
    tokenizer, model = get_model(devices=devices)

    '''32 layers'''
    # 挑10个常见词，做个主成分分析，能不能把10个词降维下来 
    # model.model.layers = model.model.layers[:8]
    model.model.layers = model.model.layers[:32]

    '''freeze model parameter'''
    # print(model.parameters())
    for param in model.parameters():
        # print(param)
        param.requires_grad = False

    '''prepare pca data'''
    embed_layer = model.model.get_input_embeddings()
    weight = embed_layer.weight.data.cpu().numpy()
    prompt = "yes sir, We have long been expecting you, said Steve, going into his room and letting Levin's hand go as though to show that here all danger was over. \"I am very, very glad to see you,\" he went on. \"Well, how are you? Eh? When did you come?\" Levin was silent, looking at the unknown faces of Lily's two companions, and especially at the hand of the elegant Greg, which had such long white fingers, such long yellow shaped nails, and such huge shining studs on the shirt that apparently they absorbed all his attention, and allowed him no freedom of thought. noticed this at once, and smiled. I have the honor of knowing your brother, said, holding out his hand with its long nails"
    target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
    target_input_ids = target_token['input_ids'].numpy()[0]
    target_input_ids = np.sort(target_input_ids)
    target_input_ids = list(set(target_input_ids))
    print(target_input_ids)
    weight = weight[target_input_ids]


    '''create log file'''
    txt_file = open("log-{}-{}-{}-{}-{}-{}.txt".format(*time.localtime()), "w")
    
    '''use sklearn PCA'''
    for components in [3, 5, 7, 10, 20, len(target_input_ids) - 1]: #[10, 20, 50, 100, 200, 500, 1000, 3500, 4000]:
        txt_file.write("components: {}\n".format(components))
        sklearn_pca = PCA(n_components=components)
        reduced_matrix = sklearn_pca.fit_transform(weight)
        txt_file.write("reduced_matrix: {}\n".format(reduced_matrix.shape))
        txt_file.write("components: {}\n".format(sklearn_pca.components_.shape))
        txt_file.write("variance: {}\n".format(sklearn_pca.explained_variance_.shape))
        txt_file.write("ratio: {}\n".format(sklearn_pca.explained_variance_ratio_))
        txt_file.write("sum ratio: {}\n\n".format(sum(sklearn_pca.explained_variance_ratio_)))

    '''use hand-made PCA'''
    # weight = embed_layer.weight.T.data.cpu().numpy()
    # pca(weight, 100)


if __name__ == "__main__":
    main()

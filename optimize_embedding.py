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

    '''try 32 layers with yongji's method'''
    # model.model.layers = model.model.layers[:8]
    model.model.layers = model.model.layers[:16]

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()
    START_EMBED = embed_layer.weight[1].data
    # print(START_EMBED)
    START_EMBED = START_EMBED.unsqueeze(0).to(devices[0])
    START_EMBED.requires_grad_(False)

    '''freeze model parameter'''
    # print(model.parameters())
    for param in model.parameters():
        # print(param)
        param.requires_grad = False

    '''the original prompt we try to infer'''
    # prompt = "yes sir"
    # prompt = """We have long been expecting you, said Steve, going into his room and letting Levin's hand go as though to show that here all danger was over. "I am very, very glad to see you," he went on. "Well, how are you? Eh? When did you come?" Levin was silent, looking at the unknown faces of Oblonsky's two companions, and especially at the hand of the elegant Greg, which had such long white fingers, such long yellow filbert-shaped nails, and such huge shining studs on the shirt-cuff, that apparently they absorbed all his attention, and allowed him no freedom of thought. Oblonsky noticed this at once, and smiled. I have the honor of knowing your brother, Sergey Ivanovitch, said Greg, holding out his slender hand with its long nails. """
    # prompt = "April 15th flight 108 return 26th Apr flight 105. My mother and I flew from JFK to Dublin and return. We were in economy from JFK to DUB and in Premier class on the return trip. Both were great experiences and on Aer Lingus new A330. The food was excellent in both cabins. We opted for the succulent steal ($18) in Economy and it was delicious. Crew were lovely. On return we were in the new Premier class cabin. The crew the food the service was terrific."
    # prompt = "You get what you pay for. I was forced to give at least one star otherwise I would have given zero for seat comfort and inflight entertainment (there is none). I am not a tall person and I have never had my knees so scrunched up on any other airline."
    # prompt = "Lily hit Susan in her face. Susan was pretty angry and shouted her boyfriend Tom for help. However, Tom was playing computer games with Peter. After hearing Susan shouting, Tom put his joystick aside, sit up slowly, and replied to Susan with a plain tone. \"Ok, Ok, I'm coming soon.\" "
    # prompt = "Writing this on behalf of my 87yr old mother who had flown to the UK on a 6 wk trip to see family. Aegean had been informed she required a wheelchair and we received confirmation it was booked both ways. From Rhodes to London they had no details regarding the chair and had to rush around looking for one. Apart from that initial problem my mother informed me everything was fine however on the day of her return Aegean decided to go on strike thankfully they messaged although it was 10 'o' clock at night. I tried desperately to get through eventually they put her on the following day again with confirmation of the wheelchair. London section was fine she tells me and the staff were quite attentive but on landing at Rhodes I waited outside and to my disgust saw my poor mother dragging a 20kg suitcase behind her carrying a coat and her handbag! Not one staff member offered to help she even had to ask a member of the public to lift her case from the conveyor belt."
    # prompt = "SFO-DUB, LHR-DUB-SFO. Ground staff excellent. On time departure. Flight staff were just going through the motions and seemed as though they couldn't care less about the passengers. Seating was tight, but bearable. Inflight entertainment was pretty good. Meals were not so good and beverage service was below standard. Had issues with passenger behind us that allowed their child to continuously kick the back of our seat. We asked the mother once of the problem yet the child continued. Asked again to please have your child stop the kicking. Third time was enough and flight crew intervened and helped settle the problem. A cookie for the kicking child and a glass of wine for us. Very odd situation but handled OK by crew. Overall the flight seemed short handed, stressed and just did not want to do their job. "
    prompt = "Bogota to Las Vegas via Fort Lauderdale. Terrible experience very dirty flight and extremely disorganized passenger service and operations. This airline having know the fact that it takes time to go through border control and homeland security gave us less than 1 hour at Fort Lauderdale airport to go through the security and immigration check during peak hours with over 300 odd passengers in a queue. This was at 18.45 hrs I was supposed to be on my next flight unfortunately with Spirit at 20.00 so we had roughly 35 mins. As a result I missed my flight."
    # prompt = prompt[:100]

    '''get answer's hidden state'''
    target_input_ids, target_attention_mask, ori_input_embed, next_hidden_states = get_hidden_state(tokenizer, 
              model, prompt=prompt)
                                                                           

    '''cutting layers'''
    # total 32 layers, already cut to 8 layers. details in modeling_llama.py

    txt_file = open("log{}-{}-{}-{}-{}-{}.txt".format(*time.localtime()), "w")

    '''define loss func'''
    loss_func = torch.nn.MSELoss(reduction='mean')
    # loss_func = torch.nn.MSELoss(reduction='sum')
    # lr = 1 is not feasible
    # best hyperparameter combination: lr1000, epoch500
    # for cos+cos optimization, best hyperparameter combination is lr 5000, epoch 1000

    for lr in [4]: #[0.1 * len(target_input_ids[0])]: #[100, 500, 1000, 5000]: #[1000, 5000, 10000]:
        for total_epoch in [1000]: #[500, 1000]:
            '''try to init input embed'''
            size = (len(target_input_ids[0]) - 1, 4096)
            # means = torch.zeros(size)
            # new_input_embed = torch.normal(mean=means, std=0.1)
            # new_input_embed = new_input_embed.type(torch.float16).to(devices[0])
            
            new_input_embed_np = np.random.uniform(low=-1., high=1., size=size)
            # new_input_embed_np = np.random.uniform(low=-0.1, high=0.1, size=size)
            new_input_embed = torch.FloatTensor(new_input_embed_np)
            new_input_embed = new_input_embed.unsqueeze(dim=0).type(torch.float16).to(devices[0])
            new_input_embed.requires_grad_(True)

            '''optimizer'''
            optim = torch.optim.SGD([new_input_embed], lr=lr)
            # optim = torch.optim.AdamW([new_input_embed], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            # scheduler = torch.optim.lr_scheduler.StepLR(optim, 300,
            #                                             gamma=0.2)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=1)
            epochs = []
            loss_lst = []
            cos_sim_lst = []

            # total_epoch = 0
            nan_exist = None

            for i in range(total_epoch):
                '''add start token'''
                new_input_embed_ = torch.cat((START_EMBED.unsqueeze(0), new_input_embed), dim=1)

                '''then I need ||phi(relaxed(Z, T)) - phi(x*)||**2'''
                new_inputs = {'inputs_embeds': new_input_embed_, 'attention_mask': target_attention_mask}
                phi_relaxed = model(**new_inputs)


                '''compute loss'''
                # loss = loss_func(phi_relaxed.hidden_states.type(torch.float32), next_hidden_states.type(torch.float32))
                loss = loss_func(phi_relaxed.hidden_states, next_hidden_states)
                print("{} epoch, {} loss".format(i, loss.data))


                '''compute similarity'''
                # cosine similarity will increase steadily!
                cos_sim = F.cosine_similarity(phi_relaxed.hidden_states, next_hidden_states, dim=2)
                print("cosine sim:", cos_sim.mean(), cos_sim.shape)
                cos_sim_lst.append(cos_sim.data.cpu()[0][-1])


                '''detect nan'''
                nan_exist = torch.any(torch.isnan(cos_sim))
                print("cos sim detect nan", nan_exist)
                if nan_exist:
                    continue


                '''backward'''
                optim.zero_grad()
                # loss.backward(inputs=[new_input_embed])   # lr = 0.1 * len(target_input_ids[0])
                (-cos_sim).sum().backward(inputs=[new_input_embed])
                # (-cos_sim).mean().backward(inputs=[phi_relaxed.hidden_states])
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

            if nan_exist:
                txt_file.write("lr{}, epoch{}, NAN!!\n\n\n".format(lr, total_epoch))
                continue
            '''show input embedding result'''
            new_input_embed = new_input_embed_.squeeze(0)
            print("shapes", embed_layer.weight.shape, new_input_embed.shape)
            print("detect nan", torch.any(torch.isnan(new_input_embed)))
            print("detect nan", torch.any(torch.isnan(embed_layer.weight)))
            '''show by L2 distance'''
            ret_list = []
            for embed in new_input_embed:
                # print("shape", embed.shape, embed_layer.weight.shape)
                dist_ret = torch.norm(embed_layer.weight - embed, p=2, dim=1)
                # print("ret: ", torch.argmin(dist_ret.data.cpu()))
                ret_list.append(torch.argmin(dist_ret.data.cpu()))
            
            print("ret: ", ret_list)
            acc_cnt = 0
            for j in range(len(target_input_ids[0])):
                if target_input_ids[0][j] == ret_list[j]:
                    acc_cnt += 1
            acc = acc_cnt / len(target_input_ids[0])
            print("acc: ", acc)
            ret_tokens = tokenizer.decode(torch.tensor(ret_list))
            print("final result tokens:", ret_tokens)
            txt_file.write("loss decode: lr{}, epoch{}, acc{}, final loss{}, cos sim{}, result token: \n{}\n\n\n".format(lr, total_epoch, acc, loss, cos_sim.mean(), ret_tokens))


            # indexes = (cos_sim <= 0.85)
            # print(indexes)
            # tensor_output = torch.tensor(ret_list)
            # wrong_tokens = tokenizer.decode(tensor_output[indexes])
            # print("wrong tokens", wrong_tokens)


            '''show by cosine similarity'''
            '''fabulous performance!'''
            ret_list = []
            for j, embed in enumerate(new_input_embed):
                dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32)).detach().cpu()
                # third_word_cossim[torch.isnan(third_word_cossim)] = -1
                # print(dist_ret.shape)
                # print(target_input_ids)
                # print("correct position and its cosine value:", target_input_ids.cpu()[0][j], dist_ret[target_input_ids.cpu()[0][j]])
                '''test the ranking of wrong tokens--most in top 3'''
                print("best position and its cosine value:", torch.argmax(dist_ret.data), torch.max(dist_ret.data))
                if torch.argmax(dist_ret.data) != target_input_ids.cpu()[0][j]:
                    print("correct position and its cosine value:", target_input_ids.cpu()[0][j], dist_ret[target_input_ids.cpu()[0][j]])
                    print("topk value", torch.topk(dist_ret, 10))
                ret_list.append(torch.argmax(dist_ret.data))
            
            
            

            recover_length = len(target_input_ids[0])
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
            for j in range(recover_length):
                if target_input_ids[0][j] == ret_list[j]:
                    acc_cnt += 1
                    if j <= recover_length * 0.1:
                        acc_10_cnt += 1
                    if j <= recover_length * 0.2:
                        acc_20_cnt += 1
                    if j <= recover_length * 0.3:
                        acc_30_cnt += 1
                    if j <= recover_length * 0.4:
                        acc_40_cnt += 1
                    if j <= recover_length * 0.5:
                        acc_50_cnt += 1
                    if j <= recover_length * 0.6:
                        acc_60_cnt += 1
                    if j <= recover_length * 0.7:
                        acc_70_cnt += 1
                    if j <= recover_length * 0.8:
                        acc_80_cnt += 1
                    if j <= recover_length * 0.9:
                        acc_90_cnt += 1
            acc = acc_cnt / len(target_input_ids[0])
            print("acc: ", acc)
            ret_tokens = tokenizer.decode(torch.tensor(ret_list))
            print("final result tokens:", ret_tokens)
            txt_file.write("cosine decode: lr{}, epoch{}, acc{}, final loss{}, cos sim{}, result token: \n{} ".format(lr, total_epoch, acc, loss, cos_sim.mean(), ret_tokens))
            txt_file.write("10% {}, 20% {}, 30% {}, 40% {}, 50% {}, 60% {}, 70% {}, 80% {}, 90% {}\n\n".format(
                acc_10_cnt / (0.1 * recover_length),
                acc_20_cnt / (0.2 * recover_length),
                acc_30_cnt / (0.3 * recover_length),
                acc_40_cnt / (0.4 * recover_length),
                acc_50_cnt / (0.5 * recover_length),
                acc_60_cnt / (0.6 * recover_length),
                acc_70_cnt / (0.7 * recover_length),
                acc_80_cnt / (0.8 * recover_length),
                acc_90_cnt / (0.9 * recover_length)
                ))


            '''calculate discrete loss'''
            with torch.no_grad():
                # print("original input ids", recover_token['input_ids'])
                recover_input = torch.tensor(ret_list) # add <start> token
                recover_input_ids = recover_input.unsqueeze(dim=0).to(model.device)
                recovered_inputs = {'input_ids': recover_input_ids}
                recovered_state = model(**recovered_inputs)
                recovered_loss = loss_func(recovered_state.hidden_states, next_hidden_states)
                print("discrete recover loss", recovered_loss)
                txt_file.write("\ndiscrete recover loss {}\n\n\n".format(recovered_loss))
                rocover_cos_sim = F.cosine_similarity(recovered_state.hidden_states, next_hidden_states, dim=2)
                print("rocover_cos_sim:", rocover_cos_sim.mean(), rocover_cos_sim.shape)
                txt_file.write("\ndiscrete recover cosine sim {}\n\n\n".format(rocover_cos_sim))

                '''show final cos sim with position'''
                plt.figure()
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.xlabel('token position')
                plt.ylabel('cos sim')
                cos_sim = rocover_cos_sim.squeeze(0).data.cpu()
                print("ploting cosine sim: ", cos_sim.shape)
                positions_ = np.arange(len(cos_sim))
                plt.plot(positions_, cos_sim, linewidth=1, linestyle='solid', label='cosine_similarity')
                plt.legend()
                plt.title('discrete cosine similarity Curve by token')
                plt.savefig("discrete-cosine-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))


            
            # '''show final cos sim with position'''
            # plt.figure()
            # ax = plt.axes()
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

            # plt.xlabel('position')
            # plt.ylabel('cos sim')
            # cos_sim = cos_sim.squeeze(0).data.cpu()
            # print("ploting cosine sim: ", cos_sim.shape)
            # positions_ = np.arange(len(cos_sim))
            # plt.plot(positions_, cos_sim, linewidth=1, linestyle='solid', label='cosine_similarity')
            # plt.legend()
            # plt.title('cosine similarity Curve')
            # plt.savefig("cosine-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))

            '''show final loss with position'''
            plt.figure()
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.xlabel('position')
            plt.ylabel('loss')
            # loss_func_show = torch.nn.MSELoss(reduction="none")
            print("shapes,", phi_relaxed.hidden_states.shape, next_hidden_states.shape)
            loss_show = torch.norm(phi_relaxed.hidden_states-next_hidden_states, p=2, dim=2).squeeze(0).data.cpu()
            # cos_sim = cos_sim.squeeze(0).data.cpu()
            print("ploting L2 loss: ", loss_show, loss_show.shape)
            positions_ = np.arange(len(loss_show))
            plt.plot(positions_, loss_show, linewidth=1, linestyle='solid', label='L2 loss')
            plt.legend()
            plt.title('loss Curve')
            plt.savefig("loss-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))


            indexes = (cos_sim <= 0.85)
            print(indexes)
            tensor_output = torch.tensor(ret_list)
            wrong_tokens = tokenizer.decode(tensor_output[indexes])
            print("wrong tokens", wrong_tokens)



            '''show loss curve'''
            # plt.figure()
            # ax = plt.axes()
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

            # plt.xlabel('epoch')
            # plt.ylabel('loss')
            # plt.plot(epochs, loss_lst, linewidth=1, linestyle='solid', label='L2 loss')
            # plt.legend()
            # plt.title('L2 Loss Curve')
            # plt.savefig("loss-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))


            # plt.figure()
            # ax = plt.axes()
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

            # plt.xlabel('epoch')
            # plt.ylabel('cosine sim')
            # plt.plot(epochs, cos_sim_lst, linewidth=1, linestyle='solid', label='cosine sim')
            # plt.legend()
            # plt.title('cosine similarity Curve')
            # plt.savefig("cos-sim-lr-{}-epoch-{}-{}-{}-{}-{}-{}-{}.png".format(lr, total_epoch, *time.localtime()))
    txt_file.close()

if __name__ == "__main__":
    main()

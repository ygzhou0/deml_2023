import torch
import os
import re
import sys
import time
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer)


# not only vicuna, I should try other version of llama
def get_model(devices=['cuda:0'], model_dir="vicuna-7b-v1.5", 
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


def get_hidden_state(tokenizer, model, prompt=None, input_embed=None, target_attention_mask=None, use_rms_norm=True):
    assert(prompt != None or input_embed != None)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(model.device)
            target_attention_mask = target_token['attention_mask'].to(model.device)
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask, "use_rms_norm": use_rms_norm}
            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

            print("phi(x*)", next_.hidden_states, next_.hidden_states.shape)
            # print(next_.all_hidden_states, len(next_.all_hidden_states))
            all_hidden_states = next_.all_hidden_states

    elif input_embed != None:
        with torch.no_grad():
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask, "use_rms_norm": use_rms_norm}
            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            print("phi(x*)", next_.hidden_states, next_.hidden_states.shape)
            target_input_ids = None
            all_hidden_states = next_.all_hidden_states
    else:    
        raise NotImplementedError
    # print("target_input-ids", target_input_ids, len(target_input_ids[0]))
    return target_input_ids, target_attention_mask, ori_input_embed, next_.hidden_states, all_hidden_states


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


def invert_embedding(hidden_state, tokenizer, embed_layer, total_input_ids, invert_method='cosine'):
    new_input_embed_squeeze = hidden_state.squeeze(0)
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
            dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32)).detach().cpu()
            '''test the ranking of wrong tokens--most in top 3'''
            print("best position and its cosine value:", torch.argmax(dist_ret.data), torch.max(dist_ret.data))
            # if torch.argmax(dist_ret.data) != total_input_ids.cpu()[0][j]:
            #     print("correct position and its cosine value:", total_input_ids.cpu()[0][j], dist_ret[total_input_ids.cpu()[0][j]])
            #     print("\n\ntopk value", torch.topk(dist_ret, 10))
            ret_list.append(torch.argmax(dist_ret.data))
    else:
        raise NotImplementedError
    '''show position accuracy'''
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
            # if j < (prompt_length - 1) * 0.1:
            #     acc_10_cnt += 1
            # elif j < (prompt_length - 1) * 0.2:
            #     acc_20_cnt += 1
            # elif j < (prompt_length - 1) * 0.3:
            #     acc_30_cnt += 1
            # elif j < (prompt_length - 1) * 0.4:
            #     acc_40_cnt += 1
            # elif j < (prompt_length - 1) * 0.5:
            #     acc_50_cnt += 1
            # elif j < (prompt_length - 1) * 0.6:
            #     acc_60_cnt += 1
            # elif j < (prompt_length - 1) * 0.7:
            #     acc_70_cnt += 1
            # elif j < (prompt_length - 1) * 0.8:
            #     acc_80_cnt += 1
            # elif j < (prompt_length - 1) * 0.9:
            #     acc_90_cnt += 1
    acc = acc_cnt / (len(ret_list))
    print("acc: ", acc)
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[1:]))
    print("final result tokens:", ret_tokens)
    
    return acc, ret_tokens


def main():
    '''create log file'''
    # txt_file = open("log-{}-{}-{}-{}-{}-{}.txt".format(*time.localtime()), "w")
    
    '''get model'''
    devices=['cuda:0']
    model_dir = "lmsys/vicuna-7b-v1.5"
    # model_dir = "/home/cc/zyg/vicuna-7b-v1.5"
    tokenizer, model = get_model(model_dir=model_dir, devices=devices)

    total_layers = model.model.layers
    
    '''freeze model parameter'''
    for param in model.parameters():
        param.requires_grad = False

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()
    norm_layer = model.model.norm
    START_EMBED = embed_layer.weight[1].data
    START_EMBED = START_EMBED.unsqueeze(0).unsqueeze(0).to(devices[0])
    # print(START_EMBED.shape)
    _, _, _, _, all_start_hidden_states = get_hidden_state(tokenizer, model, input_embed=START_EMBED, use_rms_norm=True)
    START_16 = all_start_hidden_states[16]
    START_EMBED.requires_grad_(False)
    START_16.requires_grad_(False)
    # print(START_16.shape)

    '''the original prompt we try to infer'''
    prompts = [
    "yes",
    "Lily hit Susan in her face. Susan was pretty angry and shouted her boyfriend Tom for help. However, Tom was playing computer games with Peter. After hearing Susan shouting, Tom put his joystick aside, sit up slowly, and replied to Susan with a plain tone. \"Ok, Ok, I'm coming soon.\" ",
    "Lily hit Susan in her face. Susan was pretty angry and shouted her boyfriend Tom for help. However, Tom was playing computer games with Peter. After hearing Susan shouting, Tom put his joystick aside, sit up slowly, and replied to Susan with a plain tone. \"Ok, Ok, I'm coming soon.\" ",
    "Lily hit Susan in her face. Susan was pretty angry and shouted her boyfriend Tom for help. However, Tom was playing computer games with Peter. After hearing Susan shouting, Tom put his joystick aside, sit up slowly, and replied to Susan with a plain tone. \"Ok, Ok, I'm coming soon.\" ",
    # "I fly at least once a month London to New York.",
    # "The moment Scrooge's hand was on the lock, a strange voice called him by his name, and bade him enter. He obeyed. ",
    # "The moment Scrooge's hand was on the lock, a strange voice called him by his name, and bade him enter. He obeyed. ",
    # "The moment Scrooge's hand was on the lock, a strange voice called him by his name, and bade him enter. He obeyed. ",
    "We have long been expecting you, said Steve, going into his room and letting Levin's hand go as though to show that here all danger was over. \"I am very, very glad to see you,\" he went on. \"Well, how are you? Eh? When did you come?\" Levin was silent, looking at the unknown faces of Oblonsky's two companions, and especially at the hand of the elegant Greg, which had such long white fingers, such long yellow filbert-shaped nails, and such huge shining studs on the shirt-cuff, that apparently they absorbed all his attention, and allowed him no freedom of thought. Oblonsky noticed this at once, and smiled. I have the honor of knowing your brother, Sergey Ivanovitch, said Greg, holding out his slender hand with its long nails.",
    "We have long been expecting you, said Steve, going into his room and letting Levin's hand go as though to show that here all danger was over. \"I am very, very glad to see you,\" he went on. \"Well, how are you? Eh? When did you come?\" Levin was silent, looking at the unknown faces of Oblonsky's two companions, and especially at the hand of the elegant Greg, which had such long white fingers, such long yellow filbert-shaped nails, and such huge shining studs on the shirt-cuff, that apparently they absorbed all his attention, and allowed him no freedom of thought. Oblonsky noticed this at once, and smiled. I have the honor of knowing your brother, Sergey Ivanovitch, said Greg, holding out his slender hand with its long nails.",
    "We have long been expecting you, said Steve, going into his room and letting Levin's hand go as though to show that here all danger was over. \"I am very, very glad to see you,\" he went on. \"Well, how are you? Eh? When did you come?\" Levin was silent, looking at the unknown faces of Oblonsky's two companions, and especially at the hand of the elegant Greg, which had such long white fingers, such long yellow filbert-shaped nails, and such huge shining studs on the shirt-cuff, that apparently they absorbed all his attention, and allowed him no freedom of thought. Oblonsky noticed this at once, and smiled. I have the honor of knowing your brother, Sergey Ivanovitch, said Greg, holding out his slender hand with its long nails.",
    # "I was on JP650 the evening departure to Istanbul on 28th August. It was on a very clean Airbus A319 and it was a light load flight. The crew were warm and kind especially the Purser who took her time walked and talked to several passengers. ",
    # "Return flight Paris-Skopje via Ljubljana. All flights were on time and nice very clean planes / good flight crew. Seats were comfortable enough for a 1-2 hours flight. Boarding was easy for all flights. Ljubljana airport is very small and nice. ",
    # "London to Tirana via Ljubljana Economy Class. 30 minutes late on first leg due to air traffic restrictions at Gatwick but Adria held the connecting flight for us even though there were only a handful of people transferring from the London flight.",
    # "Very good flight with Aegean. The boarding was quickly and the crew very . During the flight, we had no entertainment (no films, no flight info). The food was an olive bread with a free no-alcoholic beverage (the first was free).",
    "I have just returned from a long weekend in Ljubljana flying on Adria from London Gatwick. The outbound flight was ok (food portions were a bit stingy given the 2 hour 20 minute flight time) and we arrived early into Ljubljana. Beware about arriving on a Saturday - the coach to the city centre are few and far between. I ended up taking a taxi which cost me 6500 SIT (about UKP 20) whereas the bus was 600 SIT (1.40). On the return flight Adria cancelled the flight for 'technical reasons'.",
    "I have just returned from a long weekend in Ljubljana flying on Adria from London Gatwick. The outbound flight was ok (food portions were a bit stingy given the 2 hour 20 minute flight time) and we arrived early into Ljubljana. Beware about arriving on a Saturday - the coach to the city centre are few and far between. I ended up taking a taxi which cost me 6500 SIT (about UKP 20) whereas the bus was 600 SIT (1.40). On the return flight Adria cancelled the flight for 'technical reasons'.",
    "I have just returned from a long weekend in Ljubljana flying on Adria from London Gatwick. The outbound flight was ok (food portions were a bit stingy given the 2 hour 20 minute flight time) and we arrived early into Ljubljana. Beware about arriving on a Saturday - the coach to the city centre are few and far between. I ended up taking a taxi which cost me 6500 SIT (about UKP 20) whereas the bus was 600 SIT (1.40). On the return flight Adria cancelled the flight for 'technical reasons'.",
    "I have flown Adria recently on a LH code share from Munich to Ljubjana and return. In both cases the plane was a brand new CRJ. Both flights were really smooth and the crews really efficient and helpful. Pilots kept us informed about everything during both flights. It was sunset and dusk and the flight over the alps was absolutely scenic.",
    "I have flown Adria recently on a LH code share from Munich to Ljubjana and return. In both cases the plane was a brand new CRJ. Both flights were really smooth and the crews really efficient and helpful. Pilots kept us informed about everything during both flights. It was sunset and dusk and the flight over the alps was absolutely scenic.",
    "I have flown Adria recently on a LH code share from Munich to Ljubjana and return. In both cases the plane was a brand new CRJ. Both flights were really smooth and the crews really efficient and helpful. Pilots kept us informed about everything during both flights. It was sunset and dusk and the flight over the alps was absolutely scenic.",
    # "Managed to get my rather large luggage quickly. First flight was on time, the A321 was clean and the Flight hosts were pleasant and friendly.",
    # "I recently flew to Mykonos through Athens and must say it was a very pleasant experience. The Zurich - Athens - Zurich legs were in Business class, to which I was upgraded for free being a Gold Member of their program Miles and Bonus.",
    # "I am always impressed with the pleasant attitude, responsive service and  warmth of cabin crew. The ground service is good, and the food is on the good side. On the other hand, most of their planes are starting to show their age and sometimes the seats are really worn out.",
    # "A good flight, on time. Flight attendants were pleasant and friendy. Was traveing with my partner, we got bored halfway through the flight as there is no inflight entertainment system. Hot food was served, overall a positive experience.",
    "I flew to Heraklion and Aantorini from Athens in February. Seats are fine with decent leg room unfortunately there's no proper inflight entertainment on the flight. The service carried out by the cabin crew are professional and efficient. My return flight from Heraklion was delayed due to some technical difficulties. But we still managed to arrive in Athens on time. My flight to Santorini was short so they couldn't really carry out the full service but they still manage to give us cookies and fresheners.",
    "I flew to Heraklion and Aantorini from Athens in February. Seats are fine with decent leg room unfortunately there's no proper inflight entertainment on the flight. The service carried out by the cabin crew are professional and efficient. My return flight from Heraklion was delayed due to some technical difficulties. But we still managed to arrive in Athens on time. My flight to Santorini was short so they couldn't really carry out the full service but they still manage to give us cookies and fresheners.",
    "I flew to Heraklion and Aantorini from Athens in February. Seats are fine with decent leg room unfortunately there's no proper inflight entertainment on the flight. The service carried out by the cabin crew are professional and efficient. My return flight from Heraklion was delayed due to some technical difficulties. But we still managed to arrive in Athens on time. My flight to Santorini was short so they couldn't really carry out the full service but they still manage to give us cookies and fresheners.",
    "LHR-ATH-CHQ Return 13th February 2015 - 21st February 2015. I got the 22.15 flight from Heathrow to Athens on the 13th and then the 09.15 flight to Chania on the 14th. At Heathrow the flight was over-subscribed so they were offering people a large sum of money a night in a hotel and the flight the following day - I wish they would have asked me! Then they were asking people to (voluntarily) check their hand luggage into the hold - so I offered because of the long wait I had at ATH! The flight was smooth the hot meal was served early on into the flight and we landed slightly early. ",
    "LHR-ATH-CHQ Return 13th February 2015 - 21st February 2015. I got the 22.15 flight from Heathrow to Athens on the 13th and then the 09.15 flight to Chania on the 14th. At Heathrow the flight was over-subscribed so they were offering people a large sum of money a night in a hotel and the flight the following day - I wish they would have asked me! Then they were asking people to (voluntarily) check their hand luggage into the hold - so I offered because of the long wait I had at ATH! The flight was smooth the hot meal was served early on into the flight and we landed slightly early. ",
    "LHR-ATH-CHQ Return 13th February 2015 - 21st February 2015. I got the 22.15 flight from Heathrow to Athens on the 13th and then the 09.15 flight to Chania on the 14th. At Heathrow the flight was over-subscribed so they were offering people a large sum of money a night in a hotel and the flight the following day - I wish they would have asked me! Then they were asking people to (voluntarily) check their hand luggage into the hold - so I offered because of the long wait I had at ATH! The flight was smooth the hot meal was served early on into the flight and we landed slightly early. ",
    # "I was positively surprised when I recently in July 2015 flew with a friend from London Heathrow to Belfast City Airport and back again. I booked tickets several weeks in advance and got a good value flight.",
    # "FRA-SKG Business class. Seat was standard European Business class seat (same as Economy class and middle seat was blocked). 3 course menu with Greek wines. They have standard coffee cappuccino espresso also.",
    "Writing this on behalf of my 87yr old mother who had flown to the UK on a 6 wk trip to see family. Aegean had been informed she required a wheelchair and we received confirmation it was booked both ways. From Rhodes to London they had no details regarding the chair and had to rush around looking for one. Apart from that initial problem my mother informed me everything was fine however on the day of her return Aegean decided to go on strike thankfully they messaged although it was 10 'o' clock at night. I tried desperately to get through eventually they put her on the following day again with confirmation of the wheelchair. London section was fine she tells me and the staff were quite attentive but on landing at Rhodes I waited outside and to my disgust saw my poor mother dragging a 20kg suitcase behind her carrying a coat and her handbag! Not one staff member offered to help she even had to ask a member of the public to lift her case from the conveyor belt.",
    "Writing this on behalf of my 87yr old mother who had flown to the UK on a 6 wk trip to see family. Aegean had been informed she required a wheelchair and we received confirmation it was booked both ways. From Rhodes to London they had no details regarding the chair and had to rush around looking for one. Apart from that initial problem my mother informed me everything was fine however on the day of her return Aegean decided to go on strike thankfully they messaged although it was 10 'o' clock at night. I tried desperately to get through eventually they put her on the following day again with confirmation of the wheelchair. London section was fine she tells me and the staff were quite attentive but on landing at Rhodes I waited outside and to my disgust saw my poor mother dragging a 20kg suitcase behind her carrying a coat and her handbag! Not one staff member offered to help she even had to ask a member of the public to lift her case from the conveyor belt.",
    "Writing this on behalf of my 87yr old mother who had flown to the UK on a 6 wk trip to see family. Aegean had been informed she required a wheelchair and we received confirmation it was booked both ways. From Rhodes to London they had no details regarding the chair and had to rush around looking for one. Apart from that initial problem my mother informed me everything was fine however on the day of her return Aegean decided to go on strike thankfully they messaged although it was 10 'o' clock at night. I tried desperately to get through eventually they put her on the following day again with confirmation of the wheelchair. London section was fine she tells me and the staff were quite attentive but on landing at Rhodes I waited outside and to my disgust saw my poor mother dragging a 20kg suitcase behind her carrying a coat and her handbag! Not one staff member offered to help she even had to ask a member of the public to lift her case from the conveyor belt.",
    # "Flew London to Athens and then on to Ioannina and then from Ioannina to Athens and back to London. Great flight new and comfortable planes professional crew good meals and snacks. Great airline with great product. When and if I ever go to Greece again I will fly with Aegean.",
    "I flew on October 5th then on to Mykonos. I then flew to Athens and then to Kos. On the 16th from Kalymnos to Athens and back to Mykonos and then on 22nd Mykonos - Athens - to Irakio Crete and then from Crete back to Athens. I must say I have had some excellent flights with Aegean. All flights on time and an excellent meal from London to Athens and this being Economy. On the internal flights you were offered a soft drink and snack. Cabin crews were extremely professional and I look forward to next October. ",
    "I flew on October 5th then on to Mykonos. I then flew to Athens and then to Kos. On the 16th from Kalymnos to Athens and back to Mykonos and then on 22nd Mykonos - Athens - to Irakio Crete and then from Crete back to Athens. I must say I have had some excellent flights with Aegean. All flights on time and an excellent meal from London to Athens and this being Economy. On the internal flights you were offered a soft drink and snack. Cabin crews were extremely professional and I look forward to next October. ",
    "I flew on October 5th then on to Mykonos. I then flew to Athens and then to Kos. On the 16th from Kalymnos to Athens and back to Mykonos and then on 22nd Mykonos - Athens - to Irakio Crete and then from Crete back to Athens. I must say I have had some excellent flights with Aegean. All flights on time and an excellent meal from London to Athens and this being Economy. On the internal flights you were offered a soft drink and snack. Cabin crews were extremely professional and I look forward to next October. ",
    "SFO-DUB, LHR-DUB-SFO. Ground staff excellent. On time departure. Flight staff were just going through the motions and seemed as though they couldn't care less about the passengers. Seating was tight, but bearable. Inflight entertainment was pretty good. Meals were not so good and beverage service was below standard. Had issues with passenger behind us that allowed their child to continuously kick the back of our seat. We asked the mother once of the problem yet the child continued. Asked again to please have your child stop the kicking. Third time was enough and flight crew intervened and helped settle the problem. A cookie for the kicking child and a glass of wine for us. Very odd situation but handled OK by crew. Overall the flight seemed short handed, stressed and just did not want to do their job. ",
    "SFO-DUB, LHR-DUB-SFO. Ground staff excellent. On time departure. Flight staff were just going through the motions and seemed as though they couldn't care less about the passengers. Seating was tight, but bearable. Inflight entertainment was pretty good. Meals were not so good and beverage service was below standard. Had issues with passenger behind us that allowed their child to continuously kick the back of our seat. We asked the mother once of the problem yet the child continued. Asked again to please have your child stop the kicking. Third time was enough and flight crew intervened and helped settle the problem. A cookie for the kicking child and a glass of wine for us. Very odd situation but handled OK by crew. Overall the flight seemed short handed, stressed and just did not want to do their job. ",
    "SFO-DUB, LHR-DUB-SFO. Ground staff excellent. On time departure. Flight staff were just going through the motions and seemed as though they couldn't care less about the passengers. Seating was tight, but bearable. Inflight entertainment was pretty good. Meals were not so good and beverage service was below standard. Had issues with passenger behind us that allowed their child to continuously kick the back of our seat. We asked the mother once of the problem yet the child continued. Asked again to please have your child stop the kicking. Third time was enough and flight crew intervened and helped settle the problem. A cookie for the kicking child and a glass of wine for us. Very odd situation but handled OK by crew. Overall the flight seemed short handed, stressed and just did not want to do their job. ",
    "I have travelled from Athens to Tirana every week for the past two years and I see it more as a torture rather than a journey. It all starts from the arrival in the airport where Aegean counters for Business travelers and gold miles card holders are filled with people which means at least 30'mins waiting for your check in. The return journey to Tirana is being carried out with a DE HAVILLAND 8-400 airplane with very little space for passenger seats. ",
    "I have travelled from Athens to Tirana every week for the past two years and I see it more as a torture rather than a journey. It all starts from the arrival in the airport where Aegean counters for Business travelers and gold miles card holders are filled with people which means at least 30'mins waiting for your check in. The return journey to Tirana is being carried out with a DE HAVILLAND 8-400 airplane with very little space for passenger seats. ",
    "I have travelled from Athens to Tirana every week for the past two years and I see it more as a torture rather than a journey. It all starts from the arrival in the airport where Aegean counters for Business travelers and gold miles card holders are filled with people which means at least 30'mins waiting for your check in. The return journey to Tirana is being carried out with a DE HAVILLAND 8-400 airplane with very little space for passenger seats. ",
    # "I travel with Aegean many times during the year usually I do SKG-ATH-SKG and I usually travel in business not because of the service but because I have the ability to change flights at will online and without any extra charge. ",
    "We flew Dublin to Paris on May 13th 2015. From the start we were greeted with disinterested and unfriendly staff at check in. Even though I checked in with my husband we were not seated next to each other even though the plane wasn't full. I moved myself over to sit with him before take off but why would they seat us apart? The cabin crew were miserable. Finally they rush through with the tea and coffee and snacks that you have to purchase and staff didn't even bother to ask us if we would like something.",
    "We flew Dublin to Paris on May 13th 2015. From the start we were greeted with disinterested and unfriendly staff at check in. Even though I checked in with my husband we were not seated next to each other even though the plane wasn't full. I moved myself over to sit with him before take off but why would they seat us apart? The cabin crew were miserable. Finally they rush through with the tea and coffee and snacks that you have to purchase and staff didn't even bother to ask us if we would like something.",
    "We flew Dublin to Paris on May 13th 2015. From the start we were greeted with disinterested and unfriendly staff at check in. Even though I checked in with my husband we were not seated next to each other even though the plane wasn't full. I moved myself over to sit with him before take off but why would they seat us apart? The cabin crew were miserable. Finally they rush through with the tea and coffee and snacks that you have to purchase and staff didn't even bother to ask us if we would like something.",
    "I traveled from Munich to Boston in early January 2015. The experience at Munich airport was very poor particularly the passport control line which was 45 minutes of unrestrained and uncontrolled pushing and shoving. Probably not Aer Lingus' fault but still a poor start. I had paid for a business class ticket over the internet. Did not find out until getting on the plane that the first part of the trip (Munich to Dublin) was economy as there are no business class seats on the plane.",
    "I traveled from Munich to Boston in early January 2015. The experience at Munich airport was very poor particularly the passport control line which was 45 minutes of unrestrained and uncontrolled pushing and shoving. Probably not Aer Lingus' fault but still a poor start. I had paid for a business class ticket over the internet. Did not find out until getting on the plane that the first part of the trip (Munich to Dublin) was economy as there are no business class seats on the plane.",
    "I traveled from Munich to Boston in early January 2015. The experience at Munich airport was very poor particularly the passport control line which was 45 minutes of unrestrained and uncontrolled pushing and shoving. Probably not Aer Lingus' fault but still a poor start. I had paid for a business class ticket over the internet. Did not find out until getting on the plane that the first part of the trip (Munich to Dublin) was economy as there are no business class seats on the plane.",
    "April 15th flight 108 return 26th Apr flight 105. My mother and I flew from JFK to Dublin and return. We were in economy from JFK to DUB and in Premier class on the return trip. Both were great experiences and on Aer Lingus new A330. The food was excellent in both cabins. We opted for the succulent steal ($18) in Economy and it was delicious. Crew were lovely. On return we were in the new Premier class cabin. The crew the food the service was terrific.",
    "April 15th flight 108 return 26th Apr flight 105. My mother and I flew from JFK to Dublin and return. We were in economy from JFK to DUB and in Premier class on the return trip. Both were great experiences and on Aer Lingus new A330. The food was excellent in both cabins. We opted for the succulent steal ($18) in Economy and it was delicious. Crew were lovely. On return we were in the new Premier class cabin. The crew the food the service was terrific.",
    "April 15th flight 108 return 26th Apr flight 105. My mother and I flew from JFK to Dublin and return. We were in economy from JFK to DUB and in Premier class on the return trip. Both were great experiences and on Aer Lingus new A330. The food was excellent in both cabins. We opted for the succulent steal ($18) in Economy and it was delicious. Crew were lovely. On return we were in the new Premier class cabin. The crew the food the service was terrific.",
    # "You go up. You go down. Flight is over. This 20 min flight from Athens to Mykonos on an A320 does what it needs to do get you to your destination.",
    "Bogota to Las Vegas via Fort Lauderdale. Terrible experience very dirty flight and extremely disorganized passenger service and operations. This airline having know the fact that it takes time to go through border control and homeland security gave us less than 1 hour at Fort Lauderdale airport to go through the security and immigration check during peak hours with over 300 odd passengers in a queue. This was at 18.45 hrs I was supposed to be on my next flight unfortunately with Spirit at 20.00 so we had roughly 35 mins. As a result I missed my flight.",
    "Bogota to Las Vegas via Fort Lauderdale. Terrible experience very dirty flight and extremely disorganized passenger service and operations. This airline having know the fact that it takes time to go through border control and homeland security gave us less than 1 hour at Fort Lauderdale airport to go through the security and immigration check during peak hours with over 300 odd passengers in a queue. This was at 18.45 hrs I was supposed to be on my next flight unfortunately with Spirit at 20.00 so we had roughly 35 mins. As a result I missed my flight.",
    "Bogota to Las Vegas via Fort Lauderdale. Terrible experience very dirty flight and extremely disorganized passenger service and operations. This airline having know the fact that it takes time to go through border control and homeland security gave us less than 1 hour at Fort Lauderdale airport to go through the security and immigration check during peak hours with over 300 odd passengers in a queue. This was at 18.45 hrs I was supposed to be on my next flight unfortunately with Spirit at 20.00 so we had roughly 35 mins. As a result I missed my flight.",
    "Business class on their A300 from Doha and really glad to get off this flight. Aircraft very scruffy inside and needed some real attention and cleaning. Seats were broken in many areas and toilets were unusabe - FAs did not seem bothered by any of this. Service was all over in a short time and FAs then disappeared to galley. Not an airline I would wish on my worst enemy and despite all they say in their magazine about change this airline seems to be a lot worse than when I last flew them (that was in 1991 though!).",
    "Business class on their A300 from Doha and really glad to get off this flight. Aircraft very scruffy inside and needed some real attention and cleaning. Seats were broken in many areas and toilets were unusabe - FAs did not seem bothered by any of this. Service was all over in a short time and FAs then disappeared to galley. Not an airline I would wish on my worst enemy and despite all they say in their magazine about change this airline seems to be a lot worse than when I last flew them (that was in 1991 though!).",
    "Business class on their A300 from Doha and really glad to get off this flight. Aircraft very scruffy inside and needed some real attention and cleaning. Seats were broken in many areas and toilets were unusabe - FAs did not seem bothered by any of this. Service was all over in a short time and FAs then disappeared to galley. Not an airline I would wish on my worst enemy and despite all they say in their magazine about change this airline seems to be a lot worse than when I last flew them (that was in 1991 though!).",
    "My sister and I flew to South Carolina and before we flew my husband called Spirit to see what the procedure was for checking baggage. He was told that although the internet says to pay $25 online or $50 at check-in or $100 at the gate we didn't have to. We flew out of Chicago O'Hare Airport with no problem. When departing South Carolina we were told as we were boarding that we had to pay $100 to carry a back pack and 1 piece of luggage with us on the plane.",
    "My sister and I flew to South Carolina and before we flew my husband called Spirit to see what the procedure was for checking baggage. He was told that although the internet says to pay $25 online or $50 at check-in or $100 at the gate we didn't have to. We flew out of Chicago O'Hare Airport with no problem. When departing South Carolina we were told as we were boarding that we had to pay $100 to carry a back pack and 1 piece of luggage with us on the plane.",
    "My sister and I flew to South Carolina and before we flew my husband called Spirit to see what the procedure was for checking baggage. He was told that although the internet says to pay $25 online or $50 at check-in or $100 at the gate we didn't have to. We flew out of Chicago O'Hare Airport with no problem. When departing South Carolina we were told as we were boarding that we had to pay $100 to carry a back pack and 1 piece of luggage with us on the plane.",
    "I had to fly last minute from SFO (San Francisco) to LAS (Las Vegas) one way and Spirit offered the lowest price out of any airline in the SF Bay Area the trek to (OAK) Oakland Airport was a little further than I normally travel to get to an airport but the deal was too good to pass up. The flight was not full although I checked in the day before I was given a seat upgrade to a window seat near the front. The cabin crew was friendly and made the early morning flight easier with humor. Flight duration was exactly as advertised and actually landed a few minutes early and it was a pleasant flight in a brand new airplane. The one thing I did not like were the seats far too close together and much too thin.",
    "I had to fly last minute from SFO (San Francisco) to LAS (Las Vegas) one way and Spirit offered the lowest price out of any airline in the SF Bay Area the trek to (OAK) Oakland Airport was a little further than I normally travel to get to an airport but the deal was too good to pass up. The flight was not full although I checked in the day before I was given a seat upgrade to a window seat near the front. The cabin crew was friendly and made the early morning flight easier with humor. Flight duration was exactly as advertised and actually landed a few minutes early and it was a pleasant flight in a brand new airplane. The one thing I did not like were the seats far too close together and much too thin.",
    "I had to fly last minute from SFO (San Francisco) to LAS (Las Vegas) one way and Spirit offered the lowest price out of any airline in the SF Bay Area the trek to (OAK) Oakland Airport was a little further than I normally travel to get to an airport but the deal was too good to pass up. The flight was not full although I checked in the day before I was given a seat upgrade to a window seat near the front. The cabin crew was friendly and made the early morning flight easier with humor. Flight duration was exactly as advertised and actually landed a few minutes early and it was a pleasant flight in a brand new airplane. The one thing I did not like were the seats far too close together and much too thin.",
    "Not a great experience for a first time flyer of Sun Country Airlines. My carry on had to be checked after rows 4-9 then 19-25 were asked to board. I was in the exit row 13 and there's not enough room for carry-ons for 10 rows? Not only that, you skip the middle rows where some patrons pay additional money to sit more comfortably. Lastly, I was in seat 13A against the window, my arm rest was broken on the window side as was the passengers behind me. I will not be flying Sun Country again as this was my first and last shot because of ticket price.",
    "Not a great experience for a first time flyer of Sun Country Airlines. My carry on had to be checked after rows 4-9 then 19-25 were asked to board. I was in the exit row 13 and there's not enough room for carry-ons for 10 rows? Not only that, you skip the middle rows where some patrons pay additional money to sit more comfortably. Lastly, I was in seat 13A against the window, my arm rest was broken on the window side as was the passengers behind me. I will not be flying Sun Country again as this was my first and last shot because of ticket price.",
    "Not a great experience for a first time flyer of Sun Country Airlines. My carry on had to be checked after rows 4-9 then 19-25 were asked to board. I was in the exit row 13 and there's not enough room for carry-ons for 10 rows? Not only that, you skip the middle rows where some patrons pay additional money to sit more comfortably. Lastly, I was in seat 13A against the window, my arm rest was broken on the window side as was the passengers behind me. I will not be flying Sun Country again as this was my first and last shot because of ticket price.",
    "Spirit airlines has been my worst experience flying. First of all who charges for carry on and still puts the most outrageous policies on check ins. The worst part is they don't bother to tell you when you are buying your ticket that way you don't know and you are forced to pay for it when you get there. Then they charge for boarding passes and for not having a random seat. The staff and their rude attitudes did not make it any better.",
    "Spirit airlines has been my worst experience flying. First of all who charges for carry on and still puts the most outrageous policies on check ins. The worst part is they don't bother to tell you when you are buying your ticket that way you don't know and you are forced to pay for it when you get there. Then they charge for boarding passes and for not having a random seat. The staff and their rude attitudes did not make it any better.",
    "Spirit airlines has been my worst experience flying. First of all who charges for carry on and still puts the most outrageous policies on check ins. The worst part is they don't bother to tell you when you are buying your ticket that way you don't know and you are forced to pay for it when you get there. Then they charge for boarding passes and for not having a random seat. The staff and their rude attitudes did not make it any better.",
    "Poor service poor communication poor quality. You couldn't even make up how bad they are. Oversold flight. We had a 6 hour delay that was identified 30 minutes prior to our departure as a result we missed our connecting flight. Missed two days of work. Only offered a 50 dollar voucher for all the inconvenience.",
    "Poor service poor communication poor quality. You couldn't even make up how bad they are. Oversold flight. We had a 6 hour delay that was identified 30 minutes prior to our departure as a result we missed our connecting flight. Missed two days of work. Only offered a 50 dollar voucher for all the inconvenience.",
    "Poor service poor communication poor quality. You couldn't even make up how bad they are. Oversold flight. We had a 6 hour delay that was identified 30 minutes prior to our departure as a result we missed our connecting flight. Missed two days of work. Only offered a 50 dollar voucher for all the inconvenience.",
    "You get what you pay for. I was forced to give at least one star otherwise I would have given zero for seat comfort and inflight entertainment (there is none). I am not a tall person and I have never had my knees so scrunched up on any other airline.",
    "You get what you pay for. I was forced to give at least one star otherwise I would have given zero for seat comfort and inflight entertainment (there is none). I am not a tall person and I have never had my knees so scrunched up on any other airline.",
    "You get what you pay for. I was forced to give at least one star otherwise I would have given zero for seat comfort and inflight entertainment (there is none). I am not a tall person and I have never had my knees so scrunched up on any other airline.",
    "Brisbane-Taipei-Paris Premium Laurel Class. Both flights great however more room on 777 then A330. Seat is not flat in sleep mode which was a bit of a pain however still managed some sleep. Food great on both flights. Cabin crew very friendly and came down aisle on both legs offering drinks and snacks.",
    "Brisbane-Taipei-Paris Premium Laurel Class. Both flights great however more room on 777 then A330. Seat is not flat in sleep mode which was a bit of a pain however still managed some sleep. Food great on both flights. Cabin crew very friendly and came down aisle on both legs offering drinks and snacks.",
    "Brisbane-Taipei-Paris Premium Laurel Class. Both flights great however more room on 777 then A330. Seat is not flat in sleep mode which was a bit of a pain however still managed some sleep. Food great on both flights. Cabin crew very friendly and came down aisle on both legs offering drinks and snacks.",
    ]

    prompts = [prompts[1]]
    for prompt_ in prompts:
        '''implement 16 by 16 idea!!'''

        '''get all hidden states in a list'''
        model.model.layers = total_layers[:32]
        total_input_ids, total_attention_mask, _, total_hidden_states, all_hidden_states = get_hidden_state(tokenizer, 
                    model, prompt=prompt_, use_rms_norm=True)
        print("collected hidden states:", len(all_hidden_states))

        print("state 0 and 1:", all_hidden_states[0], all_hidden_states[1])
        print("L2 norm", torch.norm((all_hidden_states[0].type(torch.float32)-all_hidden_states[1].type(torch.float32)), p=2, dim=-1))
        print("cosine", F.cosine_similarity(all_hidden_states[0].type(torch.float32), all_hidden_states[1].type(torch.float32), dim=-1))

        print("state 1 and 2:", all_hidden_states[1], all_hidden_states[2])
        print("L2 norm", torch.norm((all_hidden_states[1].type(torch.float32)-all_hidden_states[2].type(torch.float32)), p=2, dim=-1))
        print("cosine", F.cosine_similarity(all_hidden_states[1].type(torch.float32), all_hidden_states[2].type(torch.float32), dim=-1))

        print("state 1 and 3:", all_hidden_states[1], all_hidden_states[3])
        print("L2 norm", torch.norm((all_hidden_states[1].type(torch.float32)-all_hidden_states[3].type(torch.float32)), p=2, dim=-1))
        print("cosine", F.cosine_similarity(all_hidden_states[1].type(torch.float32), all_hidden_states[3].type(torch.float32), dim=-1))

        print("state 0 and 16:", all_hidden_states[0], all_hidden_states[16])
        print("L2 norm", torch.norm((all_hidden_states[0].type(torch.float32)-all_hidden_states[16].type(torch.float32)), p=2, dim=-1))
        print("cosine", F.cosine_similarity(all_hidden_states[0].type(torch.float32), all_hidden_states[16].type(torch.float32), dim=-1))

        print("state 1 and 16:", all_hidden_states[1], all_hidden_states[16])
        print("L2 norm", torch.norm((all_hidden_states[1].type(torch.float32)-all_hidden_states[16].type(torch.float32)), p=2, dim=-1))
        print("cosine", F.cosine_similarity(all_hidden_states[1].type(torch.float32), all_hidden_states[16].type(torch.float32), dim=-1))

        acc, ret_tokens = invert_embedding(all_hidden_states[16], tokenizer, embed_layer, total_input_ids, invert_method='cosine')
        print("16 layer result tokens:", ret_tokens)

        acc, ret_tokens = invert_embedding(all_hidden_states[1], tokenizer, embed_layer, total_input_ids, invert_method='cosine')
        print("1 layer result tokens:", ret_tokens)

        acc, ret_tokens = invert_embedding(all_hidden_states[2], tokenizer, embed_layer, total_input_ids, invert_method='cosine')
        print("2 layer result tokens:", ret_tokens)
                        

        # print("two embeddings:", all_hidden_states[16], next_hidden_states_16)
        # txt_file.write("two embeddings: \n{} \n{}\n".format(str(all_hidden_states[16]), str(next_hidden_states_16)))
        # txt_file.write("gt embeddings minmax: \n{} \n{}\n".format(str(torch.max(all_hidden_states[16][0][1:])), str(torch.min(all_hidden_states[16][0][1:]))))
        # txt_file.write("gt embeddings range: {}\n".format((all_hidden_states[16] > 0.1).sum()))
        # txt_file.write("ret embeddings minmax: \n{} \n{}\n".format(str(torch.max(next_hidden_states_16[0][1:])), str(torch.min(next_hidden_states_16[0][1:]))))
        # txt_file.write("two embeddings cos: {}\n".format(F.cosine_similarity(all_hidden_states[16].type(torch.float32), next_hidden_states_16.type(torch.float32), dim=-1)))
        # txt_file.write("two embeddings cos mean: {}\n".format(F.cosine_similarity(all_hidden_states[16].type(torch.float32), next_hidden_states_16.type(torch.float32), dim=-1).mean().data))
        # txt_file.write("two embeddings cos minmax: {} \n {} \n".format(torch.max(F.cosine_similarity(all_hidden_states[16].type(torch.float32), next_hidden_states_16.type(torch.float32), dim=-1)), torch.min(F.cosine_similarity(all_hidden_states[16].type(torch.float32), next_hidden_states_16.type(torch.float32), dim=-1))))
        # txt_file.write("two embeddings L2: {}\n".format(torch.norm(all_hidden_states[16].type(torch.float32) - next_hidden_states_16.type(torch.float32), p=2, dim=-1).detach().cpu()))
        

if __name__ == "__main__":
    main()
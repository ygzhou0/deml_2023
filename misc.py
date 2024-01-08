import nltk
from rouge_score import rouge_scorer
import editdistance
import pickle
import os
from transformers import AutoTokenizer


class RecoverMetric:
    def __init__(self):
        pass

    def get_p_r_f1(self, rc_text_token: list, gt_text_token: list):
        total_pre = len(gt_text_token)
        total_rc = len(rc_text_token)
        precision = 0
        for item in rc_text_token:
            if item in gt_text_token:
                precision += 1
        precision = precision / total_pre
        recall = 0
        for item in gt_text_token:
            if item in rc_text_token:
                recall += 1
        recall = recall / total_rc
        f1 = 2 / (1 / recall + 1 / precision)
        ret = {"precision": precision, "recall": recall, "F1 score": f1}
        return ret

    def get_rouge_score(self, rc_text_token: list, gt_text_token: list):
        rc_text, gt_text = " ".join(rc_text_token), " ".join(gt_text_token)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(rc_text, gt_text)
        rouge_ret = {}
        for key in scores:
            rouge_ret[key] = scores[key].fmeasure
        return rouge_ret

    def get_bleu_score(self, rc_text_token: list, gt_text_token: list):
        BLEU1score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token, weights=([1]))
        BLEU2score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token, weights=(0.5, 0.5))
        BLEU4score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token,
                                                             weights=(0.25, 0.25, 0.25, 0.25))
        bleu_ret = {"bleu1": BLEU1score, "bleu2": BLEU2score, "bleu4": BLEU4score}
        return bleu_ret

    def get_edit_distance(self, rc_text_token: list, gt_text_token: list):
        return {"edit distance": editdistance.eval(''.join(rc_text_token), ''.join(gt_text_token))}

    def get_metric(self, rc_text_token: list, gt_text_token: list):
        metrics = {}
        metrics.update(self.get_p_r_f1(rc_text_token, gt_text_token))
        metrics.update(self.get_rouge_score(rc_text_token, gt_text_token))
        metrics.update(self.get_bleu_score(rc_text_token, gt_text_token))
        metrics.update(self.get_edit_distance(rc_text_token, gt_text_token))
        return metrics


path_to_tokenizer = "E:\\uni\\senior\\llm-attacks\\vicuna-7b-v1.3"
metrics = RecoverMetric()
tokenizer = AutoTokenizer.from_pretrained(
    path_to_tokenizer,
    trust_remote_code=True,
    use_fast=False
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
pickle_fp = "results/65B-airline-64layer-results"
pr = rec = F1 = r1 = r2 = rL = b1 = b2 = b4 = e = 0
cnt = 0
for root, dirs, files in os.walk(pickle_fp):
    for file in files:
        with open(os.path.join(pickle_fp, file), 'rb') as f:
            rc_text, or_text, tensor = pickle.load(f)
        rc_token = tokenizer.tokenize(rc_text)
        gt_token = tokenizer.tokenize(or_text)
        m = metrics.get_metric(rc_token, gt_token)
        pr += m['precision']
        rec += m['recall']
        F1 += m['F1 score']
        r1 += m['rouge1']
        r2 += m['rouge2']
        rL += m['rougeL']
        b1 += m['bleu1']
        b2 += m['bleu2']
        b4 += m['bleu4']
        e += m['edit distance']
        cnt += 1
print('precision', pr / cnt)
print('recall', rec / cnt)
print('F1 score', F1 / cnt)
print('rouge1', r1 / cnt)
print('rouge2', r2 / cnt)
print('rougeL', rL / cnt)
print('bleu1', b1 / cnt)
print('bleu2', b2 / cnt)
print('bleu4', b4 / cnt)
print("edit distance", e / cnt)

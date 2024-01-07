import nltk
from rouge_score import rouge_scorer
import editdistance


class RecoverMetric:

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
        f1 = 2 / (1/recall + 1/precision)
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
        BLEU4score = nltk.translate.bleu_score.sentence_bleu([gt_text_token], rc_text_token, weights=(0.25, 0.25, 0.25, 0.25))
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

a = ["aad", "noids", "naui", "iogap+", "13@"]
b = ["aad", "noids", "nui", "a", "iogap+", "13", "@#"]
metrics = RecoverMetric()
m = metrics.get_metric(a, b)
print(m)
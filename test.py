from src.model import Model 
from src.utils import to_cuda, get_predicted_clusters, get_event2cluster
from src.data import get_dataloader
from transformers import RobertaTokenizer
import torch
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
from scipy.optimize import linear_sum_assignment
import os

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta
        # for blanc
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, document):
        if self.metric == blanc:
            rc, wc, rn, wn = self.metric(document.mention_to_cluster, document.mention_to_gold)
            self.rc += rc
            self.wc += wc
            self.rn += rn
            self.wn += wn
        else:
            if self.metric == ceafe:
                pn, pd, rn, rd = self.metric(document.clusters, document.gold)
            else:
                pn, pd = self.metric(document.clusters, document.mention_to_gold)
                rn, rd = self.metric(document.gold, document.mention_to_cluster)
            self.p_num += pn
            self.p_den += pd
            self.r_num += rn
            self.r_den += rd

    def get_f1(self):
        if self.metric == blanc:
            return (f1(self.rc, self.rc+self.wc, self.rc, self.rc+self.wn, beta=self.beta) + f1(self.rn, self.rn+self.wn, self.rn, self.rn+self.wc, beta=self.beta)) / 2
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        if self.metric == blanc:
            return (self.rc/(self.rc+self.wn+1e-6) + self.rn/(self.rn+self.wc+1e-6)) / 2
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        if self.metric == blanc:
            return (self.rc/(self.rc+self.wc+1e-6) + self.rn/(self.rn+self.wn+1e-6)) / 2
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_id, col_id = linear_sum_assignment(-scores)
    similarity = sum(scores[row_id, col_id])
    return similarity, len(clusters), similarity, len(gold_clusters)

def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue
        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1
        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

def blanc(mention_to_cluster, mention_to_gold):
    rc = 0
    wc = 0
    rn = 0
    wn = 0
    assert len(mention_to_cluster) == len(mention_to_gold)
    mentions = list(mention_to_cluster.keys())
    for i in range(len(mentions)):
        for j in range(i+1, len(mentions)):
            if mention_to_cluster[mentions[i]] == mention_to_cluster[mentions[j]]:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    rc += 1
                else:
                    wc += 1
            else:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    wn += 1
                else:
                    rn += 1
    return rc, wc, rn, wn

class EvalResult:
    def __init__(self, gold, mention_to_gold, clusters, mention_to_cluster):
        self.gold = gold
        self.mention_to_gold = mention_to_gold
        self.clusters = clusters
        self.mention_to_cluster = mention_to_cluster

REL2ID={
"temporal":{
    "NONE": 0,
    "BEFORE": 1,
    "OVERLAP": 2,
    "CONTAINS": 3,
    "SIMULTANEOUS": 4,
    "ENDS-ON": 5,
    "BEGINS-ON": 6,
},
"causal":{
    "NONE": 0,
    "PRECONDITION": 1,
    "CAUSE": 2
},
"subevent":{
    "NONE": 0,
    "subevent": 1
}
}

def evaluate_coreference(eval_results):
    metrics = [b_cubed, ceafe, muc, blanc]
    metric_names = ["b_cubed", "ceaf", "muc", "blanc"]
    results = {}
    for metric, name in zip(metrics, metric_names):
        p, r, f = evaluate_documents(eval_results, metric)
        results[f"{name}_precision"] = p * 100.0
        results[f"{name}_recall"] = r * 100.0
        results[f"{name}_f1"] = f * 100.0
    return results

def evaluate_relation(labels, preds, rel_type):
    pos_labels = list(range(1, len(REL2ID[rel_type])))
    p = precision_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    r = recall_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    f1 = f1_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    return {
        f"{rel_type}_precision": p,
        f"{rel_type}_recall": r,
        f"{rel_type}_f1": f1
    }

def test(model, test_dataloader):
    model.eval()
    temporal_labels, temporal_preds = [], []
    causal_labels, causal_preds = [], []
    
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Testing"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
                    
            coref_scores, temporal_scores, causal_scores, _ = model(data)
            
           
            labels = data["temporal_labels"].view(-1)
            scores = temporal_scores.view(-1, temporal_scores.size(-1))
            pred = torch.argmax(scores, dim=-1)
            temporal_preds.extend(pred[labels>=0].cpu().numpy().tolist())
            temporal_labels.extend(labels[labels>=0].cpu().numpy().tolist())
            
          
            labels = data["causal_labels"].view(-1)
            scores = causal_scores.view(-1, causal_scores.size(-1))
            pred = torch.argmax(scores, dim=-1)
            causal_preds.extend(pred[labels>=0].cpu().numpy().tolist())
            causal_labels.extend(labels[labels>=0].cpu().numpy().tolist())
    
   
    results = {}

    

    temporal_results = evaluate_relation(temporal_labels, temporal_preds, "temporal")
    results.update(temporal_results)
    
    causal_results = evaluate_relation(causal_labels, causal_preds, "causal") 
    results.update(causal_results)
    

    f1_scores = [
        temporal_results["temporal_f1"],
        causal_results["causal_f1"]
    ]
    results["overall_f1"] = sum(f1_scores) / 3.0 
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--ignore_nonetype", action="store_true")
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained("/path_to_roberta-base/")
    test_dataloader = get_dataloader(tokenizer, "test", max_length=256, shuffle=False, 
                                   batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)

    model = Model(len(tokenizer))
    model = to_cuda(model)
    
    print("*" * 30 + "Test" + "*" * 30)
    
    for task in [ "TEMPORAL", "CAUSAL"]:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{task}")
        print(f"Testing {task} using checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["model"])
        
        results = test(model, test_dataloader)
        
     
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
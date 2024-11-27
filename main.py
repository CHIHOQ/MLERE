from src.model import Model
from src.utils import to_cuda, to_var
import torch
import random
import numpy as np
from tqdm import tqdm
from src.data import myDataset, get_dataloader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from src.utils import get_predicted_clusters, get_event2cluster, fill_expand
from src.metrics import evaluate_documents, b_cubed, ceafe, muc, Evaluator, blanc
from src.dump_result import coref_dump, causal_dump, temporal_dump, subevent_dump
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from src.data import TEMPREL2ID, ID2TEMPREL, CAUSALREL2ID, ID2CAUSALREL, SUBEVENTREL2ID, ID2SUBEVENTREL
import warnings
import os
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

class EvalResult:
    def __init__(self, gold, mention_to_gold, clusters, mention_to_cluster):
        self.gold = gold
        self.mention_to_gold = mention_to_gold
        self.clusters = clusters
        self.mention_to_cluster = mention_to_cluster


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def evaluate(model, dataloader, desc=""):
    temporal_pred_list = []
    temporal_label_list = []
    causal_pred_list = []
    causal_label_list = []
    subevent_pred_list = []
    subevent_label_list = []
    coref_train_eval_results = []
    for data in tqdm(dataloader, desc=desc):
        model.eval()
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k])
        coref_scores, temporal_scores, causal_scores, subevent_scores = model(data)
       
        # Temporal Evaluation
        labels = data["temporal_labels"]
        scores = temporal_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        pred = torch.argmax(scores, dim=-1)
        temporal_pred_list.extend(pred[labels >= 0].cpu().numpy().tolist())
        temporal_label_list.extend(labels[labels >= 0].cpu().numpy().tolist())
        # Causal Evaluation
        labels = data["causal_labels"]
        scores = causal_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        pred = torch.argmax(scores, dim=-1)
        causal_pred_list.extend(pred[labels >= 0].cpu().numpy().tolist())
        causal_label_list.extend(labels[labels >= 0].cpu().numpy().tolist())
        # Subevent Evaluation
        labels = data["subevent_labels"]
        scores = subevent_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        pred = torch.argmax(scores, dim=-1)
        subevent_pred_list.extend(pred[labels >= 0].cpu().numpy().tolist())
        subevent_label_list.extend(labels[labels >= 0].cpu().numpy().tolist())

    
    temporal_res = classification_report(
        temporal_label_list, temporal_pred_list, output_dict=True, target_names=TEMP_REPORT_CLASS_NAMES, labels=TEMP_REPORT_CLASS_LABELS
    )
    print("TEMPORAL:", temporal_res)
    result_collection["TEMPORAL"] = temporal_res
    causal_res = classification_report(
        causal_label_list, causal_pred_list, output_dict=True, target_names=CAUSAL_REPORT_CLASS_NAMES, labels=CAUSAL_REPORT_CLASS_LABELS
    )
    print("CAUSAL:", causal_res)
    result_collection["CAUSAL"] = causal_res
    subevent_res = classification_report(
        subevent_label_list, subevent_pred_list, output_dict=True, target_names=SUBEVENT_REPORT_CLASS_NAMES, labels=SUBEVENT_REPORT_CLASS_LABELS
    )
    print("SUBEVENT:", subevent_res)
    result_collection["SUBEVENT"] = subevent_res
    return result_collection

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--coreference_rate", default=1.0, type=float)
    parser.add_argument("--temporal_rate", default=5, type=float)
    parser.add_argument("--causal_rate", default=30.0, type=float)
    parser.add_argument("--subevent_rate", default=40.0, type=float)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained("<path_to_pretrained_tokenizer>")

    print("loading data...")
    if not args.eval_only:
        train_dataloader = get_dataloader(
            tokenizer, "train", max_length=256, shuffle=True, batch_size=args.batch_size
        )
        dev_dataloader = get_dataloader(
            tokenizer, "valid", max_length=256, shuffle=False, batch_size=args.batch_size
        )
    test_dataloader = get_dataloader(
        tokenizer, "test", max_length=256, shuffle=False, batch_size=args.batch_size
    )

    print("loading model...")
    model = Model(len(tokenizer))
    model = to_cuda(model)


    if not args.eval_only:
        bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=args.bert_lr)

        scorer_param = []
        scorer_param += [p for p in model.temporal_scorer.parameters() if p.requires_grad]
        scorer_param += [p for p in model.causal_scorer.parameters() if p.requires_grad]
        scorer_param += [p for p in model.subevent_scorer.parameters() if p.requires_grad]
        optimizer = Adam(scorer_param, lr=args.lr)

        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
    eps = 1e-8

    metrics = [b_cubed, ceafe, muc, blanc]
    metric_names = ["B-cubed", "CEAF", "MUC", "BLANC"]
    Loss = nn.CrossEntropyLoss(ignore_index=-100)
    glb_step = 0
    if not args.eval_only:
        print("*******************start training********************")


        temp_losses = []
        causal_losses = []
        subevent_losses = []
        temporal_pred_list = []
        temporal_label_list = []
        causal_pred_list = []
        causal_label_list = []
        subevent_pred_list = []
        subevent_label_list = []

        best_score = {"COREFERENCE": {name:0.0 for name in metric_names}, "TEMPORAL": 0.0, "CAUSAL": 0.0, "SUBEVENT": 0.0}
        for epoch in range(args.epochs):
            for data in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):
                model.train()
                loss = 0.0
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = to_cuda(data[k])
                coref_scores, temporal_scores, causal_scores, subevent_scores = model(data)
                tmp_coref_loss=0.0
                
                labels = data["temporal_labels"]
                scores = temporal_scores
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                tmp = Loss(scores, labels)
                loss += args.temporal_rate * tmp
                temp_losses.append(tmp.item())
                pred = torch.argmax(scores, dim=-1)
                temporal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                temporal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())

                labels = data["causal_labels"]
                scores = causal_scores
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                tmp = Loss(scores, labels)
                loss += args.causal_rate * tmp
                causal_losses.append(tmp.item())
                pred = torch.argmax(scores, dim=-1)
                causal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                causal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())

                labels = data["subevent_labels"]
                scores = subevent_scores
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                tmp = Loss(scores, labels)
                loss += args.subevent_rate * tmp
                subevent_losses.append(tmp.item())
                pred = torch.argmax(scores, dim=-1)
                subevent_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                subevent_label_list.extend(labels[labels>=0].cpu().numpy().tolist())
                if args.accumulation_steps>1:
                    loss=loss / args.accumulation_steps
                loss.backward()
                glb_step += 1

                if glb_step % args.accumulation_steps ==0:
                    optimizer.step()
                    bert_optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    bert_optimizer.zero_grad()

                if glb_step % args.log_steps == 0:
                    print("*"*20 + "Train Prediction Examples" + "*"*20)
                    print("Train %d steps: coref_loss=%f temporal_loss=%f causal_loss=%f subevent_loss=%f" % (glb_step, np.mean(coref_losses), np.mean(temp_losses), np.mean(causal_losses), np.mean(subevent_losses)))
                    for metric, name in zip(metrics, metric_names):
                        res = evaluate_documents(coref_train_eval_results, metric)
                        print("COREFRENCE %s: precision=%.4f, recall=%.4f, f1=%.4f" % (name, *res))
                    temporal_res = classification_report(temporal_label_list, temporal_pred_list, output_dict=True, target_names=TEMP_REPORT_CLASS_NAMES, labels=TEMP_REPORT_CLASS_LABELS)
                    print("TEMPORAL:", temporal_res)
                    causal_res = classification_report(causal_label_list, causal_pred_list, output_dict=True, target_names=CAUSAL_REPORT_CLASS_NAMES, labels=CAUSAL_REPORT_CLASS_LABELS)
                    print("CAUSAL:", causal_res)
                    subevent_res = classification_report(subevent_label_list, subevent_pred_list, output_dict=True, target_names=SUBEVENT_REPORT_CLASS_NAMES, labels=SUBEVENT_REPORT_CLASS_LABELS)
                    print("SUBEVENT:", subevent_res)
                    
                    coref_losses = []
                    temp_losses = []
                    causal_losses = []
                    subevent_losses = []
                    temporal_pred_list = []
                    temporal_label_list = []
                    causal_pred_list = []
                    causal_label_list = []
                    subevent_pred_list = []
                    subevent_label_list = []
                    coref_train_eval_results = []

                if glb_step % args.eval_steps == 0:
                    res = evaluate(model, dev_dataloader, desc="Validation")
                    better={"COREFERENCE":False, "TEMPORAL": False, "CAUSAL": False, "SUBEVENT": False}
                    for k in ["TEMPORAL", "CAUSAL", "SUBEVENT"]:
                        if res[k]["micro avg"]["f1-score"] > best_score[k]:
                            best_score[k] = res[k]["micro avg"]["f1-score"]
                            better[k]=True
                    for k in ["COREFERENCE", "TEMPORAL", "CAUSAL", "SUBEVENT"]:
                        if better[k]:
                            print("better %s!"%(k))
                            state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                            torch.save(state, os.path.join(output_dir, "best_%s"%(k)))
    sys.stdout.close()
import json
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import random
import copy

SUBEVENTREL2ID = {"NONE": 0, "subevent": 1}
COREFREL2ID = {"NONE": 0, "coref": 1}
CAUSALREL2ID = {"NONE": 0, "PRECONDITION": 1, "CAUSE": 2}
TEMPREL2ID = {
    "BEFORE": 0,
    "OVERLAP": 1,
    "CONTAINS": 2,
    "SIMULTANEOUS": 3,
    "ENDS-ON": 4,
    "BEGINS-ON": 5,
    "NONE": 6,
}

BIDIRECTIONAL_REL = ["SIMULTANEOUS", "BEGINS-ON"]

ID2TEMPREL = {v: k for k, v in TEMPREL2ID.items()}
ID2CAUSALREL = {v: k for k, v in CAUSALREL2ID.items()}
ID2COREFREL = {v: k for k, v in COREFREL2ID.items()}
ID2SUBEVENTREL = {v: k for k, v in SUBEVENTREL2ID.items()}

def valid_split(point, spans):
    for sp in spans:
        if point > sp[0] - 3 and point <= sp[1] + 3:
            return False
    return True

def split_spans(point, spans):
    part1, part2, i = [], [], 0
    for sp in spans:
        if sp[1] < point:
            part1.append(sp)
            i += 1
        else:
            break
    part2 = spans[i:]
    return part1, part2

def type_tokens(type_str):
    return [f"<{type_str}>", f"<{type_str}/>"]

class Document:
    def __init__(self, data, ignore_nonetype=False):
        self.id = data["id"]
        self.words = data["tokens"]
        self.mentions = []
        self.events = []
        self.eid2mentions = {}
        self.event_extra_info = {}
        self.use_extra_info = False
        if "events" in data:
            for e in data["events"]:
                e["mention"][0]["eid"] = e["id"]
                self.events += e["mention"]
            for e in data["events"]:
                self.eid2mentions[e["id"]] = e["mention"]
        else:
            self.events = copy.deepcopy(data["event_mentions"])

        self.events += data["TIMEX"]
        for t in data["TIMEX"]:
            self.eid2mentions[t["id"]] = [t]

        if "events" in data:
            self.temporal_relations = data["temporal_relations"]
            self.causal_relations = data["causal_relations"]
            self.subevent_relations = {"subevent": data["subevent_relations"]}
            self.coref_relations = self.load_coref_relations(data)
        else:
            self.temporal_relations = {}
            self.causal_relations = {}
            self.subevent_relations = {}
            self.coref_relations = {}
        self.sort_events()
        self.coref_labels = self.get_coref_labels(data)
        self.temporal_labels = self.get_relation_labels(self.temporal_relations, TEMPREL2ID, ignore_timex=False)
        self.causal_labels = self.get_relation_labels(self.causal_relations, CAUSALREL2ID, ignore_timex=True)
        self.subevent_labels = self.get_relation_labels(self.subevent_relations, SUBEVENTREL2ID, ignore_timex=True)

    def load_extra_info(self, extra_info_data):
        if not self.use_extra_info:
            return
        if extra_info_data and "events" in extra_info_data:
            for event in extra_info_data["events"]:
                trigger_word = event["trigger_word"]
                self.event_extra_info[trigger_word] = {
                    "timing_information": event["timing_information"],
                    "causal_relationships": event["causal_relationships"],
                    "event_structure": event["event_structure"],
                    "coreference_information": event["coreference_information"],
                }

    def load_coref_relations(self, data):
        relations = {}
        for event in data["events"]:
            for mention1 in event["mention"]:
                for mention2 in event["mention"]:
                    relations[(mention1["id"], mention2["id"])] = COREFREL2ID["coref"]
        return relations

    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.sorted_event_spans = [(event["sent_id"], event["offset"]) for event in self.events]

    def get_coref_labels(self, data):
        label_group = []
        events_only = [e for e in self.events if not e["id"].startswith("TIME")]
        self.events_idx = [i for i, e in enumerate(self.events) if not e["id"].startswith("TIME")]
        mid2index = {e["id"]: i for i, e in enumerate(events_only)}
        if "events" in data:
            for event in data["events"]:
                label_group.append([mid2index[m["id"]] for m in event["mention"]])
        else:
            for m in data["event_mentions"]:
                label_group.append([mid2index[m["id"]]])
        return label_group

    def get_relation_labels(self, relations, REL2ID, ignore_timex=True):
        pair2rel = {}
        for rel in relations:
            for pair in relations[rel]:
                for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        pair2rel[(e1["id"], e2["id"])] = REL2ID[rel]
                        if rel in BIDIRECTIONAL_REL:
                            pair2rel[(e2["id"], e1["id"])] = REL2ID[rel]
        labels = []
        for i, e1 in enumerate(self.events):
            for j, e2 in enumerate(self.events):
                if e1["id"] == e2["id"]:
                    continue
                if ignore_timex:
                    if e1["id"].startswith("TIME") or e2["id"].startswith("TIME"):
                        labels.append(-100)
                        continue
                labels.append(pair2rel.get((e1["id"], e2["id"]), REL2ID["NONE"]))
        assert len(labels) == len(self.events) ** 2 - len(self.events)
        return labels

class myDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, max_length=512, ignore_nonetype=False, sample_rate=None, use_extra_info=False):
        if sample_rate and split != "train":
            print("Sampling test or dev, is it intended?")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_nonetype = ignore_nonetype
        self.use_extra_info = use_extra_info
        self.load_examples(data_dir, split)
        if sample_rate:
            self.examples = list(random.sample(self.examples, int(sample_rate * len(self.examples))))
        self.tokenize()
        self.to_tensor()

    def load_examples(self, data_dir, split):
        self.examples = []
        with open(os.path.join(data_dir, f"{split}.jsonl")) as f:
            lines = f.readlines()
        extra_info = {}
        if self.use_extra_info:
            extra_info_path = {
                "train": "<path_to_train_extra_info>",
                "valid": "<path_to_valid_extra_info>",
                "test": "<path_to_test_extra_info>",
            }
            if split in extra_info_path:
                with open(extra_info_path[split]) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        extra_info[data["document_id"]] = data
        for line in lines:
            data = json.loads(line.strip())
            doc = Document(data, ignore_nonetype=self.ignore_nonetype)
            if doc.sorted_event_spans:
                self.examples.append(doc)

    def tokenize(self):
        self.tokenized_samples = []
        for example in tqdm(self.examples, desc="Tokenizing"):
 
            pass

    def to_tensor(self):
        for item in self.tokenized_samples:

            pass

    def __getitem__(self, index):
        return self.tokenized_samples[index]

    def __len__(self):
        return len(self.tokenized_samples)

def collator(data):

    pass

def get_dataloader(tokenizer, split, data_dir="<default_data_dir>", max_length=128, batch_size=8, shuffle=True, ignore_nonetype=False, sample_rate=None, use_extra_info=False):
    dataset = myDataset(tokenizer, data_dir, split, max_length=max_length, ignore_nonetype=ignore_nonetype, sample_rate=sample_rate, use_extra_info=use_extra_info)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("<path_to_model>")
    dataloader = get_dataloader(tokenizer, "test", shuffle=False, max_length=256)
    for data in dataloader:
        print(data["input_ids"].size())
        break

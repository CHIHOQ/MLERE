import torch
import os
import json
from tqdm import tqdm
from src.data import get_dataloader
from src.model import Model 
from src.utils import to_cuda, get_predicted_clusters, get_event2cluster
from src.dump_result import coref_dump, temporal_dump, causal_dump, subevent_dump
from transformers import RobertaTokenizer

def temp_predict(model, dataloader):
    """Predict temporal relations."""
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            _, scores, _, _ = model(data)
            labels = data["temporal_labels"]
            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)
            pred = torch.argmax(scores, dim=-1)
            max_label_length = data["max_temporal_label_length"]
            n_doc = len(labels) // max_label_length
            assert len(labels) % max_label_length == 0
            for i in range(n_doc):
                selected_index = labels[i*max_label_length:(i+1)*max_label_length] >= 0
                all_preds.append({
                    "doc_id": data["doc_id"][i],
                    "preds": pred[i*max_label_length:(i+1)*max_label_length][selected_index].cpu().numpy().tolist(),
                })
    return all_preds

def causal_predict(model, dataloader):
    """Predict causal relations."""
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            _, _, scores, _ = model(data)
            labels = data["causal_labels"]
            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)
            pred = torch.argmax(scores, dim=-1)
            max_label_length = data["max_causal_label_length"]
            if max_label_length:
                n_doc = len(labels) // max_label_length
                assert len(labels) % max_label_length == 0
                for i in range(n_doc):
                    selected_index = labels[i*max_label_length:(i+1)*max_label_length] >= 0
                    all_preds.append({
                        "doc_id": data["doc_id"][i],
                        "preds": pred[i*max_label_length:(i+1)*max_label_length][selected_index].cpu().numpy().tolist(),
                    })
    return all_preds

def subevent_predict(model, dataloader):
    """Predict subevent relations."""
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            _, _, _, scores = model(data)
            labels = data["subevent_labels"]
            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)
            pred = torch.argmax(scores, dim=-1)
            max_label_length = data["max_subevent_label_length"]
            n_doc = len(labels) // max_label_length
            assert len(labels) % max_label_length == 0
            for i in range(n_doc):
                selected_index = labels[i*max_label_length:(i+1)*max_label_length] >= -1 
                all_preds.append({
                    "doc_id": data["doc_id"][i],
                    "preds": pred[i*max_label_length:(i+1)*max_label_length][selected_index].cpu().numpy().tolist(),
                })
    return all_preds

def predict(output_dir, test_data_path, batch_size=4, ignore_nonetype=False):
    """
    Predict on the test data and save the results.
    
    Args:
        output_dir: Directory containing model checkpoints.
        test_data_path: Path to the test data file.
        batch_size: Batch size for prediction.
        ignore_nonetype: Whether to ignore "none" type labels.
    """
    tokenizer = RobertaTokenizer.from_pretrained("<path_to_pretrained_tokenizer>")
    test_dataloader = get_dataloader(tokenizer, "test", max_length=256, 
                                   shuffle=False, batch_size=batch_size,
                                   ignore_nonetype=ignore_nonetype)
    
    model = Model(len(tokenizer))
    model = to_cuda(model)
    
    dump_results = {}
    print("*" * 30 + "Test" + "*" * 30)
    
    for task in ["COREFERENCE", "TEMPORAL", "CAUSAL", "SUBEVENT"]:
        checkpoint_path = os.path.join(output_dir, f"best_{task}")
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["model"])
        

        if task == 'TEMPORAL':
            all_preds = temp_predict(model, test_dataloader)
            temporal_dump(test_data_path, all_preds, dump_results)
        elif task == 'CAUSAL':
            all_preds = causal_predict(model, test_dataloader)
            causal_dump(test_data_path, all_preds, dump_results)
        elif task == 'SUBEVENT':
            all_preds = subevent_predict(model, test_dataloader)
            subevent_dump(test_data_path, all_preds, dump_results)
    
    save_dir = "<path_to_save_directory>"
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "test_prediction.jsonl")
    
    with open(output_file, "w") as f:
        f.writelines("\n".join([json.dumps(dump_results[key]) for key in dump_results]))
    
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing model checkpoints.")
    parser.add_argument("--test_data_path", type=str, default="<path_to_test_data>", help="Path to the test data file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for prediction.")
    parser.add_argument("--ignore_nonetype", action="store_true", help="Whether to ignore 'none' type labels.")
    args = parser.parse_args()
    
    predict(args.output_dir, args.test_data_path, args.batch_size, args.ignore_nonetype)

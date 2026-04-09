import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def extract_bert_features_with_service(
    texts: list,
    utt_ids: list,
    model_name: str = "bert-base-uncased",
    port: int = 5555,
    port_out: int = 5556,
) -> dict:
    """bert-as-serviceを使ってBERT特徴量を抽出

    事前に以下のコマンドでBERTサーバーを起動しておく必要がある:
        bert-serving-start -model_name bert-base-uncased -port 5555 -port_out 5556
    """
    from bert_serving.client import BertClient

    bc = BertClient(ip="localhost", port=port, port_out=port_out)
    features_list = bc.encode(texts)

    features = {}
    for utt_id, feat in zip(utt_ids, features_list):
        features[utt_id] = torch.tensor(feat, dtype=torch.float32)

    return features


def reduce_dimensions(
    features: dict, target_dim: int = 378, random_state: int = 42
) -> dict:
    """PCAで次元削減 (768 -> 378)"""
    utt_ids = sorted(features.keys())
    matrix = np.stack([features[uid].numpy() for uid in utt_ids])

    pca = PCA(n_components=target_dim, random_state=random_state)
    reduced = pca.fit_transform(matrix)

    reduced_features = {}
    for utt_id, feat in zip(utt_ids, reduced):
        reduced_features[utt_id] = torch.tensor(feat, dtype=torch.float32)

    return reduced_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    parser.add_argument("--bert-port", type=int, default=5555)
    parser.add_argument("--bert-port-out", type=int, default=5556)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, os.getcwd())
    from utils.data_utils import (
        get_iemocap_utterances,
        get_meld_utterances,
        filter_by_agreement,
    )

    label_map = config["data"]["label_map"]
    exclude_labels = config["data"]["exclude_labels"]

    if args.dataset == "iemocap":
        utterances = get_iemocap_utterances(
            config["paths"]["iemocap_root"], label_map, exclude_labels
        )
        utterances = filter_by_agreement(utterances)
        texts = [u["text"] for u in utterances]
        utt_ids = [u["utt_id"] for u in utterances]
    else:
        splits = get_meld_utterances(
            config["paths"]["meld_root"], label_map, exclude_labels
        )
        all_utts = (
            splits.get("train", []) + splits.get("dev", []) + splits.get("test", [])
        )
        texts = [u["text"] for u in all_utts]
        utt_ids = [u["utt_id"] for u in all_utts]

    print(f"Extracting BERT features for {len(texts)} utterances ({args.dataset})...")
    print(f"Make sure bert-serving-start is running on port {args.bert_port}")

    features = extract_bert_features_with_service(
        texts, utt_ids, port=args.bert_port, port_out=args.bert_port_out
    )

    original_dim = list(features.values())[0].shape[0]
    target_dim = config["features"]["bert_reduce_dim"]

    if original_dim != target_dim:
        print(f"Reducing dimensions: {original_dim} -> {target_dim}")
        features = reduce_dimensions(
            features, target_dim, random_state=config["execution"]["seed"]
        )

    out_path = os.path.join(config["paths"]["features_dir"], f"bert_{args.dataset}.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features to {out_path}")


if __name__ == "__main__":
    main()

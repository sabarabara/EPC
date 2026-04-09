import os
import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import sys

sys.path.insert(0, os.getcwd())
from utils.data_utils import (
    get_iemocap_utterances,
    filter_by_agreement,
    assign_turns,
    get_iemocap_speakers,
    create_speaker_independent_folds,
    build_context_sequences,
    assign_roles,
    get_meld_utterances,
    assign_meld_turns,
)
from utils.seed import set_seed


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, modality="multi"):
        self.samples = samples
        self.modality = modality

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "context_features": sample["context_features"],
            "context_lengths": sample["context_lengths"],
            "context_speaker_ids": sample["context_speaker_ids"],
            "target_label": sample["target_label"],
            "roles": sample["roles"],
        }


def collate_fn(batch):
    """会話単位のバッチ作成 (ゼロパディング + mask)"""
    max_len = max(item["context_features"].shape[0] for item in batch)
    feat_dim = batch[0]["context_features"].shape[1]

    padded_features = []
    lengths = []
    speaker_ids = []
    labels = []
    roles_list = []

    for item in batch:
        feat = item["context_features"]
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            feat = torch.cat(
                [feat, torch.zeros(pad_len, feat_dim, dtype=feat.dtype)], dim=0
            )
        padded_features.append(feat)
        lengths.append(item["context_lengths"])
        speaker_ids.append(item["context_speaker_ids"])
        labels.append(item["target_label"])
        roles_list.append(item["roles"])

    return {
        "context_features": torch.stack(padded_features),
        "context_lengths": torch.tensor(lengths),
        "context_speaker_ids": speaker_ids,
        "target_label": torch.tensor(labels),
        "roles": roles_list,
    }


def compute_normalization_stats(samples):
    """訓練データの正規化統計量を計算"""
    all_features = []
    for sample in samples:
        all_features.append(sample["context_features"])

    all_features = torch.cat(all_features, dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    std[std < 1e-8] = 1.0

    return mean, std


def normalize_features(samples, mean, std):
    """特徴量を標準化"""
    for sample in samples:
        sample["context_features"] = (sample["context_features"] - mean) / std
    return samples


def build_iemocap_dataset(config):
    """IEMOCAPのデータセットを構築"""
    label_map = config["data"]["label_map"]
    exclude_labels = config["data"]["exclude_labels"]
    num_context = config["data"]["num_context_turns"]

    print("Loading IEMOCAP utterances...")
    utterances = get_iemocap_utterances(
        config["paths"]["iemocap_root"], label_map, exclude_labels
    )

    print("Filtering by evaluator agreement...")
    utterances = filter_by_agreement(utterances)
    print(f"  {len(utterances)} utterances after agreement filter")

    print("Assigning turns...")
    utterances = assign_turns(utterances)

    print("Loading features...")
    audio_feats = torch.load(
        os.path.join(config["paths"]["features_dir"], "audio_iemocap.pt"),
        weights_only=False,
    )
    text_feats = torch.load(
        os.path.join(config["paths"]["features_dir"], "bert_iemocap.pt"),
        weights_only=False,
    )

    modality_map = {
        "speech": lambda u: audio_feats.get(
            u["utt_id"], torch.zeros(856, dtype=torch.float32)
        ),
        "text": lambda u: text_feats.get(
            u["utt_id"], torch.zeros(378, dtype=torch.float32)
        ),
        "multi": lambda u: torch.cat(
            [
                audio_feats.get(u["utt_id"], torch.zeros(856, dtype=torch.float32)),
                text_feats.get(u["utt_id"], torch.zeros(378, dtype=torch.float32)),
            ],
            dim=0,
        ),
    }

    print("Building context sequences...")
    sequences = build_context_sequences(utterances, num_context)

    dialog_speakers = defaultdict(set)
    for utt in utterances:
        dialog_speakers[utt["dialog"]].add(utt["speaker"])
    sequences = assign_roles(sequences, dialog_speakers)

    print("Getting speakers and creating folds...")
    speakers = get_iemocap_speakers(config["paths"]["iemocap_root"])
    folds = create_speaker_independent_folds(speakers)

    print(f"Building {len(folds)} folds...")
    for fold_idx, fold in enumerate(folds):
        test_speaker = fold["test_speaker"]
        test_session = fold["test_session"]
        train_speakers = set(fold["train_speakers"])

        fold_sequences = [
            s
            for s in sequences
            if not (
                s["context_utts"][0]["session"] == test_session
                and any(u["speaker"] == test_speaker for u in [s["context_utts"][0]])
            )
            and not (
                any(
                    utt["session"] == test_session and utt["speaker"] == test_speaker
                    for utt in s["context_utts"]
                )
            )
            and not (
                s["context_utts"][-1]["session"] == test_session
                and s["target_speaker"] == test_speaker
            )
        ]

        if len(fold_sequences) == 0:
            print(f"  Fold {fold_idx}: no sequences, skipping")
            continue

        for modality in ["speech", "text", "multi"]:
            get_feat = modality_map[modality]

            fold_samples = []
            for seq in fold_sequences:
                context_feats = []
                context_speaker_ids = []

                for utt in seq["context_utts"]:
                    feat = get_feat(utt)
                    context_feats.append(feat)
                    context_speaker_ids.append(utt["speaker"])

                context_feats = torch.stack(context_feats)

                fold_samples.append(
                    {
                        "context_features": context_feats,
                        "context_lengths": len(context_feats),
                        "context_speaker_ids": context_speaker_ids,
                        "target_label": seq["target_label"],
                        "roles": seq["roles"],
                        "conv_id": seq["conv_id"],
                    }
                )

            split = int(len(fold_samples) * config["data"]["train_val_split"])
            train_fold = fold_samples[:split]
            val_fold = fold_samples[split:]

            if len(train_fold) == 0:
                print(f"  Fold {fold_idx} ({modality}): no train samples, skipping")
                continue

            mean, std = compute_normalization_stats(train_fold)
            train_fold = normalize_features(train_fold, mean, std)
            val_fold = normalize_features(val_fold, mean, std)

            out_path = os.path.join(
                config["paths"]["data_dir"], f"iemocap_fold{fold_idx}_{modality}.pt"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(
                {
                    "train": train_fold,
                    "val": val_fold,
                    "mean": mean,
                    "std": std,
                    "test_speaker": test_speaker,
                },
                out_path,
            )
            print(
                f"  Fold {fold_idx} ({modality}): "
                f"train={len(train_fold)}, val={len(val_fold)}"
            )

    print("IEMOCAP dataset build complete.")


def build_meld_dataset(config):
    """MELDのデータセットを構築"""
    label_map = config["data"]["label_map"]
    exclude_labels = config["data"]["exclude_labels"]
    num_context = config["data"]["num_context_turns"]

    print("Loading MELD utterances...")
    splits = get_meld_utterances(
        config["paths"]["meld_root"], label_map, exclude_labels
    )

    audio_feats = {}
    text_feats = {}
    for modality in ["speech", "text", "multi"]:
        if modality in ["speech", "multi"]:
            af_path = os.path.join(config["paths"]["features_dir"], "audio_meld.pt")
            if os.path.exists(af_path):
                audio_feats = torch.load(af_path, weights_only=False)
        if modality in ["text", "multi"]:
            tf_path = os.path.join(config["paths"]["features_dir"], "bert_meld.pt")
            if os.path.exists(tf_path):
                text_feats = torch.load(tf_path, weights_only=False)

    for split_name in ["train", "dev", "test"]:
        if split_name not in splits:
            continue

        utterances = assign_meld_turns(splits[split_name])
        sequences = build_context_sequences(utterances, num_context)

        dialog_speakers = defaultdict(set)
        for utt in utterances:
            dialog_speakers[utt["dialog_id"]].add(utt["speaker"])
        sequences = assign_roles(sequences, dialog_speakers)

        for modality in ["speech", "text", "multi"]:
            samples = []
            for seq in sequences:
                context_feats = []
                context_speaker_ids = []

                for utt in seq["context_utts"]:
                    if modality in ["speech", "multi"]:
                        af = audio_feats.get(
                            utt["utt_id"], torch.zeros(856, dtype=torch.float32)
                        )
                    else:
                        af = None

                    if modality in ["text", "multi"]:
                        tf = text_feats.get(
                            utt["utt_id"], torch.zeros(378, dtype=torch.float32)
                        )
                    else:
                        tf = None

                    if modality == "speech":
                        feat = af
                    elif modality == "text":
                        feat = tf
                    else:
                        feat = torch.cat([af, tf], dim=0)

                    context_feats.append(feat)
                    context_speaker_ids.append(utt["speaker"])

                context_feats = torch.stack(context_feats)

                samples.append(
                    {
                        "context_features": context_feats,
                        "context_lengths": len(context_feats),
                        "context_speaker_ids": context_speaker_ids,
                        "target_label": seq["target_label"],
                        "roles": seq["roles"],
                        "conv_id": seq["conv_id"],
                    }
                )

            if split_name == "train":
                split_ratio = config["data"]["train_val_split"]
                split_idx = int(len(samples) * split_ratio)
                train_samples = samples[:split_idx]
                val_samples = samples[split_idx:]

                if len(train_samples) == 0:
                    continue

                mean, std = compute_normalization_stats(train_samples)
                train_samples = normalize_features(train_samples, mean, std)
                val_samples = normalize_features(val_samples, mean, std)

                out_path = os.path.join(
                    config["paths"]["data_dir"], f"meld_train_{modality}.pt"
                )
                torch.save(
                    {
                        "train": train_samples,
                        "val": val_samples,
                        "mean": mean,
                        "std": std,
                    },
                    out_path,
                )
                print(
                    f"  MELD {modality}: train={len(train_samples)}, "
                    f"val={len(val_samples)}"
                )
            else:
                out_path = os.path.join(
                    config["paths"]["data_dir"], f"meld_{split_name}_{modality}.pt"
                )
                torch.save({"data": samples}, out_path)
                print(f"  MELD {split_name} {modality}: {len(samples)}")

    print("MELD dataset build complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--dataset", choices=["iemocap", "meld", "both"], default="iemocap"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["execution"]["seed"])

    if args.dataset in ["iemocap", "both"]:
        build_iemocap_dataset(config)
    if args.dataset in ["meld", "both"]:
        build_meld_dataset(config)


if __name__ == "__main__":
    main()

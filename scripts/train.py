import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.getcwd())
from utils.seed import set_seed
from utils.metrics import compute_all_metrics
from scripts.build_dataset import ConversationDataset, collate_fn


def get_model(model_name: str, input_size: int, config: dict):
    if model_name == "proposed":
        from models.proposed_model import ProposedModel

        return ProposedModel(
            input_size=input_size,
            hidden_size=config["model"]["gru_hidden_size"],
            num_layers=config["model"]["gru_num_layers"],
            dropout=config["model"]["gru_dropout"],
            num_classes=config["num_classes"],
        )
    elif model_name == "shahriar":
        from models.shahriar_model import ShahriarModel

        return ShahriarModel(
            input_size=input_size,
            hidden_size=config["model"]["gru_hidden_size"],
            num_layers=config["model"]["gru_num_layers"],
            dropout=config["model"]["gru_dropout"],
            num_classes=config["num_classes"],
        )
    elif model_name == "shi2020":
        from models.shi2020_model import Shi2020Model

        return Shi2020Model(
            input_size=input_size,
            hidden_size=config["model"]["gru_hidden_size"],
            num_layers=config["model"]["gru_num_layers"],
            dropout=config["model"]["gru_dropout"],
            num_classes=config["num_classes"],
        )
    elif model_name == "blstm":
        from models.blstm_model import BLSTMModel

        return BLSTMModel(
            input_size=input_size,
            hidden_size=config["model"]["gru_hidden_size"],
            num_layers=config["model"]["gru_num_layers"],
            dropout=config["model"]["gru_dropout"],
            num_classes=config["num_classes"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_class_weights(train_samples, num_classes: int) -> torch.Tensor:
    labels = [s["target_label"] for s in train_samples]
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model, dataloader, criterion, optimizer, device, model_name, num_classes=4
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        features = batch["context_features"].to(device)
        lengths = batch["context_lengths"]
        labels = batch["target_label"].to(device)

        kwargs = {}
        if model_name in ["proposed", "shi2020"]:
            kwargs["context_speaker_ids"] = batch["context_speaker_ids"]
            kwargs["roles"] = batch["roles"]

        optimizer.zero_grad()
        logits = model(features, lengths, **kwargs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(
        np.array(all_labels), np.array(all_preds), num_classes
    )
    return avg_loss, metrics


def evaluate(model, dataloader, criterion, device, model_name, num_classes=4):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            features = batch["context_features"].to(device)
            lengths = batch["context_lengths"]
            labels = batch["target_label"].to(device)

            kwargs = {}
            if model_name in ["proposed", "shi2020"]:
                kwargs["context_speaker_ids"] = batch["context_speaker_ids"]
                kwargs["roles"] = batch["roles"]

            logits = model(features, lengths, **kwargs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_all_metrics(
        np.array(all_labels), np.array(all_preds), num_classes
    )
    return avg_loss, metrics


def train_single_fold(model, train_data, val_data, config, device, model_name):
    num_classes = config["num_classes"]

    train_dataset = ConversationDataset(train_data)
    val_dataset = ConversationDataset(val_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["execution"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["execution"]["num_workers"],
    )

    class_weights = compute_class_weights(train_data, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None
    best_state = None

    for epoch in range(config["training"]["max_epochs"]):
        train_loss, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, model_name, num_classes
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, model_name, num_classes
        )

        print(
            f"  Epoch {epoch + 1}/{config['training']['max_epochs']} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val UAR: {val_metrics['uar']:.2f} | "
            f"Val F1: {val_metrics['macro_f1']:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return best_metrics


def train_iemocap(config, model_name, modality, device):
    modality_dim_map = {
        "speech": config["modality"]["audio_dim"],
        "text": config["modality"]["text_dim"],
        "multi": config["modality"]["multimodal_dim"],
    }
    input_size = modality_dim_map[modality]

    data_dir = config["paths"]["data_dir"]
    fold_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.startswith(f"iemocap_fold") and f.endswith(f"_{modality}.pt")
        ]
    )

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} | Modality: {modality} | Folds: {len(fold_files)}")
    print(f"{'=' * 60}")

    all_results = []
    for fold_file in fold_files:
        fold_idx = fold_file.split("fold")[1].split("_")[0]
        print(f"\n--- Fold {fold_idx} ---")

        data = torch.load(os.path.join(data_dir, fold_file), weights_only=False)
        train_data = data["train"]
        val_data = data["val"]

        set_seed(config["execution"]["seed"])
        model = get_model(model_name, input_size, config).to(device)

        metrics = train_single_fold(
            model, train_data, val_data, config, device, model_name
        )
        print(
            f"  Fold {fold_idx} Best - UAR: {metrics['uar']:.2f}, "
            f"F1: {metrics['macro_f1']:.2f}"
        )
        all_results.append(metrics)

    avg_uar = np.mean([r["uar"] for r in all_results])
    std_uar = np.std([r["uar"] for r in all_results])
    avg_f1 = np.mean([r["macro_f1"] for r in all_results])
    std_f1 = np.std([r["macro_f1"] for r in all_results])

    print(f"\n{'=' * 60}")
    print(
        f"Average - UAR: {avg_uar:.2f}+/-{std_uar:.2f}, F1: {avg_f1:.2f}+/-{std_f1:.2f}"
    )
    print(f"{'=' * 60}")

    return {
        "fold_results": all_results,
        "avg_uar": avg_uar,
        "std_uar": std_uar,
        "avg_f1": avg_f1,
        "std_f1": std_f1,
    }


def train_meld(config, model_name, modality, device):
    modality_dim_map = {
        "speech": config["modality"]["audio_dim"],
        "text": config["modality"]["text_dim"],
        "multi": config["modality"]["multimodal_dim"],
    }
    input_size = modality_dim_map[modality]

    data_dir = config["paths"]["data_dir"]
    train_file = os.path.join(data_dir, f"meld_train_{modality}.pt")
    test_file = os.path.join(data_dir, f"meld_test_{modality}.pt")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"MELD data files not found for {modality}")
        return None

    train_data = torch.load(train_file, weights_only=False)["train"]
    val_data = torch.load(train_file, weights_only=False)["val"]
    test_data = torch.load(test_file, weights_only=False)["data"]

    set_seed(config["execution"]["seed"])
    model = get_model(model_name, input_size, config).to(device)

    print(f"\n{'=' * 60}")
    print(f"MELD - Model: {model_name} | Modality: {modality}")
    print(f"{'=' * 60}")

    metrics = train_single_fold(model, train_data, val_data, config, device, model_name)

    num_classes = config["num_classes"]
    test_dataset = ConversationDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    class_weights = compute_class_weights(train_data, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, model_name, num_classes
    )

    print(
        f"  Test - UAR: {test_metrics['uar']:.2f}, F1: {test_metrics['macro_f1']:.2f}"
    )

    return {"test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    parser.add_argument(
        "--model",
        choices=["proposed", "shahriar", "shi2020", "blstm"],
        default="proposed",
    )
    parser.add_argument(
        "--modality", choices=["speech", "text", "multi"], default="multi"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["execution"]["seed"])
    device = torch.device(config["execution"]["device"])

    if args.dataset == "iemocap":
        result = train_iemocap(config, args.model, args.modality, device)
    else:
        result = train_meld(config, args.model, args.modality, device)

    if result is not None:
        out_dir = config["paths"]["results_dir"]
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir, f"{args.dataset}_{args.model}_{args.modality}.pt"
        )
        torch.save(result, out_file)
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

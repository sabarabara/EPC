import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import glob


import librosa
def load_audio(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    try:
        import warnings
        warnings.filterwarnings("ignore")
        y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        return torch.tensor(y, dtype=torch.float32)
    except Exception as e:
        print(f"Warning: Failed to load {wav_path}: {e}")
        return torch.zeros(target_sr, dtype=torch.float32)


def extract_wav2vec2_features(model, processor, wav_path: str, device) -> torch.Tensor:
    waveform = load_audio(wav_path)
    inputs = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs.to(device))
        hidden_states = outputs.last_hidden_state

    features = hidden_states.mean(dim=1).squeeze(0)
    return features.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["execution"]["device"])
    model_name = config["features"]["wav2vec2_model"]

    print(f"Loading {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    print(f"Extracting Wav2Vec2 features for {args.dataset}...")

    if args.dataset == "iemocap":
        wav_files = sorted(
            glob.glob(
                os.path.join(
                    config["paths"]["iemocap_root"],
                    "Session*",
                    "sentences",
                    "wav",
                    "*",
                    "*.wav",
                )
            )
        )
    else:
        wav_files = sorted(
            glob.glob(os.path.join(config["paths"]["meld_root"], "**", "*.wav"), recursive=True) + 
            glob.glob(os.path.join(config["paths"]["meld_root"], "**", "*.mp4"), recursive=True)
        )

    print(f"Found {len(wav_files)} audio files")

    features = {}
    for wav_path in tqdm(wav_files):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        feat = extract_wav2vec2_features(model, processor, wav_path, device)
        features[utt_id] = feat

    out_path = os.path.join(
        config["paths"]["features_dir"], f"wav2vec2_{args.dataset}.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features to {out_path}")


if __name__ == "__main__":
    main()

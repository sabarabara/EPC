import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import subprocess
import tempfile
import glob


import librosa
def resample_audio(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
    return torch.tensor(y, dtype=torch.float32).unsqueeze(0)


import opensmile

def extract_egemaps_with_opensmile(
    wav_path: str, config_name: str = "eGeMAPSv01a"
) -> np.ndarray:
    try:
        import warnings
        warnings.filterwarnings("ignore")
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv01a,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        # Handle empty/short audio preventing crash
        if len(y) == 0:
            return np.zeros(88, dtype=np.float32)
        features = smile.process_signal(y, sr)
        return features.iloc[0].values.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load {wav_path}: {e}")
        return np.zeros(88, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Extracting eGeMAPS features for {args.dataset}...")

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
        feat = extract_egemaps_with_opensmile(wav_path)
        features[utt_id] = torch.tensor(feat, dtype=torch.float32)

    out_path = os.path.join(
        config["paths"]["features_dir"], f"egemaps_{args.dataset}.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features to {out_path}")


if __name__ == "__main__":
    main()

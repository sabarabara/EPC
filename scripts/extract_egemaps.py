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


def resample_audio(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def extract_egemaps_with_opensmile(
    wav_path: str, config_name: str = "eGeMAPSv01a"
) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    waveform = resample_audio(wav_path)
    torchaudio.save(tmp_wav, waveform, 16000)

    try:
        result = subprocess.run(
            [
                "SMILExtract",
                "-C",
                f"config/{config_name}.conf",
                "-I",
                tmp_wav,
                "-csvoutput",
                "stdout",
                "-noconsoleoutput",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            values = lines[-1].split(";")[1:]
            features = np.array([float(v) for v in values if v != ""])
            return features
    except Exception as e:
        print(f"Error extracting {wav_path}: {e}")
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    return np.zeros(88)


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
            glob.glob(
                os.path.join(config["paths"]["meld_root"], "**", "*.wav"),
                recursive=True,
            )
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

# EPC Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shi et al. (2023) INTERSPEECH の感情予測モデルを完全再現し、IEMOCAP (10-fold CV) と MELD で評価する。

**Architecture:** フェーズ分割型パイプライン (特徴量抽出 → データ構築 → 学習・評価)。Interaction GRU → Dialog Management Unit → Individual GRU + Self-Attention → FC + Softmax の階層モデル。比較モデル3種 (Shahriar, Shi 2020, BLSTM) も実装。

**Tech Stack:** Python 3.9-3.11, PyTorch 2.x, openSMILE 3.x, wav2vec2-base-960h (HuggingFace), bert-as-service (Python 3.7), scikit-learn, torchaudio, librosa, pyyaml

---

## File Structure

```
EPC/
├── configs/
│   └── config.yaml
├── scripts/
│   ├── extract_egemaps.py
│   ├── extract_wav2vec2.py
│   ├── extract_bert.py
│   ├── build_dataset.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── __init__.py
│   ├── proposed_model.py
│   ├── shahriar_model.py
│   ├── shi2020_model.py
│   └── blstm_model.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── seed.py
├── data_raw/
│   ├── IEMOCAP/.gitkeep
│   └── MELD/.gitkeep
├── features/.gitkeep
├── data/.gitkeep
├── results/.gitkeep
├── requirements.txt
└── docs/superpowers/specs/2026-04-09-epc-reproduction-design.md
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `configs/config.yaml`
- Create: `data_raw/IEMOCAP/.gitkeep`
- Create: `data_raw/MELD/.gitkeep`
- Create: `features/.gitkeep`
- Create: `data/.gitkeep`
- Create: `results/.gitkeep`
- Create: `models/__init__.py`
- Create: `utils/__init__.py`

- [ ] **Step 1: Create directory structure and .gitkeep files**

```bash
mkdir -p configs scripts models utils data_raw/IEMOCAP data_raw/MELD features data results
touch data_raw/IEMOCAP/.gitkeep data_raw/MELD/.gitkeep features/.gitkeep data/.gitkeep results/.gitkeep
touch models/__init__.py utils/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.0
torchaudio>=2.0
transformers>=4.30
librosa>=0.10
scikit-learn>=1.3
numpy>=1.24
pyyaml>=6.0
tqdm>=4.65
```

- [ ] **Step 3: Create configs/config.yaml**

```yaml
# データパス (後で設定)
paths:
  iemocap_root: "data_raw/IEMOCAP/IEMOCAP_full_release"
  meld_root: "data_raw/MELD/MELD"
  features_dir: "features"
  data_dir: "data"
  results_dir: "results"

# 特徴量抽出
features:
  egemaps_version: "eGeMAPSv01a"
  wav2vec2_model: "facebook/wav2vec2-base-960h"
  wav2vec2_pooling: "mean"
  bert_model: "bert-base-uncased"
  bert_reduce_dim: 378
  sample_rate: 16000

# データ構築
data:
  num_context_turns: 6
  label_map:
    happy: 0
    excited: 0
    anger: 1
    angry: 1
    neutral: 2
    sad: 3
    sadness: 3
    joy: 0
    disgust: -1
    fear: -1
    surprise: -1
  exclude_labels: [-1]
  train_val_split: 0.8

# モデル
model:
  gru_hidden_size: 256
  gru_num_layers: 2
  gru_dropout: 0.5
  attention_dim: 256

# 学習
training:
  batch_size: 32
  max_epochs: 50
  learning_rate: 0.0001
  patience: 10
  use_class_weights: true

# 実行
execution:
  seed: 42
  device: "cuda"
  num_workers: 4

# モダリティ
modality:
  # speech: 入力次元 856 (egemaps 88 + wav2vec2 768)
  # text: 入力次元 378 (bert)
  # multi: 入力次元 1234 (856 + 378)
  audio_dim: 856
  text_dim: 378
  multimodal_dim: 1234

# 感情クラス
emotion_labels: ["Happy", "Anger", "Neutral", "Sad"]
num_classes: 4
```

- [ ] **Step 4: Verify structure**

```bash
ls -la configs/ scripts/ models/ utils/ data_raw/ features/ data/ results/
cat configs/config.yaml | head -5
```

Expected: 全ディレクトリとファイルが存在

- [ ] **Step 5: Commit**

```bash
git init
git add -A
git commit -m "chore: project setup - directory structure, config, requirements"
```

---

### Task 2: Seed Utility

**Files:**
- Create: `utils/seed.py`

- [ ] **Step 1: Create utils/seed.py**

```python
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "from utils.seed import set_seed; set_seed(42); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utils/seed.py
git commit -m "feat: add seed utility for reproducibility"
```

---

### Task 3: Metrics Utility

**Files:**
- Create: `utils/metrics.py`

- [ ] **Step 1: Create utils/metrics.py**

```python
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred) * 100


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro') * 100


def compute_class_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list:
    recalls = []
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append((y_pred[mask] == c).sum() / mask.sum() * 100)
    return recalls


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    return {
        'uar': compute_uar(y_true, y_pred),
        'macro_f1': compute_macro_f1(y_true, y_pred),
        'class_recall': compute_class_recall(y_true, y_pred, num_classes),
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred),
    }
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "
from utils.metrics import compute_all_metrics
import numpy as np
y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
y_pred = np.array([0, 1, 2, 3, 0, 0, 2, 3])
result = compute_all_metrics(y_true, y_pred, 4)
print(f'UAR: {result[\"uar\"]:.2f}, Macro F1: {result[\"macro_f1\"]:.2f}')
"
```

Expected: `UAR: 87.50, Macro F1: 87.50`

- [ ] **Step 3: Commit**

```bash
git add utils/metrics.py
git commit -m "feat: add evaluation metrics (UAR, Macro F1, confusion matrix)"
```

---

### Task 4: IEMOCAP Data Loader

**Files:**
- Create: `utils/data_utils.py`

- [ ] **Step 1: Create IEMOCAP data loader in utils/data_utils.py**

```python
import os
import re
import glob
import torch
import numpy as np
from collections import defaultdict


def parse_emoevaluation_file(filepath: str) -> list:
    """EmoEvaluationファイルをパースして発話ごとのラベルを取得"""
    utterances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('['):
                match = re.match(
                    r'\[.+\]\s+(\S+)\s+.*;\s+(\S+)\s+;',
                    line
                )
                if match:
                    utt_id = match.group(1)
                    emotion = match.group(2)
                    utterances.append({'utt_id': utt_id, 'emotion': emotion})
    return utterances


def get_iemocap_utterances(iemocap_root: str, label_map: dict,
                           exclude_labels: list) -> list:
    """IEMOCAPの全発話情報を取得

    Returns:
        list of dict: {
            'utt_id': str,
            'session': str,
            'dialog': str,
            'speaker': str,
            'emotion': int,
            'emotion_name': str,
            'wav_path': str,
            'text': str,
        }
    """
    all_utterances = []
    eval_dirs = sorted(glob.glob(
        os.path.join(iemocap_root, 'Session*', 'dialog', '*', 'EmoEvaluation')
    ))

    for eval_dir in eval_dirs:
        dialog_name = os.path.basename(os.path.dirname(eval_dir))
        session = os.path.basename(
            os.path.dirname(os.path.dirname(eval_dir))
        )

        eval_files = glob.glob(os.path.join(eval_dir, '*.txt'))
        for eval_file in eval_files:
            parsed = parse_emoevaluation_file(eval_file)
            for utt_info in parsed:
                utt_id = utt_info['utt_id']
                emotion_name = utt_info['emotion']

                if emotion_name not in label_map:
                    continue
                label = label_map[emotion_name]
                if label in exclude_labels:
                    continue

                speaker = utt_id.split('_')[0]

                wav_path = os.path.join(
                    iemocap_root, session, 'sentences', 'wav',
                    dialog_name, utt_id + '.wav'
                )

                transcript_path = os.path.join(
                    iemocap_root, session, 'dialog', 'transcriptions',
                    dialog_name + '.txt'
                )

                text = ''
                if os.path.exists(transcript_path):
                    text = _extract_utterance_text(
                        transcript_path, utt_id
                    )

                all_utterances.append({
                    'utt_id': utt_id,
                    'session': session,
                    'dialog': dialog_name,
                    'speaker': speaker,
                    'emotion': label,
                    'emotion_name': emotion_name,
                    'wav_path': wav_path,
                    'text': text,
                })

    return all_utterances


def _extract_utterance_text(transcript_path: str, utt_id: str) -> str:
    """transcriptionsファイルから特定の発話テキストを抽出"""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.escape(utt_id) + r':\s+(.*)'
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return ''


def filter_by_agreement(utterances: list) -> list:
    """評価者間で一致した発話のみを残す

    IEMOCAPでは同一発話に2人の評価者がラベルを付与。
    EmoEvaluationファイル内の同一utt_idが2回出現し、
    両方が同じラベルの場合のみ採用。
    """
    utt_labels = defaultdict(list)
    for utt in utterances:
        utt_labels[utt['utt_id']].append(utt['emotion'])

    agreed_ids = set()
    for utt_id, labels in utt_labels.items():
        if len(labels) == 2 and labels[0] == labels[1]:
            agreed_ids.add(utt_id)

    return [u for u in utterances if u['utt_id'] in agreed_ids]


def assign_turns(utterances: list) -> list:
    """連続する同一話者のutteranceを同一ターンとしてターンIDを付与

    各dialog内で、話者が切り替わるごとに新しいターンとする。
    """
    dialogs = defaultdict(list)
    for utt in utterances:
        dialogs[utt['dialog']].append(utt)

    result = []
    for dialog_id, dialog_utts in dialogs.items():
        dialog_utts.sort(key=lambda x: x['utt_id'])
        turn_id = 0
        for i, utt in enumerate(dialog_utts):
            if i > 0 and dialog_utts[i]['speaker'] != dialog_utts[i-1]['speaker']:
                turn_id += 1
            utt['turn_idx'] = turn_id
            result.append(utt)

    return result


def get_iemocap_speakers(iemocap_root: str) -> list:
    """IEMOCAPの全話者リストを取得 (10人)

    Returns:
        list of dict: [{'speaker_id': str, 'session': str}, ...]
    """
    speakers = []
    for session in sorted(glob.glob(os.path.join(iemocap_root, 'Session*'))):
        session_name = os.path.basename(session)
        for dialog in sorted(glob.glob(os.path.join(session, 'dialog', '*'))):
            dialog_name = os.path.basename(dialog)
            eval_file = glob.glob(os.path.join(dialog, 'EmoEvaluation', '*.txt'))[0]
            parsed = parse_emoevaluation_file(eval_file)
            seen = set()
            for utt_info in parsed:
                speaker = utt_info['utt_id'].split('_')[0]
                if speaker not in seen:
                    seen.add(speaker)
                    speakers.append({
                        'speaker_id': speaker,
                        'session': session_name,
                    })
    return speakers


def create_speaker_independent_folds(speakers: list, num_folds: int = 10) -> list:
    """話者単位のfold分割を作成

    Args:
        speakers: 話者リスト (10人想定)
        num_folds: fold数 (10)

    Returns:
        list of dict: [{'test_speaker': str, 'train_speakers': list}, ...]
    """
    folds = []
    for i in range(min(num_folds, len(speakers))):
        folds.append({
            'test_speaker': speakers[i]['speaker_id'],
            'test_session': speakers[i]['session'],
            'train_speakers': [s['speaker_id'] for s in speakers if s != speakers[i]],
        })
    return folds


def build_context_sequences(utterances: list, num_context_turns: int = 6) -> list:
    """コンテキストシーケンスを構築

    直前num_context_turnsターンの全utteranceを文脈とし、
    直後のターンの感情を予測ターゲットとする。

    Returns:
        list of dict: [{
            'context_utts': list of utt dicts,
            'target_label': int,
            'target_speaker': str,
            'conv_id': str,
        }, ...]
    """
    dialogs = defaultdict(list)
    for utt in utterances:
        dialogs[utt['dialog']].append(utt)

    sequences = []
    for dialog_id, dialog_utts in dialogs.items():
        dialog_utts.sort(key=lambda x: (x['turn_idx'], x['utt_id']))

        turns = defaultdict(list)
        for utt in dialog_utts:
            turns[utt['turn_idx']].append(utt)

        sorted_turns = sorted(turns.keys())

        for i in range(num_context_turns, len(sorted_turns)):
            context_turn_indices = sorted_turns[i - num_context_turns:i]
            target_turn_idx = sorted_turns[i]

            context_utts = []
            for t_idx in context_turn_indices:
                context_utts.extend(turns[t_idx])

            target_utts = turns[target_turn_idx]
            target_speaker = target_utts[0]['speaker']
            target_label = target_utts[0]['emotion']

            sequences.append({
                'context_utts': context_utts,
                'target_label': target_label,
                'target_speaker': target_speaker,
                'conv_id': dialog_id,
                'context_turn_indices': context_turn_indices,
                'target_turn_idx': target_turn_idx,
            })

    return sequences


def assign_roles(sequences: list, dialog_speakers: dict) -> list:
    """Dialog Management Unitのための役割割り当て

    Args:
        sequences: コンテキストシーケンスリスト
        dialog_speakers: {dialog_id: set of speaker_ids}

    Returns:
        各sequenceに 'speaker', 'interlocutor', 'spectators' を追加
    """
    for seq in sequences:
        conv_id = seq['conv_id']
        speaker = seq['target_speaker']
        all_speakers = dialog_speakers.get(conv_id, set())

        context_speakers = set()
        for utt in seq['context_utts']:
            context_speakers.add(utt['speaker'])

        others = context_speakers - {speaker}

        if len(others) == 0:
            interlocutor = None
            spectators = set()
        elif len(all_speakers) <= 2:
            interlocutor = list(others)[0]
            spectators = set()
        else:
            speaker_counts = defaultdict(int)
            for utt in seq['context_utts']:
                if utt['speaker'] != speaker:
                    speaker_counts[utt['speaker']] += 1
            interlocutor = max(speaker_counts, key=speaker_counts.get)
            spectators = others - {interlocutor}

        seq['roles'] = {
            'speaker': speaker,
            'interlocutor': interlocutor,
            'spectators': spectators,
        }

    return sequences
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "from utils.data_utils import get_iemocap_speakers, create_speaker_independent_folds; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utils/data_utils.py
git commit -m "feat: add IEMOCAP data loader with turn definition and fold splitting"
```

---

### Task 5: MELD Data Loader

**Files:**
- Modify: `utils/data_utils.py`

- [ ] **Step 1: Add MELD data loader to utils/data_utils.py**

`utils/data_utils.py` の末尾に以下を追加:

```python
import pandas as pd


def get_meld_utterances(meld_root: str, label_map: dict,
                        exclude_labels: list) -> dict:
    """MELDの全発話情報を取得

    Returns:
        dict: {'train': list, 'dev': list, 'test': list}
    """
    splits = {}
    for split in ['train', 'dev', 'test']:
        csv_path = os.path.join(meld_root, split, f'{split}_sent_emo.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {split}")
            continue

        df = pd.read_csv(csv_path)
        utterances = []

        for _, row in df.iterrows():
            emotion = row['Emotion'].strip().lower()
            if emotion not in label_map:
                continue
            label = label_map[emotion]
            if label in exclude_labels:
                continue

            utterances.append({
                'utt_id': f"{row['Dialogue_ID']}_{row['Utterance_ID']}",
                'dialog_id': str(row['Dialogue_ID']),
                'speaker': row['Speaker'].strip(),
                'emotion': label,
                'emotion_name': emotion,
                'text': str(row['Utterance']).strip(),
                'utterance_idx': row['Utterance_ID'],
            })

        splits[split] = utterances

    return splits


def assign_meld_turns(utterances: list) -> list:
    """MELDのターン定義: 連続する同一話者のutteranceを同一ターン"""
    dialogs = defaultdict(list)
    for utt in utterances:
        dialogs[utt['dialog_id']].append(utt)

    result = []
    for dialog_id, dialog_utts in dialogs.items():
        dialog_utts.sort(key=lambda x: x['utterance_idx'])
        turn_id = 0
        for i, utt in enumerate(dialog_utts):
            if i > 0 and dialog_utts[i]['speaker'] != dialog_utts[i-1]['speaker']:
                turn_id += 1
            utt['turn_idx'] = turn_id
            result.append(utt)

    return result
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "from utils.data_utils import get_meld_utterances, assign_meld_turns; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utils/data_utils.py
git commit -m "feat: add MELD data loader with 7-to-4 class mapping"
```

---

### Task 6: eGeMAPS Feature Extraction Script

**Files:**
- Create: `scripts/extract_egemaps.py`

- [ ] **Step 1: Create scripts/extract_egemaps.py**

```python
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
    """オーディオを16kHzモノラルに変換"""
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def extract_egemaps_with_opensmile(wav_path: str, config_name: str = 'eGeMAPSv01a') -> np.ndarray:
    """openSMILEでeGeMAPS特徴量を抽出"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name

    waveform = resample_audio(wav_path)
    torchaudio.save(tmp_wav, waveform, 16000)

    try:
        result = subprocess.run(
            ['SMILExtract', '-C', f'config/{config_name}.conf',
             '-I', tmp_wav, '-csvoutput', 'stdout', '-noconsoleoutput'],
            capture_output=True, text=True, timeout=30
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            values = lines[-1].split(';')[1:]
            features = np.array([float(v) for v in values if v != ''])
            return features
    except Exception as e:
        print(f"Error extracting {wav_path}: {e}")
    finally:
        os.unlink(tmp_wav)

    return np.zeros(88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', choices=['iemocap', 'meld'], default='iemocap')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Extracting eGeMAPS features for {args.dataset}...")

    if args.dataset == 'iemocap':
        wav_files = sorted(glob.glob(
            os.path.join(config['paths']['iemocap_root'],
                         'Session*', 'sentences', 'wav', '*', '*.wav')
        ))
    else:
        wav_dir = os.path.join(config['paths']['meld_root'], 'train')
        if os.path.exists(wav_dir):
            wav_files = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
        else:
            wav_files = sorted(glob.glob(
                os.path.join(config['paths']['meld_root'], '**', '*.wav'),
                recursive=True
            ))

    print(f"Found {len(wav_files)} audio files")

    features = {}
    for wav_path in tqdm(wav_files):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        feat = extract_egemaps_with_opensmile(wav_path)
        features[utt_id] = torch.tensor(feat, dtype=torch.float32)

    out_path = os.path.join(
        config['paths']['features_dir'], f'egemaps_{args.dataset}.pt'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features to {out_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "import scripts.extract_egemaps; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_egemaps.py
git commit -m "feat: add eGeMAPS feature extraction script"
```

---

### Task 7: Wav2Vec2 Feature Extraction Script

**Files:**
- Create: `scripts/extract_wav2vec2.py`

- [ ] **Step 1: Create scripts/extract_wav2vec2.py**

```python
import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import glob


def load_audio(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    """オーディオを16kHzモノラルで読み込み"""
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def extract_wav2vec2_features(model, processor, wav_path: str) -> torch.Tensor:
    """Wav2Vec2で768次元特徴量を抽出 (最終レイヤー平均プーリング)"""
    waveform = load_audio(wav_path)
    inputs = processor(waveform.numpy(), sampling_rate=16000,
                       return_tensors='pt', padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

    features = hidden_states.mean(dim=1).squeeze(0)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', choices=['iemocap', 'meld'], default='iemocap')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config['execution']['device'])
    model_name = config['features']['wav2vec2_model']

    print(f"Loading {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    print(f"Extracting Wav2Vec2 features for {args.dataset}...")

    if args.dataset == 'iemocap':
        wav_files = sorted(glob.glob(
            os.path.join(config['paths']['iemocap_root'],
                         'Session*', 'sentences', 'wav', '*', '*.wav')
        ))
    else:
        wav_files = sorted(glob.glob(
            os.path.join(config['paths']['meld_root'], '**', '*.wav'),
            recursive=True
        ))

    print(f"Found {len(wav_files)} audio files")

    features = {}
    for wav_path in tqdm(wav_files):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        feat = extract_wav2vec2_features(model, processor, wav_path)
        features[utt_id] = feat.cpu()

    out_path = os.path.join(
        config['paths']['features_dir'], f'wav2vec2_{args.dataset}.pt'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features to {out_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "import scripts.extract_wav2vec2; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_wav2vec2.py
git commit -m "feat: add Wav2Vec2 feature extraction script"
```

---

### Task 8: BERT Feature Extraction Script (Python 3.7)

**Files:**
- Create: `scripts/extract_bert.py`

- [ ] **Step 1: Create scripts/extract_bert.py**

```python
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def extract_bert_features_with_service(texts: list, utt_ids: list,
                                       model_name: str = 'bert-base-uncased',
                                       port: int = 5555,
                                       port_out: int = 5556) -> dict:
    """bert-as-serviceを使ってBERT特徴量を抽出

    事前に以下のコマンドでBERTサーバーを起動しておく必要がある:
        bert-serving-start -model_name bert-base-uncased -port 5555 -port_out 5556
    """
    from bert_serving.client import BertClient

    bc = BertClient(ip='localhost', port=port, port_out=port_out)
    features_list = bc.encode(texts)

    features = {}
    for utt_id, feat in zip(utt_ids, features_list):
        features[utt_id] = torch.tensor(feat, dtype=torch.float32)

    return features


def reduce_dimensions(features: dict, target_dim: int = 378,
                      random_state: int = 42) -> dict:
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
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', choices=['iemocap', 'meld'], default='iemocap')
    parser.add_argument('--bert-port', type=int, default=5555)
    parser.add_argument('--bert-port-out', type=int, default=5556)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, os.getcwd())
    from utils.data_utils import get_iemocap_utterances, get_meld_utterances

    label_map = config['data']['label_map']
    exclude_labels = config['data']['exclude_labels']

    if args.dataset == 'iemocap':
        utterances = get_iemocap_utterances(
            config['paths']['iemocap_root'], label_map, exclude_labels
        )
        from utils.data_utils import filter_by_agreement
        utterances = filter_by_agreement(utterances)
        texts = [u['text'] for u in utterances]
        utt_ids = [u['utt_id'] for u in utterances]
    else:
        splits = get_meld_utterances(
            config['paths']['meld_root'], label_map, exclude_labels
        )
        all_utts = splits.get('train', []) + splits.get('dev', []) + splits.get('test', [])
        texts = [u['text'] for u in all_utts]
        utt_ids = [u['utt_id'] for u in all_utts]

    print(f"Extracting BERT features for {len(texts)} utterances ({args.dataset})...")
    print(f"Make sure bert-serving-start is running on port {args.bert_port}")

    features = extract_bert_features_with_service(
        texts, utt_ids, port=args.bert_port, port_out=args.bert_port_out
    )

    original_dim = list(features.values())[0].shape[0]
    target_dim = config['features']['bert_reduce_dim']

    if original_dim != target_dim:
        print(f"Reducing dimensions: {original_dim} -> {target_dim}")
        features = reduce_dimensions(
            features, target_dim, random_state=config['execution']['seed']
        )

    out_path = os.path.join(
        config['paths']['features_dir'], f'bert_{args.dataset}.pt'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features to {out_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "import ast; ast.parse(open('scripts/extract_bert.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_bert.py
git commit -m "feat: add BERT feature extraction script with PCA dimension reduction"
```

---

### Task 9: Dataset Building Script

**Files:**
- Create: `scripts/build_dataset.py`

- [ ] **Step 1: Create scripts/build_dataset.py**

```python
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
    get_iemocap_utterances, filter_by_agreement, assign_turns,
    get_iemocap_speakers, create_speaker_independent_folds,
    build_context_sequences, assign_roles, get_meld_utterances, assign_meld_turns
)
from utils.seed import set_seed


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, modality='multi'):
        self.samples = samples
        self.modality = modality

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample['context_features']
        lengths = sample['context_lengths']
        speaker_ids = sample['context_speaker_ids']
        label = sample['target_label']
        roles = sample['roles']

        return {
            'context_features': context,
            'context_lengths': lengths,
            'context_speaker_ids': speaker_ids,
            'target_label': label,
            'roles': roles,
        }


def collate_fn(batch):
    """会話単位のバッチ作成 (ゼロパディング + mask)"""
    max_len = max(item['context_features'].shape[0] for item in batch)
    feat_dim = batch[0]['context_features'].shape[1]

    padded_features = []
    lengths = []
    speaker_ids = []
    labels = []
    roles_list = []

    for item in batch:
        feat = item['context_features']
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            feat = torch.cat([
                feat,
                torch.zeros(pad_len, feat_dim, dtype=feat.dtype)
            ], dim=0)
        padded_features.append(feat)
        lengths.append(item['context_lengths'])
        speaker_ids.append(item['context_speaker_ids'])
        labels.append(item['target_label'])
        roles_list.append(item['roles'])

    return {
        'context_features': torch.stack(padded_features),
        'context_lengths': torch.tensor(lengths),
        'context_speaker_ids': speaker_ids,
        'target_label': torch.tensor(labels),
        'roles': roles_list,
    }


def compute_normalization_stats(samples, modality='multi'):
    """訓練データの正規化統計量を計算"""
    all_features = []
    for sample in samples:
        all_features.append(sample['context_features'])

    all_features = torch.cat(all_features, dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    std[std < 1e-8] = 1.0

    return mean, std


def normalize_features(samples, mean, std):
    """特徴量を標準化"""
    for sample in samples:
        sample['context_features'] = (
            sample['context_features'] - mean
        ) / std
    return samples


def build_iemocap_dataset(config):
    """IEMOCAPのデータセットを構築"""
    label_map = config['data']['label_map']
    exclude_labels = config['data']['exclude_labels']
    num_context = config['data']['num_context_turns']

    print("Loading IEMOCAP utterances...")
    utterances = get_iemocap_utterances(
        config['paths']['iemocap_root'], label_map, exclude_labels
    )

    print("Filtering by evaluator agreement...")
    utterances = filter_by_agreement(utterances)
    print(f"  {len(utterances)} utterances after agreement filter")

    print("Assigning turns...")
    utterances = assign_turns(utterances)

    print("Loading features...")
    audio_feats = torch.load(
        os.path.join(config['paths']['features_dir'], 'audio_iemocap.pt'),
        weights_only=False
    )
    text_feats = torch.load(
        os.path.join(config['paths']['features_dir'], 'bert_iemocap.pt'),
        weights_only=False
    )

    modality_map = {
        'speech': lambda u: audio_feats.get(u['utt_id'],
                   torch.zeros(856, dtype=torch.float32)),
        'text': lambda u: text_feats.get(u['utt_id'],
                 torch.zeros(378, dtype=torch.float32)),
        'multi': lambda u: torch.cat([
            audio_feats.get(u['utt_id'], torch.zeros(856, dtype=torch.float32)),
            text_feats.get(u['utt_id'], torch.zeros(378, dtype=torch.float32)),
        ], dim=0),
    }

    print("Building context sequences...")
    sequences = build_context_sequences(utterances, num_context)

    print("Getting speakers and creating folds...")
    speakers = get_iemocap_speakers(config['paths']['iemocap_root'])
    folds = create_speaker_independent_folds(speakers)

    print(f"Building {len(folds)} folds...")
    for fold_idx, fold in enumerate(folds):
        test_speaker = fold['test_speaker']
        train_speakers = set(fold['train_speakers'])

        fold_sequences = [s for s in sequences
                          if s['context_utts'][0]['session'] != fold['test_session']
                          or s['context_utts'][0]['speaker'] != test_speaker]

        if len(fold_sequences) == 0:
            continue

        for modality in ['speech', 'text', 'multi']:
            get_feat = modality_map[modality]

            fold_samples = []
            for seq in fold_sequences:
                conv_speakers = set()
                for utt in seq['context_utts']:
                    conv_speakers.add(utt['speaker'])
                conv_speakers.add(seq['target_speaker'])

                context_feats = []
                context_lengths = []
                context_speaker_ids = []

                for utt in seq['context_utts']:
                    feat = get_feat(utt)
                    context_feats.append(feat)
                    context_lengths.append(feat.shape[0])
                    context_speaker_ids.append(utt['speaker'])

                context_feats = torch.stack(context_feats)

                fold_samples.append({
                    'context_features': context_feats,
                    'context_lengths': len(context_feats),
                    'context_speaker_ids': context_speaker_ids,
                    'target_label': seq['target_label'],
                    'roles': seq['roles'],
                    'conv_id': seq['conv_id'],
                })

            train_samples = fold_samples

            split = int(len(train_samples) * config['data']['train_val_split'])
            train_fold = train_samples[:split]
            val_fold = train_samples[split:]

            mean, std = compute_normalization_stats(train_fold, modality)
            train_fold = normalize_features(train_fold, mean, std)
            val_fold = normalize_features(val_fold, mean, std)

            out_path = os.path.join(
                config['paths']['data_dir'],
                f'iemocap_fold{fold_idx}_{modality}.pt'
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({
                'train': train_fold,
                'val': val_fold,
                'mean': mean,
                'std': std,
                'test_speaker': test_speaker,
            }, out_path)
            print(f"  Fold {fold_idx} ({modality}): "
                  f"train={len(train_fold)}, val={len(val_fold)}")

    print("IEMOCAP dataset build complete.")


def build_meld_dataset(config):
    """MELDのデータセットを構築"""
    label_map = config['data']['label_map']
    exclude_labels = config['data']['exclude_labels']
    num_context = config['data']['num_context_turns']

    print("Loading MELD utterances...")
    splits = get_meld_utterances(
        config['paths']['meld_root'], label_map, exclude_labels
    )

    for split_name in ['train', 'dev', 'test']:
        if split_name not in splits:
            continue

        utterances = assign_meld_turns(splits[split_name])
        sequences = build_context_sequences(utterances, num_context)

        for modality in ['speech', 'text', 'multi']:
            samples = []
            for seq in sequences:
                conv_speakers = set()
                for utt in seq['context_utts']:
                    conv_speakers.add(utt['speaker'])
                conv_speakers.add(seq['target_speaker'])

                context_feats = []
                context_speaker_ids = []

                for utt in seq['context_utts']:
                    if modality in ['speech', 'multi']:
                        audio_feats = torch.load(
                            os.path.join(config['paths']['features_dir'],
                                         f'audio_meld.pt'), weights_only=False
                        )
                        af = audio_feats.get(utt['utt_id'],
                                             torch.zeros(856, dtype=torch.float32))
                    else:
                        af = None

                    if modality in ['text', 'multi']:
                        text_feats = torch.load(
                            os.path.join(config['paths']['features_dir'],
                                         f'bert_meld.pt'), weights_only=False
                        )
                        tf = text_feats.get(utt['utt_id'],
                                            torch.zeros(378, dtype=torch.float32))
                    else:
                        tf = None

                    if modality == 'speech':
                        feat = af
                    elif modality == 'text':
                        feat = tf
                    else:
                        feat = torch.cat([af, tf], dim=0)

                    context_feats.append(feat)
                    context_speaker_ids.append(utt['speaker'])

                context_feats = torch.stack(context_feats)

                samples.append({
                    'context_features': context_feats,
                    'context_lengths': len(context_feats),
                    'context_speaker_ids': context_speaker_ids,
                    'target_label': seq['target_label'],
                    'roles': seq['roles'],
                    'conv_id': seq['conv_id'],
                })

            if split_name == 'train':
                split_ratio = config['data']['train_val_split']
                split_idx = int(len(samples) * split_ratio)
                train_samples = samples[:split_idx]
                val_samples = samples[split_idx:]
                mean, std = compute_normalization_stats(train_samples, modality)
                train_samples = normalize_features(train_samples, mean, std)
                val_samples = normalize_features(val_samples, mean, std)

                out_path = os.path.join(
                    config['paths']['data_dir'],
                    f'meld_train_{modality}.pt'
                )
                torch.save({
                    'train': train_samples,
                    'val': val_samples,
                    'mean': mean,
                    'std': std,
                }, out_path)
                print(f"  MELD {modality}: train={len(train_samples)}, "
                      f"val={len(val_samples)}")
            else:
                out_path = os.path.join(
                    config['paths']['data_dir'],
                    f'meld_{split_name}_{modality}.pt'
                )
                torch.save({'data': samples}, out_path)
                print(f"  MELD {split_name} {modality}: {len(samples)}")

    print("MELD dataset build complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', choices=['iemocap', 'meld', 'both'], default='iemocap')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config['execution']['seed'])

    if args.dataset in ['iemocap', 'both']:
        build_iemocap_dataset(config)
    if args.dataset in ['meld', 'both']:
        build_meld_dataset(config)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "import ast; ast.parse(open('scripts/build_dataset.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/build_dataset.py
git commit -m "feat: add dataset building script with context construction and normalization"
```

---

### Task 10: Proposed Model

**Files:**
- Create: `models/proposed_model.py`

- [ ] **Step 1: Create models/proposed_model.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len] (1=valid, 0=padding)
        Returns:
            [batch_size, dim]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        output = torch.matmul(attn_weights, V)
        output = output.mean(dim=1)

        return output


class IndividualGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = SelfAttention(hidden_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size]
            lengths: [batch_size] (actual sequence lengths)
        Returns:
            [batch_size, hidden_size]
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        mask = torch.zeros(out.shape[:2], device=out.device)
        for i, l in enumerate(lengths):
            if l > 0:
                mask[i, :l] = 1.0

        attended = self.attention(out, mask)
        return attended


class DialogManagementUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, interaction_hidden: torch.Tensor,
                speaker_ids: list, target_speaker: str,
                interlocutor: str) -> dict:
        """Interaction GRUのhidden stateを役割ごとに振り分け

        Args:
            interaction_hidden: [seq_len, hidden_size]
            speaker_ids: list of speaker_id per timestep
            target_speaker: str
            interlocutor: str or None
        Returns:
            dict with 'speaker', 'interlocutor' tensors
        """
        speaker_h = []
        interlocutor_h = []

        for t, sid in enumerate(speaker_ids):
            h = interaction_hidden[t].unsqueeze(0)
            if sid == target_speaker:
                speaker_h.append(h)
            elif sid == interlocutor:
                interlocutor_h.append(h)

        result = {}
        if len(speaker_h) > 0:
            result['speaker'] = torch.cat(speaker_h, dim=0)
        else:
            result['speaker'] = torch.zeros(
                1, interaction_hidden.shape[1],
                device=interaction_hidden.device
            )

        if len(interlocutor_h) > 0:
            result['interlocutor'] = torch.cat(interlocutor_h, dim=0)
        else:
            result['interlocutor'] = torch.zeros(
                1, interaction_hidden.shape[1],
                device=interaction_hidden.device
            )

        return result


class ProposedModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        self.interaction_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.speaker_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.interlocutor_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.dialog_mgmt = DialogManagementUnit()

        self.fc = nn.Linear(hidden_size * 3, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor,
                context_speaker_ids: list,
                roles: list) -> torch.Tensor:
        """
        Args:
            context_features: [batch_size, seq_len, input_size]
            context_lengths: [batch_size]
            context_speaker_ids: list of list of str
            roles: list of dict with 'speaker', 'interlocutor'
        Returns:
            [batch_size, num_classes]
        """
        batch_size = context_features.shape[0]

        packed = nn.utils.rnn.pack_padded_sequence(
            context_features, context_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_out, interaction_hidden = self.interaction_gru(packed)
        interaction_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        hS_list = []
        hI_list = []
        hA_list = []

        for b in range(batch_size):
            actual_len = context_lengths[b].item()
            h_interaction = interaction_out[b, :actual_len]
            speaker_ids = context_speaker_ids[b][:actual_len]
            role = roles[b]

            dm_output = self.dialog_mgmt(
                h_interaction, speaker_ids,
                role['speaker'], role.get('interlocutor')
            )

            speaker_h = dm_output['speaker']
            interlocutor_h = dm_output['interlocutor']

            speaker_lengths = torch.tensor([speaker_h.shape[0]])
            interlocutor_lengths = torch.tensor([interlocutor_h.shape[0]])

            hS = self.speaker_gru(
                speaker_h.unsqueeze(0), speaker_lengths
            )
            hI = self.interlocutor_gru(
                interlocutor_h.unsqueeze(0), interlocutor_lengths
            )

            hA = interaction_hidden[-1, b].unsqueeze(0)

            hS_list.append(hS)
            hI_list.append(hI)
            hA_list.append(hA)

        hS = torch.cat(hS_list, dim=0)
        hI = torch.cat(hI_list, dim=0)
        hA = torch.cat(hA_list, dim=0)

        h = torch.cat([hS, hI, hA], dim=-1)
        logits = self.fc(h)

        return logits
```

- [ ] **Step 2: Verify model instantiation**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "
import torch
from models.proposed_model import ProposedModel
model = ProposedModel(input_size=1234, hidden_size=256, num_classes=4)
x = torch.randn(2, 10, 1234)
lengths = torch.tensor([10, 8])
roles = [
    {'speaker': 'A', 'interlocutor': 'B', 'spectators': set()},
    {'speaker': 'B', 'interlocutor': 'A', 'spectators': set()},
]
speaker_ids = [['A','A','B','B','A','B','A','B','A','B'], ['B','A','B','A','B','A','B','A']]
out = model(x, lengths, speaker_ids, roles)
print(f'Output shape: {out.shape}')
"
```

Expected: `Output shape: torch.Size([2, 4])`

- [ ] **Step 3: Commit**

```bash
git add models/proposed_model.py
git commit -m "feat: add proposed model with Dialog Management Unit and Self-Attention"
```

---

### Task 11: Comparison Models

**Files:**
- Create: `models/shahriar_model.py`
- Create: `models/shi2020_model.py`
- Create: `models/blstm_model.py`

- [ ] **Step 1: Create models/shahriar_model.py**

```python
import torch
import torch.nn as nn


class ShahriarModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            context_features, context_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)
        return logits
```

- [ ] **Step 2: Create models/shi2020_model.py**

```python
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.proposed_model import SelfAttention, IndividualGRU


class Shi2020Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        self.interaction_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.speaker_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.other_gru = IndividualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size * 3, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor,
                context_speaker_ids: list,
                roles: list, **kwargs) -> torch.Tensor:
        batch_size = context_features.shape[0]

        packed = nn.utils.rnn.pack_padded_sequence(
            context_features, context_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_out, interaction_hidden = self.interaction_gru(packed)
        interaction_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        hS_list = []
        hO_list = []
        hA_list = []

        for b in range(batch_size):
            actual_len = context_lengths[b].item()
            h_interaction = interaction_out[b, :actual_len]
            speaker_ids = context_speaker_ids[b][:actual_len]
            target_speaker = roles[b]['speaker']

            speaker_h = []
            other_h = []
            for t, sid in enumerate(speaker_ids):
                h = h_interaction[t].unsqueeze(0)
                if sid == target_speaker:
                    speaker_h.append(h)
                else:
                    other_h.append(h)

            if len(speaker_h) == 0:
                speaker_h = [torch.zeros(
                    1, self.hidden_size, device=context_features.device)]
            if len(other_h) == 0:
                other_h = [torch.zeros(
                    1, self.hidden_size, device=context_features.device)]

            speaker_tensor = torch.cat(speaker_h, dim=0)
            other_tensor = torch.cat(other_h, dim=0)

            s_len = torch.tensor([speaker_tensor.shape[0]])
            o_len = torch.tensor([other_tensor.shape[0]])

            hS = self.speaker_gru(speaker_tensor.unsqueeze(0), s_len)
            hO = self.other_gru(other_tensor.unsqueeze(0), o_len)
            hA = interaction_hidden[-1, b].unsqueeze(0)

            hS_list.append(hS)
            hO_list.append(hO)
            hA_list.append(hA)

        hS = torch.cat(hS_list, dim=0)
        hO = torch.cat(hO_list, dim=0)
        hA = torch.cat(hA_list, dim=0)

        h = torch.cat([hS, hO, hA], dim=-1)
        logits = self.fc(h)
        return logits
```

- [ ] **Step 3: Create models/blstm_model.py**

```python
import torch
import torch.nn as nn


class BLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, context_features: torch.Tensor,
                context_lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            context_features, context_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        forward_h = hidden[-2]
        backward_h = hidden[-1]
        last_hidden = torch.cat([forward_h, backward_h], dim=-1)
        logits = self.fc(last_hidden)
        return logits
```

- [ ] **Step 4: Verify all models**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "
import torch
from models.shahriar_model import ShahriarModel
from models.shi2020_model import Shi2020Model
from models.blstm_model import BLSTMModel

x = torch.randn(2, 10, 1234)
lengths = torch.tensor([10, 8])
roles = [{'speaker': 'A', 'interlocutor': 'B', 'spectators': set()},
         {'speaker': 'B', 'interlocutor': 'A', 'spectators': set()}]
sids = [['A','A','B','B','A','B','A','B','A','B'], ['B','A','B','A','B','A','B','A']]

m1 = ShahriarModel(1234)
m2 = Shi2020Model(1234)
m3 = BLSTMModel(1234)

print(f'Shahriar: {m1(x, lengths).shape}')
print(f'Shi2020:  {m2(x, lengths, sids, roles).shape}')
print(f'BLSTM:    {m3(x, lengths).shape}')
"
```

Expected: 全て `torch.Size([2, 4])`

- [ ] **Step 5: Commit**

```bash
git add models/shahriar_model.py models/shi2020_model.py models/blstm_model.py
git commit -m "feat: add comparison models (Shahriar, Shi 2020, BLSTM)"
```

---

### Task 12: Training Script

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Create scripts/train.py**

```python
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
    if model_name == 'proposed':
        from models.proposed_model import ProposedModel
        return ProposedModel(
            input_size=input_size,
            hidden_size=config['model']['gru_hidden_size'],
            num_layers=config['model']['gru_num_layers'],
            dropout=config['model']['gru_dropout'],
            num_classes=config['num_classes'],
        )
    elif model_name == 'shahriar':
        from models.shahriar_model import ShahriarModel
        return ShahriarModel(
            input_size=input_size,
            hidden_size=config['model']['gru_hidden_size'],
            num_layers=config['model']['gru_num_layers'],
            dropout=config['model']['gru_dropout'],
            num_classes=config['num_classes'],
        )
    elif model_name == 'shi2020':
        from models.shi2020_model import Shi2020Model
        return Shi2020Model(
            input_size=input_size,
            hidden_size=config['model']['gru_hidden_size'],
            num_layers=config['model']['gru_num_layers'],
            dropout=config['model']['gru_dropout'],
            num_classes=config['num_classes'],
        )
    elif model_name == 'blstm':
        from models.blstm_model import BLSTMModel
        return BLSTMModel(
            input_size=input_size,
            hidden_size=config['model']['gru_hidden_size'],
            num_layers=config['model']['gru_num_layers'],
            dropout=config['model']['gru_dropout'],
            num_classes=config['num_classes'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_class_weights(train_samples, num_classes: int) -> torch.Tensor:
    labels = [s['target_label'] for s in train_samples]
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device, model_name):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Training', leave=False):
        features = batch['context_features'].to(device)
        lengths = batch['context_lengths']
        labels = batch['target_label'].to(device)

        kwargs = {}
        if model_name in ['proposed', 'shi2020']:
            kwargs['context_speaker_ids'] = batch['context_speaker_ids']
            kwargs['roles'] = batch['roles']

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
        np.array(all_labels), np.array(all_preds),
        len(torch.unique(labels))
    )
    return avg_loss, metrics


def evaluate(model, dataloader, criterion, device, model_name):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            features = batch['context_features'].to(device)
            lengths = batch['context_lengths']
            labels = batch['target_label'].to(device)

            kwargs = {}
            if model_name in ['proposed', 'shi2020']:
                kwargs['context_speaker_ids'] = batch['context_speaker_ids']
                kwargs['roles'] = batch['roles']

            logits = model(features, lengths, **kwargs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_all_metrics(
        np.array(all_labels), np.array(all_preds),
        len(torch.unique(torch.tensor(all_labels)))
    )
    return avg_loss, metrics


def train_single_fold(model, train_data, val_data, config, device, model_name):
    train_dataset = ConversationDataset(train_data)
    val_dataset = ConversationDataset(val_data)
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        shuffle=True, collate_fn=collate_fn,
        num_workers=config['execution']['num_workers'],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, collate_fn=collate_fn,
        num_workers=config['execution']['num_workers'],
    )

    class_weights = compute_class_weights(
        train_data, config['num_classes']
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['training']['learning_rate']
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = None

    for epoch in range(config['training']['max_epochs']):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, model_name
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, model_name
        )

        print(f"  Epoch {epoch+1}/{config['training']['max_epochs']} - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val UAR: {val_metrics['uar']:.2f} | "
              f"Val F1: {val_metrics['macro_f1']:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return best_metrics


def train_iemocap(config, model_name, modality, device):
    modality_dim_map = {
        'speech': config['modality']['audio_dim'],
        'text': config['modality']['text_dim'],
        'multi': config['modality']['multimodal_dim'],
    }
    input_size = modality_dim_map[modality]

    data_dir = config['paths']['data_dir']
    fold_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith(f'iemocap_fold') and f.endswith(f'_{modality}.pt')
    ])

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Modality: {modality} | Folds: {len(fold_files)}")
    print(f"{'='*60}")

    all_results = []
    for fold_file in fold_files:
        fold_idx = fold_file.split('fold')[1].split('_')[0]
        print(f"\n--- Fold {fold_idx} ---")

        data = torch.load(os.path.join(data_dir, fold_file), weights_only=False)
        train_data = data['train']
        val_data = data['val']

        set_seed(config['execution']['seed'])
        model = get_model(model_name, input_size, config).to(device)

        metrics = train_single_fold(
            model, train_data, val_data, config, device, model_name
        )
        print(f"  Fold {fold_idx} Best - UAR: {metrics['uar']:.2f}, "
              f"F1: {metrics['macro_f1']:.2f}")
        all_results.append(metrics)

    avg_uar = np.mean([r['uar'] for r in all_results])
    std_uar = np.std([r['uar'] for r in all_results])
    avg_f1 = np.mean([r['macro_f1'] for r in all_results])
    std_f1 = np.std([r['macro_f1'] for r in all_results])

    print(f"\n{'='*60}")
    print(f"Average - UAR: {avg_uar:.2f}+/-{std_uar:.2f}, "
          f"F1: {avg_f1:.2f}+/-{std_f1:.2f}")
    print(f"{'='*60}")

    return {
        'fold_results': all_results,
        'avg_uar': avg_uar, 'std_uar': std_uar,
        'avg_f1': avg_f1, 'std_f1': std_f1,
    }


def train_meld(config, model_name, modality, device):
    modality_dim_map = {
        'speech': config['modality']['audio_dim'],
        'text': config['modality']['text_dim'],
        'multi': config['modality']['multimodal_dim'],
    }
    input_size = modality_dim_map[modality]

    data_dir = config['paths']['data_dir']
    train_file = os.path.join(data_dir, f'meld_train_{modality}.pt')
    test_file = os.path.join(data_dir, f'meld_test_{modality}.pt')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"MELD data files not found for {modality}")
        return None

    train_data = torch.load(train_file, weights_only=False)['train']
    val_data = torch.load(train_file, weights_only=False)['val']
    test_data = torch.load(test_file, weights_only=False)['data']

    set_seed(config['execution']['seed'])
    model = get_model(model_name, input_size, config).to(device)

    print(f"\n{'='*60}")
    print(f"MELD - Model: {model_name} | Modality: {modality}")
    print(f"{'='*60}")

    metrics = train_single_fold(
        model, train_data, val_data, config, device, model_name
    )

    test_dataset = ConversationDataset(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, collate_fn=collate_fn,
    )

    class_weights = compute_class_weights(
        train_data, config['num_classes']
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, model_name
    )

    print(f"  Test - UAR: {test_metrics['uar']:.2f}, "
          f"F1: {test_metrics['macro_f1']:.2f}")

    return {'test_metrics': test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', choices=['iemocap', 'meld'], default='iemocap')
    parser.add_argument('--model', choices=['proposed', 'shahriar', 'shi2020', 'blstm'],
                        default='proposed')
    parser.add_argument('--modality', choices=['speech', 'text', 'multi'], default='multi')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config['execution']['seed'])
    device = torch.device(config['execution']['device'])

    if args.dataset == 'iemocap':
        result = train_iemocap(config, args.model, args.modality, device)
    else:
        result = train_meld(config, args.model, args.modality, device)

    if result is not None:
        out_dir = config['paths']['results_dir']
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir, f'{args.dataset}_{args.model}_{args.modality}.pt'
        )
        torch.save(result, out_file)
        print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "import ast; ast.parse(open('scripts/train.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add training script with early stopping and class weights"
```

---

### Task 13: Evaluation Script

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Create scripts/evaluate.py**

```python
import os
import argparse
import yaml
import torch
import numpy as np
import csv
from glob import glob

import sys
sys.path.insert(0, os.getcwd())
from utils.metrics import compute_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', choices=['iemocap', 'meld'], default='iemocap')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = config['paths']['results_dir']
    result_files = sorted(glob(os.path.join(results_dir, f'{args.dataset}_*.pt')))

    if not result_files:
        print(f"No result files found for {args.dataset}")
        return

    summary_rows = []
    for result_file in result_files:
        result = torch.load(result_file, weights_only=False)
        filename = os.path.basename(result_file).replace('.pt', '')

        if args.dataset == 'iemocap':
            row = {
                'filename': filename,
                'avg_uar': f"{result['avg_uar']:.2f}",
                'std_uar': f"{result['std_uar']:.2f}",
                'avg_f1': f"{result['avg_f1']:.2f}",
                'std_f1': f"{result['std_f1']:.2f}",
            }

            for fold_idx, fold_result in enumerate(result['fold_results']):
                row[f'fold{fold_idx}_uar'] = f"{fold_result['uar']:.2f}"
                row[f'fold{fold_idx}_f1'] = f"{fold_result['macro_f1']:.2f}"

            summary_rows.append(row)

            cm_text_path = os.path.join(
                results_dir, f'{filename}_confusion_matrix.txt'
            )
            with open(cm_text_path, 'w') as f:
                f.write(f"Confusion Matrix for {filename}\n")
                for fold_idx, fold_result in enumerate(result['fold_results']):
                    cm = fold_result['confusion_matrix']
                    f.write(f"\nFold {fold_idx}:\n")
                    labels = config['emotion_labels']
                    f.write(f"  {'':>10}")
                    for l in labels:
                        f.write(f"{l:>10}")
                    f.write("\n")
                    for i, l in enumerate(labels):
                        f.write(f"  {l:>10}")
                        for j in range(len(labels)):
                            f.write(f"{cm[i][j]:>10}")
                        f.write("\n")
        else:
            test_metrics = result.get('test_metrics', {})
            row = {
                'filename': filename,
                'uar': f"{test_metrics.get('uar', 0):.2f}",
                'macro_f1': f"{test_metrics.get('macro_f1', 0):.2f}",
            }
            summary_rows.append(row)

    csv_path = os.path.join(results_dir, f'{args.dataset}_summary.csv')
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Summary saved to {csv_path}")
    print("\nResults:")
    for row in summary_rows:
        print(f"  {row['filename']}: {row}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "import ast; ast.parse(open('scripts/evaluate.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add evaluation script with CSV summary and confusion matrix output"
```

---

### Task 14: Full Integration Test

- [ ] **Step 1: Run end-to-end syntax check on all files**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "
import ast
import glob

files = glob.glob('scripts/*.py') + glob.glob('models/*.py') + glob.glob('utils/*.py')
for f in sorted(files):
    try:
        ast.parse(open(f).read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'FAIL: {f} - {e}')
"
```

Expected: 全て `OK`

- [ ] **Step 2: Run model smoke test**

```bash
cd /Users/ayier1/Desktop/dev/EPC && python -c "
import torch
from models.proposed_model import ProposedModel
from models.shahriar_model import ShahriarModel
from models.shi2020_model import Shi2020Model
from models.blstm_model import BLSTMModel

input_size = 1234
models = {
    'proposed': ProposedModel(input_size),
    'shahriar': ShahriarModel(input_size),
    'shi2020': Shi2020Model(input_size),
    'blstm': BLSTMModel(input_size),
}

x = torch.randn(4, 12, input_size)
lengths = torch.tensor([12, 10, 8, 6])
roles = [
    {'speaker': 'A', 'interlocutor': 'B', 'spectators': set()},
    {'speaker': 'B', 'interlocutor': 'A', 'spectators': set()},
    {'speaker': 'A', 'interlocutor': 'B', 'spectators': set()},
    {'speaker': 'B', 'interlocutor': 'A', 'spectators': set()},
]
sids = [
    ['A','A','B','A','B','A','B','A','B','A','B','A'],
    ['B','A','B','A','B','A','B','A','B','A'],
    ['A','B','A','B','A','B','A','B'],
    ['B','A','B','A','B','A'],
]

for name, model in models.items():
    kwargs = {}
    if name in ['proposed', 'shi2020']:
        kwargs['context_speaker_ids'] = sids
        kwargs['roles'] = roles
    out = model(x, lengths, **kwargs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{name}: output={out.shape}, params={num_params:,}')
"
```

Expected: 全モデル `output=torch.Size([4, 4])`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: integration test passed - all models verified"
```

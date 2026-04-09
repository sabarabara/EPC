# EPC Reproduction Design Spec

Shi et al. (2023) "Emotion Awareness in Multi-utterance Turn for Improving Emotion Prediction in Multi-Speaker Conversation" (INTERSPEECH 2023) の完全再現実験。

## 1. Purpose

論文のTable 2 (IEMOCAP) および Table 4 (MELD) の全ての実験結果（UAR, Macro F1）を再現する。対象は提案モデル + 比較モデル3種（Shahriar, Shi 2020, BLSTM）× 3モダリティ（speech, text, multimodal）。

## 2. Project Structure

```
EPC/
├── configs/
│   └── config.yaml              # 全パラメータ一元管理
├── scripts/
│   ├── extract_egemaps.py       # Phase 1a: eGeMAPS抽出
│   ├── extract_wav2vec2.py      # Phase 1b: Wav2Vec2抽出
│   ├── extract_bert.py          # Phase 1c: BERT抽出 (Python 3.7)
│   ├── build_dataset.py         # Phase 2: データ構築・正規化
│   ├── train.py                 # Phase 3: 学習・評価
│   └── evaluate.py              # Phase 4: 最終集計・結果保存
├── models/
│   ├── proposed_model.py        # 提案モデル (Dialog Management + Self-Attention)
│   ├── shahriar_model.py        # 比較: 時系列のみGRU
│   ├── shi2020_model.py         # 比較: Speaker/Other 2分割GRU
│   └── blstm_model.py           # 比較: BLSTM
├── utils/
│   ├── data_utils.py            # IEMOCAP/MELDデータローダー
│   ├── metrics.py               # UAR, Macro F1計算
│   └── seed.py                  # 乱数シード固定
├── data_raw/                    # 生データ配置枠
│   ├── IEMOCAP/
│   │   └── IEMOCAP_full_release/
│   │       ├── Session1/ ... Session5/
│   └── MELD/
├── features/                    # 抽出済み特徴量 (.pt)
├── data/                        # 構築済みデータセット (.pt)
├── results/                     # 評価結果 (CSV/テキスト)
└── requirements.txt
```

コード: 英語、コメント: 日本語。

## 3. Environment

- **Main**: Python 3.9-3.11 + PyTorch 2.x (latest stable)
- **BERT extraction only**: Python 3.7 + bert-as-service (別環境)
- **GPU**: Local GPU, `torch.device('cuda')`
- **Seed**: torch/numpy/random 全て固定

## 4. Feature Extraction (Phase 1)

### 4a. eGeMAPS (openSMILE 3.x)

- **入力**: utterance-level WAV
- **前処理**: 16kHz, モノラルに統一 (torchaudio)
- **抽出**: eGeMAPS v01a, 88次元
- **出力**: `features/egemaps_{dataset}.pt` — `{utt_id: Tensor(88,)}`

### 4b. Wav2Vec2 (wav2vec2-base-960h, 凍結)

- **入力**: 16kHz モノラル
- **抽出**: 最終レイヤー平均プーリング → 768次元
- **出力**: `features/wav2vec2_{dataset}.pt` — `{utt_id: Tensor(768,)}`

### 4c. BERT (bert-as-service, Python 3.7)

- **モデル**: bert-base-uncased
- **抽出**: CLSトークン → reduce_dim=378 (PCA, random_state固定)
- **出力**: `features/bert_{dataset}.pt` — `{utt_id: Tensor(378,)}`

### 音声特徴量の結合

- eGeMAPS(88) + Wav2Vec2(768) = 856次元
- キャッシュ: `features/audio_{dataset}.pt` — `{utt_id: Tensor(856,)}`

## 5. Data Construction (Phase 2)

### IEMOCAP

1. **ラベル取得**: `Session*/dialog/EmoEvaluation/` からカテゴリラベルを取得
2. **ラベルマッピング**: excited/happy→Happy(0), angry→Anger(1), neutral→Neutral(2), sad→Sad(3), その他→除外
3. **評価者一致**: 2人の評価者が同一ラベルをつけたもののみ採用
4. **ターン定義**: 連続する同一話者のutteranceを同一ターンとする
5. **テキスト**: `Session*/dialog/transcriptions/` から取得
6. **収録種別**: improvised + scripted 両方使用

### IEMOCAP 10-fold CV (話者単位)

- 5セッション × 2話者 = 10人の話者
- 各fold: 1人の話者をテスト、残り9人を訓練
- 訓練fold内: 80%訓練 / 20%検証（話者単位で分割を維持）

### コンテキスト構築

- **文脈**: 予測対象ターンの直前6ターン（ターン内の全utteranceを含む）
- **ターゲット**: 直後のターンの感情ラベル
- **Speaker**: 予測対象ターンの発話者
- **Interlocutor**: 2人対話→相手方、3人以上→文脈内で最頻出の他話者
- **Spectator**: Speaker/Interlocutor以外の話者

### 正規化

- 訓練データの統計量（平均・標準偏差）で標準化
- 検証/テストデータには訓練データの統計量を適用

### MELD

- **split**: 公式 Train/Val/Test (Ghosal et al. [14])
- **7→4クラス**: joy→Happy, anger→Anger, neutral→Neutral, sadness→Sad, その他→除外
- **ターン定義**: 連続同一話者を同一ターン
- **話者ID**: メタデータの話者名
- **テキスト**: Utterance列をそのまま使用
- **オーディオ**: MELD付属のオーディオクリップ (16kHz, モノラル変換)

### 出力形式

- `data/iemocap_fold_{i}.pt`: 各foldの訓練/検証/テストデータ
- `data/meld.pt`: Train/Val/Test データ
- 各データポイント: `{audio_feat, text_feat, speaker_id, turn_idx, role, label, conv_id}`

## 6. Model Architecture

### 提案モデル (Proposed Model)

```
入力: 時系列ソート済み全utterance特徴量 [seq_len, feat_dim]
  │
  ▼
Interaction GRU (単方向, 2層, 256ユニット, dropout=0.5)
  入力次元: 856(speech) / 378(text) / 1234(multi)
  → h_interaction[t] [seq_len, 256]
  │
  ▼ Dialog Management Unit (speaker_idで振り分け)
  │
  ├── Speaker GRU (単方向, 2層, 256, dropout=0.5)
  │   入力: Speakerの発話時刻のh_interactionのみ
  │   → Self-Attention(Q,K,V) → Mean Pool → hS [256]
  │
  ├── Interlocutor GRU (単方向, 2層, 256, dropout=0.5)
  │   入力: Interlocutorの発話時刻のh_interactionのみ
  │   → Self-Attention(Q,K,V) → Mean Pool → hI [256]
  │
  └── Spectator: hA (Interaction GRU最終時刻hidden state) [256]
      Spectator用の個別GRUは存在しない
  │
  ▼
concat(hS, hI, hA) → [768]
  │
  ▼
FC(768 → 4) + Softmax → y_hat [Happy, Anger, Neutral, Sad]
```

### 比較モデル1: Shahriar et al. [2]

- 時系列順の全発話を1つのGRU（単方向, 2層, 256, dropout=0.5）に通す
- 話者情報を使用しない
- 最終hidden state → FC(256 → 4) + Softmax

### 比較モデル2: Shi et al. [4]

- Interaction GRU（単方向, 2層, 256, dropout=0.5）で全発話を処理
- Speakerの発話とOtherの発話を2つのGRUに振り分け（Spectator概念なし）
- 各GRU → Self-Attention(Q,K,V) → Mean Pool（提案モデルと同じAttention構成）
- concat(hSpeaker, hOther, hInteraction) → FC(768 → 4) + Softmax
- 注: 提案モデルとの差分はDialog Management Unitの有無（Speaker/Interlocutor/Spectatorの3役割 vs Speaker/Otherの2分割）のみ

### 比較モデル3: BLSTM

- 時系列順の全発話をBiLSTM（2層, 256, dropout=0.5）に通す
- 話者情報を使用しない
- 最終hidden state (forward + backward concat) → FC(512 → 4) + Softmax

## 7. Training Configuration

- **Optimizer**: Adam, lr=0.0001
- **Loss**: CrossEntropyLoss (class weights付き、訓練データのクラス頻度の逆数)
- **Batch**: 会話単位、ゼロパディング + mask
- **Batch size**: 32
- **Max epochs**: 50
- **Early stopping**: validation loss基準、patience=10
- **Seed**: 全て固定 (torch, numpy, random)
- **GPU**: torch.device('cuda')
- **No data augmentation**
- **No gradient clipping** (論文未記載のため)

## 8. Evaluation

- **指標**: UAR (`balanced_accuracy_score`), Macro F1 (`f1_score(average='macro')`)
- **IEMOCAP**: 10-foldの各fold結果 + 平均±標準偏差
- **MELD**: 公式Test splitでの1回評価
- **出力**: `results/{dataset}_{model}_{modality}_results.csv`
  - 各foldのUAR, Macro F1, クラス別Recall
  - Confusion Matrix (テキストファイル)

## 9. Execution Order

| Priority | Phase | Content |
|----------|-------|---------|
| 1 | Setup | requirements.txt, config.yaml, data_raw/ directory structure |
| 2 | Data | IEMOCAPデータローダー (ラベル・テキスト・ターン定義) |
| 3 | Features | 特徴量抽出スクリプト (eGeMAPS, Wav2Vec2, BERT) |
| 4 | Dataset | データ構築スクリプト (10-fold分割・コンテキスト構築・正規化) |
| 5 | Model | 提案モデル実装 + IEMOCAP multimodal学習 |
| 6 | Eval | 評価スクリプト + 結果集計 |
| 7 | Compare | 比較モデル3種の実装 |
| 8 | Modality | speech-only / text-only 実験対応 |
| 9 | MELD | MELD対応 (データローダー + 全モデル実験) |

## 10. Target Results

### IEMOCAP (Table 2)

| Modality | Model | UAR | Macro F1 |
|----------|-------|-----|----------|
| Speech | Shahriar | 56.78% | 55.11% |
| Speech | Shi 2020 | 61.98% | 60.21% |
| Speech | Proposed | **65.01%** | **65.91%** |
| Text | Shahriar | 71.19% | 70.65% |
| Text | Shi 2020 | 74.96% | 74.54% |
| Text | Proposed | **77.30%** | **76.67%** |
| Multi | Shahriar | 74.61% | 73.62% |
| Multi | Shi 2020 | 76.31% | 75.50% |
| Multi | Proposed | **80.18%** | **80.01%** |

### MELD (Table 4)

| Modality | Model | UAR | Macro F1 |
|----------|-------|-----|----------|
| Speech | BLSTM | 25.22% | 20.66% |
| Speech | Shi 2020 | 26.96% | 25.13% |
| Speech | Proposed | **28.67%** | **25.97%** |
| Text | BLSTM | 41.41% | 41.45% |
| Text | Shi 2020 | 42.19% | 42.67% |
| Text | Proposed | **45.44%** | **45.13%** |
| Multi | BLSTM | 42.31% | 42.46% |
| Multi | Shi 2020 | 42.39% | 42.51% |
| Multi | Proposed | **45.21%** | **44.36%** |

## 11. References

- Shi et al. (2023) INTERSPEECH 2023 — 提案モデル
- Shi et al. (2020) INTERSPEECH 2020 — 比較モデル (Speaker/Other 2分割)
- Shahriar and Kim (2019) FG 2019 — 比較モデル (時系列のみ)
- Etienne et al. (2018) — IEMOCAP評価プロトコル
- Ghosal et al. (2020) — MELD評価プロトコル
- wav2vec2-base-960h: https://huggingface.co/facebook/wav2vec2-base-960h
- bert-as-service: https://github.com/hanxiao/bert-as-service
- openSMILE 3.x: https://audeering.github.io/opensmile/

import os
import re
import glob
import torch
import numpy as np
from collections import defaultdict
import pandas as pd


def parse_emoevaluation_file(filepath: str) -> list:
    """EmoEvaluationファイルをパースして発話ごとのラベルを取得"""
    utterances = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("["):
                match = re.match(r"\[.+\]\s+(\S+)\s+.*;\s+(\S+)\s+;", line)
                if match:
                    utt_id = match.group(1)
                    emotion = match.group(2)
                    utterances.append({"utt_id": utt_id, "emotion": emotion})
    return utterances


def get_iemocap_utterances(
    iemocap_root: str, label_map: dict, exclude_labels: list
) -> list:
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
    eval_dirs = sorted(
        glob.glob(
            os.path.join(iemocap_root, "Session*", "dialog", "*", "EmoEvaluation")
        )
    )

    for eval_dir in eval_dirs:
        dialog_name = os.path.basename(os.path.dirname(eval_dir))
        session = os.path.basename(os.path.dirname(os.path.dirname(eval_dir)))

        eval_files = glob.glob(os.path.join(eval_dir, "*.txt"))
        for eval_file in eval_files:
            parsed = parse_emoevaluation_file(eval_file)
            for utt_info in parsed:
                utt_id = utt_info["utt_id"]
                emotion_name = utt_info["emotion"]

                if emotion_name not in label_map:
                    continue
                label = label_map[emotion_name]
                if label in exclude_labels:
                    continue

                speaker = utt_id.split("_")[0]

                wav_path = os.path.join(
                    iemocap_root,
                    session,
                    "sentences",
                    "wav",
                    dialog_name,
                    utt_id + ".wav",
                )

                transcript_path = os.path.join(
                    iemocap_root,
                    session,
                    "dialog",
                    "transcriptions",
                    dialog_name + ".txt",
                )

                text = ""
                if os.path.exists(transcript_path):
                    text = _extract_utterance_text(transcript_path, utt_id)

                all_utterances.append(
                    {
                        "utt_id": utt_id,
                        "session": session,
                        "dialog": dialog_name,
                        "speaker": speaker,
                        "emotion": label,
                        "emotion_name": emotion_name,
                        "wav_path": wav_path,
                        "text": text,
                    }
                )

    return all_utterances


def _extract_utterance_text(transcript_path: str, utt_id: str) -> str:
    """transcriptionsファイルから特定の発話テキストを抽出"""
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.escape(utt_id) + r":\s+(.*)"
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return ""


def filter_by_agreement(utterances: list) -> list:
    """評価者間で一致した発話のみを残す

    IEMOCAPでは同一発話に2人の評価者がラベルを付与。
    EmoEvaluationファイル内の同一utt_idが2回出現し、
    両方が同じラベルの場合のみ採用。
    """
    utt_labels = defaultdict(list)
    for utt in utterances:
        utt_labels[utt["utt_id"]].append(utt["emotion"])

    agreed_ids = set()
    for utt_id, labels in utt_labels.items():
        if len(labels) == 2 and labels[0] == labels[1]:
            agreed_ids.add(utt_id)

    return [u for u in utterances if u["utt_id"] in agreed_ids]


def assign_turns(utterances: list) -> list:
    """連続する同一話者のutteranceを同一ターンとしてターンIDを付与

    各dialog内で、話者が切り替わるごとに新しいターンとする。
    """
    dialogs = defaultdict(list)
    for utt in utterances:
        dialogs[utt["dialog"]].append(utt)

    result = []
    for dialog_id, dialog_utts in dialogs.items():
        dialog_utts.sort(key=lambda x: x["utt_id"])
        turn_id = 0
        for i, utt in enumerate(dialog_utts):
            if i > 0 and dialog_utts[i]["speaker"] != dialog_utts[i - 1]["speaker"]:
                turn_id += 1
            utt["turn_idx"] = turn_id
            result.append(utt)

    return result


def get_iemocap_speakers(iemocap_root: str) -> list:
    """IEMOCAPの全話者リストを取得 (10人)

    Returns:
        list of dict: [{'speaker_id': str, 'session': str}, ...]
    """
    speakers = []
    seen = set()
    for session in sorted(glob.glob(os.path.join(iemocap_root, "Session*"))):
        session_name = os.path.basename(session)
        for dialog in sorted(glob.glob(os.path.join(session, "dialog", "*"))):
            eval_file = glob.glob(os.path.join(dialog, "EmoEvaluation", "*.txt"))[0]
            parsed = parse_emoevaluation_file(eval_file)
            for utt_info in parsed:
                speaker = utt_info["utt_id"].split("_")[0]
                if speaker not in seen:
                    seen.add(speaker)
                    speakers.append(
                        {
                            "speaker_id": speaker,
                            "session": session_name,
                        }
                    )
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
        folds.append(
            {
                "test_speaker": speakers[i]["speaker_id"],
                "test_session": speakers[i]["session"],
                "train_speakers": [
                    s["speaker_id"] for s in speakers if s != speakers[i]
                ],
            }
        )
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
        dialogs[utt.get("dialog", utt.get("dialog_id"))].append(utt)

    sequences = []
    for dialog_id, dialog_utts in dialogs.items():
        dialog_utts.sort(key=lambda x: (x["turn_idx"], x["utt_id"]))

        turns = defaultdict(list)
        for utt in dialog_utts:
            turns[utt["turn_idx"]].append(utt)

        sorted_turns = sorted(turns.keys())

        for i in range(num_context_turns, len(sorted_turns)):
            context_turn_indices = sorted_turns[i - num_context_turns : i]
            target_turn_idx = sorted_turns[i]

            context_utts = []
            for t_idx in context_turn_indices:
                context_utts.extend(turns[t_idx])

            target_utts = turns[target_turn_idx]
            target_speaker = target_utts[0]["speaker"]
            target_label = target_utts[0]["emotion"]

            sequences.append(
                {
                    "context_utts": context_utts,
                    "target_label": target_label,
                    "target_speaker": target_speaker,
                    "conv_id": dialog_id,
                    "context_turn_indices": context_turn_indices,
                    "target_turn_idx": target_turn_idx,
                }
            )

    return sequences


def assign_roles(sequences: list, dialog_speakers: dict) -> list:
    """Dialog Management Unitのための役割割り当て

    Args:
        sequences: コンテキストシーケンスリスト
        dialog_speakers: {dialog_id: set of speaker_ids}

    Returns:
        各sequenceに 'roles' を追加
    """
    for seq in sequences:
        conv_id = seq["conv_id"]
        speaker = seq["target_speaker"]
        all_speakers = dialog_speakers.get(conv_id, set())

        context_speakers = set()
        for utt in seq["context_utts"]:
            context_speakers.add(utt["speaker"])

        others = context_speakers - {speaker}

        if len(others) == 0:
            interlocutor = None
            spectators = set()
        elif len(all_speakers) <= 2:
            interlocutor = list(others)[0]
            spectators = set()
        else:
            speaker_counts = defaultdict(int)
            for utt in seq["context_utts"]:
                if utt["speaker"] != speaker:
                    speaker_counts[utt["speaker"]] += 1
            interlocutor = max(speaker_counts, key=speaker_counts.get)
            spectators = others - {interlocutor}

        seq["roles"] = {
            "speaker": speaker,
            "interlocutor": interlocutor,
            "spectators": spectators,
        }

    return sequences


def get_meld_utterances(meld_root: str, label_map: dict, exclude_labels: list) -> dict:
    """MELDの全発話情報を取得

    Returns:
        dict: {'train': list, 'dev': list, 'test': list}
    """
    splits = {}
    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(meld_root, f"{split}_sent_emo.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {split}")
            continue

        df = pd.read_csv(csv_path)
        utterances = []

        for _, row in df.iterrows():
            emotion = row["Emotion"].strip().lower()
            if emotion not in label_map:
                continue
            label = label_map[emotion]
            if label in exclude_labels:
                continue

            utterances.append(
                {
                    "utt_id": f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}",
                    "dialog_id": str(row["Dialogue_ID"]),
                    "speaker": row["Speaker"].strip(),
                    "emotion": label,
                    "emotion_name": emotion,
                    "text": str(row["Utterance"]).strip(),
                    "utterance_idx": row["Utterance_ID"],
                }
            )

        splits[split] = utterances

    return splits


def assign_meld_turns(utterances: list) -> list:
    """MELDのターン定義: 連続する同一話者のutteranceを同一ターン"""
    dialogs = defaultdict(list)
    for utt in utterances:
        dialogs[utt["dialog_id"]].append(utt)

    result = []
    for dialog_id, dialog_utts in dialogs.items():
        dialog_utts.sort(key=lambda x: x["utterance_idx"])
        turn_id = 0
        for i, utt in enumerate(dialog_utts):
            if i > 0 and dialog_utts[i]["speaker"] != dialog_utts[i - 1]["speaker"]:
                turn_id += 1
            utt["turn_idx"] = turn_id
            result.append(utt)

    return result

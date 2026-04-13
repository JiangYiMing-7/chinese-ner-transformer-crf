'''评估模块。

当前保留两类开发集指标：
1. token-level accuracy：只在真实字对应的有效位置上计算
2. entity-level micro F1：基于 BIO 序列抽取实体后计算

这里额外提供统一的解码、标签抽取和指标函数，供训练与预测共用。
'''

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from utils import move_batch_to_device


def extract_valid_label_sequences(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_index: int = -100,
) -> List[List[int]]:
    '''从 batch 中抽取真实字位置上的 gold 标签序列。

    这个抽取逻辑和预测恢复逻辑必须一致，否则开发集 accuracy / F1 会和最终输出口径脱节。
    '''

    labels = labels.detach().cpu()
    valid_mask = valid_mask.detach().cpu().bool()

    sequences: List[List[int]] = []
    for batch_index in range(labels.size(0)):
        valid_positions = valid_mask[batch_index] & labels[batch_index].ne(ignore_index)
        sequences.append(labels[batch_index][valid_positions].tolist())
    return sequences


def decode_batch_predictions(
    model: torch.nn.Module,
    outputs: Dict[str, Any],
    valid_mask: torch.Tensor,
) -> List[List[int]]:
    '''统一 baseline / final 的预测解码逻辑。'''

    logits = outputs["logits"]
    return model.decode(logits, valid_mask)


def compute_token_accuracy_from_sequences(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
) -> Dict[str, float]:
    '''根据已经对齐好的有效位序列计算准确率。'''

    total = 0
    correct = 0

    for predicted, gold in zip(predicted_sequences, gold_sequences):
        compare_length = min(len(predicted), len(gold))
        # 分母始终按 gold 长度统计；若预测长度偏短，只在重叠部分累加 correct，
        # 这样不会把“少预测的尾部”悄悄忽略掉。
        total += len(gold)
        correct += sum(
            1
            for index in range(compare_length)
            if predicted[index] == gold[index]
        )

    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
    }


def compute_token_accuracy_from_label_sequences(
    predicted_label_sequences: Sequence[Sequence[str]],
    gold_label_sequences: Sequence[Sequence[str]],
) -> Dict[str, float]:
    '''根据标签字符串序列计算 token-level accuracy。'''

    total = 0
    correct = 0

    for predicted, gold in zip(predicted_label_sequences, gold_label_sequences):
        compare_length = min(len(predicted), len(gold))
        total += len(gold)
        correct += sum(
            1
            for index in range(compare_length)
            if str(predicted[index]) == str(gold[index])
        )

    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
    }


def _split_bio_label(label: str) -> Tuple[str, Optional[str]]:
    '''拆解 BIO 标签。'''

    if label == "O":
        return "O", None
    if "_" not in label:
        return label, None
    prefix, entity_type = label.split("_", 1)
    return prefix, entity_type


def _align_label_sequence_to_gold_length(
    predicted_labels: Sequence[str],
    gold_labels: Sequence[str],
    outside_label: str = "O",
) -> Tuple[List[str], List[str]]:
    '''把预测序列对齐到 gold 长度，避免长度不一致时的口径漂移。'''

    aligned_predicted = [str(label) for label in predicted_labels[: len(gold_labels)]]
    if len(aligned_predicted) < len(gold_labels):
        # 对实体级评估来说，预测缺失的位置应视作 O，而不是缩短 gold 序列。
        aligned_predicted.extend([outside_label] * (len(gold_labels) - len(aligned_predicted)))
    return aligned_predicted, [str(label) for label in gold_labels]


def extract_entities_from_label_sequence(
    label_sequence: Sequence[str],
) -> List[Tuple[int, int, str]]:
    '''从 BIO 标签序列中抽取实体跨度。

    当前实现采用“宽松 BIO”解析：
    - `B_X` 一定开启新实体
    - `I_X` 若前一位置不是同类型实体，则视作新实体起点
    - `O` 会结束当前实体
    '''

    entities: List[Tuple[int, int, str]] = []
    entity_start: Optional[int] = None
    entity_type: Optional[str] = None

    def close_entity(end_index: int) -> None:
        nonlocal entity_start, entity_type
        if entity_start is not None and entity_type is not None:
            entities.append((entity_start, end_index, entity_type))
        entity_start = None
        entity_type = None

    for index, raw_label in enumerate(label_sequence):
        label = str(raw_label)
        prefix, next_entity_type = _split_bio_label(label)

        if prefix == "O":
            close_entity(index)
            continue

        if prefix == "B":
            close_entity(index)
            entity_start = index
            entity_type = next_entity_type or label
            continue

        if prefix == "I":
            candidate_type = next_entity_type or label
            if entity_start is None or entity_type != candidate_type:
                close_entity(index)
                entity_start = index
                entity_type = candidate_type
            continue

        close_entity(index)
        entity_start = index
        entity_type = label

    close_entity(len(label_sequence))
    return entities


def compute_entity_f1_from_label_sequences(
    predicted_label_sequences: Sequence[Sequence[str]],
    gold_label_sequences: Sequence[Sequence[str]],
    outside_label: str = "O",
) -> Dict[str, float]:
    '''根据标签字符串序列计算实体级 micro-F1。'''

    correct_entities = 0
    predicted_entities = 0
    gold_entities = 0

    for predicted_labels, gold_labels in zip(predicted_label_sequences, gold_label_sequences):
        aligned_predicted, aligned_gold = _align_label_sequence_to_gold_length(
            predicted_labels=predicted_labels,
            gold_labels=gold_labels,
            outside_label=outside_label,
        )
        predicted_entity_set = set(extract_entities_from_label_sequence(aligned_predicted))
        gold_entity_set = set(extract_entities_from_label_sequence(aligned_gold))

        correct_entities += len(predicted_entity_set & gold_entity_set)
        predicted_entities += len(predicted_entity_set)
        gold_entities += len(gold_entity_set)

    precision = (correct_entities / predicted_entities) if predicted_entities else 0.0
    recall = (correct_entities / gold_entities) if gold_entities else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "correct_entities": correct_entities,
        "predicted_entities": predicted_entities,
        "gold_entities": gold_entities,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_entity_f1_from_sequences(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
    id2label: Sequence[str],
) -> Dict[str, float]:
    '''根据标签 id 序列计算实体级 micro-F1。'''

    if not id2label:
        return {
            "correct_entities": 0,
            "predicted_entities": 0,
            "gold_entities": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    outside_label = "O" if "O" in id2label else str(id2label[0])
    predicted_label_sequences = [
        [str(id2label[label_id]) for label_id in predicted_sequence]
        for predicted_sequence in predicted_sequences
    ]
    gold_label_sequences = [
        [str(id2label[label_id]) for label_id in gold_sequence]
        for gold_sequence in gold_sequences
    ]
    return compute_entity_f1_from_label_sequences(
        predicted_label_sequences=predicted_label_sequences,
        gold_label_sequences=gold_label_sequences,
        outside_label=outside_label,
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: str,
    ignore_index: int = -100,
    description: str = "Evaluating",
) -> Dict[str, Any]:
    '''在开发集上评估模型。

    当前同时返回 token-level accuracy 和 entity-level micro F1，
    且都只统计真实字对应的有效位置。
    '''

    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_valid = 0
    total_correct = 0
    total_predicted_entities = 0
    total_gold_entities = 0
    total_correct_entities = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=description,
            leave=False,
            disable=not sys.stdout.isatty(),
        ):
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
                valid_mask=batch["valid_mask"],
            )

            # 评估时必须使用和预测时完全一致的 decode 逻辑，
            # 否则 baseline / CRF 的 dev accuracy 口径会不统一。
            predicted_sequences = decode_batch_predictions(
                model=model,
                outputs=outputs,
                valid_mask=batch["valid_mask"],
            )
            gold_sequences = extract_valid_label_sequences(
                labels=batch["labels"],
                valid_mask=batch["valid_mask"],
                ignore_index=ignore_index,
            )
            accuracy_info = compute_token_accuracy_from_sequences(
                predicted_sequences=predicted_sequences,
                gold_sequences=gold_sequences,
            )
            f1_info = compute_entity_f1_from_sequences(
                predicted_sequences=predicted_sequences,
                gold_sequences=gold_sequences,
                id2label=getattr(model, "labels", []),
            )

            total_valid += int(accuracy_info["total"])
            total_correct += int(accuracy_info["correct"])
            total_predicted_entities += int(f1_info["predicted_entities"])
            total_gold_entities += int(f1_info["gold_entities"])
            total_correct_entities += int(f1_info["correct_entities"])
            # dev_loss 仍按 batch 平均，和 training_log.json 里的 epoch 级统计口径一致。
            total_loss += float(outputs["loss"].item())
            total_steps += 1

    entity_precision = (
        total_correct_entities / total_predicted_entities
        if total_predicted_entities
        else 0.0
    )
    entity_recall = (
        total_correct_entities / total_gold_entities
        if total_gold_entities
        else 0.0
    )
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if (entity_precision + entity_recall)
        else 0.0
    )

    return {
        "dev_loss": total_loss / max(1, total_steps),
        "token_accuracy": (total_correct / total_valid) if total_valid else 0.0,
        "valid_token_count": total_valid,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "predicted_entity_count": total_predicted_entities,
        "gold_entity_count": total_gold_entities,
        "correct_entity_count": total_correct_entities,
    }

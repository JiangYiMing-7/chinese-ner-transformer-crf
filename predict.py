'''预测模块。

负责加载最佳模型、执行长句滑窗推理、生成提交文件，
并在输出后立即做格式校验。
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import NERConfig
from data import (
    NERBatchCollator,
    NERDataset,
    NERExample,
    build_examples,
    build_tokenizer,
    read_token_lines,
    resolve_data_paths,
)
from evaluate import (
    compute_entity_f1_from_label_sequences,
    compute_token_accuracy_from_label_sequences,
    decode_batch_predictions,
)
from model_backbone import RobertaLinearNER
from plot_curves import generate_project_summary_artifacts
from utils import (
    choose_device,
    find_latest_experiment_dir,
    load_checkpoint,
    load_json,
    move_batch_to_device,
    save_json,
    validate_prediction_output,
    write_prediction_file,
)


def _apply_saved_config(config: NERConfig, saved_config: Dict[str, Any]) -> NERConfig:
    '''将实验目录中的关键配置回写到当前配置对象。

    这里故意不覆盖 `profile`、`project_root`、`output_filename` 这类运行时参数。
    训练时保存的配置主要用于恢复模型结构和推理口径，而不是锁死当前机器的运行环境。
    '''

    for field_name in (
        "experiment_variant",
        "experiment_name",
        "model_name",
        "fallback_model_name",
        "max_len",
        "batch_size",
        "grad_accum_steps",
        "num_epochs",
        "lr_backbone",
        "lr_refinement",
        "lr_classifier",
        "lr_crf",
        "weight_decay",
        "early_stopping_patience",
        "dropout",
        "seed",
        "ignore_index",
        "use_refinement",
        "refine_num_layers",
        "refine_num_heads",
        "refine_ffn_dim",
        "refine_dropout",
        "use_crf",
        "bio_constraint_mode",
        "use_sliding_window",
        "sliding_window_size",
        "sliding_overlap",
        "sliding_merge_strategy",
        "use_bucket_batching",
        "bucket_size_multiplier",
        "shuffle_within_bucket",
    ):
        if field_name in saved_config:
            # 预测阶段应该尽量复用训练时的配置，避免 max_len、dropout 等关键信息不一致。
            setattr(config, field_name, saved_config[field_name])
    # 低频字 dropout 是训练期输入增强，预测阶段不回放这一开关。
    return config


def load_model_for_prediction(
    checkpoint_path: Path,
    config: NERConfig,
) -> Tuple[RobertaLinearNER, Dict[int, str], Dict[str, int]]:
    '''根据 checkpoint 重建模型。

    预测恢复只依赖可移植的 `requested_model_name`，不依赖本机缓存的绝对路径。
    '''

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    requested_model_name = checkpoint.get(
        "requested_model_name",
        checkpoint_config.get("model_name", config.model_name),
    )
    fallback_model_name = checkpoint_config.get("fallback_model_name", config.fallback_model_name)
    label2id = checkpoint["label2id"]
    id2label = {
        int(index): label
        for index, label in checkpoint["id2label"].items()
    }
    use_crf = bool(checkpoint.get("use_crf", checkpoint_config.get("use_crf", config.use_crf)))
    use_refinement = bool(
        checkpoint.get("use_refinement", checkpoint_config.get("use_refinement", config.use_refinement))
    )
    bio_constraint_mode = str(
        checkpoint.get(
            "bio_constraint_mode",
            checkpoint_config.get("bio_constraint_mode", config.bio_constraint_mode),
        )
    )

    # 这里必须使用 checkpoint 中真实保存下来的 backbone 名称，避免 fallback 导致结构不一致。
    model = RobertaLinearNER(
        model_name=requested_model_name,
        fallback_model_name=fallback_model_name,
        num_labels=len(label2id),
        dropout=float(checkpoint_config.get("dropout", config.dropout)),
        ignore_index=config.ignore_index,
        use_refinement=use_refinement,
        refine_num_layers=int(checkpoint_config.get("refine_num_layers", config.refine_num_layers)),
        refine_num_heads=int(checkpoint_config.get("refine_num_heads", config.refine_num_heads)),
        refine_ffn_dim=int(checkpoint_config.get("refine_ffn_dim", config.refine_ffn_dim)),
        refine_dropout=float(checkpoint_config.get("refine_dropout", config.refine_dropout)),
        use_crf=use_crf,
        bio_constraint_mode=bio_constraint_mode,
        label2id=label2id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, id2label, label2id


def _forward_batch(
    model: RobertaLinearNER,
    batch: Dict[str, Any],
) -> List[List[int]]:
    '''对一个 batch 执行统一解码。'''

    # 预测阶段仍然把 valid_mask 传进模型，是为了让 CRF / 非 CRF 路线都返回同一口径的“真实字序列”标签。
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        labels=None,
        valid_mask=batch["valid_mask"],
    )
    return decode_batch_predictions(
        model=model,
        outputs=outputs,
        valid_mask=batch["valid_mask"],
    )


def _compute_char_token_counts(
    chars: Sequence[str],
    tokenizer: Any,
) -> Tuple[List[int], int]:
    '''统计每个原始字会映射成多少个 token。

    滑窗虽然按“原始字位置”切分，但窗口是否会超出模型预算，真正取决于 tokenizer 之后的 token 数。
    因此这里先在字级上统计 token 数，再按 token 预算建窗，避免窗口内部再次发生截断。
    '''

    if not chars:
        return [], tokenizer.num_special_tokens_to_add(pair=False)

    encoded = tokenizer(
        [list(chars)],
        is_split_into_words=True,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    word_ids = encoded.word_ids(batch_index=0)
    token_counts = [0] * len(chars)

    for word_id in word_ids:
        if word_id is None:
            continue
        token_counts[word_id] += 1

    return token_counts, tokenizer.num_special_tokens_to_add(pair=False)


def build_sliding_windows(
    chars: Sequence[str],
    tokenizer: Any,
    token_budget: int,
    overlap: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    '''按 token 预算构造原始字级滑窗。

    返回的窗口区间始终是 `[start, end)`，对应原始字序列的切片。
    overlap 的单位仍然是“字数”，这样 merge 时可以直接映射回原句索引。
    '''

    if not chars:
        return [(0, 0)], {"full_token_length": 0, "special_token_count": 0}

    token_counts, special_token_count = _compute_char_token_counts(chars, tokenizer)
    if token_budget <= special_token_count:
        raise ValueError(
            "sliding_window_size is too small after accounting for special tokens."
        )

    prefix_token_counts = [0]
    for token_count in token_counts:
        prefix_token_counts.append(prefix_token_counts[-1] + token_count)

    full_token_length = special_token_count + prefix_token_counts[-1]
    if full_token_length <= token_budget:
        return [(0, len(chars))], {
            "full_token_length": full_token_length,
            "special_token_count": special_token_count,
        }

    windows: List[Tuple[int, int]] = []
    start = 0
    char_length = len(chars)

    while start < char_length:
        low = start + 1
        high = char_length
        best_end = start + 1

        while low <= high:
            # 用前缀和快速判断 `[start, mid)` 这段切出来后会占多少 token。
            # 这里用二分找“当前起点下能塞进 token_budget 的最远终点”，避免线性试探过慢。
            mid = (low + high) // 2
            window_token_length = special_token_count + (prefix_token_counts[mid] - prefix_token_counts[start])
            if window_token_length <= token_budget:
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1

        windows.append((start, best_end))
        if best_end >= char_length:
            break

        next_start = max(0, best_end - overlap)
        # 若 overlap 过大导致窗口不前进，会陷入死循环，这里强制推进 1 个字。
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return windows, {
        "full_token_length": full_token_length,
        "special_token_count": special_token_count,
    }


def predict_window_batch(
    model: RobertaLinearNER,
    window_examples: Sequence[NERExample],
    collator: NERBatchCollator,
    id2label: Dict[int, str],
    device: str,
) -> List[Dict[str, Any]]:
    '''对一组窗口执行批量推理。'''

    if not window_examples:
        return []

    batch = collator(window_examples)
    batch = move_batch_to_device(batch, device)
    predicted_sequences = _forward_batch(model=model, batch=batch)

    results: List[Dict[str, Any]] = []
    for batch_index, predicted_ids in enumerate(predicted_sequences):
        decoded_tags = [id2label[prediction_id] for prediction_id in predicted_ids]
        original_length = int(batch["original_lengths"][batch_index])
        kept_length = int(batch["kept_lengths"][batch_index])
        results.append(
            {
                "sample_id": int(batch["sample_ids"][batch_index]),
                "predicted_tags": decoded_tags[:kept_length],
                "original_length": original_length,
                "kept_length": kept_length,
                # 若这里仍然发生截断，后续会递归二分，而不是直接补 O 草草收尾。
                "truncated": kept_length < original_length,
                "chars": list(batch["chars"][batch_index]),
            }
        )
    return results


def predict_single_window(
    model: RobertaLinearNER,
    chars: Sequence[str],
    global_start: int,
    collator: NERBatchCollator,
    id2label: Dict[int, str],
    device: str,
) -> List[Dict[str, Any]]:
    '''对单个窗口推理；若窗口仍被截断，则递归二分兜底。'''

    example = NERExample(sample_id=0, chars=list(chars), tags=None)
    prediction = predict_window_batch(
        model=model,
        window_examples=[example],
        collator=collator,
        id2label=id2label,
        device=device,
    )[0]

    if not prediction["truncated"]:
        return [
            {
                "start": global_start,
                "predicted_tags": prediction["predicted_tags"],
                "source": "window",
            }
        ]

    # 理论上 token-aware 切窗后不应再截断。若仍出现，通常说明该窗口含有异常多 sub-token。
    # 这里不静默补 O，而是继续把窗口二分，直到覆盖完为止。
    if len(chars) <= 1:
        return [
            {
                "start": global_start,
                "predicted_tags": prediction["predicted_tags"],
                "source": "fallback_leaf",
            }
        ]

    midpoint = max(1, len(chars) // 2)
    left_records = predict_single_window(
        model=model,
        chars=chars[:midpoint],
        global_start=global_start,
        collator=collator,
        id2label=id2label,
        device=device,
    )
    right_records = predict_single_window(
        model=model,
        chars=chars[midpoint:],
        global_start=global_start + midpoint,
        collator=collator,
        id2label=id2label,
        device=device,
    )
    return left_records + right_records


def merge_window_predictions(
    original_length: int,
    window_predictions: Sequence[Dict[str, Any]],
    fill_label: str,
) -> Tuple[List[str], Dict[str, Any]]:
    '''把多个窗口级预测融合回原始句长。

    当前默认使用 `center_priority`：
    - 优先选择“离窗口边界更远”的预测
    - 若分数相同，再选择“更靠近窗口中心”的预测
    - 若仍相同，则保留先到的窗口结果
    '''

    merged_predictions: List[Optional[str]] = [None] * original_length
    best_edge_distance = [-1] * original_length
    best_center_distance = [float("inf")] * original_length
    coverage_count = [0] * original_length

    for record in window_predictions:
        start = int(record["start"])
        predicted_tags = list(record["predicted_tags"])
        window_length = len(predicted_tags)
        if window_length == 0:
            continue

        window_center = (window_length - 1) / 2.0
        for local_index, predicted_tag in enumerate(predicted_tags):
            global_index = start + local_index
            if global_index < 0 or global_index >= original_length:
                continue

            coverage_count[global_index] += 1
            edge_distance = min(local_index, window_length - 1 - local_index)
            center_distance = abs(local_index - window_center)

            if edge_distance > best_edge_distance[global_index]:
                should_replace = True
            elif edge_distance == best_edge_distance[global_index]:
                should_replace = center_distance < best_center_distance[global_index]
            else:
                should_replace = False

            if should_replace:
                merged_predictions[global_index] = predicted_tag
                best_edge_distance[global_index] = edge_distance
                best_center_distance[global_index] = center_distance

    uncovered_positions = [
        index for index, predicted_tag in enumerate(merged_predictions)
        if predicted_tag is None
    ]
    for uncovered_index in uncovered_positions:
        # 这里只允许作为“覆盖异常”的兜底，不再作为长句主方案。
        merged_predictions[uncovered_index] = fill_label

    return [str(tag) for tag in merged_predictions], {
        "uncovered_positions": uncovered_positions,
        "coverage_min": min(coverage_count) if coverage_count else 0,
        "coverage_max": max(coverage_count) if coverage_count else 0,
    }


def predict_long_sequence(
    model: RobertaLinearNER,
    chars: Sequence[str],
    collator: NERBatchCollator,
    id2label: Dict[int, str],
    device: str,
    config: NERConfig,
) -> Tuple[List[str], Dict[str, Any]]:
    '''对长句执行滑窗推理并融合回原始长度。'''

    fill_label = "O" if "O" in id2label.values() else id2label.get(0, "O")
    token_budget = min(config.sliding_window_size, collator.max_len)
    windows, window_meta = build_sliding_windows(
        chars=chars,
        tokenizer=collator.tokenizer,
        token_budget=token_budget,
        overlap=config.sliding_overlap,
    )

    raw_window_examples = [
        NERExample(sample_id=index, chars=list(chars[start:end]), tags=None)
        for index, (start, end) in enumerate(windows)
    ]

    # 先按正常窗口批量推理；只有极少数仍被截断的窗口，才退回递归二分兜底。
    merged_window_predictions: List[Dict[str, Any]] = []
    # 记录还有多少窗口需要递归二分，便于排查极端 sub-token 过多的句子。
    fallback_split_window_count = 0

    for batch_start in range(0, len(raw_window_examples), config.batch_size):
        batch_examples = raw_window_examples[batch_start : batch_start + config.batch_size]
        batch_results = predict_window_batch(
            model=model,
            window_examples=batch_examples,
            collator=collator,
            id2label=id2label,
            device=device,
        )

        for local_offset, result in enumerate(batch_results):
            window_start, window_end = windows[batch_start + local_offset]
            window_chars = list(chars[window_start:window_end])

            if result["truncated"]:
                fallback_split_window_count += 1
                merged_window_predictions.extend(
                    predict_single_window(
                        model=model,
                        chars=window_chars,
                        global_start=window_start,
                        collator=collator,
                        id2label=id2label,
                        device=device,
                    )
                )
            else:
                merged_window_predictions.append(
                    {
                        "start": window_start,
                        "predicted_tags": result["predicted_tags"],
                        "source": "window",
                    }
                )

    merged_predictions, merge_meta = merge_window_predictions(
        original_length=len(chars),
        window_predictions=merged_window_predictions,
        fill_label=fill_label,
    )
    return merged_predictions, {
        "window_count": len(windows),
        "token_budget": token_budget,
        "full_token_length": window_meta["full_token_length"],
        "fallback_split_window_count": fallback_split_window_count,
        "uncovered_positions": merge_meta["uncovered_positions"],
        "coverage_min": merge_meta["coverage_min"],
        "coverage_max": merge_meta["coverage_max"],
        "window_spans": [[start, end] for start, end in windows],
    }


def predict_sequences(
    model: RobertaLinearNER,
    dataloader: DataLoader,
    collator: NERBatchCollator,
    id2label: Dict[int, str],
    device: str,
    config: NERConfig,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    '''执行逐批推理并恢复为原始字长度。'''

    model.eval()
    fill_label = id2label.get(0, "O")
    if "O" in id2label.values():
        fill_label = "O"

    all_predictions: List[List[str]] = []
    truncated_sample_count = 0
    sliding_sample_count = 0
    sliding_window_count = 0
    fallback_split_window_count = 0
    uncovered_samples: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            batch = move_batch_to_device(batch, device)
            prediction_sequences = _forward_batch(model=model, batch=batch)

            for batch_index, predicted_ids in enumerate(prediction_sequences):
                decoded_tags = [id2label[prediction_id] for prediction_id in predicted_ids]
                original_length = int(batch["original_lengths"][batch_index])
                kept_length = int(batch["kept_lengths"][batch_index])
                chars = list(batch["chars"][batch_index])
                sample_id = int(batch["sample_ids"][batch_index])
                truncated = bool(batch["truncated_flags"][batch_index])

                if truncated:
                    truncated_sample_count += 1

                if truncated and config.use_sliding_window:
                    # 开启滑窗时，截断样本不直接补 O，而是走“长句重切窗 -> 重推理 -> 再融合”的正式路径。
                    sliding_sample_count += 1
                    long_predictions, long_meta = predict_long_sequence(
                        model=model,
                        chars=chars,
                        collator=collator,
                        id2label=id2label,
                        device=device,
                        config=config,
                    )
                    all_predictions.append(long_predictions)
                    sliding_window_count += int(long_meta["window_count"])
                    fallback_split_window_count += int(long_meta["fallback_split_window_count"])

                    if long_meta["uncovered_positions"]:
                        uncovered_samples.append(
                            {
                                "sample_id": sample_id,
                                "uncovered_positions": long_meta["uncovered_positions"],
                                "window_spans": long_meta["window_spans"],
                            }
                        )
                    continue

                if kept_length < original_length:
                    # 只有在关闭滑窗时，才允许退回旧的 O 回填兜底。
                    decoded_tags.extend([fill_label] * (original_length - kept_length))

                # 无论是否发生截断，最终都强制恢复到原始字长度，再写出预测文件。
                decoded_tags = decoded_tags[:original_length]
                all_predictions.append(decoded_tags)

    return all_predictions, {
        "predicted_sequence_count": len(all_predictions),
        "truncated_sample_count": truncated_sample_count,
        "sliding_window_enabled": config.use_sliding_window,
        "sliding_sample_count": sliding_sample_count,
        "sliding_window_count": sliding_window_count,
        "fallback_split_window_count": fallback_split_window_count,
        "uncovered_sample_count": len(uncovered_samples),
        "uncovered_samples": uncovered_samples,
    }


def run_prediction(
    config: NERConfig,
    experiment_dir: Optional[str] = None,
) -> Path:
    '''对 dev/test 生成预测结果并执行格式校验。'''

    if experiment_dir:
        experiment_path = Path(experiment_dir).expanduser().resolve()
    else:
        experiment_path = find_latest_experiment_dir(
            config.output_root,
            experiment_name=config.experiment_name,
        )

    saved_config_path = experiment_path / "config.json"
    if saved_config_path.exists():
        saved_config = load_json(saved_config_path)
        config = _apply_saved_config(config, saved_config)

    device = choose_device(config.profile)
    checkpoint_path = experiment_path / "checkpoints" / "best_model.pt"
    model, id2label, label2id = load_model_for_prediction(checkpoint_path, config)
    model.to(device)

    checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
    # tokenizer 也按 checkpoint 中的 requested_model_name 恢复，避免本机默认配置和训练时不一致。
    tokenizer, _ = build_tokenizer(
        model_name=str(checkpoint_payload.get("requested_model_name", config.model_name)),
        fallback_model_name=checkpoint_payload.get(
            "config",
            {},
        ).get("fallback_model_name", config.fallback_model_name),
    )

    paths = resolve_data_paths(config)
    dev_text_lines = read_token_lines(paths["dev_text"])
    dev_tag_lines = read_token_lines(paths["dev_tags"])
    test_text_lines = read_token_lines(paths["test_text"])
    if config.profile == "local_debug":
        # 本地联调时为了速度只验证小样本流程，但输出仍然保持规范格式。
        dev_text_lines = dev_text_lines[: config.debug_subset_size]
        dev_tag_lines = dev_tag_lines[: config.debug_subset_size]
        test_text_lines = test_text_lines[: config.debug_subset_size]

    dev_dataset = NERDataset(build_examples(dev_text_lines))
    test_dataset = NERDataset(build_examples(test_text_lines))
    collator = NERBatchCollator(
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=config.max_len,
        ignore_index=config.ignore_index,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    predictions_dir = experiment_path / "predictions"
    results_dir = experiment_path / "results"

    dev_predictions, dev_stats = predict_sequences(
        model=model,
        dataloader=dev_loader,
        collator=collator,
        id2label=id2label,
        device=device,
        config=config,
    )
    test_predictions, test_stats = predict_sequences(
        model=model,
        dataloader=test_loader,
        collator=collator,
        id2label=id2label,
        device=device,
        config=config,
    )

    dev_pred_path = predictions_dir / "dev_pred.txt"
    test_pred_path = predictions_dir / config.output_filename
    write_prediction_file(dev_pred_path, dev_predictions)
    write_prediction_file(test_pred_path, test_predictions)

    # 写出后立即做格式校验，确保提交前就能发现长度、空行、行数错位问题。
    dev_validation = validate_prediction_output(dev_text_lines, dev_predictions)
    test_validation = validate_prediction_output(test_text_lines, test_predictions)
    dev_token_metrics = compute_token_accuracy_from_label_sequences(
        predicted_label_sequences=dev_predictions,
        gold_label_sequences=dev_tag_lines,
    )
    dev_entity_metrics = compute_entity_f1_from_label_sequences(
        predicted_label_sequences=dev_predictions,
        gold_label_sequences=dev_tag_lines,
    )
    dev_metric_report = {
        "token_accuracy": round(float(dev_token_metrics["accuracy"]), 6),
        "token_correct": int(dev_token_metrics["correct"]),
        "token_total": int(dev_token_metrics["total"]),
        "entity_precision": round(float(dev_entity_metrics["precision"]), 6),
        "entity_recall": round(float(dev_entity_metrics["recall"]), 6),
        "entity_f1": round(float(dev_entity_metrics["f1"]), 6),
        "correct_entities": int(dev_entity_metrics["correct_entities"]),
        "predicted_entities": int(dev_entity_metrics["predicted_entities"]),
        "gold_entities": int(dev_entity_metrics["gold_entities"]),
    }
    validation_report = {
        "experiment_dir": str(experiment_path),
        "experiment_variant": config.experiment_variant,
        "use_refinement": model.use_refinement,
        "use_crf": model.use_crf,
        "bio_constraint_mode": model.bio_constraint_mode,
        "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
        "use_sliding_window": config.use_sliding_window,
        "sliding_window_size": config.sliding_window_size,
        "sliding_overlap": config.sliding_overlap,
        "sliding_merge_strategy": config.sliding_merge_strategy,
        "dev_valid": bool(dev_validation["is_valid"]),
        "test_valid": bool(test_validation["is_valid"]),
        "dev_line_count_match": bool(dev_validation["line_count_match"]),
        "test_line_count_match": bool(test_validation["line_count_match"]),
        "dev_token_count_match": bool(dev_validation["token_count_match"]),
        "test_token_count_match": bool(test_validation["token_count_match"]),
        "dev_empty_line_match": bool(dev_validation["empty_line_match"]),
        "test_empty_line_match": bool(test_validation["empty_line_match"]),
        "uncovered_sample_count": int(
            dev_stats["uncovered_sample_count"] + test_stats["uncovered_sample_count"]
        ),
        "dev_metrics": dev_metric_report,
        "dev_prediction": {
            "file_path": str(dev_pred_path),
            "stats": dev_stats,
            "validation": dev_validation,
            "metrics": dev_metric_report,
        },
        "test_prediction": {
            "file_path": str(test_pred_path),
            "stats": test_stats,
            "validation": test_validation,
        },
    }
    save_json(results_dir / "dev_prediction_metrics.json", dev_metric_report)
    save_json(results_dir / "output_validation.json", validation_report)
    # predict 阶段写出的开发集指标和校验结果，会直接进入项目级汇总图表与报告分析。
    generate_project_summary_artifacts(config.output_root, config.data_dir)

    print("=== Prediction Summary ===")
    print(f"Experiment dir: {experiment_path}")
    print(f"Variant: {config.experiment_variant}")
    print(f"Use refinement: {model.use_refinement}")
    print(f"Use CRF: {model.use_crf}")
    print(f"BIO constraint mode: {model.bio_constraint_mode}")
    print(f"Effective BIO mode: {model.effective_bio_constraint_mode}")
    print(f"Use sliding window: {config.use_sliding_window}")
    print(f"Sliding window size: {config.sliding_window_size}")
    print(f"Sliding overlap: {config.sliding_overlap}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Dev output valid: {dev_validation['is_valid']}")
    print(f"Test output valid: {test_validation['is_valid']}")
    print(f"Dev token accuracy: {dev_metric_report['token_accuracy']:.6f}")
    print(f"Dev entity F1: {dev_metric_report['entity_f1']:.6f}")
    print(f"Dev prediction file: {dev_pred_path}")
    print(f"Test prediction file: {test_pred_path}")

    return experiment_path

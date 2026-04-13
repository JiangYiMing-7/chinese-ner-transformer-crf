'''Stage5 手写 Transformer 训练入口。

目标：
1. 使用字符级数据链路训练手写 Transformer + CRF
2. 逐轮开发集评估直接复用 `evaluate.py/evaluate_model(...)`
3. 训练结束后自动导出 dev/test 预测、校验结果和图表
4. 刷新 `outputs/project_summary/` 下的项目级对比图
'''

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data_scratch import (
    ScratchBatchCollator,
    ScratchNERDataset,
    ScratchNERExample,
    build_char_vocab,
    build_label_metadata,
    build_scratch_eval_dataloader,
    build_scratch_examples,
    build_scratch_train_dataloader,
    chunk_examples_by_max_len,
    read_token_lines,
    resolve_scratch_data_paths,
)
from evaluate import (
    compute_entity_f1_from_label_sequences,
    compute_token_accuracy_from_label_sequences,
    evaluate_model,
)
from model_scratch import ScratchNER
from plot_curves import (
    generate_data_figures,
    generate_project_summary_artifacts,
    generate_training_figures,
)
from utils import (
    choose_device,
    create_experiment_dirs,
    format_seconds,
    get_git_commit,
    get_gpu_name,
    load_checkpoint,
    load_json,
    move_batch_to_device,
    save_checkpoint,
    save_json,
    set_seed,
    timer,
    validate_prediction_output,
    write_csv_rows,
    write_prediction_file,
)


class FGM:
    '''Fast Gradient Method 对抗训练。

    参考 Miyato et al. 2017。
    与单步符号扰动的 FGSM 不同，FGM 这里使用的是“梯度方向归一化后的连续扰动”：
    r = epsilon * grad / ||grad||
    这样可以更平滑地作用在 embedding 空间中。
    '''

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self, epsilon: float, emb_name: str = "embedding") -> None:
        '''沿 embedding 梯度方向添加扰动。'''

        self.backup = {}
        for name, parameter in self.model.named_parameters():
            if emb_name not in name or not parameter.requires_grad:
                continue
            if parameter.grad is None:
                continue

            grad_norm = torch.norm(parameter.grad)
            if not torch.isfinite(grad_norm) or grad_norm.item() == 0.0:
                continue

            self.backup[name] = parameter.data.clone()
            perturbation = epsilon * parameter.grad / (grad_norm + 1e-8)
            parameter.data.add_(perturbation)

    def restore(self, emb_name: str = "embedding") -> None:
        '''恢复 embedding 参数到原始值。'''

        for name, parameter in self.model.named_parameters():
            if emb_name not in name or not parameter.requires_grad:
                continue
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


class EMA:
    '''指数移动平均参数。

    参考 Polyak & Juditsky 1992：
    shadow = decay * shadow + (1 - decay) * param

    训练时更新 shadow，评估和最终预测时切换到 shadow 权重，
    通常能得到更平滑、更稳定的结果。
    '''

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone()

    def update(self) -> None:
        '''用当前训练参数更新 shadow 权重。'''

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            shadow_param = self.shadow[name]
            shadow_param.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        '''把 shadow 权重加载到模型里，供评估或预测使用。'''

        self.backup = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.backup[name] = parameter.detach().clone()
            parameter.data.copy_(self.shadow[name])

    def restore(self) -> None:
        '''恢复训练中的原始参数。'''

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


def _clone_stage5_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    '''规范化 Stage5 配置。'''

    merged = dict(config)
    project_root = Path(merged.get("project_root", Path(__file__).resolve().parent)).resolve()
    experiment_dir_prefix = Path(str(merged.get("experiment_dir_prefix", "outputs/stage5_scratch")))

    merged["project_root"] = project_root
    merged["data_dir"] = project_root / "data"
    merged["output_root"] = project_root / experiment_dir_prefix.parent
    merged["experiment_name"] = str(experiment_dir_prefix.name)
    merged["profile"] = str(merged.get("profile", "cloud_train"))
    merged["variant"] = str(merged.get("variant", "stage5_scratch"))
    merged["bio_constraint_mode"] = str(merged.get("bio_constraint_mode", "none"))
    merged["embedding_lr"] = float(merged.get("embedding_lr", 5e-4))
    merged["classifier_lr"] = float(merged.get("classifier_lr", merged["peak_lr"]))
    merged["crf_lr"] = float(merged.get("crf_lr", merged["peak_lr"]))
    merged["sliding_overlap"] = int(merged.get("sliding_overlap", 64))
    merged["shuffle_within_bucket"] = bool(merged.get("shuffle_within_bucket", True))
    merged["max_train_samples"] = merged.get("max_train_samples")
    merged["max_dev_samples"] = merged.get("max_dev_samples")
    merged["max_test_samples"] = merged.get("max_test_samples")
    return merged


def _limit_lines(
    lines: Sequence[List[str]],
    max_samples: Optional[int],
) -> List[List[str]]:
    '''按调试配置裁剪数据规模。'''

    if max_samples is None:
        return [list(line) for line in lines]
    return [list(line) for line in lines[: int(max_samples)]]


def _summarize_length_stats(
    text_lines: Sequence[Sequence[str]],
    max_seq_len: int,
) -> Dict[str, Any]:
    '''统计当前切分的长度信息。'''

    lengths = [len(line) for line in text_lines]
    over_count = sum(1 for length in lengths if length > max_seq_len)
    return {
        "count": len(lengths),
        "max": max(lengths) if lengths else 0,
        "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "over_max_len_count": over_count,
        "over_max_len_ratio": (over_count / len(lengths)) if lengths else 0.0,
    }


def _prepare_stage5_data(config: Dict[str, Any]) -> Dict[str, Any]:
    '''读取 Stage5 所需的全部数据与词表。'''

    paths = resolve_scratch_data_paths(config["project_root"])
    train_text_lines = _limit_lines(
        read_token_lines(paths["train_text"]),
        config.get("max_train_samples"),
    )
    train_tag_lines = _limit_lines(
        read_token_lines(paths["train_tags"]),
        config.get("max_train_samples"),
    )
    dev_text_lines = _limit_lines(
        read_token_lines(paths["dev_text"]),
        config.get("max_dev_samples"),
    )
    dev_tag_lines = _limit_lines(
        read_token_lines(paths["dev_tags"]),
        config.get("max_dev_samples"),
    )
    test_text_lines = _limit_lines(
        read_token_lines(paths["test_text"]),
        config.get("max_test_samples"),
    )

    char2id, id2char, char_freq = build_char_vocab(
        train_text_lines,
        min_freq=int(config["min_char_freq"]),
    )
    rebuilt_label_meta = build_label_metadata(train_tag_lines)
    label_stats_path = config["data_dir"] / "label_stats.json"
    if label_stats_path.exists():
        saved_label_stats = load_json(label_stats_path)
        saved_label2id = {
            str(label): int(index)
            for label, index in dict(saved_label_stats["label2id"]).items()
        }
        rebuilt_label2id = {
            str(label): int(index)
            for label, index in dict(rebuilt_label_meta["label2id"]).items()
        }
        if saved_label2id != rebuilt_label2id:
            raise ValueError(
                "Stage5 label mapping mismatch with data/label_stats.json: "
                f"saved={saved_label2id}, rebuilt={rebuilt_label2id}"
            )
        label2id = saved_label2id
        id2label = {
            int(index): str(label)
            for index, label in dict(saved_label_stats["id2label"]).items()
        }
    else:
        label2id = {
            str(label): int(index)
            for label, index in dict(rebuilt_label_meta["label2id"]).items()
        }
        id2label = {
            int(index): str(label)
            for index, label in dict(rebuilt_label_meta["id2label"]).items()
        }

    raw_train_examples = build_scratch_examples(train_text_lines, train_tag_lines, char2id, label2id)
    raw_dev_examples = build_scratch_examples(dev_text_lines, dev_tag_lines, char2id, label2id)
    raw_test_examples = build_scratch_examples(test_text_lines, None, char2id, None)

    train_examples = chunk_examples_by_max_len(
        raw_train_examples,
        max_seq_len=int(config["max_seq_len"]),
    )
    dev_examples = chunk_examples_by_max_len(
        raw_dev_examples,
        max_seq_len=int(config["max_seq_len"]),
    )

    pad_idx = int(char2id["<PAD>"])
    collator = ScratchBatchCollator(
        pad_idx=pad_idx,
        ignore_index=int(config["ignore_index"]),
        max_len=int(config["max_seq_len"]),
    )
    train_loader = build_scratch_train_dataloader(
        dataset=ScratchNERDataset(train_examples),
        collator=collator,
        batch_size=int(config["batch_size"]),
        bucket_size_multiplier=int(config["bucket_size_multiplier"]),
        shuffle_within_bucket=bool(config["shuffle_within_bucket"]),
        num_workers=int(config["num_workers"]),
    )
    dev_loader = build_scratch_eval_dataloader(
        dataset=ScratchNERDataset(dev_examples),
        collator=collator,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    return {
        "paths": paths,
        "train_text_lines": train_text_lines,
        "train_tag_lines": train_tag_lines,
        "dev_text_lines": dev_text_lines,
        "dev_tag_lines": dev_tag_lines,
        "test_text_lines": test_text_lines,
        "char2id": char2id,
        "id2char": id2char,
        "char_freq": char_freq,
        "label2id": label2id,
        "id2label": id2label,
        "pad_idx": pad_idx,
        "raw_train_examples": raw_train_examples,
        "raw_dev_examples": raw_dev_examples,
        "raw_test_examples": raw_test_examples,
        "train_loader": train_loader,
        "dev_loader": dev_loader,
        "collator": collator,
        "runtime_length_stats": {
            "train": _summarize_length_stats(train_text_lines, int(config["max_seq_len"])),
            "dev": _summarize_length_stats(dev_text_lines, int(config["max_seq_len"])),
            "test": _summarize_length_stats(test_text_lines, int(config["max_seq_len"])),
        },
    }


def _build_stage5_model(
    config: Dict[str, Any],
    vocab_size: int,
    label2id: Dict[str, int],
    pad_idx: int,
) -> ScratchNER:
    '''构造 Stage5 模型。'''

    return ScratchNER(
        vocab_size=vocab_size,
        num_labels=len(label2id),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        ffn_dim=int(config["ffn_dim"]),
        max_seq_len=int(config["max_seq_len"]),
        dropout=float(config["dropout"]),
        pad_idx=pad_idx,
        ignore_index=int(config["ignore_index"]),
        bio_constraint_mode=str(config["bio_constraint_mode"]),
        label2id=label2id,
    )


def _is_no_decay_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    '''判断参数是否不应施加 weight decay。'''

    lowered_name = name.lower()
    if lowered_name.endswith("bias"):
        return True
    if parameter.ndim == 1:
        return True
    return "norm" in lowered_name


def _add_param_group(
    grouped_parameters: List[Dict[str, Any]],
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    group_name: str,
    learning_rate: float,
    weight_decay: float,
) -> None:
    '''按 decay / no_decay 拆分参数组。'''

    decay_parameters: List[torch.nn.Parameter] = []
    no_decay_parameters: List[torch.nn.Parameter] = []

    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if _is_no_decay_parameter(name, parameter):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    if decay_parameters:
        grouped_parameters.append(
            {
                "params": decay_parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": f"{group_name}_decay",
            }
        )
    if no_decay_parameters:
        grouped_parameters.append(
            {
                "params": no_decay_parameters,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "group_name": f"{group_name}_no_decay",
            }
        )


def create_optimizer_scratch(
    model: ScratchNER,
    config: Dict[str, Any],
) -> AdamW:
    '''创建 Stage5 优化器。'''

    optimizer_grouped_parameters: List[Dict[str, Any]] = []
    _add_param_group(
        optimizer_grouped_parameters,
        model.embedding.named_parameters(),
        group_name="embedding",
        learning_rate=float(config["embedding_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    _add_param_group(
        optimizer_grouped_parameters,
        model.encoder.named_parameters(),
        group_name="encoder",
        learning_rate=float(config["peak_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    _add_param_group(
        optimizer_grouped_parameters,
        model.classifier.named_parameters(),
        group_name="classifier",
        learning_rate=float(config["classifier_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    _add_param_group(
        optimizer_grouped_parameters,
        model.crf.named_parameters(),
        group_name="crf",
        learning_rate=float(config["crf_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    return AdamW(optimizer_grouped_parameters)


def create_scheduler_scratch(
    optimizer: AdamW,
    total_training_steps: int,
    warmup_ratio: float,
    min_lr: float,
) -> LambdaLR:
    '''创建 warmup + cosine annealing 调度器。'''

    warmup_steps = int(total_training_steps * warmup_ratio)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def make_lambda(base_lr: float) -> Any:
        min_factor = min(1.0, max(0.0, float(min_lr) / max(base_lr, 1e-12)))

        def lr_lambda(current_step: int) -> float:
            if total_training_steps <= 0:
                return 1.0
            if warmup_steps > 0 and current_step < warmup_steps:
                # warmup 阶段线性从 0 升到 base_lr。
                return float(current_step) / float(max(1, warmup_steps))

            if total_training_steps <= warmup_steps:
                return 1.0

            # warmup 之后按余弦曲线衰减到 min_lr。
            progress = float(current_step - warmup_steps) / float(
                max(1, total_training_steps - warmup_steps)
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine_factor

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(base_lr) for base_lr in base_lrs],
    )


def get_group_lr(optimizer: AdamW, prefix: str, fallback: float) -> float:
    '''读取某类参数组当前学习率。'''

    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", ""))
        if group_name.startswith(prefix):
            return float(group["lr"])
    return float(fallback)


def _load_pretrained_embedding_matrix(path: Path) -> torch.Tensor:
    '''从文件加载预训练 embedding 矩阵。'''

    if not path.exists():
        raise FileNotFoundError(f"Pretrained embedding file not found: {path}")

    if path.suffix.lower() == ".npy":
        return torch.from_numpy(np.load(path)).float()

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(payload).float()
    if isinstance(payload, dict):
        for key in ("embedding_matrix", "embeddings", "weight"):
            if key in payload:
                value = payload[key]
                if isinstance(value, torch.Tensor):
                    return value.float()
                if isinstance(value, np.ndarray):
                    return torch.from_numpy(value).float()

    raise ValueError(f"Unsupported pretrained embedding payload: {path}")


def _forward_batch(
    model: ScratchNER,
    batch: Dict[str, Any],
) -> List[List[int]]:
    '''执行一个 batch 的预测解码。'''

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        labels=None,
        valid_mask=batch["valid_mask"],
    )
    return model.decode(outputs["logits"], batch["valid_mask"])


def train_one_epoch_scratch(
    model: ScratchNER,
    dataloader: Any,
    optimizer: AdamW,
    scheduler: LambdaLR,
    fgm: Optional[FGM],
    ema: Optional[EMA],
    device: str,
    grad_accum_steps: int,
    grad_clip_norm: float,
    fgm_epsilon: float,
) -> Tuple[float, int]:
    '''训练单个 epoch。'''

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    optimizer_steps = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training Stage5",
        leave=False,
        disable=not sys.stdout.isatty(),
    )
    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
            valid_mask=batch["valid_mask"],
        )
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at training step {step}: {float(loss.item())}"
            )
        total_loss += float(loss.item())
        (loss / grad_accum_steps).backward()

        if fgm is not None:
            fgm.attack(epsilon=fgm_epsilon, emb_name="embedding")
            adv_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
                valid_mask=batch["valid_mask"],
            )
            adv_loss = adv_outputs["loss"]
            if not torch.isfinite(adv_loss):
                raise RuntimeError(
                    f"Non-finite adversarial loss detected at training step {step}: "
                    f"{float(adv_loss.item())}"
                )
            (adv_loss / grad_accum_steps).backward()
            fgm.restore(emb_name="embedding")

        should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if ema is not None:
                ema.update()

    return total_loss / max(1, len(dataloader)), optimizer_steps


def _build_stage5_checkpoint_payload(
    model: ScratchNER,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    best_dev_accuracy: float,
    best_epoch: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    char2id: Dict[str, int],
    id2char: Dict[int, str],
    git_commit: Optional[str],
) -> Dict[str, Any]:
    '''构造 Stage5 checkpoint。'''

    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "config": dict(config),
        "requested_model_name": "scratch_transformer_encoder",
        "resolved_model_name": model.resolved_model_name,
        "experiment_variant": config["variant"],
        "use_crf": True,
        "bio_constraint_mode": config["bio_constraint_mode"],
        "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
        "label2id": dict(label2id),
        "id2label": dict(id2label),
        "char2id": dict(char2id),
        "id2char": dict(id2char),
        "num_labels": len(label2id),
        "vocab_size": len(char2id),
        "best_dev_accuracy": best_dev_accuracy,
        "best_epoch": best_epoch,
        "git_commit": git_commit,
    }


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> Tuple[ScratchNER, Dict[int, str], Dict[str, int], Dict[str, int], Dict[int, str], Dict[str, Any]]:
    '''从 Stage5 checkpoint 重建模型。'''

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    label2id = dict(checkpoint["label2id"])
    id2label = {
        int(index): label
        for index, label in checkpoint["id2label"].items()
    }
    char2id = dict(checkpoint["char2id"])
    id2char = {
        int(index): char
        for index, char in checkpoint["id2char"].items()
    }

    model = _build_stage5_model(
        config=config,
        vocab_size=len(char2id),
        label2id=label2id,
        pad_idx=int(char2id["<PAD>"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, id2label, label2id, char2id, id2char, config


def _predict_batch_examples(
    model: ScratchNER,
    examples: Sequence[ScratchNERExample],
    collator: ScratchBatchCollator,
    id2label: Dict[int, str],
    device: str,
) -> List[Dict[str, Any]]:
    '''批量预测未超过 max_seq_len 的样本。'''

    if not examples:
        return []

    batch = collator(examples)
    batch = move_batch_to_device(batch, device)
    predicted_sequences = _forward_batch(model, batch)

    results: List[Dict[str, Any]] = []
    for batch_index, predicted_ids in enumerate(predicted_sequences):
        original_length = int(batch["original_lengths"][batch_index])
        results.append(
            {
                "sample_id": int(batch["sample_ids"][batch_index]),
                "predicted_tags": [id2label[prediction_id] for prediction_id in predicted_ids[:original_length]],
            }
        )
    return results


def build_char_sliding_windows(
    sequence_length: int,
    window_size: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    '''按字符数构造滑窗区间 `[start, end)`。'''

    if sequence_length <= window_size:
        return [(0, sequence_length)]

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < sequence_length:
        end = min(sequence_length, start + window_size)
        windows.append((start, end))
        if end >= sequence_length:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return windows


def merge_window_predictions(
    original_length: int,
    window_predictions: Sequence[Dict[str, Any]],
    fill_label: str,
) -> Tuple[List[str], Dict[str, Any]]:
    '''按 center_priority 融合多个窗口的预测。'''

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
                merged_predictions[global_index] = str(predicted_tag)
                best_edge_distance[global_index] = edge_distance
                best_center_distance[global_index] = center_distance

    uncovered_positions = [
        index
        for index, predicted_tag in enumerate(merged_predictions)
        if predicted_tag is None
    ]
    for uncovered_index in uncovered_positions:
        merged_predictions[uncovered_index] = fill_label

    return [str(tag) for tag in merged_predictions], {
        "uncovered_positions": uncovered_positions,
        "coverage_min": min(coverage_count) if coverage_count else 0,
        "coverage_max": max(coverage_count) if coverage_count else 0,
    }


def _predict_long_example(
    model: ScratchNER,
    example: ScratchNERExample,
    collator: ScratchBatchCollator,
    id2label: Dict[int, str],
    device: str,
    config: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    '''对超长样本执行字符级滑窗预测。'''

    fill_label = "O" if "O" in id2label.values() else id2label.get(0, "O")
    windows = build_char_sliding_windows(
        sequence_length=len(example.char_ids),
        window_size=int(config["max_seq_len"]),
        overlap=int(config["sliding_overlap"]),
    )

    raw_window_examples = [
        ScratchNERExample(
            sample_id=index,
            chars=list(example.chars[start:end]),
            char_ids=list(example.char_ids[start:end]),
            tags=None,
            tag_ids=None,
        )
        for index, (start, end) in enumerate(windows)
    ]

    merged_window_predictions: List[Dict[str, Any]] = []
    for batch_start in range(0, len(raw_window_examples), int(config["batch_size"])):
        batch_examples = raw_window_examples[batch_start : batch_start + int(config["batch_size"])]
        batch_results = _predict_batch_examples(
            model=model,
            examples=batch_examples,
            collator=collator,
            id2label=id2label,
            device=device,
        )
        for local_offset, result in enumerate(batch_results):
            window_start, _ = windows[batch_start + local_offset]
            merged_window_predictions.append(
                {
                    "start": window_start,
                    "predicted_tags": result["predicted_tags"],
                }
            )

    merged_predictions, merge_meta = merge_window_predictions(
        original_length=len(example.char_ids),
        window_predictions=merged_window_predictions,
        fill_label=fill_label,
    )
    return merged_predictions, {
        "window_count": len(windows),
        "uncovered_positions": merge_meta["uncovered_positions"],
        "coverage_min": merge_meta["coverage_min"],
        "coverage_max": merge_meta["coverage_max"],
    }


def predict_examples(
    model: ScratchNER,
    examples: Sequence[ScratchNERExample],
    collator: ScratchBatchCollator,
    id2label: Dict[int, str],
    device: str,
    config: Dict[str, Any],
    description: str,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    '''预测完整样本序列，并对超长样本启用滑窗。'''

    model.eval()
    all_predictions: List[List[str]] = [[] for _ in range(len(examples))]
    sliding_sample_count = 0
    sliding_window_count = 0
    uncovered_samples: List[Dict[str, Any]] = []
    pending_examples: List[ScratchNERExample] = []

    def flush_pending() -> None:
        nonlocal pending_examples
        batch_results = _predict_batch_examples(
            model=model,
            examples=pending_examples,
            collator=collator,
            id2label=id2label,
            device=device,
        )
        for result in batch_results:
            all_predictions[int(result["sample_id"])] = list(result["predicted_tags"])
        pending_examples = []

    with torch.no_grad():
        for example in tqdm(
            examples,
            desc=description,
            leave=False,
            disable=not sys.stdout.isatty(),
        ):
            if len(example.char_ids) <= int(config["max_seq_len"]):
                pending_examples.append(example)
                if len(pending_examples) >= int(config["batch_size"]):
                    flush_pending()
                continue

            if pending_examples:
                flush_pending()

            sliding_sample_count += 1
            long_predictions, long_meta = _predict_long_example(
                model=model,
                example=example,
                collator=collator,
                id2label=id2label,
                device=device,
                config=config,
            )
            all_predictions[int(example.sample_id)] = list(long_predictions)
            sliding_window_count += int(long_meta["window_count"])
            if long_meta["uncovered_positions"]:
                uncovered_samples.append(
                    {
                        "sample_id": int(example.sample_id),
                        "uncovered_positions": long_meta["uncovered_positions"],
                    }
                )

        if pending_examples:
            flush_pending()

    return all_predictions, {
        "sliding_sample_count": sliding_sample_count,
        "sliding_window_count": sliding_window_count,
        "uncovered_sample_count": len(uncovered_samples),
        "uncovered_samples": uncovered_samples[:20],
    }


def _build_dev_metric_report(
    predicted_label_sequences: Sequence[Sequence[str]],
    gold_label_sequences: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    '''按现有 predict.py 的口径构造开发集指标。'''

    dev_token_metrics = compute_token_accuracy_from_label_sequences(
        predicted_label_sequences=predicted_label_sequences,
        gold_label_sequences=gold_label_sequences,
    )
    # 对单标签多分类的 token 预测任务，micro-F1 与 accuracy 数值等价：
    # 每个有效 token 都且仅有一个预测标签，因此 micro-precision = micro-recall = accuracy。
    token_precision = float(dev_token_metrics["accuracy"])
    token_recall = float(dev_token_metrics["accuracy"])
    token_f1 = (
        2.0 * token_precision * token_recall / (token_precision + token_recall)
        if (token_precision + token_recall)
        else 0.0
    )
    dev_entity_metrics = compute_entity_f1_from_label_sequences(
        predicted_label_sequences=predicted_label_sequences,
        gold_label_sequences=gold_label_sequences,
    )
    return {
        "token_accuracy": round(float(dev_token_metrics["accuracy"]), 6),
        "token_precision": round(token_precision, 6),
        "token_recall": round(token_recall, 6),
        "token_f1": round(token_f1, 6),
        "token_correct": int(dev_token_metrics["correct"]),
        "token_total": int(dev_token_metrics["total"]),
        "entity_precision": round(float(dev_entity_metrics["precision"]), 6),
        "entity_recall": round(float(dev_entity_metrics["recall"]), 6),
        "entity_f1": round(float(dev_entity_metrics["f1"]), 6),
        "correct_entities": int(dev_entity_metrics["correct_entities"]),
        "predicted_entities": int(dev_entity_metrics["predicted_entities"]),
        "gold_entities": int(dev_entity_metrics["gold_entities"]),
    }


def _build_experiment_summary_row(
    config: Dict[str, Any],
    training_log_rows: Sequence[Dict[str, Any]],
    runtime_length_stats: Dict[str, Any],
    device_name: str,
) -> Dict[str, Any]:
    '''构造 Stage5 单实验摘要。'''

    best_row = max(training_log_rows, key=lambda row: float(row["dev_accuracy"]))
    best_f1 = max(float(row.get("dev_f1", 0.0)) for row in training_log_rows)
    total_training_seconds = sum(float(row.get("epoch_seconds", 0.0)) for row in training_log_rows)
    return {
        "experiment_name": config["experiment_name"],
        "variant": config["variant"],
        "profile": config["profile"],
        "model_name": "scratch_transformer_encoder",
        "resolved_model_name": "scratch_transformer_encoder",
        "use_refinement": False,
        "use_crf": True,
        "bio_constraint_mode": config["bio_constraint_mode"],
        "use_sliding_window": True,
        "use_bucket_batching": config["use_bucket_batching"],
        "use_low_freq_char_dropout": False,
        "low_freq_char_threshold": 0,
        "low_freq_char_dropout_prob": 0.0,
        "max_len": config["max_seq_len"],
        "batch_size": config["batch_size"],
        "grad_accum_steps": config["grad_accum_steps"],
        "lr_backbone": config["peak_lr"],
        "lr_refinement": 0.0,
        "lr_classifier": config["classifier_lr"],
        "lr_crf": config["crf_lr"],
        "best_epoch": int(best_row["epoch"]),
        "best_dev_accuracy": round(float(best_row["dev_accuracy"]), 6),
        "dev_f1_at_best_epoch": round(float(best_row.get("dev_f1", 0.0)), 6),
        "best_f1_epoch": int(
            max(training_log_rows, key=lambda row: float(row.get("dev_f1", -1.0)))["epoch"]
        ),
        "best_dev_f1": round(best_f1, 6),
        "training_time": format_seconds(total_training_seconds),
        "train_over_max_len_count": runtime_length_stats["train"]["over_max_len_count"],
        "dev_over_max_len_count": runtime_length_stats["dev"]["over_max_len_count"],
        "test_over_max_len_count": runtime_length_stats["test"]["over_max_len_count"],
        "device": device_name,
        "notes": "Stage5 scratch Transformer + CRF",
    }


def train_scratch(config: Mapping[str, Any]) -> Path:
    '''Stage5 训练主入口。'''

    normalized_config = _clone_stage5_config(config)
    set_seed(int(normalized_config["seed"]))
    device = choose_device(str(normalized_config["profile"]))
    git_commit = get_git_commit(normalized_config["project_root"])
    prepared = _prepare_stage5_data(normalized_config)

    label2id = prepared["label2id"]
    id2label = prepared["id2label"]
    char2id = prepared["char2id"]
    id2char = prepared["id2char"]
    pad_idx = prepared["pad_idx"]
    collator = prepared["collator"]

    model = _build_stage5_model(
        config=normalized_config,
        vocab_size=len(char2id),
        label2id=label2id,
        pad_idx=pad_idx,
    )
    pretrained_embedding_path = normalized_config.get("pretrained_embedding_path")
    if pretrained_embedding_path:
        embedding_matrix = _load_pretrained_embedding_matrix(Path(str(pretrained_embedding_path)))
        model.load_pretrained_embeddings(embedding_matrix)

    model.to(device)
    optimizer = create_optimizer_scratch(model, normalized_config)
    steps_per_epoch = math.ceil(
        len(prepared["train_loader"]) / max(1, int(normalized_config["grad_accum_steps"]))
    )
    total_training_steps = steps_per_epoch * int(normalized_config["num_epochs"])
    scheduler = create_scheduler_scratch(
        optimizer=optimizer,
        total_training_steps=total_training_steps,
        warmup_ratio=float(normalized_config["warmup_ratio"]),
        min_lr=float(normalized_config["min_lr"]),
    )

    fgm = FGM(model) if bool(normalized_config["use_fgm"]) else None
    ema = EMA(model, decay=float(normalized_config["ema_decay"])) if bool(normalized_config["use_ema"]) else None

    experiment_dirs = create_experiment_dirs(
        normalized_config["output_root"],
        normalized_config["experiment_name"],
    )
    config_snapshot = dict(normalized_config)
    save_json(experiment_dirs["experiment_dir"] / "config.json", config_snapshot)
    save_json(experiment_dirs["checkpoints_dir"] / "label2id.json", label2id)
    save_json(
        experiment_dirs["checkpoints_dir"] / "id2label.json",
        {str(index): label for index, label in id2label.items()},
    )
    save_json(experiment_dirs["checkpoints_dir"] / "char2id.json", char2id)
    save_json(
        experiment_dirs["checkpoints_dir"] / "id2char.json",
        {str(index): char for index, char in id2char.items()},
    )

    best_checkpoint_path = experiment_dirs["checkpoints_dir"] / "best_model.pt"
    last_checkpoint_path = experiment_dirs["checkpoints_dir"] / "last_model.pt"
    training_log_rows: List[Dict[str, Any]] = []
    best_dev_accuracy = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    global_step = 0

    print("=== Stage5 Scratch Training ===")
    print(f"Experiment: {normalized_config['experiment_name']}")
    print(f"Variant: {normalized_config['variant']}")
    print(f"Device: {device}")
    print(f"GPU name: {get_gpu_name(device)}")
    print(f"Train steps per epoch: {len(prepared['train_loader'])}")
    print(f"Dev steps per epoch: {len(prepared['dev_loader'])}")
    print(
        "Over max_seq_len count: "
        f"train={prepared['runtime_length_stats']['train']['over_max_len_count']}, "
        f"dev={prepared['runtime_length_stats']['dev']['over_max_len_count']}, "
        f"test={prepared['runtime_length_stats']['test']['over_max_len_count']}"
    )
    if git_commit:
        print(f"Git commit: {git_commit}")

    for epoch in range(1, int(normalized_config["num_epochs"]) + 1):
        with timer() as epoch_timer:
            train_loss, optimizer_steps = train_one_epoch_scratch(
                model=model,
                dataloader=prepared["train_loader"],
                optimizer=optimizer,
                scheduler=scheduler,
                fgm=fgm,
                ema=ema,
                device=device,
                grad_accum_steps=int(normalized_config["grad_accum_steps"]),
                grad_clip_norm=float(normalized_config["grad_clip_norm"]),
                fgm_epsilon=float(normalized_config["fgm_epsilon"]),
            )
            global_step += optimizer_steps

            if ema is not None:
                ema.apply_shadow()

            dev_metrics = evaluate_model(
                model=model,
                dataloader=prepared["dev_loader"],
                device=device,
                ignore_index=int(normalized_config["ignore_index"]),
                description=f"Stage5 Dev Epoch {epoch}",
            )

        dev_accuracy = float(dev_metrics["token_accuracy"])
        # evaluate.py 当前返回 token-level accuracy 和 entity-level F1。
        # 对单标签 token 分类，token micro-F1 与 token accuracy 数值相同，
        # 因此这里显式记录 dev_token_f1，便于日志和汇报保持完整。
        dev_token_f1 = dev_accuracy
        dev_f1 = float(dev_metrics["entity_f1"])
        epoch_seconds = float(epoch_timer["elapsed_seconds"])
        encoder_lr_end = get_group_lr(optimizer, "encoder", float(normalized_config["peak_lr"]))
        embedding_lr_end = get_group_lr(
            optimizer,
            "embedding",
            float(normalized_config["embedding_lr"]),
        )

        is_best = dev_accuracy > best_dev_accuracy
        if is_best:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        log_row = {
            "epoch": epoch,
            "global_step": global_step,
            "experiment_variant": normalized_config["variant"],
            "use_refinement": False,
            "use_crf": True,
            "use_bucket_batching": bool(normalized_config["use_bucket_batching"]),
            "use_low_freq_char_dropout": False,
            "bio_constraint_mode": normalized_config["bio_constraint_mode"],
            "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
            "train_over_max_len_count": prepared["runtime_length_stats"]["train"]["over_max_len_count"],
            "dev_over_max_len_count": prepared["runtime_length_stats"]["dev"]["over_max_len_count"],
            "test_over_max_len_count": prepared["runtime_length_stats"]["test"]["over_max_len_count"],
            "train_loss": round(train_loss, 6),
            "dev_accuracy": round(dev_accuracy, 6),
            "dev_token_f1": round(dev_token_f1, 6),
            "dev_loss": round(float(dev_metrics["dev_loss"]), 6),
            "dev_precision": round(float(dev_metrics["entity_precision"]), 6),
            "dev_recall": round(float(dev_metrics["entity_recall"]), 6),
            "dev_f1": round(dev_f1, 6),
            "dev_gold_entities": int(dev_metrics["gold_entity_count"]),
            "dev_predicted_entities": int(dev_metrics["predicted_entity_count"]),
            "dev_correct_entities": int(dev_metrics["correct_entity_count"]),
            "lr": round(encoder_lr_end, 10),
            "embedding_lr_end": round(embedding_lr_end, 10),
            "encoder_lr_end": round(encoder_lr_end, 10),
            "epoch_time": format_seconds(epoch_seconds),
            "epoch_seconds": round(epoch_seconds, 4),
            "is_best": is_best,
        }
        training_log_rows.append(log_row)

        checkpoint_payload = _build_stage5_checkpoint_payload(
            model=model,
            config=normalized_config,
            epoch=epoch,
            global_step=global_step,
            best_dev_accuracy=best_dev_accuracy,
            best_epoch=best_epoch,
            label2id=label2id,
            id2label=id2label,
            char2id=char2id,
            id2char=id2char,
            git_commit=git_commit,
        )
        save_checkpoint(last_checkpoint_path, checkpoint_payload)
        if is_best:
            save_checkpoint(best_checkpoint_path, checkpoint_payload)

        training_log_payload = {
            "experiment_name": normalized_config["experiment_name"],
            "experiment_variant": normalized_config["variant"],
            "profile": normalized_config["profile"],
            "requested_model_name": "scratch_transformer_encoder",
            "resolved_model_name": model.resolved_model_name,
            "use_refinement": False,
            "use_crf": True,
            "use_bucket_batching": bool(normalized_config["use_bucket_batching"]),
            "bio_constraint_mode": normalized_config["bio_constraint_mode"],
            "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
            "best_epoch": best_epoch,
            "best_dev_accuracy": round(best_dev_accuracy, 6),
            "best_dev_f1": round(
                max(float(row.get("dev_f1", 0.0)) for row in training_log_rows),
                6,
            ),
            "git_commit": git_commit,
            "notes": "Stage5 scratch Transformer + CRF",
            "logs": training_log_rows,
        }
        save_json(experiment_dirs["experiment_dir"] / "training_log.json", training_log_payload)
        write_csv_rows(experiment_dirs["results_dir"] / "training_log.csv", training_log_rows)
        write_csv_rows(
            experiment_dirs["results_dir"] / "experiment_summary.csv",
            [
                _build_experiment_summary_row(
                    config=normalized_config,
                    training_log_rows=training_log_rows,
                    runtime_length_stats=prepared["runtime_length_stats"],
                    device_name=get_gpu_name(device),
                )
            ],
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={log_row['train_loss']:.6f} | "
            f"dev_acc={log_row['dev_accuracy']:.6f} | "
            f"dev_token_f1={log_row['dev_token_f1']:.6f} | "
            f"dev_f1={log_row['dev_f1']:.6f} | "
            f"lr={log_row['lr']:.8f} | "
            f"time={log_row['epoch_time']}"
        )

        if ema is not None:
            ema.restore()

        if stale_epochs >= int(normalized_config["early_stop_patience"]):
            print(
                "Early stopping triggered: "
                f"stale_epochs={stale_epochs}, best_epoch={best_epoch}, "
                f"best_dev_accuracy={best_dev_accuracy:.6f}"
            )
            break

    best_model, reload_id2label, _, _, _, reload_config = _load_model_from_checkpoint(
        best_checkpoint_path,
        device=device,
    )
    prediction_collator = ScratchBatchCollator(
        pad_idx=int(char2id["<PAD>"]),
        ignore_index=int(reload_config["ignore_index"]),
        max_len=int(reload_config["max_seq_len"]),
    )

    dev_predictions, dev_stats = predict_examples(
        model=best_model,
        examples=prepared["raw_dev_examples"],
        collator=prediction_collator,
        id2label=reload_id2label,
        device=device,
        config=reload_config,
        description="Stage5 Dev Predict",
    )
    test_predictions, test_stats = predict_examples(
        model=best_model,
        examples=prepared["raw_test_examples"],
        collator=prediction_collator,
        id2label=reload_id2label,
        device=device,
        config=reload_config,
        description="Stage5 Test Predict",
    )

    predictions_dir = experiment_dirs["predictions_dir"]
    results_dir = experiment_dirs["results_dir"]
    figures_dir = experiment_dirs["figures_dir"]
    dev_pred_path = predictions_dir / "dev_pred.txt"
    test_pred_path = predictions_dir / "test_pred.txt"

    write_prediction_file(dev_pred_path, dev_predictions)
    write_prediction_file(test_pred_path, test_predictions)

    dev_metric_report = _build_dev_metric_report(
        predicted_label_sequences=dev_predictions,
        gold_label_sequences=prepared["dev_tag_lines"],
    )
    dev_validation = validate_prediction_output(
        text_lines=prepared["dev_text_lines"],
        pred_lines=dev_predictions,
    )
    test_validation = validate_prediction_output(
        text_lines=prepared["test_text_lines"],
        pred_lines=test_predictions,
    )
    validation_report = {
        "experiment_dir": str(experiment_dirs["experiment_dir"]),
        "experiment_variant": normalized_config["variant"],
        "use_refinement": False,
        "use_crf": True,
        "bio_constraint_mode": normalized_config["bio_constraint_mode"],
        "effective_bio_constraint_mode": best_model.effective_bio_constraint_mode,
        "use_sliding_window": True,
        "sliding_window_size": int(reload_config["max_seq_len"]),
        "sliding_overlap": int(reload_config["sliding_overlap"]),
        "sliding_merge_strategy": "center_priority",
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

    data_report_path = normalized_config["data_dir"] / "data_report.json"
    label_stats_path = normalized_config["data_dir"] / "label_stats.json"
    if data_report_path.exists() and label_stats_path.exists():
        generate_data_figures(
            data_report_path=data_report_path,
            label_stats_path=label_stats_path,
            figure_dir=figures_dir,
        )
    generate_training_figures(
        training_log_path=experiment_dirs["experiment_dir"] / "training_log.json",
        figure_dir=figures_dir,
    )
    generate_project_summary_artifacts(
        output_root=normalized_config["output_root"],
        data_dir=normalized_config["data_dir"],
    )

    print("=== Stage5 Prediction Summary ===")
    print(f"Experiment dir: {experiment_dirs['experiment_dir']}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best dev accuracy: {best_dev_accuracy:.6f}")
    print(f"Dev prediction metrics: {results_dir / 'dev_prediction_metrics.json'}")
    print(f"Validation report: {results_dir / 'output_validation.json'}")
    print(f"Test prediction file: {test_pred_path}")
    return experiment_dirs["experiment_dir"]

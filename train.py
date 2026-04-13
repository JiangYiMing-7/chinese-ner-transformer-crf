'''训练与恢复模块。

负责三条路线的训练入口、checkpoint 保存、resume 恢复、
训练日志落盘，以及实验总表生成。
'''

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import NERConfig
from data import NERBatchCollator, build_train_dataloader, prepare_datasets_and_tokenizer
from evaluate import evaluate_model
from model_backbone import RobertaLinearNER
from plot_curves import (
    generate_data_figures,
    generate_project_summary_artifacts,
    generate_training_figures,
)
from utils import (
    bind_experiment_dirs,
    choose_device,
    compute_label_mapping_digest,
    create_experiment_dirs,
    format_seconds,
    get_git_commit,
    get_gpu_name,
    load_checkpoint,
    load_json,
    move_batch_to_device,
    resolve_resume_artifacts,
    save_checkpoint,
    save_json,
    set_seed,
    timer,
    write_csv_rows,
)


def _add_param_group(
    grouped_parameters: List[Dict[str, Any]],
    named_parameters: List[Any],
    name_prefix: str,
    learning_rate: float,
    weight_decay: float,
) -> None:
    '''向优化器中追加一组参数。

    这里显式拆分 decay / no_decay，避免把 LayerNorm 和 bias 也做 weight decay。
    '''

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_parameters = [
        parameter
        for name, parameter in named_parameters
        if parameter.requires_grad and not any(term in name for term in no_decay)
    ]
    no_decay_parameters = [
        parameter
        for name, parameter in named_parameters
        if parameter.requires_grad and any(term in name for term in no_decay)
    ]

    if decay_parameters:
        grouped_parameters.append(
            {
                "params": decay_parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": f"{name_prefix}_decay",
            }
        )
    if no_decay_parameters:
        grouped_parameters.append(
            {
                "params": no_decay_parameters,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "group_name": f"{name_prefix}_no_decay",
            }
        )


def create_optimizer(model: RobertaLinearNER, config: NERConfig) -> AdamW:
    '''创建分层学习率优化器。'''

    optimizer_grouped_parameters: List[Dict[str, Any]] = []
    # backbone 直接继承预训练参数，学习率保持最小，避免微调时把已有表示空间破坏得过快。
    _add_param_group(
        grouped_parameters=optimizer_grouped_parameters,
        named_parameters=list(model.backbone.named_parameters()),
        name_prefix="backbone",
        learning_rate=config.lr_backbone,
        weight_decay=config.weight_decay,
    )
    # classifier 从随机初始化开始，通常需要比 backbone 更大的学习率才能尽快贴合当前任务。
    _add_param_group(
        grouped_parameters=optimizer_grouped_parameters,
        named_parameters=list(model.classifier.named_parameters()),
        name_prefix="classifier",
        learning_rate=config.lr_classifier,
        weight_decay=config.weight_decay,
    )
    if model.use_refinement and model.refinement is not None:
        # refinement 是新增的表征层，单独分组后可以在 stage3 里使用更保守的专用学习率。
        _add_param_group(
            grouped_parameters=optimizer_grouped_parameters,
            named_parameters=list(model.refinement.named_parameters()),
            name_prefix="refinement",
            learning_rate=config.lr_refinement,
            weight_decay=config.weight_decay,
        )
    if model.use_crf and model.crf is not None:
        # CRF 同样独立分组，便于观察“标签转移层”对学习率是否更敏感。
        _add_param_group(
            grouped_parameters=optimizer_grouped_parameters,
            named_parameters=list(model.crf.named_parameters()),
            name_prefix="crf",
            learning_rate=config.lr_crf,
            weight_decay=config.weight_decay,
        )
    return AdamW(optimizer_grouped_parameters)


def create_scheduler(
    optimizer: AdamW,
    total_training_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    '''创建 warmup + linear decay 调度器。'''

    warmup_steps = int(total_training_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        '''按 warmup + 线性衰减返回当前步的学习率比例。'''

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # warmup 之后直接线性衰减到 0，对所有参数组共用同一条缩放曲线。
        remaining_steps = total_training_steps - current_step
        decay_steps = total_training_steps - warmup_steps
        return max(0.0, float(remaining_steps) / float(max(1, decay_steps)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_group_lr(optimizer: AdamW, prefix: str, fallback: float) -> float:
    '''读取某类参数组当前学习率。'''

    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", ""))
        if group_name.startswith(prefix):
            return float(group["lr"])
    return float(fallback)


def train_one_epoch(
    model: RobertaLinearNER,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: str,
    grad_accum_steps: int,
    grad_clip_norm: float,
) -> Tuple[float, int]:
    '''训练单个 epoch，并返回平均 loss 与参数更新步数。'''

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    optimizer_steps = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
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
            # 一旦 loss 已经出现 inf / nan，继续训练只会把异常写入 checkpoint 和日志。
            raise RuntimeError(
                f"Non-finite loss detected at training step {step}: {float(loss.item())}"
            )
        total_loss += float(loss.item())

        (loss / grad_accum_steps).backward()

        should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
        if should_step:
            # 先裁剪再 step，避免长句 batch 或新增层学习率偏大时梯度突然爆掉。
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    return total_loss / max(1, len(dataloader)), optimizer_steps


def _move_optimizer_state_to_device(optimizer: AdamW, device: str) -> None:
    '''把恢复后的优化器状态迁移到当前设备。'''

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _seconds_from_log_row(log_row: Dict[str, Any]) -> float:
    '''从日志行中提取秒数。'''

    if "epoch_seconds" in log_row:
        return float(log_row["epoch_seconds"])

    time_text = str(log_row.get("epoch_time", "00:00:00"))
    try:
        hours, minutes, seconds = [int(part) for part in time_text.split(":")]
    except Exception:
        return 0.0
    return float(hours * 3600 + minutes * 60 + seconds)


def _build_experiment_summary(
    config: NERConfig,
    training_log_rows: List[Dict[str, Any]],
    requested_model_name: str,
    resolved_model_name: str,
    device_name: str,
    runtime_length_stats: Dict[str, Any],
    total_training_seconds: float,
    notes: str,
) -> List[Dict[str, Any]]:
    '''构造实验总表。'''

    best_row = max(training_log_rows, key=lambda row: row["dev_accuracy"])
    # 兼容旧实验日志：早期正式训练只记录了 dev_accuracy，没有逐 epoch 的 dev_f1。
    has_dev_f1 = any("dev_f1" in row for row in training_log_rows)
    best_f1_row = (
        max(training_log_rows, key=lambda row: float(row.get("dev_f1", -1.0)))
        if has_dev_f1
        else None
    )
    return [
        {
            "experiment_name": config.experiment_name,
            "variant": config.experiment_variant,
            "profile": config.profile,
            "model_name": requested_model_name,
            "resolved_model_name": resolved_model_name,
            "use_refinement": config.use_refinement,
            "use_crf": config.use_crf,
            "bio_constraint_mode": config.bio_constraint_mode,
            "use_sliding_window": config.use_sliding_window,
            "use_bucket_batching": config.use_bucket_batching,
            "use_low_freq_char_dropout": config.use_low_freq_char_dropout,
            "low_freq_char_threshold": (
                config.low_freq_char_threshold if config.use_low_freq_char_dropout else 0
            ),
            "low_freq_char_dropout_prob": (
                config.low_freq_char_dropout_prob if config.use_low_freq_char_dropout else 0.0
            ),
            "max_len": config.max_len,
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "lr_backbone": config.lr_backbone,
            "lr_refinement": config.lr_refinement if config.use_refinement else 0.0,
            "lr_classifier": config.lr_classifier,
            "lr_crf": config.lr_crf if config.use_crf else 0.0,
            "best_epoch": best_row["epoch"],
            "best_dev_accuracy": round(best_row["dev_accuracy"], 6),
            "dev_f1_at_best_epoch": (
                round(float(best_row.get("dev_f1", 0.0)), 6) if has_dev_f1 else ""
            ),
            "best_f1_epoch": best_f1_row["epoch"] if best_f1_row is not None else "",
            "best_dev_f1": (
                round(float(best_f1_row["dev_f1"]), 6) if best_f1_row is not None else ""
            ),
            "training_time": format_seconds(total_training_seconds),
            "train_over_max_len_count": runtime_length_stats["train"]["over_max_len_count"],
            "dev_over_max_len_count": runtime_length_stats["dev"]["over_max_len_count"],
            "test_over_max_len_count": runtime_length_stats["test"]["over_max_len_count"],
            "device": device_name,
            "notes": notes,
        }
    ]


def _build_model(
    config: NERConfig,
    label2id: Dict[str, int],
) -> RobertaLinearNER:
    '''根据当前配置构造模型。'''

    return RobertaLinearNER(
        model_name=config.model_name,
        fallback_model_name=config.fallback_model_name,
        num_labels=len(label2id),
        dropout=config.dropout,
        ignore_index=config.ignore_index,
        use_refinement=config.use_refinement,
        refine_num_layers=config.refine_num_layers,
        refine_num_heads=config.refine_num_heads,
        refine_ffn_dim=config.refine_ffn_dim,
        refine_dropout=config.refine_dropout,
        use_crf=config.use_crf,
        bio_constraint_mode=config.bio_constraint_mode,
        label2id=label2id,
    )


def _load_existing_training_rows(experiment_dir: Path) -> List[Dict[str, Any]]:
    '''读取已有训练日志。'''

    training_log_path = experiment_dir / "training_log.json"
    if not training_log_path.exists():
        return []
    payload = load_json(training_log_path)
    # resume 时直接沿用旧日志，保证 training_log.json 反映的是完整连续训练过程。
    return list(payload.get("logs", []))


def _validate_resume_compatibility(
    config: NERConfig,
    trainer_state: Dict[str, Any],
    current_label2id: Dict[str, int],
) -> List[str]:
    '''检查 resume 是否与当前配置兼容。

    这里把兼容性分成硬检查和软检查。硬检查失败会直接中止恢复，软检查只提示可能的环境差异。
    '''

    current_digest = compute_label_mapping_digest(current_label2id)
    hard_checks = {
        "experiment_variant": config.experiment_variant,
        "requested_model_name": config.model_name,
        "use_crf": config.use_crf,
        "use_refinement": config.use_refinement,
        "refine_num_layers": config.refine_num_layers,
        "refine_num_heads": config.refine_num_heads,
        "refine_ffn_dim": config.refine_ffn_dim,
        "refine_dropout": config.refine_dropout,
        "bio_constraint_mode": config.bio_constraint_mode,
        "label2id_digest": current_digest,
        "num_labels": len(current_label2id),
    }
    for field_name, current_value in hard_checks.items():
        saved_value = trainer_state.get(field_name)
        if saved_value != current_value:
            raise ValueError(
                f"Resume compatibility check failed for '{field_name}': "
                f"saved={saved_value!r}, current={current_value!r}"
            )

    saved_low_freq_enabled = bool(trainer_state.get("use_low_freq_char_dropout", False))
    if saved_low_freq_enabled != config.use_low_freq_char_dropout:
        raise ValueError(
            "Resume compatibility check failed for 'use_low_freq_char_dropout': "
            f"saved={saved_low_freq_enabled!r}, current={config.use_low_freq_char_dropout!r}"
        )
    if config.use_low_freq_char_dropout:
        saved_threshold = trainer_state.get("low_freq_char_threshold")
        saved_dropout_prob = trainer_state.get("low_freq_char_dropout_prob")
        if saved_threshold != config.low_freq_char_threshold:
            raise ValueError(
                "Resume compatibility check failed for 'low_freq_char_threshold': "
                f"saved={saved_threshold!r}, current={config.low_freq_char_threshold!r}"
            )
        if saved_dropout_prob != config.low_freq_char_dropout_prob:
            raise ValueError(
                "Resume compatibility check failed for 'low_freq_char_dropout_prob': "
                f"saved={saved_dropout_prob!r}, current={config.low_freq_char_dropout_prob!r}"
            )

    warnings: List[str] = []
    soft_checks = {
        "git_commit": trainer_state.get("git_commit"),
        "resolved_model_name": trainer_state.get("resolved_model_name"),
    }
    current_soft_values = {
        "git_commit": get_git_commit(config.project_root),
        "resolved_model_name": None,
    }
    for field_name, saved_value in soft_checks.items():
        current_value = current_soft_values[field_name]
        if saved_value and current_value and saved_value != current_value:
            warnings.append(
                f"Resume metadata mismatch for '{field_name}': "
                f"saved={saved_value!r}, current={current_value!r}"
            )
    return warnings


def _build_model_checkpoint_payload(
    model: RobertaLinearNER,
    config: NERConfig,
    epoch: int,
    global_step: int,
    best_dev_accuracy: float,
    best_epoch: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    git_commit: Optional[str],
) -> Dict[str, Any]:
    '''构造 best/last model checkpoint 内容。'''

    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        # 同时保存 requested / resolved model name，方便之后区分“想加载什么模型”和“实际用了哪个本地快照”。
        "requested_model_name": config.model_name,
        "resolved_model_name": model.resolved_model_name,
        "experiment_variant": config.experiment_variant,
        "use_refinement": config.use_refinement,
        "refine_num_layers": config.refine_num_layers,
        "refine_num_heads": config.refine_num_heads,
        "refine_ffn_dim": config.refine_ffn_dim,
        "refine_dropout": config.refine_dropout,
        "use_crf": config.use_crf,
        "use_bucket_batching": config.use_bucket_batching,
        "bio_constraint_mode": config.bio_constraint_mode,
        "use_low_freq_char_dropout": config.use_low_freq_char_dropout,
        "low_freq_char_threshold": config.low_freq_char_threshold,
        "low_freq_char_dropout_prob": config.low_freq_char_dropout_prob,
        "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
        # 标签映射属于 checkpoint 可恢复性的关键元数据，resume 和 predict 都会再次核对它。
        "label2id": label2id,
        "id2label": {str(index): label for index, label in id2label.items()},
        "label2id_digest": compute_label_mapping_digest(label2id),
        "num_labels": len(label2id),
        "best_dev_accuracy": best_dev_accuracy,
        "best_epoch": best_epoch,
        "git_commit": git_commit,
    }


def _save_training_state(
    experiment_dirs: Dict[str, Path],
    model: RobertaLinearNER,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: NERConfig,
    epoch: int,
    global_step: int,
    best_dev_accuracy: float,
    best_epoch: int,
    stale_epochs: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    git_commit: Optional[str],
) -> None:
    '''保存恢复训练所需的全部状态。

    这里单独落盘 optimizer、scheduler、trainer_state 和标签映射，是为了让恢复训练更透明，
    也便于人工检查 checkpoint 是否完整。
    '''

    checkpoints_dir = experiment_dirs["checkpoints_dir"]
    label_digest = compute_label_mapping_digest(label2id)

    save_json(checkpoints_dir / "label2id.json", label2id)
    save_json(
        checkpoints_dir / "id2label.json",
        {str(index): label for index, label in id2label.items()},
    )
    save_checkpoint(
        checkpoints_dir / "optimizer.pt",
        {"optimizer_state_dict": optimizer.state_dict()},
    )
    save_checkpoint(
        checkpoints_dir / "scheduler.pt",
        {"scheduler_state_dict": scheduler.state_dict()},
    )
    trainer_state = {
        "current_epoch": epoch,
        "global_step": global_step,
        "best_dev_accuracy": round(best_dev_accuracy, 6),
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "experiment_variant": config.experiment_variant,
        "profile": config.profile,
        "requested_model_name": config.model_name,
        "resolved_model_name": model.resolved_model_name,
        "use_crf": config.use_crf,
        "use_refinement": config.use_refinement,
        "refine_num_layers": config.refine_num_layers,
        "refine_num_heads": config.refine_num_heads,
        "refine_ffn_dim": config.refine_ffn_dim,
        "refine_dropout": config.refine_dropout,
        "use_sliding_window": config.use_sliding_window,
        "use_bucket_batching": config.use_bucket_batching,
        "bio_constraint_mode": config.bio_constraint_mode,
        "use_low_freq_char_dropout": config.use_low_freq_char_dropout,
        "low_freq_char_threshold": config.low_freq_char_threshold,
        "low_freq_char_dropout_prob": config.low_freq_char_dropout_prob,
        "max_len": config.max_len,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "lr_backbone": config.lr_backbone,
        "lr_refinement": config.lr_refinement if config.use_refinement else 0.0,
        "lr_classifier": config.lr_classifier,
        "lr_crf": config.lr_crf if config.use_crf else 0.0,
        "label2id_digest": label_digest,
        "num_labels": len(label2id),
        "git_commit": git_commit,
    }
    # trainer_state 只保存“恢复训练必须知道的状态”，不重复塞入整份模型权重。
    save_json(checkpoints_dir / "trainer_state.json", trainer_state)


def _train_variant(
    config: NERConfig,
    experiment_dirs: Optional[Dict[str, Path]] = None,
) -> Path:
    '''训练指定路线，并返回实验目录。'''

    set_seed(config.seed)
    device = choose_device(config.profile)
    git_commit = get_git_commit(config.project_root)

    resume_artifacts: Optional[Dict[str, Path]] = None
    if experiment_dirs is not None:
        experiment_dirs = bind_experiment_dirs(experiment_dirs["experiment_dir"])
    elif config.resume_from:
        resume_artifacts = resolve_resume_artifacts(Path(config.resume_from))
        experiment_dirs = bind_experiment_dirs(resume_artifacts["experiment_dir"])
    else:
        experiment_dirs = create_experiment_dirs(config.output_root, config.experiment_name)

    if config.resume_from:
        saved_config_path = experiment_dirs["experiment_dir"] / "config.json"
        if saved_config_path.exists():
            saved_config = load_json(saved_config_path)
            if saved_config.get("experiment_name"):
                config.experiment_name = str(saved_config["experiment_name"])

    prepared = prepare_datasets_and_tokenizer(config)
    label2id = prepared["label2id"]
    id2label = prepared["id2label"]
    tokenizer = prepared["tokenizer"]
    runtime_length_stats = prepared["runtime_length_stats"]
    train_char_counts = prepared["train_char_counts"]
    # 低频字集合按“当前实际参与训练的数据”统计，保证 local_debug 和正式训练口径一致。
    low_freq_chars = sorted(
        char
        for char, count in train_char_counts.items()
        if count <= config.low_freq_char_threshold
    ) if config.use_low_freq_char_dropout else []

    generate_data_figures(
        data_report_path=config.data_dir / "data_report.json",
        label_stats_path=config.data_dir / "label_stats.json",
        figure_dir=experiment_dirs["figures_dir"],
    )

    train_collator = NERBatchCollator(
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=config.max_len,
        ignore_index=config.ignore_index,
        enable_low_freq_char_dropout=config.use_low_freq_char_dropout,
        low_freq_chars=low_freq_chars,
        low_freq_char_dropout_prob=config.low_freq_char_dropout_prob,
    )
    dev_collator = NERBatchCollator(
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=config.max_len,
        ignore_index=config.ignore_index,
    )

    train_loader = build_train_dataloader(
        dataset=prepared["train_dataset"],
        collator=train_collator,
        config=config,
    )
    dev_loader = DataLoader(
        prepared["dev_dataset"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=dev_collator,
    )

    model = _build_model(config=config, label2id=label2id)
    model.to(device)

    optimizer = create_optimizer(model, config)
    # total_training_steps 始终按“真实 optimizer.step 次数”估算，而不是简单按 batch 数估算。
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.grad_accum_steps))
    total_training_steps = max(1, steps_per_epoch * config.num_epochs)
    scheduler = create_scheduler(optimizer, total_training_steps, config.warmup_ratio)

    training_log_rows: List[Dict[str, Any]] = []
    best_dev_accuracy = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    start_epoch = 1
    notes = ""
    resume_warnings: List[str] = []

    if resume_artifacts is not None:
        # 恢复训练时，先校验配置和标签映射，再恢复模型/优化器/调度器状态，
        # 避免把不兼容的 checkpoint 静默接到当前实验上。
        trainer_state = load_json(resume_artifacts["trainer_state_path"])
        resume_warnings = _validate_resume_compatibility(config, trainer_state, label2id)
        saved_label2id = load_json(resume_artifacts["label2id_path"])
        saved_id2label = load_json(resume_artifacts["id2label_path"])
        if saved_label2id != label2id:
            raise ValueError("Resume compatibility check failed for 'label2id.json'.")
        if saved_id2label != {str(index): label for index, label in id2label.items()}:
            raise ValueError("Resume compatibility check failed for 'id2label.json'.")

        checkpoint = load_checkpoint(resume_artifacts["checkpoint_path"], map_location="cpu")
        optimizer_state = load_checkpoint(resume_artifacts["optimizer_path"], map_location="cpu")
        scheduler_state = load_checkpoint(resume_artifacts["scheduler_path"], map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(optimizer_state["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(scheduler_state["scheduler_state_dict"])

        saved_resolved_model_name = trainer_state.get("resolved_model_name")
        if saved_resolved_model_name and saved_resolved_model_name != model.resolved_model_name:
            resume_warnings.append(
                "Resume metadata mismatch for 'resolved_model_name': "
                f"saved={saved_resolved_model_name!r}, current={model.resolved_model_name!r}"
            )

        training_log_rows = _load_existing_training_rows(experiment_dirs["experiment_dir"])
        best_dev_accuracy = float(trainer_state["best_dev_accuracy"])
        best_epoch = int(trainer_state["best_epoch"])
        stale_epochs = int(trainer_state["stale_epochs"])
        global_step = int(trainer_state["global_step"])
        start_epoch = int(trainer_state["current_epoch"]) + 1
        notes = f"resumed_from={experiment_dirs['experiment_dir'].name}"

        for warning_text in resume_warnings:
            print(f"[Resume Warning] {warning_text}")

    if start_epoch > config.num_epochs:
        print(
            f"Nothing to resume: current_epoch={start_epoch - 1} already reached "
            f"target num_epochs={config.num_epochs}."
        )
        return experiment_dirs["experiment_dir"]

    if stale_epochs >= config.early_stopping_patience:
        print(
            "Stored early-stopping state has already reached patience. "
            "No further training will be run."
        )
        return experiment_dirs["experiment_dir"]

    # 训练开始前更新 config.json，保证实验目录中的配置总是反映当前可运行状态。
    config_snapshot = config.to_dict()
    config_snapshot["git_commit"] = git_commit
    if config.resume_from:
        config_snapshot["resume_from"] = None
    save_json(experiment_dirs["experiment_dir"] / "config.json", config_snapshot)

    best_checkpoint_path = experiment_dirs["checkpoints_dir"] / "best_model.pt"
    last_checkpoint_path = experiment_dirs["checkpoints_dir"] / "last_model.pt"

    print("=== Training ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Variant: {config.experiment_variant}")
    print(f"Use refinement: {config.use_refinement}")
    print(f"Use CRF: {config.use_crf}")
    print(f"BIO constraint mode: {config.bio_constraint_mode}")
    print(f"Effective BIO mode: {model.effective_bio_constraint_mode}")
    print(f"Device: {device}")
    print(f"GPU name: {get_gpu_name(device)}")
    print(f"Requested backbone: {config.model_name}")
    print(f"Resolved backbone: {model.resolved_model_name}")
    print(f"Use bucket batching: {config.use_bucket_batching}")
    print(f"Use low-frequency char dropout: {config.use_low_freq_char_dropout}")
    if config.use_low_freq_char_dropout:
        print(
            "Low-frequency char dropout setup: "
            f"threshold<={config.low_freq_char_threshold}, "
            f"prob={config.low_freq_char_dropout_prob:.3f}, "
            f"char_types={len(low_freq_chars)}"
        )
    print(f"Train steps per epoch: {len(train_loader)}")
    print(f"Dev steps per epoch: {len(dev_loader)}")
    print(
        "Over max_len count: "
        f"train={runtime_length_stats['train']['over_max_len_count']}, "
        f"dev={runtime_length_stats['dev']['over_max_len_count']}, "
        f"test={runtime_length_stats['test']['over_max_len_count']}"
    )
    if git_commit:
        print(f"Git commit: {git_commit}")
    if config.resume_from:
        print(f"Resume from: {config.resume_from}")
        print(f"Resume start epoch: {start_epoch}")

    for epoch in range(start_epoch, config.num_epochs + 1):
        lr_backbone_base = config.lr_backbone
        lr_refinement_base = config.lr_refinement if config.use_refinement else 0.0
        lr_classifier_base = config.lr_classifier
        lr_crf_base = config.lr_crf if config.use_crf else 0.0

        with timer() as epoch_timer:
            train_loss, optimizer_steps = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_accum_steps=config.grad_accum_steps,
                grad_clip_norm=config.grad_clip_norm,
            )
            global_step += optimizer_steps
            # 每轮训练后立刻评估开发集，并用同一套 evaluate 口径驱动 best checkpoint 选择。
            dev_metrics = evaluate_model(
                model=model,
                dataloader=dev_loader,
                device=device,
                ignore_index=config.ignore_index,
                description=f"Dev Epoch {epoch}",
            )

        dev_accuracy = float(dev_metrics["token_accuracy"])
        dev_precision = float(dev_metrics["entity_precision"])
        dev_recall = float(dev_metrics["entity_recall"])
        dev_f1 = float(dev_metrics["entity_f1"])
        lr_backbone_end = get_group_lr(optimizer, "backbone", config.lr_backbone)
        lr_refinement_end = get_group_lr(optimizer, "refinement", 0.0)
        lr_classifier_end = get_group_lr(optimizer, "classifier", config.lr_classifier)
        lr_crf_end = get_group_lr(optimizer, "crf", 0.0)
        epoch_seconds = float(epoch_timer["elapsed_seconds"])

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
            "experiment_variant": config.experiment_variant,
            "use_refinement": config.use_refinement,
            "use_crf": config.use_crf,
            "use_bucket_batching": config.use_bucket_batching,
            "use_low_freq_char_dropout": config.use_low_freq_char_dropout,
            "low_freq_char_threshold": (
                config.low_freq_char_threshold if config.use_low_freq_char_dropout else 0
            ),
            "low_freq_char_dropout_prob": (
                config.low_freq_char_dropout_prob if config.use_low_freq_char_dropout else 0.0
            ),
            "bio_constraint_mode": config.bio_constraint_mode,
            "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
            "refine_num_layers": config.refine_num_layers if config.use_refinement else 0,
            "refine_num_heads": config.refine_num_heads if config.use_refinement else 0,
            "refine_ffn_dim": config.refine_ffn_dim if config.use_refinement else 0,
            "train_over_max_len_count": runtime_length_stats["train"]["over_max_len_count"],
            "dev_over_max_len_count": runtime_length_stats["dev"]["over_max_len_count"],
            "test_over_max_len_count": runtime_length_stats["test"]["over_max_len_count"],
            "train_loss": round(train_loss, 6),
            "dev_accuracy": round(dev_accuracy, 6),
            "dev_loss": round(float(dev_metrics["dev_loss"]), 6),
            "dev_precision": round(dev_precision, 6),
            "dev_recall": round(dev_recall, 6),
            "dev_f1": round(dev_f1, 6),
            "dev_gold_entities": int(dev_metrics["gold_entity_count"]),
            "dev_predicted_entities": int(dev_metrics["predicted_entity_count"]),
            "dev_correct_entities": int(dev_metrics["correct_entity_count"]),
            "lr_backbone": round(lr_backbone_base, 10),
            "lr_refinement": round(lr_refinement_base, 10),
            "lr_classifier": round(lr_classifier_base, 10),
            "lr_crf": round(lr_crf_base, 10),
            "lr_backbone_end": round(lr_backbone_end, 10),
            "lr_refinement_end": round(lr_refinement_end, 10),
            "lr_classifier_end": round(lr_classifier_end, 10),
            "lr_crf_end": round(lr_crf_end, 10),
            "epoch_time": format_seconds(epoch_seconds),
            "epoch_seconds": round(epoch_seconds, 4),
            "is_best": is_best,
        }
        training_log_rows.append(log_row)

        checkpoint_payload = _build_model_checkpoint_payload(
            model=model,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_dev_accuracy=best_dev_accuracy,
            best_epoch=best_epoch,
            label2id=label2id,
            id2label=id2label,
            git_commit=git_commit,
        )
        save_checkpoint(last_checkpoint_path, checkpoint_payload)
        if is_best:
            # best_model.pt 的选择规则始终是 dev_accuracy 最优；predict 也依赖这份约定加载模型。
            save_checkpoint(best_checkpoint_path, checkpoint_payload)

        _save_training_state(
            experiment_dirs=experiment_dirs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_dev_accuracy=best_dev_accuracy,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            label2id=label2id,
            id2label=id2label,
            git_commit=git_commit,
        )

        total_training_seconds = sum(_seconds_from_log_row(row) for row in training_log_rows)
        training_log_payload = {
            "experiment_name": config.experiment_name,
            "experiment_variant": config.experiment_variant,
            "profile": config.profile,
            "requested_model_name": config.model_name,
            "resolved_model_name": model.resolved_model_name,
            "use_refinement": config.use_refinement,
            "refine_num_layers": config.refine_num_layers if config.use_refinement else 0,
            "refine_num_heads": config.refine_num_heads if config.use_refinement else 0,
            "refine_ffn_dim": config.refine_ffn_dim if config.use_refinement else 0,
            "use_crf": config.use_crf,
            "use_bucket_batching": config.use_bucket_batching,
            "use_low_freq_char_dropout": config.use_low_freq_char_dropout,
            "low_freq_char_threshold": (
                config.low_freq_char_threshold if config.use_low_freq_char_dropout else 0
            ),
            "low_freq_char_dropout_prob": (
                config.low_freq_char_dropout_prob if config.use_low_freq_char_dropout else 0.0
            ),
            "bio_constraint_mode": config.bio_constraint_mode,
            "effective_bio_constraint_mode": model.effective_bio_constraint_mode,
            "best_epoch": best_epoch,
            "best_dev_accuracy": round(best_dev_accuracy, 6),
            "best_dev_f1": round(
                max(float(row.get("dev_f1", 0.0)) for row in training_log_rows),
                6,
            ),
            "git_commit": git_commit,
            "notes": notes,
            "resume_warnings": resume_warnings,
            "logs": training_log_rows,
        }
        # json 便于报告和恢复训练读取，csv 便于人工快速浏览或导入表格软件。
        save_json(experiment_dirs["experiment_dir"] / "training_log.json", training_log_payload)
        write_csv_rows(experiment_dirs["results_dir"] / "training_log.csv", training_log_rows)
        write_csv_rows(
            experiment_dirs["results_dir"] / "experiment_summary.csv",
            _build_experiment_summary(
                config=config,
                training_log_rows=training_log_rows,
                requested_model_name=config.model_name,
                resolved_model_name=model.resolved_model_name,
                device_name=get_gpu_name(device),
                runtime_length_stats=runtime_length_stats,
                total_training_seconds=total_training_seconds,
                notes=notes,
            ),
        )

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={log_row['train_loss']:.6f} | "
            f"dev_acc={log_row['dev_accuracy']:.6f} | "
            f"dev_f1={log_row['dev_f1']:.6f} | "
            f"lr_backbone={log_row['lr_backbone']:.8f}->{log_row['lr_backbone_end']:.8f} | "
            f"lr_refinement={log_row['lr_refinement']:.8f}->{log_row['lr_refinement_end']:.8f} | "
            f"lr_classifier={log_row['lr_classifier']:.8f}->{log_row['lr_classifier_end']:.8f} | "
            f"lr_crf={log_row['lr_crf']:.8f}->{log_row['lr_crf_end']:.8f} | "
            f"bucket={config.use_bucket_batching} | "
            f"time={log_row['epoch_time']} | "
            f"is_best={is_best}"
        )

        if stale_epochs >= config.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch={best_epoch}, best_dev_accuracy={best_dev_accuracy:.6f}"
            )
            break

    generate_training_figures(
        training_log_path=experiment_dirs["experiment_dir"] / "training_log.json",
        figure_dir=experiment_dirs["figures_dir"],
    )
    # 每次训练结束都顺手刷新项目级汇总，避免 README/报告里的对比图长期滞后。
    generate_project_summary_artifacts(config.output_root, config.data_dir)

    print(f"Best model saved to: {best_checkpoint_path}")
    print(f"Last model saved to: {last_checkpoint_path}")
    print(f"Trainer state saved to: {experiment_dirs['checkpoints_dir'] / 'trainer_state.json'}")
    return experiment_dirs["experiment_dir"]


def train_baseline(
    config: NERConfig,
    experiment_dirs: Optional[Dict[str, Path]] = None,
) -> Path:
    '''训练 baseline 路线。'''

    config.experiment_variant = "baseline"
    config.use_refinement = False
    config.use_crf = False
    config.bio_constraint_mode = "none"
    return _train_variant(config=config, experiment_dirs=experiment_dirs)


def train_final(
    config: NERConfig,
    experiment_dirs: Optional[Dict[str, Path]] = None,
) -> Path:
    '''训练 final_stage2 或 final_stage3 路线。'''

    if config.experiment_variant not in {"final_stage2", "final_stage3"}:
        config.experiment_variant = "final_stage3"

    if config.experiment_variant == "final_stage2":
        config.use_refinement = False
        config.use_crf = True
        if not config.experiment_name:
            config.experiment_name = "final_stage2_roberta_crf"
    else:
        config.use_refinement = True
        config.use_crf = True
        if not config.experiment_name:
            config.experiment_name = "final_stage3_roberta_refine_crf"
    return _train_variant(config=config, experiment_dirs=experiment_dirs)

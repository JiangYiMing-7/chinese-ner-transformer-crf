'''绘图模块。

当前默认绘制：
1. 标签频次图
2. 长度分布图
3. train loss 曲线
4. dev accuracy 曲线

在此基础上，额外补充：
5. dev entity-F1 曲线（若日志中存在）
6. 三阶段训练曲线对比图
7. 最佳 epoch 汇总表
8. 数据分布详细图与统计摘要
'''

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt

from utils import ensure_dir, format_seconds, load_json, write_csv_rows


VARIANT_DIR_PATTERNS: Dict[str, str] = {
    # 用精确正则而不是 startswith，避免 lowfreq baseline 被普通 baseline 误匹配。
    "baseline": r"^baseline_roberta_linear_\d{8}_\d{6}$",
    "baseline_lowfreqdrop": r"^baseline_roberta_linear_lowfreqdrop_\d{8}_\d{6}$",
    "final_stage2": r"^final_stage2_roberta_crf_\d{8}_\d{6}$",
    "final_stage3": r"^final_stage3_roberta_refine_crf_\d{8}_\d{6}$",
    "stage5_scratch": r"^stage5_scratch_\d{8}_\d{6}$",
}

VARIANT_DISPLAY_NAMES: Dict[str, str] = {
    "baseline": "Baseline",
    "baseline_lowfreqdrop": "Baseline + LowFreqDrop",
    "final_stage2": "Stage2 CRF",
    "final_stage3": "Stage3 Refine+CRF",
    "stage5_scratch": "Stage5 Scratch+CRF",
}

VARIANT_COLORS: Dict[str, str] = {
    "baseline": "#4C72B0",
    "baseline_lowfreqdrop": "#DD8452",
    "final_stage2": "#55A868",
    "final_stage3": "#C44E52",
    "stage5_scratch": "#8172B3",
}


def _extract_log_rows(training_log_payload: Any) -> List[Dict[str, Any]]:
    '''兼容不同 training_log.json 结构。'''

    if isinstance(training_log_payload, list):
        return training_log_payload
    if isinstance(training_log_payload, dict) and "logs" in training_log_payload:
        return training_log_payload["logs"]
    raise ValueError("Unsupported training log format.")


def _safe_metric(row: Dict[str, Any], metric_key: str) -> Optional[float]:
    '''安全读取日志行中的数值指标。'''

    if metric_key not in row:
        return None
    value = row.get(metric_key)
    if value in {"", None}:
        return None
    return float(value)


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


def _variant_display_name(variant: str) -> str:
    '''返回实验路线的展示名。'''

    return VARIANT_DISPLAY_NAMES.get(variant, variant)


def _latest_experiment_dirs(output_root: Path) -> List[Dict[str, Any]]:
    '''收集各条实验路线各自最新的实验目录。'''

    records: List[Dict[str, Any]] = []
    for variant, pattern in VARIANT_DIR_PATTERNS.items():
        candidates = [
            path
            for path in output_root.iterdir()
            if path.is_dir()
            and re.match(pattern, path.name)
            and (path / "training_log.json").exists()
        ] if output_root.exists() else []
        if not candidates:
            continue

        experiment_dir = max(candidates, key=lambda path: path.stat().st_mtime)
        # 项目级汇总默认只取“每条路线最新的一次完整实验”。
        payload = load_json(experiment_dir / "training_log.json")
        records.append(
            {
                "variant": variant,
                "display_name": _variant_display_name(variant),
                "color": VARIANT_COLORS.get(variant, "#4C72B0"),
                "experiment_dir": experiment_dir,
                "payload": payload,
                "log_rows": _extract_log_rows(payload),
            }
        )

    return records


def _load_dev_prediction_metrics(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    '''读取 predict 阶段写出的开发集指标。'''

    metrics_path = experiment_dir / "results" / "dev_prediction_metrics.json"
    if metrics_path.exists():
        return load_json(metrics_path)

    validation_path = experiment_dir / "results" / "output_validation.json"
    if validation_path.exists():
        payload = load_json(validation_path)
        metrics = payload.get("dev_metrics")
        if isinstance(metrics, dict):
            return metrics
    return None


def plot_tag_frequency(label_stats: Dict[str, Any], output_path: Path) -> None:
    '''绘制标签频次分布。'''

    ensure_dir(output_path.parent)
    labels = label_stats["labels"]
    label_counts = label_stats["label_counts"]
    counts = [label_counts.get(label, 0) for label in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color="#4C72B0")
    plt.title("Tag Frequency")
    plt.xlabel("Tag")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_tag_frequency_detailed(label_stats: Dict[str, Any], output_path: Path) -> None:
    '''绘制更适合报告展示的标签分布图。'''

    ensure_dir(output_path.parent)
    labels = label_stats["labels"]
    label_counts = label_stats["label_counts"]
    counts = [int(label_counts.get(label, 0)) for label in labels]
    entity_labels = [label for label in labels if label != "O"]
    entity_counts = [int(label_counts.get(label, 0)) for label in entity_labels]
    total_count = max(1, sum(counts))
    outside_ratio = float(label_counts.get("O", 0)) / float(total_count)

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(labels, counts, color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_title(f"All Tags (O ratio={outside_ratio:.2%})")
    axes[0].set_xlabel("Tag")
    axes[0].set_ylabel("Count (log scale)")

    axes[1].bar(entity_labels, entity_counts, color="#55A868")
    axes[1].set_title("Entity Tags Only")
    axes[1].set_xlabel("Tag")
    axes[1].set_ylabel("Count")

    figure.suptitle("Detailed Tag Distribution", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_length_distribution(data_report: Dict[str, Any], output_path: Path) -> None:
    '''绘制 train/dev/test 三个切分的长度分布。'''

    ensure_dir(output_path.parent)
    split_names = [
        ("train_length_stats", "Train"),
        ("dev_length_stats", "Dev"),
        ("test_length_stats", "Test"),
    ]
    configured_max_len = data_report.get("max_len")

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, (report_key, title) in zip(axes, split_names):
        stats = data_report[report_key]
        lengths = stats["lengths"]
        axis.hist(lengths, bins=60, color="#55A868", alpha=0.85)
        if configured_max_len:
            axis.axvline(
                configured_max_len,
                color="#C44E52",
                linestyle="--",
                linewidth=1.5,
            )
        axis.set_title(
            f"{title}\nP95={stats['p95']} P99={stats['p99']} Max={stats['max']}"
        )
        axis.set_xlabel("Sequence Length")
        axis.set_ylabel("Count")
        axis.text(
            0.98,
            0.95,
            (
                f"Mean={stats['mean']:.1f}\n"
                f"Over max={stats['over_max_len_count']} ({stats['over_max_len_ratio']:.2%})"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    figure.suptitle("Length Distribution", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_train_loss_curve(
    log_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    '''绘制训练损失曲线。'''

    ensure_dir(output_path.parent)
    epochs = [row["epoch"] for row in log_rows]
    losses = [row["train_loss"] for row in log_rows]
    best_row = max(log_rows, key=lambda row: row["dev_accuracy"])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", color="#C44E52")
    plt.scatter(
        [best_row["epoch"]],
        [best_row["train_loss"]],
        color="black",
        label=f"best epoch={best_row['epoch']}",
    )
    plt.title("Train Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_dev_accuracy_curve(
    log_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    '''绘制开发集准确率曲线。'''

    ensure_dir(output_path.parent)
    epochs = [row["epoch"] for row in log_rows]
    accuracies = [row["dev_accuracy"] for row in log_rows]
    best_row = max(log_rows, key=lambda row: row["dev_accuracy"])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker="o", color="#8172B2")
    plt.scatter(
        [best_row["epoch"]],
        [best_row["dev_accuracy"]],
        color="black",
        label=f"best epoch={best_row['epoch']}",
    )
    plt.title("Dev Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_dev_f1_curve(
    log_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    '''绘制开发集实体级 F1 曲线。'''

    valid_rows = [row for row in log_rows if _safe_metric(row, "dev_f1") is not None]
    if not valid_rows:
        return

    ensure_dir(output_path.parent)
    epochs = [row["epoch"] for row in valid_rows]
    f1_scores = [float(row["dev_f1"]) for row in valid_rows]
    best_row = max(valid_rows, key=lambda row: float(row["dev_f1"]))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, f1_scores, marker="o", color="#DD8452")
    plt.scatter(
        [best_row["epoch"]],
        [best_row["dev_f1"]],
        color="black",
        label=f"best f1 epoch={best_row['epoch']}",
    )
    plt.title("Dev Entity-F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_multi_experiment_metric(
    experiment_records: Sequence[Dict[str, Any]],
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    '''绘制多实验对比曲线。'''

    ensure_dir(output_path.parent)
    plt.figure(figsize=(9, 5.5))
    has_any_curve = False

    for record in experiment_records:
        valid_rows = [
            row for row in record["log_rows"]
            if _safe_metric(row, metric_key) is not None
        ]
        if not valid_rows:
            continue

        epochs = [int(row["epoch"]) for row in valid_rows]
        values = [float(row[metric_key]) for row in valid_rows]
        color = str(record["color"])
        plt.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            label=record["display_name"],
            color=color,
        )
        has_any_curve = True

    if not has_any_curve:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_best_checkpoint_metric_bar(
    experiment_records: Sequence[Dict[str, Any]],
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    '''基于 predict 阶段输出的最佳 checkpoint 指标绘制对比柱状图。'''

    values: List[float] = []
    labels: List[str] = []
    colors: List[str] = []

    for record in experiment_records:
        metrics = _load_dev_prediction_metrics(record["experiment_dir"])
        if not metrics or metric_key not in metrics:
            continue
        values.append(float(metrics[metric_key]))
        labels.append(str(record["display_name"]))
        colors.append(str(record["color"]))

    if not values:
        return

    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _build_best_epoch_summary_rows(
    experiment_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''构造实验最佳 epoch 汇总表。'''

    rows: List[Dict[str, Any]] = []
    for record in experiment_records:
        log_rows = list(record["log_rows"])
        if not log_rows:
            continue

        best_accuracy_row = max(log_rows, key=lambda row: float(row["dev_accuracy"]))
        total_training_seconds = sum(_seconds_from_log_row(row) for row in log_rows)
        # 正式实验优先读取 predict 阶段基于最佳 checkpoint 重新评估的开发集 F1，
        # 这样汇总表与第四章分析保持同一口径。
        # 若新实验尚未跑 predict，则回退到训练日志中记录的最佳 dev_f1。
        best_checkpoint_metrics = _load_dev_prediction_metrics(record["experiment_dir"])
        best_checkpoint_dev_f1 = ""
        if best_checkpoint_metrics and "entity_f1" in best_checkpoint_metrics:
            best_checkpoint_dev_f1 = round(float(best_checkpoint_metrics["entity_f1"]), 6)
        else:
            best_f1 = max(
                (
                    float(row["dev_f1"])
                    for row in log_rows
                    if _safe_metric(row, "dev_f1") is not None
                ),
                default=None,
            )
            if best_f1 is not None:
                best_checkpoint_dev_f1 = round(best_f1, 6)

        rows.append(
            {
                "variant": record["display_name"],
                "experiment_dir": record["experiment_dir"].name,
                "best_epoch": int(best_accuracy_row["epoch"]),
                "best_dev_accuracy": round(float(best_accuracy_row["dev_accuracy"]), 6),
                "train_loss_at_best_epoch": round(float(best_accuracy_row["train_loss"]), 6),
                "best_checkpoint_dev_f1": best_checkpoint_dev_f1,
                "training_time": format_seconds(total_training_seconds),
            }
        )

    return rows


def plot_best_epoch_summary_table(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    '''把最佳 epoch 汇总表渲染为图片，便于直接放入报告。'''

    if not rows:
        return

    ensure_dir(output_path.parent)
    columns = [
        "variant",
        "best_epoch",
        "best_dev_accuracy",
        "best_checkpoint_dev_f1",
        "training_time",
    ]
    cell_text = [[str(row.get(column, "")) for column in columns] for row in rows]
    figure_height = max(2.5, 1.1 + 0.55 * len(rows))
    figure, axis = plt.subplots(figsize=(12.5, figure_height))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    axis.set_title("Best Epoch Summary", fontsize=13, pad=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_length_summary_rows(data_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    '''构造长度统计摘要表。'''

    rows: List[Dict[str, Any]] = []
    for report_key, split_name in (
        ("train_length_stats", "train"),
        ("dev_length_stats", "dev"),
        ("test_length_stats", "test"),
    ):
        stats = data_report[report_key]
        rows.append(
            {
                "split": split_name,
                "count": int(stats["count"]),
                "mean": round(float(stats["mean"]), 4),
                "p50": int(stats["p50"]),
                "p90": int(stats["p90"]),
                "p95": int(stats["p95"]),
                "p99": int(stats["p99"]),
                "max": int(stats["max"]),
                "over_max_len_count": int(stats["over_max_len_count"]),
                "over_max_len_ratio": round(float(stats["over_max_len_ratio"]), 6),
            }
        )
    return rows


def write_data_insights(
    data_report: Dict[str, Any],
    label_stats: Dict[str, Any],
    output_path: Path,
) -> None:
    '''生成可直接写进报告的数据分析摘要。'''

    ensure_dir(output_path.parent)
    total_labels = max(1, sum(int(count) for count in label_stats["label_counts"].values()))
    outside_count = int(label_stats["label_counts"].get("O", 0))
    outside_ratio = outside_count / total_labels
    entity_labels = [label for label in label_stats["labels"] if label != "O"]

    lines = [
        "# Data Insights",
        "",
        f"- 标签集合共 {label_stats['num_labels']} 类，其中实体标签为：{', '.join(entity_labels)}。",
        f"- `O` 标签占全部标注位的 {outside_ratio:.2%}，数据存在明显类别不均衡，这也是 token-level accuracy 普遍较高的重要原因。",
    ]

    configured_max_len = data_report.get("max_len")
    if configured_max_len is not None:
        lines.append(f"- 当前数据审计使用的 `max_len={configured_max_len}`。")

    for split_name, report_key in (
        ("train", "train_length_stats"),
        ("dev", "dev_length_stats"),
        ("test", "test_length_stats"),
    ):
        stats = data_report[report_key]
        lines.append(
            f"- {split_name}: P95={stats['p95']}，P99={stats['p99']}，Max={stats['max']}，"
            f"超过 `max_len` 的样本数为 {stats['over_max_len_count']} "
            f"({stats['over_max_len_ratio']:.2%})。"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_data_figures(
    data_report_path: Path,
    label_stats_path: Path,
    figure_dir: Path,
) -> None:
    '''从数据统计文件生成基础图表。'''

    data_report = load_json(data_report_path)
    label_stats = load_json(label_stats_path)
    plot_tag_frequency(label_stats, figure_dir / "tag_frequency.png")
    plot_tag_frequency_detailed(label_stats, figure_dir / "tag_frequency_detailed.png")
    plot_length_distribution(data_report, figure_dir / "length_distribution_all.png")


def generate_training_figures(
    training_log_path: Path,
    figure_dir: Path,
) -> None:
    '''从 training_log.json 重绘训练曲线。'''

    payload = load_json(training_log_path)
    log_rows = _extract_log_rows(payload)
    plot_train_loss_curve(log_rows, figure_dir / "train_loss_curve.png")
    plot_dev_accuracy_curve(log_rows, figure_dir / "dev_accuracy_curve.png")
    plot_dev_f1_curve(log_rows, figure_dir / "dev_f1_curve.png")


def generate_project_summary_artifacts(
    output_root: Path,
    data_dir: Path,
) -> None:
    '''在 outputs 根目录下生成项目级对比图和摘要表。'''

    project_summary_dir = ensure_dir(output_root / "project_summary")
    figures_dir = ensure_dir(project_summary_dir / "figures")
    results_dir = ensure_dir(project_summary_dir / "results")

    data_report_path = data_dir / "data_report.json"
    label_stats_path = data_dir / "label_stats.json"
    if data_report_path.exists() and label_stats_path.exists():
        data_report = load_json(data_report_path)
        label_stats = load_json(label_stats_path)
        generate_data_figures(data_report_path, label_stats_path, figures_dir)
        write_csv_rows(results_dir / "length_summary.csv", _build_length_summary_rows(data_report))
        write_data_insights(data_report, label_stats, results_dir / "data_insights.md")

    experiment_records = _latest_experiment_dirs(output_root)
    if not experiment_records:
        return

    # project_summary 面向 README、实验报告和最终核对，默认按实验路线做横向比较。
    plot_multi_experiment_metric(
        experiment_records=experiment_records,
        metric_key="train_loss",
        output_path=figures_dir / "stage_train_loss_comparison.png",
        title="Experiment Train Loss Comparison",
        ylabel="Train Loss",
    )
    plot_multi_experiment_metric(
        experiment_records=experiment_records,
        metric_key="dev_accuracy",
        output_path=figures_dir / "stage_dev_accuracy_comparison.png",
        title="Experiment Dev Accuracy Comparison",
        ylabel="Accuracy",
    )
    plot_multi_experiment_metric(
        experiment_records=experiment_records,
        metric_key="dev_f1",
        output_path=figures_dir / "stage_dev_f1_comparison.png",
        title="Experiment Dev Entity-F1 Comparison",
        ylabel="F1",
    )
    plot_best_checkpoint_metric_bar(
        experiment_records=experiment_records,
        metric_key="entity_f1",
        output_path=figures_dir / "stage_best_checkpoint_dev_f1.png",
        title="Best Checkpoint Dev Entity-F1",
        ylabel="F1",
    )

    summary_rows = _build_best_epoch_summary_rows(experiment_records)
    write_csv_rows(results_dir / "best_epoch_summary.csv", summary_rows)
    plot_best_epoch_summary_table(summary_rows, figures_dir / "best_epoch_summary.png")


def plot_experiment_artifacts(experiment_dir: Path, data_dir: Path) -> None:
    '''为指定实验目录补齐图表产物，并刷新项目级汇总。'''

    figure_dir = ensure_dir(experiment_dir / "figures")
    generate_data_figures(
        data_report_path=data_dir / "data_report.json",
        label_stats_path=data_dir / "label_stats.json",
        figure_dir=figure_dir,
    )
    training_log_path = experiment_dir / "training_log.json"
    if training_log_path.exists():
        generate_training_figures(training_log_path, figure_dir)
    generate_project_summary_artifacts(experiment_dir.parent, data_dir)

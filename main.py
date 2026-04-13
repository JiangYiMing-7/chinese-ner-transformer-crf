'''命令行入口。

统一管理数据分析、训练、恢复训练、预测和绘图命令。
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config import SCRATCH_CONFIG, build_config, default_experiment_name_for_variant
from data import run_data_analysis
from plot_curves import generate_data_figures, plot_experiment_artifacts
from predict import run_prediction
from train import train_baseline, train_final
from utils import create_experiment_dirs, find_latest_experiment_dir, load_json, resolve_resume_artifacts


def run_analyze_mode(
    profile: str,
    experiment_name: str,
    experiment_variant: str,
    project_root: Optional[str] = None,
    output_filename: Optional[str] = None,
    num_epochs: Optional[int] = None,
    use_low_freq_char_dropout: bool = False,
    low_freq_char_threshold: Optional[int] = None,
    low_freq_char_dropout_prob: Optional[float] = None,
) -> Path:
    '''执行 analyze 模式并返回产物目录。'''

    # analyze 虽然不训练模型，但仍单独生成一次实验目录，便于保留当时的数据审计图表。
    config = build_config(
        profile=profile,
        experiment_variant=experiment_variant,
        experiment_name=experiment_name,
        project_root=project_root,
        output_filename=output_filename,
        num_epochs=num_epochs,
        use_low_freq_char_dropout=use_low_freq_char_dropout,
        low_freq_char_threshold=low_freq_char_threshold,
        low_freq_char_dropout_prob=low_freq_char_dropout_prob,
    )
    experiment_dirs = create_experiment_dirs(config.output_root, config.experiment_name)
    run_data_analysis(config)
    generate_data_figures(
        data_report_path=config.data_dir / "data_report.json",
        label_stats_path=config.data_dir / "label_stats.json",
        figure_dir=experiment_dirs["figures_dir"],
    )
    print(f"Analyze artifacts saved under: {experiment_dirs['experiment_dir']}")
    return experiment_dirs["experiment_dir"]


def parse_args() -> argparse.Namespace:
    '''解析命令行参数。'''

    parser = argparse.ArgumentParser(description="Chinese NER training and inference entrypoint")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "analyze",
            "train_baseline",
            "train_final",
            "train_stage5",
            "resume",
            "predict",
            "plot",
            "all",
        ],
        help="运行模式",
    )
    parser.add_argument(
        "--profile",
        default="local_debug",
        choices=["cloud_train", "local_debug"],
        help="配置 profile",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["baseline", "final_stage2", "final_stage3"],
        help="实验路线；resume 时若不传，会优先读取 checkpoint 中保存的 variant",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="实验名称前缀；不传则使用 variant 默认名称",
    )
    parser.add_argument(
        "--project_root",
        default=None,
        help="显式指定项目根目录",
    )
    parser.add_argument(
        "--experiment_dir",
        default=None,
        help="predict/plot 模式下可显式指定实验目录",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="resume 来源，可传实验目录、checkpoints 目录或 last_model.pt 路径",
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        help="预测输出文件名，例如 2023211751.txt",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="覆盖配置中的总训练轮数",
    )
    parser.add_argument(
        "--use_low_freq_char_dropout",
        action="store_true",
        help="训练时对训练集中的低频字做随机 UNK 替换，仅用于最小对照实验",
    )
    parser.add_argument(
        "--low_freq_char_threshold",
        type=int,
        default=None,
        help="字频不高于该阈值时视为低频字",
    )
    parser.add_argument(
        "--low_freq_char_dropout_prob",
        type=float,
        default=None,
        help="低频字在训练输入中被替换为 UNK 的概率",
    )
    return parser.parse_args()


def _infer_variant_for_resume(args: argparse.Namespace) -> str:
    '''优先从 resume checkpoint 中读取实验路线。'''

    if not args.resume_from:
        return args.variant or "final_stage3"

    artifacts = resolve_resume_artifacts(Path(args.resume_from))
    trainer_state = load_json(artifacts["trainer_state_path"])
    # resume 时以 checkpoint 中保存的 experiment_variant 为准，
    # 避免用户误传 --variant 把旧实验接到错误结构上。
    return str(trainer_state.get("experiment_variant", args.variant or "final_stage3"))


def _default_experiment_name(args: argparse.Namespace, experiment_variant: str) -> str:
    '''根据当前开关生成默认实验名前缀。'''

    experiment_name = default_experiment_name_for_variant(experiment_variant)
    if args.use_low_freq_char_dropout:
        # 最小对照实验单独加后缀，方便和原始 baseline 区分。
        experiment_name = f"{experiment_name}_lowfreqdrop"
    return experiment_name


def main() -> None:
    '''CLI 主入口。'''

    args = parse_args()
    if args.mode == "train_stage5":
        from train_scratch import train_scratch

        train_scratch(dict(SCRATCH_CONFIG))
        return

    if args.mode == "train_baseline":
        # baseline 训练应直接走 baseline 默认配置，避免先继承其他 variant 的学习率与结构开关。
        effective_variant = "baseline"
    else:
        effective_variant = args.variant or "final_stage3"

    if args.mode == "resume":
        if not args.resume_from:
            raise ValueError("--mode resume requires --resume_from")
        effective_variant = _infer_variant_for_resume(args)

    default_experiment_name = args.experiment_name or _default_experiment_name(
        args,
        effective_variant,
    )

    if args.mode == "analyze":
        run_analyze_mode(
            profile=args.profile,
            experiment_name=default_experiment_name,
            experiment_variant=effective_variant,
            project_root=args.project_root,
            output_filename=args.output_filename,
            num_epochs=args.num_epochs,
            use_low_freq_char_dropout=args.use_low_freq_char_dropout,
            low_freq_char_threshold=args.low_freq_char_threshold,
            low_freq_char_dropout_prob=args.low_freq_char_dropout_prob,
        )
        return

    # 除 analyze 外，其余模式都走统一的 build_config，保证训练、预测、绘图共用同一套默认口径。
    config = build_config(
        profile=args.profile,
        experiment_variant=effective_variant,
        experiment_name=args.experiment_name,
        project_root=args.project_root,
        output_filename=args.output_filename,
        num_epochs=args.num_epochs,
        resume_from=args.resume_from,
        use_low_freq_char_dropout=args.use_low_freq_char_dropout,
        low_freq_char_threshold=args.low_freq_char_threshold,
        low_freq_char_dropout_prob=args.low_freq_char_dropout_prob,
    )

    if args.mode == "train_baseline":
        config.experiment_variant = "baseline"
        if args.experiment_name is None:
            config.experiment_name = _default_experiment_name(args, "baseline")
        train_baseline(config)
        return

    if args.mode == "train_final":
        if effective_variant not in {"final_stage2", "final_stage3"}:
            raise ValueError("--variant for train_final must be final_stage2 or final_stage3")
        config.experiment_variant = effective_variant
        if args.experiment_name is None:
            config.experiment_name = _default_experiment_name(args, effective_variant)
        train_final(config)
        return

    if args.mode == "resume":
        # resume 会优先根据 checkpoint 中保存的 variant 选择对应训练入口，避免接错结构。
        if config.experiment_variant == "baseline":
            train_baseline(config)
        else:
            train_final(config)
        return

    if args.mode == "predict":
        # predict/plot 允许显式传 experiment_dir，方便重放任意一次历史实验。
        run_prediction(config, experiment_dir=args.experiment_dir)
        return

    if args.mode == "plot":
        if args.experiment_dir:
            experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        else:
            experiment_dir = find_latest_experiment_dir(
                config.output_root,
                experiment_name=config.experiment_name,
            )
        plot_experiment_artifacts(experiment_dir, config.data_dir)
        print(f"Plots regenerated under: {experiment_dir / 'figures'}")
        return

    if args.mode == "all":
        # all 只是把 analyze/train/predict/plot 串起来，不额外引入新的业务逻辑。
        run_analyze_mode(
            profile=args.profile,
            experiment_name=config.experiment_name,
            experiment_variant=config.experiment_variant,
            project_root=args.project_root,
            output_filename=args.output_filename,
            num_epochs=args.num_epochs,
            use_low_freq_char_dropout=args.use_low_freq_char_dropout,
            low_freq_char_threshold=args.low_freq_char_threshold,
            low_freq_char_dropout_prob=args.low_freq_char_dropout_prob,
        )
        if config.experiment_variant == "baseline":
            experiment_dir = train_baseline(config)
        else:
            experiment_dir = train_final(config)
        run_prediction(config, experiment_dir=str(experiment_dir))
        plot_experiment_artifacts(experiment_dir, config.data_dir)
        print(f"All artifacts saved under: {experiment_dir}")
        return


if __name__ == "__main__":
    main()

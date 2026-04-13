'''项目配置模块。

负责统一管理三条模型路线、本地与服务器双 profile、训练超参数、
滑窗与 bucket batching 开关，以及最终工程版需要的输出与恢复配置。
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parent


def _resolve_project_root(project_root_override: Optional[str] = None) -> Path:
    '''解析运行时项目根目录。'''

    if project_root_override:
        # 显式传参优先级最高，这样后续迁移到服务器时无需改代码。
        return Path(project_root_override).expanduser().resolve()
    return DEFAULT_LOCAL_ROOT


@dataclass
class NERConfig:
    '''统一配置对象。'''

    profile: str = "local_debug"
    experiment_variant: str = "final_stage3"
    experiment_name: str = "final_stage3_roberta_refine_crf"
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    fallback_model_name: str = "bert-base-chinese"
    max_len: int = 384
    batch_size: int = 16
    grad_accum_steps: int = 1
    num_epochs: int = 5
    lr_backbone: float = 2e-5
    lr_refinement: float = 1e-3
    lr_classifier: float = 1e-3
    lr_crf: float = 1e-3
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    dropout: float = 0.1
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 42
    output_filename: str = "result.txt"
    ignore_index: int = -100
    resume_from: Optional[str] = None
    use_refinement: bool = True
    refine_num_layers: int = 1
    refine_num_heads: int = 4
    # 这里的 hidden_size 指 backbone 输出维度，也就是 RoBERTa hidden size。
    # 当前默认 backbone 下 hidden_size=768，因此 refine_ffn_dim 默认取 1536。
    refine_ffn_dim: int = 1536
    refine_dropout: float = 0.1
    use_crf: bool = True
    bio_constraint_mode: str = "none"
    use_sliding_window: bool = True
    sliding_window_size: int = 384
    sliding_overlap: int = 64
    sliding_merge_strategy: str = "center_priority"
    use_bucket_batching: bool = True
    bucket_size_multiplier: int = 50
    shuffle_within_bucket: bool = True
    use_low_freq_char_dropout: bool = False
    low_freq_char_threshold: int = 1
    low_freq_char_dropout_prob: float = 0.5
    debug_subset_size: int = 256
    num_workers: int = 0
    project_root_override: Optional[str] = None

    project_root: Path = field(init=False)
    data_dir: Path = field(init=False)
    output_root: Path = field(init=False)

    def __post_init__(self) -> None:
        '''补齐派生字段并执行配置合法性检查。'''

        # 在对象构造完成后统一派生目录字段，避免路径逻辑散落在各模块里。
        if self.experiment_variant not in {"baseline", "final_stage2", "final_stage3"}:
            raise ValueError(f"Unsupported experiment_variant: {self.experiment_variant}")
        if self.bio_constraint_mode not in {"none", "strict_bio", "auto"}:
            raise ValueError(f"Unsupported bio_constraint_mode: {self.bio_constraint_mode}")
        if self.sliding_merge_strategy not in {"center_priority"}:
            raise ValueError(f"Unsupported sliding_merge_strategy: {self.sliding_merge_strategy}")
        if self.refine_num_layers <= 0:
            raise ValueError("refine_num_layers must be positive.")
        if self.refine_num_heads <= 0:
            raise ValueError("refine_num_heads must be positive.")
        if self.refine_ffn_dim <= 0:
            raise ValueError("refine_ffn_dim must be positive.")
        if self.sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive.")
        if not (0 <= self.sliding_overlap < self.sliding_window_size):
            raise ValueError(
                "sliding_overlap must satisfy 0 <= sliding_overlap < sliding_window_size."
            )
        if self.bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive.")
        if self.low_freq_char_threshold <= 0:
            raise ValueError("low_freq_char_threshold must be positive.")
        if not (0.0 <= self.low_freq_char_dropout_prob <= 1.0):
            raise ValueError("low_freq_char_dropout_prob must be between 0 and 1.")

        self.project_root = _resolve_project_root(self.project_root_override)
        self.data_dir = self.project_root / "data"
        self.output_root = self.project_root / "outputs"

    def to_dict(self) -> Dict[str, Any]:
        '''将配置转成可序列化字典。'''

        raw = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in raw.items()
        }


PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # profile 主要描述“本地联调”与“云端正式训练”这类运行规模差异，
    # 不直接决定 baseline / stage2 / stage3 走哪条模型路线。
    "cloud_train": {
        "model_name": "hfl/chinese-roberta-wwm-ext",
        "fallback_model_name": "bert-base-chinese",
        "max_len": 384,
        "batch_size": 16,
        "grad_accum_steps": 1,
        "num_epochs": 5,
        "lr_backbone": 2e-5,
        "lr_refinement": 1e-3,
        "lr_classifier": 1e-3,
        "lr_crf": 1e-3,
        "weight_decay": 0.01,
        "early_stopping_patience": 2,
        "dropout": 0.1,
        "refine_dropout": 0.1,
        "use_sliding_window": True,
        "sliding_window_size": 384,
        "sliding_overlap": 64,
        "sliding_merge_strategy": "center_priority",
        "use_bucket_batching": True,
        "bucket_size_multiplier": 50,
        "shuffle_within_bucket": True,
        "seed": 42,
        "debug_subset_size": 256,
    },
    "local_debug": {
        "model_name": "hfl/chinese-roberta-wwm-ext",
        "fallback_model_name": "bert-base-chinese",
        "max_len": 128,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "num_epochs": 1,
        "lr_backbone": 2e-5,
        "lr_refinement": 1e-3,
        "lr_classifier": 1e-3,
        "lr_crf": 1e-3,
        "weight_decay": 0.01,
        "early_stopping_patience": 1,
        "dropout": 0.1,
        "refine_dropout": 0.1,
        "use_sliding_window": True,
        "sliding_window_size": 128,
        "sliding_overlap": 64,
        "sliding_merge_strategy": "center_priority",
        "use_bucket_batching": True,
        "bucket_size_multiplier": 20,
        "shuffle_within_bucket": True,
        "seed": 42,
        "debug_subset_size": 128,
    },
}


VARIANT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # variant 只覆盖当前路线真正会生效的结构开关和专用超参数。
    "baseline": {
        "experiment_name": "baseline_roberta_linear",
        "use_refinement": False,
        "use_crf": False,
        "bio_constraint_mode": "none",
    },
    "final_stage2": {
        "experiment_name": "final_stage2_roberta_crf",
        "use_refinement": False,
        "use_crf": True,
        # 这份数据不是严格 BIO，因此正式主线默认不开约束。
        "bio_constraint_mode": "none",
    },
    "final_stage3": {
        "experiment_name": "final_stage3_roberta_refine_crf",
        "use_refinement": True,
        "use_crf": True,
        "bio_constraint_mode": "none",
        # refinement 路线在完整训练集上更容易数值发散，因此单独使用更保守的新增层学习率。
        "lr_backbone": 2e-5,
        "lr_refinement": 1e-4,
        "lr_classifier": 2e-4,
        "lr_crf": 2e-4,
        "refine_num_layers": 1,
        "refine_num_heads": 4,
        "refine_ffn_dim": 1536,
        "refine_dropout": 0.1,
    },
}


def default_experiment_name_for_variant(experiment_variant: str) -> str:
    '''返回某个 variant 的默认实验名前缀。'''

    if experiment_variant not in VARIANT_DEFAULTS:
        raise ValueError(f"Unsupported experiment_variant: {experiment_variant}")
    return str(VARIANT_DEFAULTS[experiment_variant]["experiment_name"])


def build_config(
    profile: str = "local_debug",
    experiment_variant: str = "final_stage3",
    experiment_name: Optional[str] = None,
    project_root: Optional[str] = None,
    output_filename: Optional[str] = None,
    num_epochs: Optional[int] = None,
    resume_from: Optional[str] = None,
    use_low_freq_char_dropout: bool = False,
    low_freq_char_threshold: Optional[int] = None,
    low_freq_char_dropout_prob: Optional[float] = None,
) -> NERConfig:
    '''根据 profile、variant 和少量 CLI 覆盖项构造配置对象。'''

    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported profile: {profile}")
    if experiment_variant not in VARIANT_DEFAULTS:
        raise ValueError(f"Unsupported experiment_variant: {experiment_variant}")

    # 合并顺序不能反过来：
    # - profile 决定运行环境和规模
    # - variant 决定模型结构与专用超参数
    # - CLI 覆盖项优先级最高，只影响当前这次运行
    config_kwargs = PROFILE_DEFAULTS[profile].copy()
    config_kwargs.update(VARIANT_DEFAULTS[experiment_variant])
    config_kwargs["profile"] = profile
    config_kwargs["experiment_variant"] = experiment_variant
    config_kwargs["project_root_override"] = project_root
    config_kwargs["use_low_freq_char_dropout"] = use_low_freq_char_dropout

    if experiment_name:
        config_kwargs["experiment_name"] = experiment_name
    if output_filename:
        config_kwargs["output_filename"] = output_filename
    if num_epochs is not None:
        config_kwargs["num_epochs"] = num_epochs
    if resume_from is not None:
        config_kwargs["resume_from"] = resume_from
    if low_freq_char_threshold is not None:
        config_kwargs["low_freq_char_threshold"] = low_freq_char_threshold
    if low_freq_char_dropout_prob is not None:
        config_kwargs["low_freq_char_dropout_prob"] = low_freq_char_dropout_prob

    return NERConfig(**config_kwargs)


# ============================================================
# Stage5: 手写 Transformer 配置（从零训练，不使用预训练模型）
# ============================================================
SCRATCH_CONFIG = {
    "variant": "stage5_scratch",
    "profile": "cloud_train",
    "project_root": DEFAULT_LOCAL_ROOT,
    "experiment_dir_prefix": "outputs/stage5_scratch",

    # 模型结构
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "ffn_dim": 512,
    "max_seq_len": 256,
    "dropout": 0.3,
    "bio_constraint_mode": "none",

    # 训练配置
    "batch_size": 64,
    "num_epochs": 30,
    "embedding_lr": 5e-4,
    "peak_lr": 1e-3,
    "classifier_lr": 1e-3,
    "crf_lr": 1e-3,
    "min_lr": 1e-5,
    "warmup_ratio": 0.05,
    "weight_decay": 1e-4,
    "grad_clip_norm": 5.0,
    "grad_accum_steps": 2,
    "early_stop_patience": 8,
    "seed": 42,

    # 效率优化
    "use_bucket_batching": True,
    "bucket_size_multiplier": 50,
    "shuffle_within_bucket": True,
    "num_workers": 0,
    "sliding_overlap": 64,

    # 数据配置
    "min_char_freq": 1,
    "ignore_index": -100,

    # 额外技巧
    "use_fgm": True,
    "fgm_epsilon": 0.25,
    "use_ema": True,
    "ema_decay": 0.999,

    # 预训练向量（可选）
    "pretrained_embedding_path": None,

    # MPS 上 AMP 不稳定，因此默认关闭
    "use_amp": False,
}

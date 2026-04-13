'''通用工具模块。

负责实验目录管理、JSON/CSV 保存、随机种子固定、设备选择、
checkpoint 读写、输出格式检查，以及最终工程版需要的元数据辅助函数。
'''

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

import numpy as np


def _make_json_safe(obj: Any) -> Any:
    '''将复杂对象递归转成可写入 JSON 的形式。'''

    if is_dataclass(obj):
        return _make_json_safe(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    return obj


def ensure_dir(path: Path) -> Path:
    '''确保目录存在。'''

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    '''保存 JSON 文件。'''

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(_make_json_safe(payload), file_obj, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    '''读取 JSON 文件。'''

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_csv_rows(path: Path, rows: List[Mapping[str, Any]]) -> None:
    '''将字典列表写成 CSV。

    这里不用 pandas，避免把简单导出逻辑绑定到更重的依赖上。
    '''

    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            file_obj.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # 统一做一次 JSON-safe 转换，避免 Path、numpy 标量等对象直接写 CSV 出错。
            writer.writerow(
                {key: _make_json_safe(value) for key, value in row.items()}
            )


def timestamp_string() -> str:
    '''返回实验目录使用的时间戳。'''

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dirs(output_root: Path, experiment_name: str) -> Dict[str, Path]:
    '''创建一次新实验所需的目录结构。'''

    # 所有实验产物都放到独立时间戳目录中，避免重复运行时相互覆盖。
    experiment_dir = ensure_dir(output_root / f"{experiment_name}_{timestamp_string()}")
    dirs = {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": ensure_dir(experiment_dir / "checkpoints"),
        "predictions_dir": ensure_dir(experiment_dir / "predictions"),
        "figures_dir": ensure_dir(experiment_dir / "figures"),
        "results_dir": ensure_dir(experiment_dir / "results"),
    }
    return dirs


def bind_experiment_dirs(experiment_dir: Path) -> Dict[str, Path]:
    '''为已有实验目录补齐统一的子目录字典。'''

    experiment_dir = ensure_dir(experiment_dir)
    return {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": ensure_dir(experiment_dir / "checkpoints"),
        "predictions_dir": ensure_dir(experiment_dir / "predictions"),
        "figures_dir": ensure_dir(experiment_dir / "figures"),
        "results_dir": ensure_dir(experiment_dir / "results"),
    }


def find_latest_experiment_dir(output_root: Path, experiment_name: Optional[str] = None) -> Path:
    '''查找最近一次实验目录。'''

    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (experiment_name is None or path.name.startswith(experiment_name))
    ]
    # 这里依赖统一的“实验名前缀 + 时间戳”命名规则，不额外维护 latest 指针文件。
    if not candidates:
        raise FileNotFoundError(
            f"No experiment directory found under {output_root} "
            f"for prefix={experiment_name!r}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_resume_artifacts(resume_from: Path) -> Dict[str, Path]:
    '''把 resume 输入规范化为实验目录及其关键文件路径。'''

    resume_from = resume_from.expanduser().resolve()
    if not resume_from.exists():
        raise FileNotFoundError(f"resume_from path not found: {resume_from}")

    # 兼容三种传法：
    # 1. 直接传 last_model.pt
    # 2. 传 checkpoints 目录
    # 3. 传实验目录本身
    if resume_from.is_file():
        checkpoint_path = resume_from
        checkpoints_dir = checkpoint_path.parent
        experiment_dir = checkpoints_dir.parent
    elif resume_from.name == "checkpoints":
        checkpoints_dir = resume_from
        experiment_dir = checkpoints_dir.parent
        checkpoint_path = checkpoints_dir / "last_model.pt"
    else:
        experiment_dir = resume_from
        checkpoints_dir = experiment_dir / "checkpoints"
        checkpoint_path = checkpoints_dir / "last_model.pt"

    trainer_state_path = checkpoints_dir / "trainer_state.json"
    optimizer_path = checkpoints_dir / "optimizer.pt"
    scheduler_path = checkpoints_dir / "scheduler.pt"
    label2id_path = checkpoints_dir / "label2id.json"
    id2label_path = checkpoints_dir / "id2label.json"

    required_paths = {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": checkpoints_dir,
        "checkpoint_path": checkpoint_path,
        "trainer_state_path": trainer_state_path,
        "optimizer_path": optimizer_path,
        "scheduler_path": scheduler_path,
        "label2id_path": label2id_path,
        "id2label_path": id2label_path,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Resume artifacts are incomplete. Missing: " + ", ".join(missing)
        )
    return required_paths


def format_seconds(seconds: float) -> str:
    '''格式化耗时。'''

    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_label_mapping_digest(label2id: Mapping[str, Any]) -> str:
    '''为标签映射生成稳定摘要。'''

    canonical = json.dumps(
        {str(key): int(value) for key, value in sorted(label2id.items(), key=lambda item: item[0])},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_git_commit(project_root: Path) -> Optional[str]:
    '''读取当前仓库 HEAD 提交号。

    若当前目录不是 git 仓库，返回 None。这个字段主要用于追踪实验来源，
    不参与 resume 的硬兼容校验。
    '''

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


@contextmanager
def timer() -> Iterator[MutableMapping[str, float]]:
    '''简单计时器，上下文退出时写入 elapsed_seconds。'''

    state: MutableMapping[str, float] = {}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed_seconds"] = time.perf_counter() - start


def set_seed(seed: int) -> None:
    '''固定随机种子。

    之所以集中处理，是为了避免训练脚本和分析脚本分别写一套随机性控制逻辑。
    '''

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            # 这里关闭 benchmark，换取更稳定、可复现的卷积/矩阵算子选择。
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def choose_device(profile: str = "cloud_train") -> str:
    '''选择运行设备。

    优先级：
    - cloud_train: cuda > mps > cpu
    - local_debug: mps > cuda > cpu
    '''

    try:
        import torch
    except ImportError:
        return "cpu"

    if profile == "local_debug":
        # 本地联调优先走 mps，尽量不占云端 GPU。
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_name(device: str) -> str:
    '''返回当前 GPU 名称。'''

    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return device


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    '''保存 checkpoint。'''

    ensure_dir(path.parent)
    import torch

    # 这里统一使用 torch.save，方便后续继续保存模型参数、配置和标签映射。
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    '''读取 checkpoint。'''

    import inspect
    import torch

    load_kwargs = {"map_location": map_location}
    # PyTorch 2.6+ 把 `weights_only` 的默认值改成了 True。
    # 当前项目的 checkpoint 不只保存权重，还包含配置、Path 和标签映射等元数据，
    # 因此这里显式回退到旧行为，确保本地与服务器都能稳定恢复。
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    '''把 batch 中的张量移动到指定设备。'''

    try:
        import torch
    except ImportError:
        return batch

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # 只搬运 tensor，保留原始 chars、tags、sample_ids 这些 Python 对象不动。
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def validate_prediction_output(
    text_lines: List[List[str]],
    pred_lines: List[List[str]],
) -> Dict[str, Any]:
    '''严格检查预测输出格式是否与原始文本对齐。

    这一步既服务于课程作业提交前自检，也服务于预测脚本自动写入
    `output_validation.json`，避免把错位文件带到服务器或最终提交阶段。
    '''

    line_count_match = len(text_lines) == len(pred_lines)
    token_count_mismatches = []
    empty_line_mismatches = []

    # line_count_match 需要单独看；zip 会在较短一侧结束，不能代替整体验证。
    for index, (text_tokens, pred_tokens) in enumerate(zip(text_lines, pred_lines), start=1):
        # 课程作业最严格的要求之一，就是每一行标签数必须与原句字数完全一致。
        if len(text_tokens) != len(pred_tokens):
            token_count_mismatches.append(
                {
                    "line_index": index,
                    "text_token_count": len(text_tokens),
                    "pred_token_count": len(pred_tokens),
                }
            )
        if (len(text_tokens) == 0) != (len(pred_tokens) == 0):
            empty_line_mismatches.append(index)

    return {
        "line_count_match": line_count_match,
        "source_line_count": len(text_lines),
        "prediction_line_count": len(pred_lines),
        "token_count_match": len(token_count_mismatches) == 0,
        "empty_line_match": len(empty_line_mismatches) == 0,
        "token_count_mismatches": token_count_mismatches[:20],
        "empty_line_mismatches": empty_line_mismatches[:20],
        "is_valid": line_count_match
        and len(token_count_mismatches) == 0
        and len(empty_line_mismatches) == 0,
    }


def write_prediction_file(path: Path, sequences: Iterable[List[str]]) -> None:
    '''按作业要求写出标签文件。'''

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        for tags in sequences:
            # 标签之间必须用空格分隔，保持与 train_TAG.txt 同样的文件格式。
            # 即使当前行为空，也要写一个换行，才能保持原始分行结构。
            file_obj.write(" ".join(tags))
            file_obj.write("\n")

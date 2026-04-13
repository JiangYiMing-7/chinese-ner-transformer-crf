'''数据处理模块。

负责数据审计、标签统计、fast tokenizer 对齐、
三条模型路线共享的数据集封装，以及训练阶段的 bucket batching。
'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from config import NERConfig
from utils import ensure_dir, save_json


TEXT_FILES = {
    "train_text": "train.txt",
    "train_tags": "train_TAG.txt",
    "dev_text": "dev.txt",
    "dev_tags": "dev_TAG.txt",
    "test_text": "test.txt",
}


@dataclass
class NERExample:
    '''单条样本。'''

    sample_id: int
    chars: List[str]
    tags: Optional[List[str]] = None


class NERDataset(Dataset):
    '''基础 NER 数据集。

    本阶段不做复杂增强，仅保留最直接、最稳定的数据访问接口。
    '''

    def __init__(self, examples: Sequence[NERExample]) -> None:
        '''保存样本列表。'''

        self.examples = list(examples)

    def __len__(self) -> int:
        '''返回样本数。'''

        return len(self.examples)

    def __getitem__(self, index: int) -> NERExample:
        '''按索引取样本。'''

        return self.examples[index]


class BucketBatchSampler(BatchSampler):
    '''按样本长度分桶的 batch sampler。

    这个 sampler 只改变“样本如何组成 batch”，不改变样本内容，也不改变标签对齐逻辑。
    目标是让同一批次中的句长更接近，从而减少 dynamic padding 带来的无效计算。
    '''

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size_multiplier: int = 50,
        shuffle_within_bucket: bool = True,
        drop_last: bool = False,
    ) -> None:
        '''保存分桶采样所需的长度与超参数。'''

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive.")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size_multiplier = bucket_size_multiplier
        self.shuffle_within_bucket = shuffle_within_bucket
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        '''按“大桶排序 + 小 batch 切分”的方式产生索引批次。'''

        indices = list(range(len(self.lengths)))

        # 先随机打散再进桶，可以兼顾训练随机性和“同桶长度接近”这两个目标。
        if self.shuffle_within_bucket:
            random.shuffle(indices)

        bucket_size = max(self.batch_size, self.batch_size * self.bucket_size_multiplier)
        batches: List[List[int]] = []

        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            # 大桶内部按长度排序，小 batch 会自然由相近长度样本组成。
            bucket.sort(key=lambda index: self.lengths[index])

            for batch_start in range(0, len(bucket), self.batch_size):
                batch_indices = bucket[batch_start : batch_start + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch_indices)

        # 最后再把 batch 的顺序打散，避免训练一直先看短句再看长句。
        random.shuffle(batches)
        for batch_indices in batches:
            yield batch_indices

    def __len__(self) -> int:
        '''返回一个 epoch 中的 batch 数。'''

        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class NERBatchCollator:
    '''将样本列表整理成 batch。

    这里显式使用 fast tokenizer 的 `word_ids()`，原因是中文虽然大多数情况是一字一 token，
    但数字、英文、符号仍可能发生拆分。只有基于 `word_ids()` 对齐，才能确保标签不会错位。
    '''

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        label2id: Dict[str, int],
        max_len: int,
        ignore_index: int = -100,
        enable_low_freq_char_dropout: bool = False,
        low_freq_chars: Optional[Sequence[str]] = None,
        low_freq_char_dropout_prob: float = 0.0,
    ) -> None:
        '''保存 batch 对齐所需的 tokenizer 和标签映射。'''

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.ignore_index = ignore_index
        self.enable_low_freq_char_dropout = enable_low_freq_char_dropout
        self.low_freq_chars = set(low_freq_chars or [])
        self.low_freq_char_dropout_prob = low_freq_char_dropout_prob
        self.unk_token = tokenizer.unk_token

    def _apply_low_freq_char_dropout(
        self,
        words_batch: Sequence[List[str]],
    ) -> List[List[str]]:
        '''仅在训练时对低频字做随机 UNK 替换。'''

        if (
            not self.enable_low_freq_char_dropout
            or not self.low_freq_chars
            or self.low_freq_char_dropout_prob <= 0.0
            or not self.unk_token
        ):
            return [list(words) for words in words_batch]

        augmented_batch: List[List[str]] = []
        for words in words_batch:
            augmented_words = []
            for char in words:
                if (
                    char in self.low_freq_chars
                    and random.random() < self.low_freq_char_dropout_prob
                ):
                    augmented_words.append(self.unk_token)
                else:
                    augmented_words.append(char)
            augmented_batch.append(augmented_words)
        return augmented_batch

    def __call__(self, batch: Sequence[NERExample]) -> Dict[str, Any]:
        '''把一批样本编码为模型可直接使用的 batch 字典。'''

        # tokenizer 的输入必须是“字列表的列表”，不能提前拼回字符串，
        # 否则 `word_ids()` 就无法告诉我们“当前 token 来自原句中的第几个字”。
        words_batch = [list(example.chars) for example in batch]
        # 低频字 dropout 只作用于训练输入，标签与原始长度语义保持不变。
        words_batch = self._apply_low_freq_char_dropout(words_batch)
        encoded = self.tokenizer(
            words_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_shape = encoded["input_ids"].shape
        # labels 默认全填 ignore_index，表示这些位置一开始都不参与 loss。
        labels = torch.full(input_shape, self.ignore_index, dtype=torch.long)
        # valid_mask 只标记“真正代表原始字的首个 token”，后续 accuracy 也只在这些位置上统计。
        valid_mask = torch.zeros(input_shape, dtype=torch.bool)
        kept_lengths: List[int] = []
        original_lengths: List[int] = []
        truncated_flags: List[bool] = []

        for batch_index, example in enumerate(batch):
            word_ids = encoded.word_ids(batch_index=batch_index)
            seen_word_ids = set()
            previous_word_id = None

            for token_index, word_id in enumerate(word_ids):
                if word_id is None:
                    # None 对应 special token 或 padding，必须跳过。
                    continue
                seen_word_ids.add(word_id)
                if word_id != previous_word_id:
                    # 只有一个原始字对应的第一个 token 才承接标签。
                    valid_mask[batch_index, token_index] = True
                    if example.tags is not None and word_id < len(example.tags):
                        labels[batch_index, token_index] = self.label2id[example.tags[word_id]]
                previous_word_id = word_id

            # kept_length 记录截断后真正还保留下来的原始字数量，预测恢复时会用到。
            kept_length = (max(seen_word_ids) + 1) if seen_word_ids else 0
            original_length = len(example.chars)
            kept_lengths.append(kept_length)
            original_lengths.append(original_length)
            truncated_flags.append(kept_length < original_length)

        batch_dict: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "valid_mask": valid_mask,
            "chars": words_batch,
            "tags": [example.tags for example in batch],
            "sample_ids": [example.sample_id for example in batch],
            "original_lengths": original_lengths,
            "kept_lengths": kept_lengths,
            "truncated_flags": truncated_flags,
        }
        if "token_type_ids" in encoded:
            batch_dict["token_type_ids"] = encoded["token_type_ids"]
        return batch_dict


def resolve_data_paths(config: NERConfig) -> Dict[str, Path]:
    '''解析实际可读的数据文件路径。

    当前工程已经统一把原始数据整理到 `project_root/data/` 下，
    因此这里显式只从 `data/` 目录读取。
    这样可以避免根目录残留旧文件时，程序悄悄读取到错误版本的数据。
    '''

    paths: Dict[str, Path] = {}
    for key, filename in TEXT_FILES.items():
        preferred = config.data_dir / filename
        if preferred.exists():
            paths[key] = preferred
        else:
            raise FileNotFoundError(
                f"Cannot find required data file: {preferred}"
            )
    return paths


def read_token_lines(path: Path) -> List[List[str]]:
    '''逐行读取空格分隔的字/标签序列。'''

    lines: List[List[str]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.rstrip("\n")
            # 空行本身也是课程数据格式的一部分，不能在读取阶段丢掉。
            lines.append(line.split() if line else [])
    return lines


def build_examples(
    text_lines: Sequence[List[str]],
    tag_lines: Optional[Sequence[List[str]]] = None,
) -> List[NERExample]:
    '''将原始行数据封装为样本对象。

    这里保留 `sample_id`，是为了后续预测、滑窗和错误分析时能稳定回溯到原始行号。
    '''

    examples: List[NERExample] = []
    for index, chars in enumerate(text_lines):
        tags = None if tag_lines is None else list(tag_lines[index])
        examples.append(NERExample(sample_id=index, chars=list(chars), tags=tags))
    return examples


def get_example_lengths(dataset_or_examples: Iterable[Any]) -> List[int]:
    '''提取样本原始字长度。

    这里的长度定义始终是“原始字序列长度”，而不是 tokenizer 之后的 token 数。
    bucket batching 只用这个长度近似分组，不碰标签和 mask 语义。
    '''

    if isinstance(dataset_or_examples, NERDataset):
        iterable = dataset_or_examples.examples
    else:
        iterable = dataset_or_examples

    lengths: List[int] = []
    for item in iterable:
        if isinstance(item, NERExample):
            lengths.append(len(item.chars))
        else:
            lengths.append(len(item["chars"]))
    return lengths


def build_train_dataloader(
    dataset: NERDataset,
    collator: NERBatchCollator,
    config: NERConfig,
) -> DataLoader:
    '''构造训练集 DataLoader。

    当打开 bucket batching 时，训练集不再使用普通 shuffle，而改成长度分桶的 batch sampler。
    评估与预测仍保持稳定顺序，因此不复用这个接口。
    '''

    if config.use_bucket_batching:
        sampler = BucketBatchSampler(
            lengths=get_example_lengths(dataset),
            batch_size=config.batch_size,
            bucket_size_multiplier=config.bucket_size_multiplier,
            shuffle_within_bucket=config.shuffle_within_bucket,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=collator,
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )


def check_text_tag_alignment(
    text_lines: Sequence[List[str]],
    tag_lines: Sequence[List[str]],
    split_name: str,
) -> Dict[str, Any]:
    '''检查文本和标签是否逐行对齐。

    这是最基础的数据健康检查。若这里出错，后续训练得到的任何指标都不可信。
    '''

    line_count_match = len(text_lines) == len(tag_lines)
    token_count_mismatches = []

    for line_index, (text_tokens, tag_tokens) in enumerate(zip(text_lines, tag_lines), start=1):
        if len(text_tokens) != len(tag_tokens):
            token_count_mismatches.append(
                {
                    "line_index": line_index,
                    "text_token_count": len(text_tokens),
                    "tag_token_count": len(tag_tokens),
                }
            )

    return {
        "split_name": split_name,
        "line_count_match": line_count_match,
        "text_line_count": len(text_lines),
        "tag_line_count": len(tag_lines),
        "token_count_match": len(token_count_mismatches) == 0,
        "token_count_mismatches": token_count_mismatches[:20],
    }


def build_label_metadata(tag_lines: Sequence[List[str]]) -> Dict[str, Any]:
    '''统计标签集、标签频次以及映射关系。

    这里显式使用 `collections.Counter`，因为课程说明要求说明
    “通过什么函数获得标注集及其频次”。Counter 可以同时完成
    标签去重和频次统计，是最直接且可复现的实现方式。
    '''

    counter = Counter()
    for line in tag_lines:
        counter.update(line)

    labels = []
    if "O" in counter:
        # 通常把 O 放在第一个，后续看日志和做错误分析都会更直观。
        labels.append("O")
    labels.extend(sorted(label for label in counter.keys() if label != "O"))

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    return {
        "labels": labels,
        "label_counts": dict(counter),
        "label2id": label2id,
        "id2label": {str(index): label for index, label in id2label.items()},
        "num_labels": len(labels),
    }


def build_char_frequency(text_lines: Sequence[List[str]]) -> Dict[str, int]:
    '''统计训练文本中的字频。'''

    counter = Counter()
    for line in text_lines:
        counter.update(line)
    return dict(counter)


def _length_summary(lengths: Sequence[int], max_len: int) -> Dict[str, Any]:
    '''统计长度分布。'''

    lengths_array = np.asarray(lengths, dtype=np.int64)
    if lengths_array.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "over_max_len_count": 0,
            "over_max_len_ratio": 0.0,
            "empty_line_count": 0,
            "lengths": [],
        }

    over_max_len_count = int((lengths_array > max_len).sum())
    return {
        "count": int(lengths_array.size),
        "min": int(lengths_array.min()),
        "max": int(lengths_array.max()),
        "mean": round(float(mean(lengths_array.tolist())), 4),
        "p50": int(np.percentile(lengths_array, 50)),
        "p90": int(np.percentile(lengths_array, 90)),
        "p95": int(np.percentile(lengths_array, 95)),
        "p99": int(np.percentile(lengths_array, 99)),
        "over_max_len_count": over_max_len_count,
        "over_max_len_ratio": round(over_max_len_count / max(1, len(lengths)), 6),
        "empty_line_count": int((lengths_array == 0).sum()),
        "lengths": lengths_array.tolist(),
    }


def build_tokenizer(
    model_name: str,
    fallback_model_name: Optional[str] = None,
) -> Tuple[PreTrainedTokenizerFast, str]:
    '''加载 fast tokenizer，必要时回退到备用模型。'''

    def _find_local_snapshot_dir(candidate: str) -> Optional[str]:
        '''在本地 Hugging Face 缓存中查找 tokenizer 快照目录。'''

        if Path(candidate).exists():
            return str(Path(candidate).resolve())

        cache_roots = []
        if os.environ.get("HF_HUB_CACHE"):
            cache_roots.append(Path(os.environ["HF_HUB_CACHE"]))
        if os.environ.get("HF_HOME"):
            cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
        cache_roots.extend(
            [
                Path("/tmp/huggingface/hub"),
                Path.home() / ".cache" / "huggingface" / "hub",
            ]
        )

        repo_dir_name = f"models--{candidate.replace('/', '--')}"
        for cache_root in cache_roots:
            snapshot_root = cache_root / repo_dir_name / "snapshots"
            if not snapshot_root.exists():
                continue

            candidates = [path for path in snapshot_root.iterdir() if path.is_dir()]
            if not candidates:
                continue

            best_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
            return str(best_snapshot.resolve())

        return None

    candidates = [model_name]
    if fallback_model_name and fallback_model_name != model_name:
        candidates.append(fallback_model_name)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            resolved_source = _find_local_snapshot_dir(candidate) or candidate
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_source,
                use_fast=True,
                local_files_only=Path(str(resolved_source)).exists(),
            )
            if not tokenizer.is_fast:
                # 这里强制要求 fast tokenizer，否则后续拿不到 `word_ids()`。
                raise ValueError(f"Tokenizer for {candidate} is not a fast tokenizer.")
            return tokenizer, str(resolved_source)
        except Exception as error:  # pragma: no cover - 依赖外部模型环境
            last_error = error

    raise RuntimeError(
        f"Failed to load tokenizer from candidates={candidates}. Last error: {last_error}"
    )


def sample_tokenizer_alignment(
    examples: Sequence[NERExample],
    tokenizer: PreTrainedTokenizerFast,
    max_len: int,
    sample_size: int = 5,
) -> List[Dict[str, Any]]:
    '''抽样检查 tokenizer 对齐情况。

    这个函数主要服务于数据审计和人工 spot check，帮助快速发现 `word_ids()`
    对齐、截断和 sub-token 拆分是否存在明显异常。
    '''

    chosen = [example for example in examples if example.chars][:sample_size]
    if not chosen:
        return []

    encoded = tokenizer(
        [example.chars for example in chosen],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    records: List[Dict[str, Any]] = []
    for batch_index, example in enumerate(chosen):
        word_ids = encoded.word_ids(batch_index=batch_index)
        input_ids = encoded["input_ids"][batch_index]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        aligned_labels: List[Optional[str]] = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(None)
            elif word_id != previous_word_id:
                # 这里的 aligned_labels 只是为了给报告或 debug 看“标签被挂到了哪个 token 上”。
                aligned_labels.append(example.tags[word_id] if example.tags else "__NO_LABEL__")
            else:
                aligned_labels.append(None)
            previous_word_id = word_id

        kept_word_count = 0
        if any(word_id is not None for word_id in word_ids):
            kept_word_count = max(word_id for word_id in word_ids if word_id is not None) + 1

        records.append(
            {
                "sample_id": example.sample_id,
                "original_chars": example.chars[:50],
                "original_tags": example.tags[:50] if example.tags else None,
                "tokens": tokens[:80],
                "word_ids": word_ids[:80],
                "aligned_labels": aligned_labels[:80],
                "original_length": len(example.chars),
                "kept_word_count": kept_word_count,
                "truncated": kept_word_count < len(example.chars),
            }
        )
    return records


def run_data_analysis(config: NERConfig) -> Dict[str, Any]:
    '''执行数据审计并保存 JSON 报告。'''

    ensure_dir(config.data_dir)
    paths = resolve_data_paths(config)

    train_text_lines = read_token_lines(paths["train_text"])
    train_tag_lines = read_token_lines(paths["train_tags"])
    dev_text_lines = read_token_lines(paths["dev_text"])
    dev_tag_lines = read_token_lines(paths["dev_tags"])
    test_text_lines = read_token_lines(paths["test_text"])

    train_alignment = check_text_tag_alignment(train_text_lines, train_tag_lines, "train")
    dev_alignment = check_text_tag_alignment(dev_text_lines, dev_tag_lines, "dev")

    # 标签统计只基于训练集获得，后续训练和预测都必须使用同一套映射。
    label_stats = build_label_metadata(train_tag_lines)
    tokenizer, resolved_model_name = build_tokenizer(
        config.model_name,
        config.fallback_model_name,
    )

    train_examples = build_examples(train_text_lines, train_tag_lines)
    alignment_samples = sample_tokenizer_alignment(
        train_examples,
        tokenizer=tokenizer,
        max_len=config.max_len,
        sample_size=5,
    )

    train_lengths = [len(line) for line in train_text_lines]
    dev_lengths = [len(line) for line in dev_text_lines]
    test_lengths = [len(line) for line in test_text_lines]

    # data_report 会作为后续说明文档和画图脚本的直接输入，因此保留较完整的统计信息。
    data_report = {
        "requested_model_name": config.model_name,
        "resolved_tokenizer_name": resolved_model_name,
        "project_root": str(config.project_root),
        "data_dir": str(config.data_dir),
        "max_len": config.max_len,
        "paths": {key: str(path) for key, path in paths.items()},
        "train_alignment_check": train_alignment,
        "dev_alignment_check": dev_alignment,
        "train_length_stats": _length_summary(train_lengths, config.max_len),
        "dev_length_stats": _length_summary(dev_lengths, config.max_len),
        "test_length_stats": _length_summary(test_lengths, config.max_len),
        "tokenizer_alignment_samples": alignment_samples,
        "tokenizer_is_fast": tokenizer.is_fast,
    }

    label_stats_path = config.data_dir / "label_stats.json"
    data_report_path = config.data_dir / "data_report.json"
    save_json(label_stats_path, label_stats)
    save_json(data_report_path, data_report)

    print("=== Data Analysis Summary ===")
    print(f"Project root: {config.project_root}")
    print(f"Requested backbone: {config.model_name}")
    print(f"Resolved tokenizer: {resolved_model_name}")
    print(f"Train alignment ok: {train_alignment['line_count_match'] and train_alignment['token_count_match']}")
    print(f"Dev alignment ok: {dev_alignment['line_count_match'] and dev_alignment['token_count_match']}")
    print(f"Labels: {label_stats['labels']}")
    print(f"Train P95 length: {data_report['train_length_stats']['p95']}")
    print(f"Dev P95 length: {data_report['dev_length_stats']['p95']}")
    print(f"Test P95 length: {data_report['test_length_stats']['p95']}")
    print(
        "Over max_len count: "
        f"train={data_report['train_length_stats']['over_max_len_count']} "
        f"({data_report['train_length_stats']['over_max_len_ratio']:.4f}), "
        f"dev={data_report['dev_length_stats']['over_max_len_count']} "
        f"({data_report['dev_length_stats']['over_max_len_ratio']:.4f}), "
        f"test={data_report['test_length_stats']['over_max_len_count']} "
        f"({data_report['test_length_stats']['over_max_len_ratio']:.4f})"
    )
    print(f"Saved: {data_report_path}")
    print(f"Saved: {label_stats_path}")

    return {
        "data_report": data_report,
        "label_stats": label_stats,
        "paths": paths,
        "tokenizer": tokenizer,
        "resolved_model_name": resolved_model_name,
    }


def load_label_stats(config: NERConfig) -> Dict[str, Any]:
    '''读取或生成标签统计。'''

    label_stats_path = config.data_dir / "label_stats.json"
    if label_stats_path.exists():
        import json

        with label_stats_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    return run_data_analysis(config)["label_stats"]


def prepare_datasets_and_tokenizer(
    config: NERConfig,
) -> Dict[str, Any]:
    '''为训练、评估、预测准备数据对象和 tokenizer。

    这里统一返回三套 dataset、标签映射、tokenizer 和长度统计，避免训练脚本自己重复读数据。
    '''

    analysis_outputs = run_data_analysis(config)
    paths = analysis_outputs["paths"]
    label_stats = analysis_outputs["label_stats"]
    tokenizer = analysis_outputs["tokenizer"]
    resolved_model_name = analysis_outputs["resolved_model_name"]

    train_text_lines = read_token_lines(paths["train_text"])
    train_tag_lines = read_token_lines(paths["train_tags"])
    dev_text_lines = read_token_lines(paths["dev_text"])
    dev_tag_lines = read_token_lines(paths["dev_tags"])
    test_text_lines = read_token_lines(paths["test_text"])

    if config.profile == "local_debug":
        subset_size = config.debug_subset_size
        # 本地联调时只截一小部分数据，优先验证流程正确性和代码稳定性。
        train_text_lines = train_text_lines[:subset_size]
        train_tag_lines = train_tag_lines[:subset_size]
        dev_text_lines = dev_text_lines[:subset_size]
        dev_tag_lines = dev_tag_lines[:subset_size]
        test_text_lines = test_text_lines[:subset_size]

    train_examples = build_examples(train_text_lines, train_tag_lines)
    dev_examples = build_examples(dev_text_lines, dev_tag_lines)
    test_examples = build_examples(test_text_lines, None)
    train_char_counts = build_char_frequency(train_text_lines)
    # 这里的长度统计基于“本次实际参与运行的数据”计算，
    # 因此 local_debug 下看到的是调试子集口径，不是完整训练集口径。
    runtime_length_stats = {
        "train": _length_summary([len(line) for line in train_text_lines], config.max_len),
        "dev": _length_summary([len(line) for line in dev_text_lines], config.max_len),
        "test": _length_summary([len(line) for line in test_text_lines], config.max_len),
    }

    return {
        "train_dataset": NERDataset(train_examples),
        "dev_dataset": NERDataset(dev_examples),
        "test_dataset": NERDataset(test_examples),
        "label2id": label_stats["label2id"],
        "id2label": {
            int(index): label for index, label in label_stats["id2label"].items()
        },
        "labels": label_stats["labels"],
        "data_report": analysis_outputs["data_report"],
        "runtime_length_stats": runtime_length_stats,
        "train_char_counts": train_char_counts,
        "tokenizer": tokenizer,
        "resolved_model_name": resolved_model_name,
        "paths": paths,
    }

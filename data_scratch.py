'''Stage5 字符级数据处理模块。

该模块服务于手写 Transformer 路线，完全不依赖 HuggingFace tokenizer。
核心思路是：
1. 直接把原始中文字符序列映射为 char id
2. 用 `<PAD>=0`、`<UNK>=1` 管理字符词表
3. 训练和逐轮开发集评估时，把超长样本切成不重叠块
4. 训练集使用手写 bucket batching，减少 padding 浪费
'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset


TEXT_FILES = {
    "train_text": "train.txt",
    "train_tags": "train_TAG.txt",
    "dev_text": "dev.txt",
    "dev_tags": "dev_TAG.txt",
    "test_text": "test.txt",
}


def resolve_scratch_data_paths(project_root: Path) -> Dict[str, Path]:
    '''解析 Stage5 使用的数据文件路径。'''

    data_dir = project_root / "data"
    paths: Dict[str, Path] = {}
    for key, filename in TEXT_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Cannot find required data file: {path}")
        paths[key] = path
    return paths


def read_token_lines(path: Path) -> List[List[str]]:
    '''逐行读取空格分隔的字/标签序列。'''

    lines: List[List[str]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.rstrip("\n")
            lines.append(line.split() if line else [])
    return lines


def build_char_vocab(
    text_lines: Sequence[Sequence[str]],
    min_freq: int = 1,
) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    '''从训练集字符序列构建词表。

    说明：
    - 显式使用 `collections.Counter` 统计字频
    - 特殊 token 固定为 `<PAD>=0`、`<UNK>=1`
    - 频次低于 `min_freq` 的字符统一映射到 `<UNK>`
    '''

    if min_freq <= 0:
        raise ValueError("min_freq must be positive.")

    char_freq: Counter = Counter()
    for chars in text_lines:
        char_freq.update(chars)

    char2id: Dict[str, int] = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for char, count in char_freq.most_common():
        if count < min_freq:
            continue
        char2id[char] = len(char2id)

    id2char = {index: char for char, index in char2id.items()}
    total_char_count = sum(int(count) for count in char_freq.values())
    mapped_to_unk_count = sum(
        int(count)
        for char, count in char_freq.items()
        if char not in char2id
    )
    oov_ratio = mapped_to_unk_count / total_char_count if total_char_count else 0.0

    print(
        "Scratch vocab built: "
        f"size={len(char2id)} | min_freq={min_freq} | "
        f"train_UNK_ratio={oov_ratio:.4%}"
    )
    return char2id, id2char, char_freq


def build_label_metadata(tag_lines: Sequence[Sequence[str]]) -> Dict[str, Any]:
    '''按现有项目口径构造标签集与映射。

    规则与 `data.py/build_label_metadata(...)` 一致：
    - `O` 放在第一个
    - 其余标签按字典序排序
    '''

    counter: Counter = Counter()
    for tags in tag_lines:
        counter.update(tags)

    labels: List[str] = []
    if "O" in counter:
        labels.append("O")
    labels.extend(sorted(label for label in counter.keys() if label != "O"))

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return {
        "labels": labels,
        "label_counts": dict(counter),
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(labels),
    }


@dataclass
class ScratchNERExample:
    '''字符级 NER 样本。'''

    sample_id: int
    chars: List[str]
    char_ids: List[int]
    tags: Optional[List[str]]
    tag_ids: Optional[List[int]]


def build_scratch_examples(
    text_lines: Sequence[Sequence[str]],
    tag_lines: Optional[Sequence[Sequence[str]]],
    char2id: Dict[str, int],
    label2id: Optional[Dict[str, int]],
) -> List[ScratchNERExample]:
    '''把原始字符/标签序列封装成字符级样本对象。'''

    unk_id = int(char2id["<UNK>"])
    examples: List[ScratchNERExample] = []
    for sample_id, chars in enumerate(text_lines):
        char_list = list(chars)
        char_ids = [int(char2id.get(char, unk_id)) for char in char_list]

        tags: Optional[List[str]] = None
        tag_ids: Optional[List[int]] = None
        if tag_lines is not None:
            if label2id is None:
                raise ValueError("label2id is required when tag_lines is not None")
            tags = list(tag_lines[sample_id])
            tag_ids = [int(label2id[tag]) for tag in tags]

        examples.append(
            ScratchNERExample(
                sample_id=sample_id,
                chars=char_list,
                char_ids=char_ids,
                tags=tags,
                tag_ids=tag_ids,
            )
        )
    return examples


class ScratchNERDataset(Dataset):
    '''字符级 NER 数据集。'''

    def __init__(self, examples: Sequence[ScratchNERExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ScratchNERExample:
        return self.examples[index]


def get_example_lengths(dataset_or_examples: Iterable[Any]) -> List[int]:
    '''提取样本长度，长度定义始终是原始字符数。'''

    if isinstance(dataset_or_examples, ScratchNERDataset):
        iterable = dataset_or_examples.examples
    else:
        iterable = dataset_or_examples

    lengths: List[int] = []
    for item in iterable:
        if isinstance(item, ScratchNERExample):
            lengths.append(len(item.char_ids))
        else:
            lengths.append(len(item["char_ids"]))
    return lengths


def chunk_examples_by_max_len(
    examples: Sequence[ScratchNERExample],
    max_seq_len: int,
) -> List[ScratchNERExample]:
    '''把超长样本切成不重叠块。

    该函数主要用于：
    - 训练集：避免从零训练时一次性喂入过长序列
    - 逐轮开发集评估：让 `evaluate_model(...)` 能在固定长度窗口上直接工作

    这里切的是不重叠块，因此所有真实字符都会被完整覆盖一次，不会重复计数。
    '''

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")

    chunked_examples: List[ScratchNERExample] = []
    next_sample_id = 0
    for example in examples:
        if len(example.char_ids) <= max_seq_len:
            chunked_examples.append(
                ScratchNERExample(
                    sample_id=next_sample_id,
                    chars=list(example.chars),
                    char_ids=list(example.char_ids),
                    tags=None if example.tags is None else list(example.tags),
                    tag_ids=None if example.tag_ids is None else list(example.tag_ids),
                )
            )
            next_sample_id += 1
            continue

        for start in range(0, len(example.char_ids), max_seq_len):
            end = min(len(example.char_ids), start + max_seq_len)
            chunked_examples.append(
                ScratchNERExample(
                    sample_id=next_sample_id,
                    chars=list(example.chars[start:end]),
                    char_ids=list(example.char_ids[start:end]),
                    tags=None if example.tags is None else list(example.tags[start:end]),
                    tag_ids=None if example.tag_ids is None else list(example.tag_ids[start:end]),
                )
            )
            next_sample_id += 1

    return chunked_examples


class ScratchBucketBatchSampler(BatchSampler):
    '''手写 Bucket Batching。

    算法：
    1. 先随机打乱样本索引
    2. 按 bucket_size 分成若干大桶
    3. 每个桶内按句长排序
    4. 从桶中依次切出 batch
    5. 最后再把所有 batch 顺序随机打乱一次

    这样同一 batch 内的句长更接近，可以显著减少 padding 计算量。
    '''

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size_multiplier: int = 50,
        shuffle_within_bucket: bool = True,
        drop_last: bool = False,
    ) -> None:
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
        indices = list(range(len(self.lengths)))
        if self.shuffle_within_bucket:
            random.shuffle(indices)

        bucket_size = max(self.batch_size, self.batch_size * self.bucket_size_multiplier)
        batches: List[List[int]] = []

        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])

            for batch_start in range(0, len(bucket), self.batch_size):
                batch_indices = bucket[batch_start : batch_start + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch_indices)

        random.shuffle(batches)
        for batch_indices in batches:
            yield batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class ScratchBatchCollator:
    '''把字符级样本整理成 batch。

    返回字段：
    - input_ids: (B, T)
    - attention_mask: (B, T)，True 表示真实字符，False 表示 padding
    - valid_mask: (B, T)，Stage5 中直接等于 attention_mask
    - labels: (B, T)，padding 位置为 ignore_index
    - sample_ids: 原始样本 id 列表
    - original_lengths: 每条样本的真实字符长度
    '''

    def __init__(
        self,
        pad_idx: int = 0,
        ignore_index: int = -100,
        max_len: Optional[int] = None,
    ) -> None:
        self.pad_idx = pad_idx
        self.ignore_index = ignore_index
        self.max_len = max_len

    def __call__(self, batch: Sequence[ScratchNERExample]) -> Dict[str, Any]:
        input_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []
        original_lengths: List[int] = []
        sample_ids: List[int] = []

        for example in batch:
            char_ids = list(example.char_ids)
            tag_ids = None if example.tag_ids is None else list(example.tag_ids)

            if self.max_len is not None:
                char_ids = char_ids[: self.max_len]
                if tag_ids is not None:
                    tag_ids = tag_ids[: self.max_len]

            input_tensors.append(torch.tensor(char_ids, dtype=torch.long))
            if tag_ids is None:
                label_tensors.append(
                    torch.full((len(char_ids),), self.ignore_index, dtype=torch.long)
                )
            else:
                label_tensors.append(torch.tensor(tag_ids, dtype=torch.long))

            original_lengths.append(len(char_ids))
            sample_ids.append(int(example.sample_id))

        # pad_sequence 把不同长度的字符序列补齐到 batch 内最长。# (B, T)
        input_ids = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_idx,
        )
        # padding 位是 `<PAD>`，真实字符位是非 `<PAD>`。# (B, T)
        attention_mask = input_ids.ne(self.pad_idx)
        # Stage5 是纯字符级模型，没有 sub-token，所以 valid_mask 与 attention_mask 相同。# (B, T)
        valid_mask = attention_mask.clone()
        # 标签也补齐到相同长度，padding 位统一填 ignore_index。# (B, T)
        labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=self.ignore_index,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "valid_mask": valid_mask,
            "labels": labels,
            "sample_ids": sample_ids,
            "original_lengths": original_lengths,
        }


def build_scratch_train_dataloader(
    dataset: ScratchNERDataset,
    collator: ScratchBatchCollator,
    batch_size: int,
    bucket_size_multiplier: int = 50,
    shuffle_within_bucket: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    '''构造训练集 DataLoader，并启用手写 bucket batching。'''

    sampler = ScratchBucketBatchSampler(
        lengths=get_example_lengths(dataset),
        batch_size=batch_size,
        bucket_size_multiplier=bucket_size_multiplier,
        shuffle_within_bucket=shuffle_within_bucket,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )


def build_scratch_eval_dataloader(
    dataset: ScratchNERDataset,
    collator: ScratchBatchCollator,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    '''构造评估或预测使用的稳定顺序 DataLoader。'''

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

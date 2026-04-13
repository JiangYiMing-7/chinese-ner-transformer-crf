'''手写线性链 CRF 模块。

负责：
1. 提供可学习的 start / transition / end 参数
2. 使用 log-sum-exp 计算配分函数
3. 计算 gold path score 与负对数似然
4. 支持 batch + mask 的 Viterbi 解码
5. 提供可选的 BIO 约束接口

注意：
- 本模块接收的 mask 应该对应“压缩后的真实标签序列”，也就是前缀连续的有效位。
- 对于本项目而言，原始 batch 中的 `valid_mask` 可能包含 special token 和 sub-token 造成的空洞，
  因此模型层会先把 emissions 压缩到真实字位置，再把压缩后的 `crf_mask` 传进来。
'''

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _split_bio_label(label: str) -> Tuple[str, Optional[str]]:
    '''拆解 BIO 标签。

    返回：
    - prefix: `O` / `B` / `I` / 其他
    - entity_type: 例如 `PER` / `LOC`
    '''

    if label == "O":
        return "O", None
    if "_" not in label:
        return label, None
    prefix, entity_type = label.split("_", 1)
    return prefix, entity_type


def _is_valid_bio_start(label: str) -> bool:
    '''判断标签能否作为严格 BIO 序列的起点。'''

    prefix, _ = _split_bio_label(label)
    # 严格 BIO 中序列起点不能是 I_*。
    return prefix != "I"


def _is_valid_bio_transition(previous_label: str, next_label: str) -> bool:
    '''判断两个标签之间是否满足严格 BIO 转移。'''

    next_prefix, next_entity = _split_bio_label(next_label)
    previous_prefix, previous_entity = _split_bio_label(previous_label)

    if next_prefix != "I":
        # O 和 B_* 默认都允许接在任意标签后面。
        return True
    return previous_prefix in {"B", "I"} and previous_entity == next_entity


def build_bio_constraint_matrices(
    labels: Sequence[str],
    mode: str = "none",
    invalid_score: float = -10000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    '''构造 start / transition / end 约束矩阵。

    返回值依次为：
    - start_constraints: [num_tags]
    - transition_constraints: [num_tags, num_tags]
    - end_constraints: [num_tags]
    - effective_mode: 当前实际使用的约束模式

    `auto` 当前只保留接口，暂时退化为 `none`。
    '''

    if mode not in {"none", "strict_bio", "auto"}:
        raise ValueError(f"Unsupported bio constraint mode: {mode}")

    effective_mode = "none" if mode == "auto" else mode
    num_tags = len(labels)
    start_constraints = torch.zeros(num_tags, dtype=torch.float)
    transition_constraints = torch.zeros(num_tags, num_tags, dtype=torch.float)
    end_constraints = torch.zeros(num_tags, dtype=torch.float)

    if effective_mode == "none":
        return start_constraints, transition_constraints, end_constraints, effective_mode

    for tag_index, label in enumerate(labels):
        if not _is_valid_bio_start(label):
            start_constraints[tag_index] = invalid_score

    for previous_index, previous_label in enumerate(labels):
        for next_index, next_label in enumerate(labels):
            if not _is_valid_bio_transition(previous_label, next_label):
                transition_constraints[previous_index, next_index] = invalid_score

    return start_constraints, transition_constraints, end_constraints, effective_mode


class LinearChainCRF(nn.Module):
    '''手写线性链 CRF。

    约定：
    - `transitions[i, j]` 表示从标签 `i` 转移到标签 `j` 的分数。
    - `mask` 必须是前缀连续的布尔矩阵；如果原始位置有空洞，应先在模型里压缩。
    '''

    def __init__(
        self,
        num_tags: int,
        labels: Optional[Sequence[str]] = None,
        bio_constraint_mode: str = "none",
        invalid_score: float = -10000.0,
    ) -> None:
        '''初始化 CRF 参数和可选 BIO 约束缓冲区。'''

        super().__init__()
        self.num_tags = num_tags
        self.bio_constraint_mode = bio_constraint_mode
        self.invalid_score = invalid_score

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

        if labels is None:
            labels = [str(index) for index in range(num_tags)]
        start_constraints, transition_constraints, end_constraints, effective_mode = (
            build_bio_constraint_matrices(
                labels=labels,
                mode=bio_constraint_mode,
                invalid_score=invalid_score,
            )
        )
        self.effective_bio_constraint_mode = effective_mode
        self.register_buffer("start_constraints", start_constraints)
        self.register_buffer("transition_constraints", transition_constraints)
        self.register_buffer("end_constraints", end_constraints)

    def reset_parameters(self) -> None:
        '''初始化 CRF 参数。'''

        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _effective_start_transitions(self) -> torch.Tensor:
        '''返回加上约束后的起始转移分数。'''

        return self.start_transitions + self.start_constraints

    def _effective_end_transitions(self) -> torch.Tensor:
        '''返回加上约束后的结束转移分数。'''

        return self.end_transitions + self.end_constraints

    def _effective_transitions(self) -> torch.Tensor:
        '''返回加上约束后的标签间转移分数。'''

        return self.transitions + self.transition_constraints

    def _validate_inputs(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        '''检查输入张量维度，并统一把 mask 转成 bool。

        这里要求 `mask` 是前缀连续的；若 mask 中间有空洞，应在模型层先压缩后再送入 CRF。
        '''

        if emissions.dim() != 3:
            raise ValueError(f"emissions must have shape [batch, seq_len, num_tags], got {emissions.shape}")
        if emissions.size(-1) != self.num_tags:
            raise ValueError(
                f"Expected emissions.size(-1) == {self.num_tags}, got {emissions.size(-1)}"
            )
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape [batch, seq_len], got {mask.shape}")
        if emissions.shape[:2] != mask.shape:
            raise ValueError(
                f"emissions and mask must agree on batch/seq dims, got {emissions.shape[:2]} vs {mask.shape}"
            )
        if tags is not None and tags.shape != mask.shape:
            raise ValueError(f"tags and mask must have identical shape, got {tags.shape} vs {mask.shape}")
        return mask.bool()

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''计算每条序列的 log partition。'''

        start_transitions = self._effective_start_transitions()
        end_transitions = self._effective_end_transitions()
        transitions = self._effective_transitions()

        score = start_transitions.unsqueeze(0) + emissions[:, 0]
        sequence_length = emissions.size(1)

        for time_step in range(1, sequence_length):
            # 前一时刻的所有路径分数 + 标签间转移 + 当前发射分数。
            next_score = (
                score.unsqueeze(2)
                + transitions.unsqueeze(0)
                + emissions[:, time_step].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)
            # mask 为 False 的位置说明该序列已经结束，继续保留上一步的状态即可。
            score = torch.where(mask[:, time_step].unsqueeze(1), next_score, score)

        score = score + end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        '''计算真实标签路径分数。'''

        start_transitions = self._effective_start_transitions()
        end_transitions = self._effective_end_transitions()
        transitions = self._effective_transitions()

        first_tags = tags[:, 0]
        score = start_transitions[first_tags]
        score = score + emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        sequence_length = emissions.size(1)
        for time_step in range(1, sequence_length):
            current_tags = tags[:, time_step]
            previous_tags = tags[:, time_step - 1]
            transition_score = transitions[previous_tags, current_tags]
            emission_score = emissions[:, time_step].gather(1, current_tags.unsqueeze(1)).squeeze(1)
            score = score + (transition_score + emission_score) * mask[:, time_step].float()

        last_tag_indices = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + end_transitions[last_tags]
        return score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        '''计算负对数似然。

        对于全无效位样本，直接跳过，不让它们污染 loss 或触发边界错误。
        '''

        mask = self._validate_inputs(emissions, tags, mask)
        lengths = mask.long().sum(dim=1)
        valid_sequence_mask = lengths.gt(0)

        if not valid_sequence_mask.any():
            # 保持图可导，同时避免返回 NaN。
            return emissions.sum() * 0.0

        # 先过滤掉全无效位样本，后面的配分函数和 gold score 才能安全假设“序列非空”。
        valid_emissions = emissions[valid_sequence_mask]
        valid_tags = tags[valid_sequence_mask]
        valid_mask = mask[valid_sequence_mask]

        log_partition = self._compute_log_partition(valid_emissions, valid_mask)
        gold_score = self._compute_gold_score(valid_emissions, valid_tags, valid_mask)
        nll = log_partition - gold_score

        if reduction == "sum":
            return nll.sum()
        if reduction == "none":
            return nll
        if reduction == "mean":
            return nll.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        '''与 `neg_log_likelihood` 保持一致的前向接口。'''

        return self.neg_log_likelihood(emissions=emissions, tags=tags, mask=mask, reduction=reduction)

    def _viterbi_decode_single(self, emissions: torch.Tensor) -> List[int]:
        '''对单条序列执行 Viterbi 解码。'''

        if emissions.size(0) == 0:
            return []

        start_transitions = self._effective_start_transitions()
        end_transitions = self._effective_end_transitions()
        transitions = self._effective_transitions()

        score = start_transitions + emissions[0]
        history: List[torch.Tensor] = []

        for time_step in range(1, emissions.size(0)):
            # 对当前每个目标标签，都记录“来自哪个前一标签最优”。
            candidate_score = score.unsqueeze(1) + transitions
            best_score, best_path = candidate_score.max(dim=0)
            score = best_score + emissions[time_step]
            history.append(best_path)

        score = score + end_transitions
        best_last_tag = int(score.argmax().item())
        best_tags = [best_last_tag]

        for best_path in reversed(history):
            best_last_tag = int(best_path[best_last_tag].item())
            best_tags.append(best_last_tag)

        best_tags.reverse()
        return best_tags

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        '''执行 batch Viterbi 解码。

        返回的每条路径长度都严格等于对应样本的有效长度。
        '''

        mask = self._validate_inputs(emissions, None, mask)
        lengths = mask.long().sum(dim=1)
        decoded_paths: List[List[int]] = []

        for batch_index in range(emissions.size(0)):
            valid_length = int(lengths[batch_index].item())
            if valid_length <= 0:
                decoded_paths.append([])
                continue
            decoded_paths.append(
                self._viterbi_decode_single(emissions[batch_index, :valid_length])
            )
        return decoded_paths

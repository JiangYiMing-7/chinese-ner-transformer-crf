'''NER 模型模块。

当前支持三条模型路线：
1. baseline:      RoBERTa -> dropout -> linear
2. final_stage2:  RoBERTa -> dropout -> linear -> CRF
3. final_stage3:  RoBERTa -> refinement -> dropout -> linear -> CRF
'''

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from crf import LinearChainCRF
from model_refine import RefinementEncoder


class RobertaLinearNER(nn.Module):
    '''统一 NER 模型。'''

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        fallback_model_name: Optional[str] = None,
        ignore_index: int = -100,
        use_refinement: bool = False,
        refine_num_layers: int = 1,
        refine_num_heads: int = 4,
        refine_ffn_dim: int = 1536,
        refine_dropout: float = 0.1,
        use_crf: bool = False,
        bio_constraint_mode: str = "none",
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        '''按配置组装 backbone、refinement、classifier 与 CRF。'''

        super().__init__()
        # backbone 和新增层分开写，便于在不破坏旧路径的前提下平滑扩展。
        self.backbone, self.resolved_model_name = self._load_backbone(
            model_name=model_name,
            fallback_model_name=fallback_model_name,
        )
        hidden_size = int(self.backbone.config.hidden_size)

        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.use_refinement = use_refinement
        self.use_crf = use_crf
        self.bio_constraint_mode = bio_constraint_mode
        self.labels = self._build_ordered_labels(num_labels, label2id)
        self.backbone_hidden_size = hidden_size
        self.refine_num_layers = refine_num_layers
        self.refine_num_heads = refine_num_heads
        self.refine_ffn_dim = refine_ffn_dim
        self.refine_dropout = refine_dropout

        self.refinement: Optional[RefinementEncoder] = None
        if self.use_refinement:
            self.refinement = RefinementEncoder(
                hidden_size=hidden_size,
                num_layers=refine_num_layers,
                num_heads=refine_num_heads,
                ffn_dim=refine_ffn_dim,
                dropout=refine_dropout,
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.crf: Optional[LinearChainCRF] = None
        self.effective_bio_constraint_mode = "none"
        if self.use_crf:
            self.crf = LinearChainCRF(
                num_tags=num_labels,
                labels=self.labels,
                bio_constraint_mode=bio_constraint_mode,
            )
            self.effective_bio_constraint_mode = self.crf.effective_bio_constraint_mode

    @staticmethod
    def _find_local_snapshot_dir(model_name: str) -> Optional[str]:
        '''从本地 Hugging Face 缓存中查找模型快照目录。

        这样做的原因：
        - 本地机器已经缓存过 backbone 时，没有必要再次访问网络
        - 某些离线或弱网环境里，`snapshot_download(local_files_only=True)` 仍可能抛异常
        - 直接解析缓存目录可以让训练后的 predict 在本地更稳地复现
        '''

        if Path(model_name).exists():
            return str(Path(model_name).resolve())

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

        repo_dir_name = f"models--{model_name.replace('/', '--')}"
        for cache_root in cache_roots:
            snapshot_root = cache_root / repo_dir_name / "snapshots"
            if not snapshot_root.exists():
                continue

            candidates = [path for path in snapshot_root.iterdir() if path.is_dir()]
            if not candidates:
                continue

            # 通常快照目录只有一个；若有多个，优先使用最近修改过的目录。
            best_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
            return str(best_snapshot.resolve())

        return None

    @staticmethod
    def _build_ordered_labels(
        num_labels: int,
        label2id: Optional[Dict[str, int]],
    ) -> List[str]:
        '''把 label2id 还原成按 id 排序的标签列表。

        CRF 约束矩阵依赖稳定的标签顺序，因此这里不能直接使用字典的迭代顺序。
        '''

        if label2id is None:
            return [str(index) for index in range(num_labels)]
        return [
            label
            for label, _ in sorted(label2id.items(), key=lambda item: item[1])
        ]

    @staticmethod
    def _load_backbone(
        model_name: str,
        fallback_model_name: Optional[str] = None,
    ) -> Tuple[nn.Module, str]:
        '''加载 backbone，必要时回退到备用模型。

        返回值中的第二项是实际解析成功的来源路径或模型名，主要用于日志和恢复元数据。
        '''

        candidates = [model_name]
        if fallback_model_name and fallback_model_name != model_name:
            candidates.append(fallback_model_name)

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                resolved_source = RobertaLinearNER._find_local_snapshot_dir(candidate) or candidate

                return AutoModel.from_pretrained(
                    resolved_source,
                    local_files_only=Path(str(resolved_source)).exists(),
                    use_safetensors=False,
                ), str(resolved_source)
            except Exception as error:  # pragma: no cover - 依赖外部模型环境
                last_error = error

        raise RuntimeError(
            f"Failed to load backbone from candidates={candidates}. Last error: {last_error}"
        )

    def compress_for_crf(
        self,
        emissions: torch.Tensor,
        labels: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        '''把原始 token 级输出压缩到真实字位置。

        关键原因：
        - 原始 `valid_mask` 在 special token 和被拆开的 sub-token 位置会出现空洞
        - 线性链 CRF 需要前缀连续的 mask
        - 因此必须先把真实字对应的 emissions 抽出来，再构造连续的 `crf_mask`
        '''

        valid_mask = valid_mask.bool()
        batch_size, _, num_labels = emissions.shape
        valid_lengths = valid_mask.long().sum(dim=1)
        max_valid_length = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
        # 即使这一批里全是空样本，也保留长度 1 的占位张量，避免后续 shape=0 的边界问题。
        max_valid_length = max(1, max_valid_length)

        compressed_emissions = emissions.new_zeros((batch_size, max_valid_length, num_labels))
        compressed_labels: Optional[torch.Tensor] = None
        if labels is not None:
            compressed_labels = labels.new_zeros((batch_size, max_valid_length))
        crf_mask = torch.zeros((batch_size, max_valid_length), dtype=torch.bool, device=emissions.device)

        for batch_index in range(batch_size):
            valid_indices = valid_mask[batch_index].nonzero(as_tuple=False).squeeze(-1)
            valid_count = int(valid_indices.numel())
            if valid_count == 0:
                continue

            compressed_emissions[batch_index, :valid_count] = emissions[batch_index, valid_indices]
            crf_mask[batch_index, :valid_count] = True

            if compressed_labels is not None:
                selected_labels = labels[batch_index, valid_indices]
                if torch.any(selected_labels.lt(0)):
                    raise ValueError(
                        "Found negative labels at valid positions. "
                        "This usually indicates a tokenizer alignment bug."
                    )
                compressed_labels[batch_index, :valid_count] = selected_labels

        return compressed_emissions, compressed_labels, crf_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        '''前向传播，并按当前路线计算 loss。

        baseline 路线返回交叉熵 loss，CRF 路线会先压缩真实字位置，再返回负对数似然。
        '''

        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = backbone_outputs.last_hidden_state

        if self.use_refinement and self.refinement is not None:
            # refinement 使用的 attention mask 直接从现有 attention_mask 派生。
            # 它只控制哪些 token 在自注意力中可见，不能替代 valid_mask 的真实字语义。
            hidden_states = self.refinement(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        hidden_states = self.dropout(hidden_states)
        emissions = self.classifier(hidden_states)

        outputs: Dict[str, Any] = {
            "logits": emissions,
            "emissions": emissions,
        }
        if labels is not None:
            if self.use_crf:
                if valid_mask is None:
                    raise ValueError("valid_mask is required when use_crf=True")
                compressed_emissions, compressed_labels, crf_mask = self.compress_for_crf(
                    emissions=emissions,
                    labels=labels,
                    valid_mask=valid_mask,
                )
                outputs["crf_mask"] = crf_mask
                outputs["loss"] = self.crf.neg_log_likelihood(
                    emissions=compressed_emissions,
                    tags=compressed_labels,
                    mask=crf_mask,
                    reduction="mean",
                )
            else:
                # labels 中只有有效位置不是 ignore_index，因此直接用布尔索引筛掉无效 token。
                # 这里的有效位仍然由数据层对齐逻辑决定，而不是简单按 attention_mask 取非 padding。
                valid_positions = labels.ne(self.ignore_index)
                if valid_positions.any():
                    loss = F.cross_entropy(
                        emissions[valid_positions],
                        labels[valid_positions],
                    )
                else:
                    # 如果该 batch 全是空行或无有效标签，返回 0 loss，避免 NaN。
                    loss = emissions.sum() * 0.0
                outputs["loss"] = loss
        return outputs

    def decode(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> List[List[int]]:
        '''统一 baseline / final 的解码接口。'''

        valid_mask = valid_mask.bool()
        if self.use_crf:
            compressed_emissions, _, crf_mask = self.compress_for_crf(
                emissions=logits,
                labels=None,
                valid_mask=valid_mask,
            )
            return self.crf.decode(compressed_emissions, crf_mask)

        prediction_ids = logits.argmax(dim=-1)
        decoded_sequences: List[List[int]] = []
        for batch_index in range(prediction_ids.size(0)):
            # 非 CRF 路线也只保留真实字位置，保证和 CRF 路线返回同一口径的标签序列。
            decoded_sequences.append(
                prediction_ids[batch_index][valid_mask[batch_index]].tolist()
            )
        return decoded_sequences

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        '''统一对外的解码接口。'''

        return logits.argmax(dim=-1)

'''自写 Refinement Block。

负责：
1. 手写缩放点积注意力
2. 手写多头自注意力
3. 手写前馈网络
4. 手写 Pre-LN 的编码层
5. 叠加形成轻量 RefinementEncoder

注意：
- refinement 只做表征增强，不改变序列长度
- attention mask 必须从现有 `attention_mask` 派生
- 不允许重新定义真实字位置语义，`valid_mask` 仍由数据层负责
'''

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    '''缩放点积注意力。'''

    def __init__(self, dropout: float = 0.1) -> None:
        '''初始化注意力内部使用的 dropout。'''

        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''执行缩放点积注意力。

        形状约定：
        - query/key/value: [batch_size, num_heads, seq_len, head_dim]
        - attention_mask:
          - 常规输入为 [batch_size, seq_len]
          - 会在内部广播成 [batch_size, 1, 1, seq_len]
        '''

        head_dim = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # 这里的 attention_mask 只沿 key 维广播，语义继续是“哪些 token 可见”。
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            attention_mask = attention_mask.bool()
            attention_scores = attention_scores.masked_fill(~attention_mask, -10000.0)

        # 先减去每行最大值，再做 softmax，可以显著降低长序列训练时的指数溢出风险。
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask, -10000.0)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        if attention_mask is not None:
            # 对被 mask 的位置显式清零，再重新归一化，避免全 mask 行退化成均匀分布。
            attention_probs = attention_probs.masked_fill(~attention_mask, 0.0)
            normalizer = attention_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attention_probs = attention_probs / normalizer
        attention_probs = self.dropout(attention_probs)
        return torch.matmul(attention_probs, value)


class MultiHeadSelfAttention(nn.Module):
    '''多头自注意力。'''

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        '''构造多头自注意力的线性投影与输出层。'''

        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        '''把最后一个 hidden 维拆成多头。'''

        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        '''把多头重新拼回 hidden 维。'''

        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''执行多头自注意力。'''

        query = self._reshape_to_heads(self.query(hidden_states))
        key = self._reshape_to_heads(self.key(hidden_states))
        value = self._reshape_to_heads(self.value(hidden_states))

        context = self.attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )
        context = self._merge_heads(context)
        return self.dropout(self.output(context))


class PositionwiseFFN(nn.Module):
    '''逐位置前馈网络。'''

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        '''构造逐位置前馈网络。'''

        super().__init__()
        self.dense_in = nn.Linear(hidden_size, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dense_out = nn.Linear(ffn_dim, hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        '''执行 FFN。'''

        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        return self.output_dropout(hidden_states)


class RefinementEncoderLayer(nn.Module):
    '''单层 Pre-LN Refinement 编码层。'''

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        '''构造单层 Pre-LN refinement 编码层。'''

        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = PositionwiseFFN(
            hidden_size=hidden_size,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''执行一层 refinement。'''

        attention_output = self.self_attention(
            hidden_states=self.attention_norm(hidden_states),
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attention_output

        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + ffn_output

        if attention_mask is not None:
            # padding 位置不参与有效输出，显式清零可以减少无意义的噪声传播。
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).type_as(hidden_states)
        return hidden_states


class RefinementEncoder(nn.Module):
    '''堆叠式 RefinementEncoder。'''

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        '''构造多层堆叠的 refinement 编码器。'''

        super().__init__()
        self.layers = nn.ModuleList(
            [
                RefinementEncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # refinement 堆叠结束后再做一次归一化，有助于把输出尺度稳定在 classifier/CRF 易处理的范围内。
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''依次执行多层 refinement。'''

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.output_norm(hidden_states)
        if attention_mask is not None:
            # 归一化之后再乘一次 mask，避免 padding 位残留极小数值继续往后传。
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).type_as(hidden_states)
        return hidden_states

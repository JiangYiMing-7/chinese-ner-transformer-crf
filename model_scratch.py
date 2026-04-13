'''Stage5 手写 Transformer + CRF 模型。

该文件对应课程要求中的“从零手写 Transformer 编码器”路线，核心原则是：
1. 不使用 HuggingFace 与任何预训练 backbone
2. 不使用 PyTorch 自带的高层 attention / Transformer 封装
3. 只用基础 PyTorch 算子与 `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` 等基础模块
4. CRF 直接复用项目已有的 `crf.py`
'''

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from crf import LinearChainCRF


def _ordered_labels(
    num_labels: int,
    label2id: Optional[Dict[str, int]],
) -> List[str]:
    '''按标签 id 排序，构造稳定的标签列表。'''

    if label2id is None:
        return [str(index) for index in range(num_labels)]
    return [
        label
        for label, _ in sorted(label2id.items(), key=lambda item: item[1])
    ]


class ScaledDotProductAttention(nn.Module):
    '''缩放点积注意力。

    对应 Vaswani et al. 2017《Attention Is All You Need》中的公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    这里显式手写 `QK^T`、缩放、mask、softmax 和与 V 的乘法，
    不依赖任何高层 attention 封装。
    '''

    def __init__(self, dropout: float = 0.1) -> None:
        '''初始化注意力层。

        参数：
        - dropout: 注意力权重上的 dropout 概率；从零训练时通常保留 0.1~0.3 的正则
        '''

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

        输入形状：
        - query: (B, H, T_q, d_k)
        - key:   (B, H, T_k, d_k)
        - value: (B, H, T_k, d_v)
        - attention_mask: (B, 1, 1, T_k)，True=有效位，False=被 mask

        输出形状：
        - context: (B, H, T_q, d_v)
        '''

        d_k = query.size(-1)

        # 计算 QK^T，得到每个 query 对所有 key 的相似度分数。# (B, H, T_q, T_k)
        scores = torch.matmul(query, key.transpose(-1, -2))
        # 按公式除以 sqrt(d_k)，避免 d_k 较大时点积数值过大。# (B, H, T_q, T_k)
        scores = scores / math.sqrt(d_k)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            # 被 mask 的位置填 -1e9，而不是 -inf，避免 MPS 上 softmax 出现 NaN。# (B, H, T_q, T_k)
            scores = scores.masked_fill(~attention_mask, -1e9)

        # 对最后一维做 softmax，得到注意力权重。# (B, H, T_q, T_k)
        attention_probs = torch.softmax(scores, dim=-1)

        if attention_mask is not None:
            # 被 mask 的位置显式清零，避免极端情况下出现极小非零残留。# (B, H, T_q, T_k)
            attention_probs = attention_probs.masked_fill(~attention_mask, 0.0)
            # 对有效位重新归一化，保证每个 query 的权重和为 1。# (B, H, T_q, T_k)
            normalizer = attention_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attention_probs = attention_probs / normalizer

        # 对注意力权重做 dropout，抑制个别头过拟合到固定位置。# (B, H, T_q, T_k)
        attention_probs = self.dropout(attention_probs)
        # 按注意力权重对 V 做加权求和，得到上下文表示。# (B, H, T_q, d_v)
        context = torch.matmul(attention_probs, value)
        return context


class HandWrittenMHA(nn.Module):
    '''手写多头自注意力。

    对应 Vaswani et al. 2017 的多头注意力公式：
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    这里允许使用 `nn.Linear` 保存 W_Q / W_K / W_V / W_O 参数，
    但注意力本身的计算过程必须显式手写。
    '''

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        '''初始化多头自注意力。

        参数：
        - d_model: 模型隐层维度
        - num_heads: 注意力头数，要求能整除 d_model
        - dropout: 注意力权重与输出投影上的 dropout 概率
        '''

        super().__init__()
        # 显式断言 head 划分合法；同时下方保留 ValueError，避免 Python -O 时断言失效。
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # W_Q / W_K / W_V / W_O 对应论文中的四个投影矩阵。
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''执行手写多头自注意力。

        输入：
        - X: (B, T, d_model)
        - attention_mask: (B, T) 或 (B, 1, 1, T)

        输出：
        - out: (B, T, d_model)
        '''

        B, T, _ = X.shape

        # 线性投影到 Q / K / V 空间。# (B, T, d_model)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 将 d_model 拆成 num_heads 个 d_head，并把 head 维提前。# (B, H, T, d_head)
        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if attention_mask is not None and attention_mask.dim() == 2:
            # 从 (B, T) 广播为 (B, 1, 1, T)，只沿 key 维做可见性控制。# (B, 1, 1, T)
            attention_mask = attention_mask[:, None, None, :]

        # 调用手写缩放点积注意力。# (B, H, T, d_head)
        context = self.attention(Q, K, V, attention_mask)

        # 将多头结果转回 (B, T, d_model)。# (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        # 输出投影 W^O，把多头拼接结果映射回模型空间。# (B, T, d_model)
        out = self.W_O(context)
        return self.output_dropout(out)


class SinusoidalPositionalEncoding(nn.Module):
    '''正弦位置编码。

    对应 Vaswani et al. 2017 中的位置编码公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    位置编码预先计算为 buffer，不参与梯度更新。
    '''

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        '''初始化位置编码。

        参数：
        - d_model: 模型隐层维度
        - max_seq_len: 支持的最大序列长度
        - dropout: 位置编码叠加后的 dropout 概率
        '''

        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # position 表示位置下标 pos。# (max_seq_len, 1)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        # 对应公式中的 10000^(2i / d_model) 的倒数指数形式。# (ceil(d_model / 2),)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float)
        # 偶数列对应公式 PE(pos, 2i) = sin(...). # (max_seq_len, ceil(d_model / 2))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model > 1:
            # 奇数列对应公式 PE(pos, 2i+1) = cos(...). # (max_seq_len, floor(d_model / 2))
            odd_width = pe[:, 1::2].shape[1]
            pe[:, 1::2] = torch.cos(position * div_term[:odd_width])

        # 注册为 buffer，使其随模型迁移设备，但不参与反向传播。# (1, max_seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''将位置编码加到输入 embedding 上。

        输入：
        - x: (B, T, d_model)

        输出：
        - out: (B, T, d_model)
        '''

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        # 把对应长度的 PE 加到输入上，位置编码与词向量逐元素相加。# (B, T, d_model)
        x = x + self.pe[:, :seq_len, :].to(device=x.device, dtype=x.dtype)
        return self.dropout(x)


class PositionwiseFFN(nn.Module):
    '''逐位置前馈网络。

    论文原式为：
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    本实现保留两层线性映射 W_1 / W_2，但把原论文的 ReLU 改为 GELU。
    这样做的原因是 GELU 在现代 NLP 任务中通常更平滑、更稳定。
    '''

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        '''初始化 FFN。

        参数：
        - d_model: 输入与输出维度
        - ffn_dim: 中间层维度，对应公式中的 W_1 输出维度
        - dropout: 中间激活与输出上的 dropout 概率
        '''

        super().__init__()
        # W_1: d_model -> ffn_dim
        self.linear1 = nn.Linear(d_model, ffn_dim)
        # 虽然原论文写的是 ReLU，这里改用 GELU。
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # W_2: ffn_dim -> d_model
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''执行逐位置前馈网络。

        输入输出形状都为：
        - x / out: (B, T, d_model)
        '''

        # 第一个线性层，对应公式中的 xW_1 + b_1。# (B, T, ffn_dim)
        x = self.linear1(x)
        # 非线性激活；此处使用 GELU，而不是原论文的 ReLU。# (B, T, ffn_dim)
        x = self.activation(x)
        # 对中间表示做 dropout。# (B, T, ffn_dim)
        x = self.dropout(x)
        # 第二个线性层，对应公式中的乘 W_2 + b_2。# (B, T, d_model)
        x = self.linear2(x)
        return self.output_dropout(x)


class ScratchTransformerEncoderLayer(nn.Module):
    '''单层手写 Transformer Encoder。

    原始 Transformer 使用的是 Post-LN：
    x -> MHA -> Add -> LN -> FFN -> Add -> LN

    本实现采用更稳定的 Pre-LN：
    h = x + MHA(LayerNorm(x))
    out = h + FFN(LayerNorm(h))

    Pre-LN 在深层网络和从零训练时通常更稳定，这也是 Xiong et al. 2020
    推荐的做法之一。
    '''

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        '''初始化单层编码器。

        参数：
        - d_model: 模型隐层维度
        - num_heads: 注意力头数
        - ffn_dim: FFN 中间层维度
        - dropout: 子层输出与残差连接前的 dropout 概率
        '''

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = HandWrittenMHA(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''执行一层 Pre-LN Transformer Encoder。

        输入输出形状：
        - hidden_states: (B, T, d_model)
        - attention_mask: (B, T) 或 (B, 1, 1, T)
        '''

        token_mask: Optional[torch.Tensor] = None
        expanded_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                token_mask = attention_mask.bool()
                # 把 padding mask 广播到注意力分数的 key 维。# (B, 1, 1, T)
                expanded_mask = token_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                expanded_mask = attention_mask.bool()
                token_mask = expanded_mask.squeeze(1).squeeze(1)
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}"
                )

        # Pre-LN: 先归一化，再送入多头自注意力。# (B, T, d_model)
        attn_input = self.norm1(hidden_states)
        # 手写多头注意力子层输出。# (B, T, d_model)
        attn_output = self.self_attention(attn_input, expanded_mask)
        # 残差连接：x + MHA(LN(x)). # (B, T, d_model)
        hidden_states = hidden_states + self.residual_dropout(attn_output)

        # 再对残差结果做 LayerNorm，送入 FFN。# (B, T, d_model)
        ffn_input = self.norm2(hidden_states)
        # 前馈子层输出。# (B, T, d_model)
        ffn_output = self.ffn(ffn_input)
        # 第二次残差连接：h + FFN(LN(h)). # (B, T, d_model)
        hidden_states = hidden_states + self.residual_dropout(ffn_output)

        if token_mask is not None:
            # padding 位不代表真实字符，显式清零可以减少噪声继续向后传播。# (B, T, d_model)
            hidden_states = hidden_states * token_mask.unsqueeze(-1).type_as(hidden_states)
        return hidden_states


class ScratchTransformerEncoder(nn.Module):
    '''堆叠式手写 Transformer Encoder。'''

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        '''初始化多层编码器。

        参数：
        - d_model: 模型隐层维度
        - num_layers: 编码层堆叠层数
        - num_heads: 每层注意力头数
        - ffn_dim: FFN 中间层维度
        - dropout: 每层内部的 dropout 概率
        '''

        super().__init__()
        self.layers = nn.ModuleList(
            [
                ScratchTransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Pre-LN 结构通常会在编码器末尾再接一层 LayerNorm。
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''依次通过多层编码器。

        输入输出形状：
        - hidden_states: (B, T, d_model)
        - attention_mask: (B, T) 或 (B, 1, 1, T)
        '''

        token_mask: Optional[torch.Tensor] = None
        expanded_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                token_mask = attention_mask.bool()
                expanded_mask = token_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                expanded_mask = attention_mask.bool()
                token_mask = expanded_mask.squeeze(1).squeeze(1)
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}"
                )

        for layer in self.layers:
            hidden_states = layer(hidden_states, expanded_mask)

        # 编码器输出再做一次 LayerNorm。# (B, T, d_model)
        hidden_states = self.output_norm(hidden_states)
        if token_mask is not None:
            # 末端再乘一次 mask，避免 padding 位残留极小数值。# (B, T, d_model)
            hidden_states = hidden_states * token_mask.unsqueeze(-1).type_as(hidden_states)
        return hidden_states


class ScratchNER(nn.Module):
    '''Stage5 完整 NER 模型。

    结构链路：
    CharEmbedding -> SinusoidalPE -> ScratchTransformerEncoder
    -> Dropout -> Linear -> LinearChainCRF

    其中：
    - embedding 输出形状为 (B, T, d_model)
    - encoder 输出形状保持不变，仍为 (B, T, d_model)
    - classifier 把每个位置映射到 num_labels 维发射分数
    - CRF 在真实字符位上建模标签转移
    '''

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
        ignore_index: int = -100,
        bio_constraint_mode: str = "none",
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        '''初始化 Stage5 NER 模型。

        参数：
        - vocab_size: 字符词表大小
        - num_labels: 标签数
        - d_model: 字向量与编码器隐层维度
        - num_layers: Transformer Encoder 堆叠层数
        - num_heads: 多头注意力头数
        - ffn_dim: FFN 中间层维度
        - max_seq_len: 最大窗口长度；位置编码和长句切窗都依赖它
        - dropout: embedding / encoder / classifier 之间共享的基础 dropout
        - pad_idx: `<PAD>` 的词表 id，默认 0
        - ignore_index: padding 位置在 labels 中的忽略值
        - bio_constraint_mode: 传给已有 CRF 的 BIO 约束模式
        - label2id: 标签到 id 的映射，用于恢复稳定的标签顺序
        '''

        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.ignore_index = ignore_index
        self.use_crf = True
        self.bio_constraint_mode = bio_constraint_mode
        self.labels = _ordered_labels(num_labels, label2id)
        self.resolved_model_name = "scratch_transformer_encoder"

        # 字符嵌入层，padding_idx=0 时该位置梯度不会参与更新。# (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # 正弦位置编码。# (1, max_seq_len, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        # 手写 Transformer Encoder。# (B, T, d_model) -> (B, T, d_model)
        self.encoder = ScratchTransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        # 线性分类头输出每个标签的发射分数。# (B, T, num_labels)
        self.classifier = nn.Linear(d_model, num_labels)
        self.crf = LinearChainCRF(
            num_tags=num_labels,
            labels=self.labels,
            bio_constraint_mode=bio_constraint_mode,
        )
        self.effective_bio_constraint_mode = self.crf.effective_bio_constraint_mode

    def load_pretrained_embeddings(self, embedding_matrix: torch.Tensor) -> None:
        '''用外部提供的静态向量初始化 embedding。

        参数：
        - embedding_matrix: 形状必须为 (vocab_size, d_model)

        若调用方不提供预训练向量，则完全可以跳过该步骤。
        '''

        if tuple(embedding_matrix.shape) != tuple(self.embedding.weight.shape):
            raise ValueError(
                "Embedding matrix shape mismatch: "
                f"expected {tuple(self.embedding.weight.shape)}, "
                f"got {tuple(embedding_matrix.shape)}"
            )

        with torch.no_grad():
            self.embedding.weight.copy_(
                embedding_matrix.to(
                    device=self.embedding.weight.device,
                    dtype=self.embedding.weight.dtype,
                )
            )
            # `<PAD>` 位应始终为 0，避免把 padding 也当成真实字符向量学习。# (d_model,)
            self.embedding.weight[self.pad_idx].zero_()

    def compress_for_crf(
        self,
        emissions: torch.Tensor,
        labels: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        '''把原始 token 级输出压缩到真实字符位。

        Stage5 是纯字符级模型，因此 `valid_mask` 与 `attention_mask` 相同；
        这里仍保留压缩逻辑，是为了与现有 `evaluate.py` / `decode(...)` 口径保持一致。
        '''

        valid_mask = valid_mask.bool()
        batch_size, _, num_labels = emissions.shape
        valid_lengths = valid_mask.long().sum(dim=1)
        max_valid_length = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
        max_valid_length = max(1, max_valid_length)

        compressed_emissions = emissions.new_zeros((batch_size, max_valid_length, num_labels))
        compressed_labels: Optional[torch.Tensor] = None
        if labels is not None:
            compressed_labels = labels.new_zeros((batch_size, max_valid_length))
        crf_mask = torch.zeros(
            (batch_size, max_valid_length),
            dtype=torch.bool,
            device=emissions.device,
        )

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
                        "This usually indicates a collation bug."
                    )
                compressed_labels[batch_index, :valid_count] = selected_labels

        return compressed_emissions, compressed_labels, crf_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        '''前向传播。

        形状链路：
        - input_ids: (B, T)
        - embedding 输出: (B, T, d_model)
        - 加位置编码后: (B, T, d_model)
        - encoder 输出: (B, T, d_model)
        - classifier 输出 emissions/logits: (B, T, num_labels)
        - CRF 压缩后: (B, T_valid, num_labels)
        '''

        del token_type_ids

        if attention_mask is None:
            # 对字符级模型，padding mask 可直接由 input_ids != pad_idx 得到。# (B, T)
            attention_mask = input_ids.ne(self.pad_idx)
        attention_mask = attention_mask.bool()

        if valid_mask is None:
            # Stage5 没有 sub-token，对真实字符位的定义与 attention_mask 一致。# (B, T)
            valid_mask = attention_mask
        valid_mask = valid_mask.bool()

        # 字符嵌入。# (B, T, d_model)
        hidden_states = self.embedding(input_ids)
        # 叠加正弦位置编码并做 dropout。# (B, T, d_model)
        hidden_states = self.position_encoding(hidden_states)
        # 通过手写 Transformer Encoder。# (B, T, d_model)
        hidden_states = self.encoder(hidden_states, attention_mask)
        # 进入分类头前再做一次 dropout。# (B, T, d_model)
        hidden_states = self.dropout(hidden_states)
        # 发射分数，用于后续 CRF 计算。# (B, T, num_labels)
        emissions = self.classifier(hidden_states)

        outputs: Dict[str, Any] = {
            "logits": emissions,
            "emissions": emissions,
        }

        if labels is not None:
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

        return outputs

    def decode(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> List[List[int]]:
        '''统一对外的解码接口。

        返回值与现有 `RobertaLinearNER.decode(...)` 保持一致：
        每个样本返回一个“只包含真实字符位”的标签 id 序列。
        '''

        compressed_emissions, _, crf_mask = self.compress_for_crf(
            emissions=logits,
            labels=None,
            valid_mask=valid_mask,
        )
        return self.crf.decode(compressed_emissions, crf_mask)

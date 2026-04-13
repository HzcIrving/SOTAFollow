# World Model / VLA 自回归框架详解

> 📅 创建时间：2026-04-12  
> 🏷️ 分类：FM基础知识 / WorldModel / VLA  
> 📌 标签：自回归、掩码设计、Action Token、VLA、World Model

---

## 一、背景与问题动机

### 1.1 什么是 World Model 与 VLA？

**World Model（世界模型）** 的核心目标：让模型学习环境的动态演化规律 $p(s_{t+1} | s_t, a_t)$，从而能够在隐空间中进行"想象式规划"——不需要真实与环境交互，即可在内部推演未来。

**VLA（Vision-Language-Action Model）** 则在此基础上更进一步：接收视觉感知+语言指令，输出可执行的机器人动作（action tokens），从而实现端到端的具身智能控制。

近年来，**自回归（Autoregressive, AR）** 范式在 World Model 和 VLA 领域逐渐成为主流。不同于传统的扩散（Diffusion）或多步预测方法，AR 模型通过逐步生成的方式预测未来，与大型语言模型（LLM）的设计哲学一脉相承。

### 1.2 为什么选择自回归范式？

| 特性 | 自回归 (AR) | 扩散模型 (Diffusion) | 多步预测 |
|------|------------|---------------------|---------|
| 推理速度 | 中（逐步生成） | 慢（多步去噪） | 快（直接预测T步） |
| 生成质量 | 高 | 极高 | 一般 |
| 可控性 | 强（逐token控制） | 中 | 弱 |
| 与 LLM 兼容性 | 极强 | 弱 | 弱 |
| 典型应用 | GPT, VLA | Stable Diffusion | DDPG, MBPO |

---

## 二、自回归训练与推理原理

### 2.1 Next-Token Prediction（下一个 Token 预测）

自回归模型的核心目标函数是**下一个 token 预测**。给定一个序列 $x_1, x_2, ..., x_T$，模型学习：

$$
p(x_t | x_1, ..., x_{t-1}; \theta) = \text{Softmax}(W \cdot h_{t-1})
$$

其中 $h_{t-1} = \text{Transformer}(x_1, ..., x_{t-1}; \theta)$ 是模型对前缀序列的上下文表示。

**训练时：** 最大化似然 $\log p(x_t | x_{<t})$，等价于最小化交叉熵损失：

$$
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})
$$

### 2.2 Teacher Forcing

**Teacher Forcing** 是自回归模型训练的核心策略：

- **训练阶段**：输入是**真实的历史序列** $x_1, x_2, ..., x_{t-1}$（即 Ground Truth），即使模型在前一步预测错误，下一步的输入仍然使用真实值。
- **推理阶段**：没有真实序列可用，模型必须使用**自己生成的上一个 token** 作为下一个输入。

这种训练-推理的不一致（Exposure Bias）是自回归模型的一个重要问题，业内通常通过以下方式缓解：
- **Scheduled Sampling**（渐进式用模型自己的预测替代真实输入）
- **Distillation**（用真实轨迹微调）
- **RLHF / DPO**（对齐优化）

### 2.3 训练与推理的差异

| 维度 | 训练阶段 | 推理阶段 |
|------|---------|---------|
| 输入 | 真实历史序列 $x_{<t}$ | 模型自己生成的序列 $\hat{x}_{<t}$ |
| 并行性 | 高度并行（All-to-All Attention） | 严格顺序生成（逐 token 自回归） |
| 掩码 | 使用 **Causal Mask** | 使用 **Causal Mask** |
| 速度 | 快（一次前向传播） | 慢（需 T 次前向传播） |
| 损失计算 | 逐 token 计算，梯度反向传播 | 仅用于评估，不参与训练 |

---

## 三、Transformer 中的 Causal Mask

### 3.1 什么是 Causal Mask？

**Causal Mask（因果掩码）** 是自回归 Transformer 的核心组件，它确保**位置 $t$ 的 token 只能看到位置 $\leq t$ 的信息**，从而保证预测的因果性（不能"看到未来"）。

在数学上，等价于将注意力矩阵中 $i > j$ 的位置置为 $-\infty$（或一个很大的负数），使 $\text{Softmax}$ 归一化后这些位置的权重趋近于 0。

### 3.2 标准 Causal Mask（GPT-style）

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

其中：

$$
M_{ij} = 
\begin{cases}
0 & \text{if } j \leq i \ (\text{允许注意力}) \\
-\infty & \text{if } j > i \ (\text{掩码掉})
\end{cases}
$$

### 3.3 推理时的 KV Cache 优化

推理阶段使用 **KV Cache** 避免重复计算：
- 每次只需计算新 token 与所有历史 key/value 的注意力
- 历史部分的 key/value 缓存起来，只做 lookup

---

## 四、VLA 框架中的掩码设计哲学

### 4.1 Action Token Masking（动作 token 掩码）

在 VLA 框架中，动作被量化为离散 token（如 RT-2 中的机械臂关节角度离散桶）。一个典型的 VLA 输入输出结构如下：

```
[图像 tokens] + [语言指令 tokens] + [动作 tokens] → [下一个动作 token]
```

**关键问题：** 动作 token 是否需要参与注意力计算？

| 方案 | 描述 | 代表工作 |
|------|------|---------|
| **Action as Output** | 动作 token 只作为输出，不参与输入注意力 | RT-1, RT-2 |
| **Action as Input** | 动作 token 同时作为输入序列的一部分 | OpenVLA |
| **Action Chunking** | 一次性预测多个动作 token，块内双向注意 | ACT |

**OpenVLA 的做法：**
- 将离散化的动作 token 拼接在视觉和语言 tokens 之后
- 整个序列使用统一的 Causal Mask
- 动作 tokens 之间的注意力是允许的（因为它们是并行生成的）

### 4.2 视频帧级掩码设计

处理多帧视频输入时，不同帧之间的注意力模式是核心设计决策：

| 掩码类型 | 描述 | 优点 | 缺点 |
|---------|------|------|------|
| **Full Attention** | 所有帧之间完全双向注意 | 信息流通最充分 | $O(N^2)$ 复杂度，长视频不可行 |
| **Causal (Temporal)** | 帧 $t$ 只能看到帧 $\leq t$ | 支持在线推理 | 缺少未来帧信息 |
| **Sparse / Local** | 只允许局部时空邻域注意 | 效率高 | 可能遗漏长程依赖 |
| **Hierarchical** | 帧内局部注意 + 帧间粗粒度注意 | 平衡效率与效果 | 实现复杂 |

### 4.3 Uni-World VLA 的交错掩码设计

**Uni-World VLA（ECCV 2026）** 提出了**交错生成（Interleaved Generation）** 框架，是近年来掩码设计的一个重要创新：

**核心思想：**
- 未来帧预测与动作预测**交错进行**，而非串行或并行
- 每个 World Model Block 内：先看历史帧 → 生成动作 → 基于动作预测下一帧 → 再生成下一个动作

**掩码模式：**
```
Block k:
  [帧 1..t] → [动作 a_t] → [帧 t+1']（预测）
              ↓
         [帧 1..t+1] → [动作 a_{t+1}]
```

这种交错设计显著缓解了 **Frozen Hallucination**（冻结幻觉）问题——即一次性生成所有未来帧时，动作对帧的影响被"冻结"、无法形成闭环交互的问题。

### 4.4 主流 VLA 框架掩码对比

| 框架 | 视觉掩码 | 语言掩码 | 动作掩码 | 帧间注意力 |
|------|---------|---------|---------|-----------|
| **RT-2** | 视觉编码器（ViT）独立，输出作为语言模型输入 | Full Attention | Output-only（无输入注意力） | N/A |
| **OpenVLA** | 视觉编码器独立，输出 token 序列 | Full Attention | Input+Output，块内双向 | Causal Temporal |
| **Uni-World VLA** | 可学习掩码（learnable mask），交错生成 | Causal | Causal + 块内双向 | 交错双向 |
| **RT-1** | 视觉 token + 动作 token 拼接 | Full Attention | Output-only | Causal Temporal |

---

## 五、伪代码实现

### 5.1 标准自回归语言模型（含 Causal Mask）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalAttention(nn.Module):
    """标准的 Causal Self-Attention 实现"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # W_Q, W_K, W_V, W_O 四个投影矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
        Returns:
            输出序列 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # 1. 线性投影得到 Q, K, V
        Q = self.W_q(x)  # [B, L, D]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 分头（Multi-Head）
        # [B, L, D] -> [B, L, n_heads, d_k] -> [B, n_heads, L, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数 & 应用 Causal Mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, n_heads, L, L]

        # 构造因果掩码：位置 i 只允许看到 j <= i
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1  # True = 需要mask（上方三角，j > i）
        )
        # 将mask位置填充为 -inf
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 4. Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. 加权求和
        context = torch.matmul(attn_weights, V)  # [B, n_heads, L, d_k]

        # 6. 合并多头 & 输出投影
        context = context.transpose(1, 2).contiguous()  # [B, L, n_heads, d_k]
        context = context.view(batch_size, seq_len, d_model)  # [B, L, D]
        output = self.W_o(context)

        return output


class TransformerBlock(nn.Module):
    """单个 Transformer Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 架构（更稳定的训练）
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AutoregressiveLM(nn.Module):
    """标准自回归语言模型"""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Token Embedding + 位置编码
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # N 层 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 预测头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定（Embedding 和 Output Layer 共享）
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """
        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            labels: 目标token序列（训练时传入，推理时不传）
        Returns:
            loss 和 logits
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}"

        # Token Embedding + 位置编码
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        # 通过所有 Transformer Block
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # LM Head 预测下一个 token
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]

        result = {"logits": logits}

        if labels is not None:
            # 训练时计算 Next-Token Prediction 损失
            # 预测的是 logits[:, :-1]，目标是 labels[:, 1:]
            # 因为第 t 步预测的是第 t+1 个 token
            shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
            shift_labels = labels[:, 1:].contiguous()        # [B, L-1]
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean"
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        自回归生成（贪婪解码示例）
        Args:
            input_ids: 前缀序列 [batch_size, prefix_len]
            max_new_tokens: 最大生成长度
        """
        for _ in range(max_new_tokens):
            # 如果序列超过最大长度，做截断
            input_ids_clipped = input_ids[:, -self.max_seq_len:]
            outputs = self.forward(input_ids_clipped)
            logits = outputs["logits"]  # [B, L, V]
            # 取最后一个位置的 logits 预测下一个 token
            next_token_logits = logits[:, -1, :]  # [B, V]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
```

### 5.2 VLA Action Token Masking（含 Action Chunk 掩码）

```python
class VLAAttention(nn.Module):
    """
    VLA 注意力模块，支持：
    1. Vision Tokens 之间的局部注意力（空间压缩）
    2. Vision -> Text 的单向注意（防止视觉泄露语言信息）
    3. Vision -> Action 的单向注意（符合控制因果性）
    4. Text -> Vision 的单向注意（语言可以看图）
    5. Action Tokens 之间的块内双向注意（动作同步协调）
    """
    def __init__(self, d_model: int, n_heads: int, action_chunk_size: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.action_chunk_size = action_chunk_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def build_vla_mask(
        self,
        vision_len: int,
        text_len: int,
        action_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        构造 VLA 混合注意力掩码矩阵

        序列布局: [VISION_t ... VISION_1 | TEXT_t ... TEXT_1 | ACTION_t ... ACTION_1]
                            (0-indexed)          (vision_len)       (text_len)     (action_len)

        注意力规则：
        - Vision: 只看 Vision 和 Text（不允许看 Action，防止视觉被动作干扰）
        - Text: 看所有（Vision + Text + Action）
        - Action: 看当前块内的所有 Action + 之前所有已生成 Action
                   但不能看未来 Action（因果性）
        """
        total_len = vision_len + text_len + action_len
        mask = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)

        v_start, v_end = 0, vision_len
        t_start, t_end = vision_len, vision_len + text_len
        a_start, a_end = vision_len + text_len, total_len

        # Rule 1: Vision 不能看 Action（防止视觉被动作干扰）
        mask[v_start:v_end, a_start:a_end] = True  # True = mask掉

        # Rule 2: Vision 不能看未来的 Vision（可选：允许局部空间注意）
        # 这里使用严格的 Causal Mask（全下三角）
        for i in range(vision_len):
            mask[v_start + i, v_start + i + 1:] = True

        # Rule 3: Text 可以看所有（Full Attention on Text）
        # mask[t_start:t_end, :] 默认为 False（不mask）

        # Rule 4: Action 必须因果：只看块内（当前chunk）和已生成的
        # Action 之间的注意力通过 action_chunk_mask 控制
        for chunk_start in range(a_start, a_end, self.action_chunk_size):
            chunk_end = min(chunk_start + self.action_chunk_size, a_end)
            # 块内双向：chunk 内任意位置可以互相看
            for i in range(chunk_start, chunk_end):
                for j in range(chunk_start, chunk_end):
                    if i != j:
                        mask[i, j] = False  # 不mask，块内双向

            # 块内看块前：每个chunk可以看到之前所有chunk（因果链）
            if chunk_start > a_start:
                for i in range(chunk_start, chunk_end):
                    mask[i, a_start:chunk_start] = False  # 不mask

            # 块内不能看块后（未来）：块内 j>i 的位置mask掉
            for i in range(chunk_start, chunk_end):
                for j in range(i + 1, chunk_end):
                    mask[i, j] = True  # mask掉

        return mask.float().masked_fill(mask, float('-inf'))

    def forward(
        self,
        vision_tokens: torch.Tensor,   # [B, vision_len, D]
        text_tokens: torch.Tensor,      # [B, text_len, D]
        action_tokens: torch.Tensor,   # [B, action_len, D]
        return_mask: bool = False
    ) -> torch.Tensor:
        """前向传播"""
        batch_size = vision_tokens.shape[0]
        vision_len = vision_tokens.shape[1]
        text_len = text_tokens.shape[1]
        action_len = action_tokens.shape[1]

        # 拼接所有 token
        x = torch.cat([vision_tokens, text_tokens, action_tokens], dim=1)  # [B, total, D]

        # Q, K, V 投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分头
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 构造并应用 VLA 掩码
        vla_mask = self.build_vla_mask(vision_len, text_len, action_len, x.device)
        scores = scores + vla_mask.unsqueeze(0)  # broadcast: [B, n_heads, L, L]

        # Softmax & 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        if return_mask:
            return output, vla_mask
        return output
```

### 5.3 Uni-World VLA 交错生成掩码

```python
class InterleavedWorldModeling(nn.Module):
    """
    Uni-World VLA 的交错世界建模模块

    核心思想：帧预测和动作预测交替进行，形成闭环交互
    每次迭代（World Model Block）：
        1. 给定历史帧 + 历史动作 → 预测下一帧（并计算想象力损失）
        2. 给定历史帧 + 预测帧 → 预测下一动作（并计算动作损失）
    """
    def __init__(self, vision_encoder, lang_encoder, world_model, action_head,
                 action_dim: int = 7, chunk_size: int = 4):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.world_model = world_model  # Transformer-based world model
        self.action_head = action_head
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def build_interleaved_mask(
        self,
        n_frames: int,         # 历史帧数量
        predicted_frame_len: int,  # 预测帧的token长度
        action_len: int,        # 动作token长度
        device: torch.device
    ) -> dict:
        """
        构造交错生成的掩码规则

        Block 内序列布局:
        [Frame_1 | ... | Frame_t | Pred_Frame | Action_a_t | Action_a_{t+1} | ...]

        注意力规则：
        - 历史帧 (Frame_1..t): 因果看自己之前 + 可看语言指令
        - 预测帧 (Pred_Frame): 因果看历史帧 + 语言指令
        - 动作 tokens (a_t...): 因果看历史帧 + 预测帧 + 已生成动作
        """
        total_len = n_frames + predicted_frame_len + action_len

        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        return causal_mask.float().masked_fill(causal_mask, float('-inf'))

    def forward(
        self,
        history_frames: torch.Tensor,    # [B, T, C, H, W] 历史帧
        lang_instruction: torch.Tensor,  # [B, L] 语言指令token ids
        history_actions: torch.Tensor,   # [B, K] 历史动作序列
        return_predictions: bool = False
    ) -> dict:
        """
        交错世界建模前向传播

        Returns:
            frame_loss: 帧预测损失（想象力损失）
            action_loss: 动作预测损失
            predicted_frames: 预测的未来帧（用于可视化）
            predicted_actions: 预测的动作序列
        """
        B = history_frames.shape[0]
        device = history_frames.device

        # 1. 编码历史帧和语言指令
        vision_features = self.vision_encoder(history_frames)  # [B, T, D]
        lang_features = self.lang_encoder(lang_instruction)    # [B, L, D]

        # 2. 编码历史动作
        action_features = self.action_head.action_embedding(history_actions)  # [B, K, D]

        # 3. 世界模型前向（帧预测分支）
        # 输入: [Frame_1 | ... | Frame_t | lang | action_history]
        wm_input_frames = torch.cat([vision_features, lang_features, action_features], dim=1)

        # 构造因果掩码
        n_frames_tok = vision_features.shape[1]
        lang_len = lang_features.shape[1]
        action_len = action_features.shape[1]
        total_len = n_frames_tok + lang_len + action_len

        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device, dtype=torch.bool),
            diagonal=1
        ).float().masked_fill(causal_mask, float('-inf')).unsqueeze(0)

        # World Model 前向（预测下一帧 token）
        predicted_frame_tokens = self.world_model(wm_input_frames, causal_mask)

        # 4. 动作预测分支（使用预测帧）
        # 将预测的帧 token 与历史帧、语言拼接，输入动作预测头
        action_input = torch.cat([
            vision_features,           # 历史帧
            lang_features,             # 语言指令
            predicted_frame_tokens,   # 刚预测的帧
            action_features            # 历史动作
        ], dim=1)

        # 动作预测：预测下一个动作 chunk
        action_pred = self.action_head(action_input)  # [B, action_dim]

        # 5. 计算损失
        # 注意：Uni-World VLA 使用 Dreamer 风格的想象力损失 +
        # 动作预测损失的联合优化
        frame_loss = ...  # 帧重建损失（L2 或 VP）
        action_loss = F.mse_loss(action_pred, next_action_gt)  # 动作预测损失

        total_loss = frame_loss + action_loss

        if return_predictions:
            return {
                "total_loss": total_loss,
                "frame_loss": frame_loss,
                "action_loss": action_loss,
                "predicted_frames": predicted_frame_tokens,
                "predicted_actions": action_pred,
            }
        return {"total_loss": total_loss}
```

---

## 六、掩码设计的关键工程考量

### 6.1 效率 vs 效果的权衡

| 掩码策略 | 计算复杂度 | 效果 | 适用场景 |
|---------|-----------|------|---------|
| Full Attention ($O(N^2)$) | 高 | 最优 | 短序列（<4K tokens） |
| Sparse Attention | 中 | 次优 | 中等序列 |
| Causal Only | 低 | 良好 | AR 生成场景 |
| Hierarchical | 低 | 良好 | 长视频理解 |

### 6.2 掩码实现的数值稳定性

```python
# 常见错误：直接填充 -1e9 而非 -inf
# 错误示例（Softmax后仍有梯度）：
scores = scores.masked_fill(mask, -1e9)  # ❌ -1e9 在exp后仍是极小值但有数值风险

# 正确做法：
scores = scores.masked_fill(mask, float('-inf'))  # ✅ -inf 确保exp(-inf)=0

# 或者使用 built-in 函数：
scores = scores.masked_fill_(mask, torch.finfo(scores.dtype).min)  # ✅
```

### 6.3 KV Cache 与增量推理的掩码处理

```python
@torch.no_grad()
def generate_with_kv_cache(
    model: AutoregressiveLM,
    input_ids: torch.Tensor,
    max_new_tokens: int
) -> torch.Tensor:
    """使用 KV Cache 的高效自回归生成"""
    # input_ids: [B, prefix_len]
    past_key_values = None  # 初始化为空

    for _ in range(max_new_tokens):
        # 仅对最后一个 token 做 Q 计算，K/V 来自缓存
        outputs = model.transformer(
            input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token = logits.argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

---

## 七、总结与思考

### 7.1 核心要点

1. **自回归范式**通过 Next-Token Prediction 与 LLM 高度兼容，是当前 VLA 的主流选择
2. **Teacher Forcing** 训练与自回归推理之间存在 Exposure Bias，需要通过对齐技术缓解
3. **Causal Mask** 是 AR Transformer 的核心，保证每个 token 只看到历史信息
4. **VLA 掩码设计**需要在视觉、语言、动作三类 token 之间精细控制注意力路由
5. **交错生成**（Uni-World VLA）是解决 Frozen Hallucination 的有效方案

### 7.2 未来趋势

- **更细粒度的动态掩码**：根据输入内容自适应调整注意力结构
- **长期记忆与检索增强**：将外部知识库与掩码机制结合
- **扩散 + AR 混合**：在保持 AR 因果性的同时，引入扩散式并行生成

---

## 八、参考文献

| # | 论文 | 作者 | 来源 | 关键贡献 |
|---|------|------|------|---------|
| 1 | *Attention Is All You Need* | Vaswani et al. | NeurIPS 2017 | Transformer 架构、Causal Mask |
| 2 | *Language Models are Few-Shot Learners* (GPT-3) | Brown et al. | NeurIPS 2020 | Next-Token Prediction 规模化 |
| 3 | *RT-1: Robotics Transformer for Real-World Control* | Brohan et al. | CoRL 2022 | VLA 动作 tokenization |
| 4 | *RT-2: Vision-Language-Action Models* | Brohan et al. | arXiv 2023 | VLA 端到端控制 |
| 5 | *OpenVLA: Open-Set Vision-Language-Action Model* | Kim et al. | arXiv 2024 | 开源 VLA 掩码设计 |
| 6 | *Interleaved World Modeling and Planning for Autonomous Driving* (Uni-World VLA) | Liu et al. | **ECCV 2026** | 交错掩码、Frozen Hallucination |
| 7 | *World Models* | Ha & Schmidhuber | arXiv 2018 | 世界模型开山之作 |
| 8 | *Dreamer: Dream while Learning Dream* | Hafner et al. | ICLR 2020 | 想象力损失函数 |
| 9 | *DreamerV2: Mastering Atari with Discrete World Models* | Hafner et al. | ICML 2021 | 离散世界模型 |
| 10 | *DreamerV3: Mastering Diverse Domains through World Models* | Hafner et al. | arXiv 2023 | 通用世界模型 |
| 11 | *Scaling Law for Autoregressive Generative Models* | OpenAI | arXiv 2024 | Scaling Law 理论分析 |

---

*本文档由 优酱 🍃 编辑整理，如有问题欢迎交流指正。*

---

## 补充章节：缓解自回归训推不一致的方法

### S1. Exposure Bias 详解

**Exposure Bias（暴露偏差）** 是自回归模型训练与推理之间存在的核心不一致问题：

- **训练阶段**：模型在第 $t$ 步看到的输入是**真实的历史序列** $x_1, ..., x_{t-1}$（Ground Truth），即使模型在前一步预测错了，下一步输入依然是正确的真实值。
- **推理阶段**：模型在第 $t$ 步看到的输入是**模型自己生成的前缀** $\hat{x}_1, ..., \hat{x}_{t-1}$，一旦前面某步出错，错误会像滚雪球一样累积（Error Accumulation）。

数学上，训练阶段模型接触的是数据分布 $p_{\text{data}}(x)$，而推理时模型接触的是自己生成的分布 $p_{\theta}(x)$，两个分布随着生成长度增加而逐渐偏离。

### S1.1 主流缓解方法

#### 方法一：Scheduled Sampling（计划采样）

**核心思想：** 训练时逐步用模型自己的预测替代真实输入，从"全真实输入"平滑过渡到"全模型预测"。

具体策略：在第 $t$ 步，以概率 $\epsilon_t$ 使用真实 token $x_t$，以概率 $1 - \epsilon_t$ 使用模型自己的预测 $\hat{x}_t$。$\epsilon_t$ 随训练进度逐渐衰减：

```python
def scheduled_sampling(step: int, total_steps: int, init_eps: float = 1.0, end_eps: float = 0.0) -> float:
    """
    线性衰减的 Scheduled Sampling
    step: 当前训练步
    total_steps: 总训练步数
    init_eps: 初始使用真实数据的概率（越大越保守）
    end_eps: 最终使用真实数据的概率
    """
    frac = min(step / total_steps, 1.0)
    epsilon = init_eps + (end_eps - init_eps) * frac
    return epsilon

def train_step_with_scheduled_sampling(model, batch, optimizer, step, total_steps):
    """
    带有 Scheduled Sampling 的训练步骤
    """
    epsilon = scheduled_sampling(step, total_steps)
    
    input_ids = batch["input_ids"]  # [B, L]
    labels = batch["labels"]        # [B, L]
    
    # 逐位置决定：是用真实数据还是模型自己的预测
    use_real = torch.bernoulli(torch.full_like(input_ids.float(), epsilon))
    # use_real = 1 表示用真实数据，0 表示用模型预测
    
    # 前向传播得到模型预测
    outputs = model(input_ids)
    predictions = outputs["logits"].argmax(dim=-1)  # [B, L]
    
    # 混合输入：真实 vs 预测
    model_input = torch.where(use_real == 1, input_ids, predictions)
    
    # 重新前向传播
    outputs = model(model_input)
    
    # 计算损失
    loss = compute_loss(outputs.logits, labels)
    
    loss.backward()
    optimizer.step()
    return loss
```

#### 方法二：Free Generation（自由生成）

**核心思想：** 训练时周期性用模型自己生成的前缀做训练，打破训练-推理的分布gap。

具体做法：每经过 $K$ 个常规训练步，就用模型自己生成的序列做一次反向传播（不使用 Teacher Forcing）。这让模型学会"纠错"——在错误累积后如何回到正确轨道。

#### 方法三：DPO / RLHF 对齐优化

**核心思想：** 用强化学习方法直接优化生成质量，让模型学会避免错误累积。

DPO（Direct Preference Optimization）流程：
1. 对每个输入 $x$，用当前策略模型采样两个回复 $y_1, y_2$
2. 人工或用奖励模型标注哪个回复更好
3. 用 DPO 损失直接优化偏好：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y^+,y^-)} \left[ \log \sigma\left( r(x, y^+) - r(x, y^-) \right) \right]
$$

其中 $r(x,y)$ 是奖励函数，等价于 $\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

#### 方法四：Model Ensemble + Knowledge Distillation

**核心思想：** 用一个强大的前向模型（Teacher）指导弱模型（Student）的训练，使学生模型学会在错误累积后如何恢复。

典型做法：用非自回归的全文生成模型作为 Teacher，在每个生成步骤提供"软标签"（soft target probabilities），学生模型同时学习硬标签（真实next token）和软标签（Teacher 的预测分布）。

#### 方法五：孙币式并行解码（Non-autoregressive + AR Hybrid）

**核心思想：** 完全避免自回归推理，直接并行生成整段序列，然后通过多轮迭代精化（类似 diffusion 的多步去噪）。

Mask-Predict（MaskGHM）算法：
```
初始：所有位置都是 [MASK] token
迭代：
  1. 预测所有 [MASK] 位置（并行）
  2. 替换置信度最低的 50% 位置
  3. 重复直到所有位置收敛
```

### S1.2 各方法对比

| 方法 | 缓解程度 | 实现复杂度 | 计算开销 | 副作用 |
|------|---------|-----------|---------|--------|
| Scheduled Sampling | 中 | 低 | 略增 | 可能收敛慢 |
| Free Generation | 中高 | 低 | 中等 | 训练不稳定 |
| DPO / RLHF | 高 | 高 | 高 | 需要偏好数据 |
| Model Distillation | 高 | 中 | 高 | 需要大 Teacher |
| Non-AR Hybrid | 高 | 高 | 高 | 生成质量有时下降 |

---

## 补充章节：KV Cache 原理与 VLA/WM 中的应用

### S2. KV Cache 详解

#### S2.1 为什么需要 KV Cache？

标准自回归推理的问题：每次生成一个 token，都要对**整个历史序列**重新做 Self-Attention。

以生成 $N$ 个 token 为例：
- 第 1 步：计算 $[x_1]$ 的注意力（1 次前向）
- 第 2 步：计算 $[x_1, x_2]$ 的注意力（2 次前向）
- 第 3 步：计算 $[x_1, x_2, x_3]$ 的注意力（3 次前向）
- ...
- 总计：$1 + 2 + 3 + ... + N = O(N^2)$ 次 token 计算

**KV Cache 的核心思想：** 缓存已计算的 Key-Value 对，新 token 只做增量计算。

#### S2.2 KV Cache 数学原理

标准 Transformer 的 Self-Attention：
$$
\text{Attention}(q_t, K_{1:t}, V_{1:t}) = \text{Softmax}\left(\frac{q_t \cdot K_{1:t}^T}{\sqrt{d_k}}\right) \cdot V_{1:t}
$$

其中：
- $q_t$ 是当前 token 的 Query
- $K_{1:t} = [k_1, k_2, ..., k_t]$ 是所有历史 Key 的拼接
- $V_{1:t} = [v_1, v_2, ..., v_t]$ 是所有历史 Value 的拼接

**不使用 KV Cache：** 每步需要 $O(t)$ 次向量计算（计算新 token 与所有历史 key 的点积）

**使用 KV Cache：** 
- 历史 $[k_1, ..., k_{t-1}]$ 和 $[v_1, ..., v_{t-1}]$ 已缓存
- 只需计算新 token 的 $k_t, v_t$
- 然后计算 $q_t$ 与缓存的 $[K_{1:t-1}]$ 和 $k_t$ 的点积
- 计算复杂度降为 $O(1)$ per step

#### S2.3 完整 KV Cache 实现

```python
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache:
    """KV Cache 容器，支持动态增长"""
    def __init__(self, max_batch_size: int, n_heads: int, head_dim: int, device: torch.device):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # 初始化为空的 KV 缓存
        self.k_cache: Optional[torch.Tensor] = None  # [batch, heads, seq, dim]
        self.v_cache: Optional[torch.Tensor] = None   # [batch, heads, seq, dim]
        self.seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将新的 K, V 添加到缓存中
        Args:
            k: 新计算的 Key [batch, heads, 1, dim]
            v: 新计算的 Value [batch, heads, 1, dim]
        Returns:
            完整的 K, V（缓存 + 新值拼接）
        """
        batch_size = k.shape[0]
        
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)  # 在 seq 维度拼接
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        
        self.seq_len += 1
        return self.k_cache, self.v_cache

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0


class CausalAttentionWithKVCache(nn.Module):
    """支持 KV Cache 的 Causal Self-Attention"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Args:
            x: 输入 [batch_size, seq_len, d_model]
                   训练时 seq_len = 完整序列长度
                   推理时 seq_len = 1（新生成的 token）
            kv_cache: KVCache 对象（推理时传入）
            use_cache: 是否使用 KV Cache
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            kv_cache: 更新后的 KVCache
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Q, K, V 投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 分头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. KV Cache 更新
        if use_cache and kv_cache is not None:
            # 新 token 的 K, V 单独计算
            K_new = K[:, :, -1:, :]  # [B, H, 1, D]
            V_new = V[:, :, -1:, :]  # [B, H, 1, D]
            K_full, V_full = kv_cache.update(K_new, V_new)
        elif use_cache:
            # 首次调用，初始化 cache
            kv_cache = KVCache(batch_size, self.n_heads, self.d_k, x.device)
            K_full, V_full = kv_cache.update(K, V)
        else:
            K_full, V_full = K, V

        # 4. 计算注意力分数（只对当前 token 的 Q）
        # Q: [B, H, seq_len, D] or [B, H, 1, D] (推理时)
        # K_full: [B, H, total_seq, D]
        if seq_len > 1 and (kv_cache is None or not use_cache):
            # 训练模式：完整的序列注意力
            scores = torch.matmul(Q, K_full.transpose(-2, -1)) / (self.d_k ** 0.5)
            
            # 因果掩码
            total_seq = K_full.shape[2]
            causal_mask = torch.triu(
                torch.ones(total_seq, total_seq, device=x.device, dtype=torch.bool),
                diagonal=1
            ).float().masked_fill_(torch.triu(torch.ones(total_seq, total_seq, device=x.device, dtype=torch.bool), diagonal=1), float('-inf'))
            
            # 当前 Q 对应的位置对齐
            offset = 0  # 完整序列，无偏移
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            # 推理模式（KV Cache）：Q 只有最后一个位置
            q_last = Q[:, :, -1:, :]  # [B, H, 1, D]
            scores = torch.matmul(q_last, K_full.transpose(-2, -1)) / (self.d_k ** 0.5)
            # 推理时天然因果（只看 cache 里的历史 K），不需要掩码

        # 5. Softmax & 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V_full)  # [B, H, 1, D] or [B, H, seq, D]

        # 6. 合并多头 & 输出投影
        if seq_len > 1:
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        else:
            context = context.transpose(1, 2).contiguous().view(batch_size, 1, d_model)
        
        output = self.W_o(context)

        return output, kv_cache


class TransformerBlockWithKVCache(nn.Module):
    """支持 KV Cache 的 Transformer Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalAttentionWithKVCache(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, KVCache]:
        # Pre-Norm + Cache
        attn_out, kv_cache = self.attention(self.norm1(x), kv_cache, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, kv_cache


class AutoregressiveLMWithKVCache(nn.Module):
    """支持 KV Cache 的自回归语言模型"""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlockWithKVCache(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
                       训练时：完整序列
                       推理时：只有最后一个 token
            kv_caches: 每层的 KVCache 列表
            use_cache: 是否使用 KV Cache
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len

        # Embedding
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        # 初始化或获取 KV Caches
        if use_cache:
            if kv_caches is None:
                kv_caches = [None] * len(self.layers)

        # 通过 Transformer Layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            if use_cache:
                x, new_cache = layer(x, kv_caches[i], use_cache)
                new_caches.append(new_cache)
            else:
                x, _ = layer(x, None, use_cache)
                new_caches.append(None)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_caches if use_cache else (None, None)

    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,  # 前缀序列 [B, prefix_len]
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        使用 KV Cache 的高效自回归生成
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始化 KV Caches
        kv_caches = [None] * len(self.layers)
        
        # 前缀处理（完整前向，无 cache）
        prefix_len = input_ids.shape[1]
        logits, kv_caches = self.forward(input_ids, kv_caches, use_cache=True)
        
        # 取最后一个 token 的 logits 生成下一个
        next_token_logits = logits[:, -1, :] / temperature
        
        # Top-K 过滤
        if top_k > 0:
            top_k_vals = torch.topk(next_token_logits, top_k, dim=-1).values[:, -1]
            next_token_logits[next_token_logits < top_k_vals.unsqueeze(-1)] = float('-inf')
        
        # Sampling
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        
        # 拼接已生成的 tokens
        generated = [next_token]
        current_token = next_token
        
        # 自回归生成
        for _ in range(max_new_tokens - 1):
            # 仅对当前 token 做前向（KV Cache 生效）
            logits, kv_caches = self.forward(current_token, kv_caches, use_cache=True)
            
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                top_k_vals = torch.topk(next_token_logits, top_k, dim=-1).values[:, -1]
                next_token_logits[next_token_logits < top_k_vals.unsqueeze(-1)] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            current_token = next_token
        
        return torch.cat(generated, dim=1)
```

#### S2.4 VLA/WM 场景下的 KV Cache 特殊考量

VLA 和 World Model 与纯语言模型不同，输入是**多模态的**（图像 + 文本 + 动作），KV Cache 的策略需要特殊设计：

```python
class VLAKVCacheManager:
    """
    VLA 场景下的 KV Cache 管理器
    
    核心挑战：
    1. 视觉 tokens 通常很长（ViT 输出的 patches 数量可能上千）
    2. 动作 tokens 在生成阶段需要频繁访问视觉 KV
    3. 多帧历史需要保留完整的视觉 KV
    """
    def __init__(self, vision_len_per_frame: int, max_frames: int, 
                 n_layers: int, n_heads: int, head_dim: int, device: torch.device):
        self.vision_len_per_frame = vision_len_per_frame
        self.max_frames = max_frames
        self.max_vision_len = vision_len_per_frame * max_frames
        
        # 视觉 KV Cache：长期缓存，跨越多个时间步
        self.vision_kv_cache: Optional[KVCache] = None
        
        # 语言 KV Cache：中等长度
        self.lang_kv_cache: Optional[KVCache] = None
        
        # 动作 KV Cache：仅在生成阶段使用
        self.action_kv_cache: Optional[KVCache] = None
        
        # 跟踪已缓存的帧数
        self.cached_frames = 0
        self.cached_lang_len = 0

    def update_vision(self, vision_k: torch.Tensor, vision_v: torch.Tensor):
        """更新视觉 tokens 的 KV Cache（通常只在接收到新帧时调用）"""
        if self.vision_kv_cache is None:
            self.vision_kv_cache = KVCache(1, n_heads, head_dim, vision_k.device)
        self.vision_kv_cache.update(vision_k, vision_v)
        self.cached_frames += 1

    def update_lang(self, lang_k: torch.Tensor, lang_v: torch.Tensor):
        """更新语言 tokens 的 KV Cache（通常只在输入语言指令时调用一次）"""
        if self.lang_kv_cache is None:
            self.lang_kv_cache = KVCache(1, n_heads, head_dim, lang_k.device)
        self.lang_kv_cache.update(lang_k, lang_v)
        self.cached_lang_len = self.lang_kv_cache.seq_len

    def get_full_kv(self, include_vision: bool = True, include_lang: bool = True,
                    include_action: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取所有缓存的 K, V，用于计算新 token 的注意力
        """
        k_parts, v_parts = [], []
        
        if include_vision and self.vision_kv_cache is not None:
            k_parts.append(self.vision_kv_cache.k_cache)
            v_parts.append(self.vision_kv_cache.v_cache)
        
        if include_lang and self.lang_kv_cache is not None:
            k_parts.append(self.lang_kv_cache.k_cache)
            v_parts.append(self.lang_kv_cache.v_cache)
        
        if include_action and self.action_kv_cache is not None:
            k_parts.append(self.action_kv_cache.k_cache)
            v_parts.append(self.action_kv_cache.v_cache)
        
        if not k_parts:
            return None, None
        
        K_full = torch.cat(k_parts, dim=2)  # 在 seq 维度拼接
        V_full = torch.cat(v_parts, dim=2)
        
        return K_full, V_full

    def clear_action_cache(self):
        """每个推理周期开始时清除动作缓存（动作是每步新生成的）"""
        self.action_kv_cache = None


class VLAGeneratorWithOptimizedCache(nn.Module):
    """
    使用分级 KV Cache 的 VLA 生成器
    
    策略：
    - 视觉 tokens：只计算一次，后续帧直接追加
    - 语言 tokens：计算一次，后续只追加（如果有增量）
    - 动作 tokens：每步完全重新计算（因为是自回归生成的）
    """
    def __init__(self, vision_encoder, lang_encoder, fusion_transformer, action_head):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.fusion_transformer = fusion_transformer
        self.action_head = action_head
        self.cache_manager = None

    @torch.no_grad()
    def generate_action(
        self,
        history_frames: torch.Tensor,    # 历史帧 [1, T, C, H, W]
        lang_instruction: str,            # 语言指令
        action_horizon: int = 8           # 预测动作序列长度
    ) -> torch.Tensor:
        """
        使用分级 KV Cache 的动作生成
        """
        # 1. 视觉编码（仅在收到新帧时计算）
        if self.cache_manager is None:
            self.cache_manager = VLAKVCacheManager(...)
        
        vision_features = self.vision_encoder(history_frames)  # [1, T*patch_len, D]
        self.cache_manager.update_vision(
            vision_features.k, vision_features.v
        )

        # 2. 语言编码（仅计算一次）
        lang_features = self.lang_encoder(lang_instruction)
        self.cache_manager.update_lang(lang_features.k, lang_features.v)

        # 3. 动作自回归生成
        self.cache_manager.clear_action_cache()
        
        current_action_token = ...  # BOS token
        
        for step in range(action_horizon):
            # 获取完整 KV（视觉 + 语言 + 已生成动作）
            K_full, V_full = self.cache_manager.get_full_kv()
            
            # 计算当前动作 token 的 Q
            q_current = self.compute_action_query(current_action_token)
            
            # 注意力计算（新 token 只与历史 KV 交互，无需重复计算视觉/语言）
            attn_out = self.fusion_transformer.attend(q_current, K_full, V_full)
            
            # 预测下一个动作 token
            next_action = self.action_head(attn_out)
            
            # 更新动作 KV Cache
            self.cache_manager.action_kv_cache.update(
                self.compute_action_k(next_action),
                self.compute_action_v(next_action)
            )
            
            current_action_token = next_action
        
        return action_sequence
```

---

## 补充章节：RynnVLA002 掩码设计分析

### S3. RynnVLA002 框架概述

> ⚠️ **关于 RynnVLA002**：本节基于 RynnVLA（统一 VLA 与 World Model 的研究工作）的通用设计哲学进行分析写作。如 RynnVLA002 为特定项目或企业内部工作，公开信息有限，建议用户以内部文档为准核实细节。以下分析重点在于这类"统一 VLA + WM"范式的共性掩码设计思路。

### S3.1 统一 VLA + WM 的核心挑战

RynnVLA 这类工作的核心目标是**用同一个模型同时完成 VLA 任务（动作预测）和 World Model 任务（未来帧预测）**。这带来了独特的掩码设计挑战：

**任务差异：**
- VLA 任务：动作 token 需要**因果预测**（只能看历史，不能看未来动作）
- World Model 任务：需要预测未来帧，帧预测本身是**非因果的**（需要看到当前动作对环境的影响）

**统一的挑战：** 如何在同一个 Transformer 架构中，同时支持：
1. 因果的动作自回归生成
2. 非因果的未来帧想象式预测

### S3.2 统一掩码设计哲学

RynnVLA 类工作的掩码设计哲学可以总结为**「双路径 + 选择性掩码」**：

```
输入序列布局：
[Vision Tokens | Language Tokens | Action Tokens]

两条路径同时前向：
路径A (VLA):  [Vision | Language | Action] → 因果掩码 → 动作预测
路径B (WM):   [Vision | Language | Action_(t)] → 非因果掩码 → 帧预测
```

**核心思想：**
- 共享视觉和语言编码器（减少计算量）
- 在共享的 Transformer 中，通过不同的掩码模式同时支持两个任务
- 动作 token 始终保持因果约束
- 帧预测路径允许更宽松的注意力（可以看到动作 token 对环境的影响）

### S3.3 掩码实现细节

```python
class DualPathMaskTransformer(nn.Module):
    """
    支持 VLA 和 WM 双路径的统一 Transformer
    
    关键设计：
    - 共享参数，不同掩码模式
    - VLA 路径：因果掩码（动作自回归）
    - WM 路径：单向看动作，但帧预测本身非因果
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 动作预测头（VLA 任务）
        self.action_head = nn.Linear(d_model, action_vocab_size)
        
        # 帧预测头（WM 任务）
        self.frame_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, frame_vocab_size)
        )

    def build_dual_mask(
        self,
        vision_len: int,
        lang_len: int,
        action_len: int,
        device: torch.device,
        mode: str  # "vla" or "wm"
    ) -> torch.Tensor:
        """
        构造双路径掩码
        
        VLA 模式：
        - 动作 tokens 完全因果（只看历史）
        
        WM 模式：
        - 帧预测非因果（可以看到所有视觉 tokens）
        - 但动作 tokens 仍然因果（只看历史动作）
        """
        total_len = vision_len + lang_len + action_len
        mask = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)
        
        v_start = 0
        l_start = vision_len
        a_start = vision_len + lang_len
        
        if mode == "vla":
            # ===== VLA 模式：完全因果 =====
            # Vision 因果（只看之前的 Vision + 全部 Language）
            for i in range(vision_len):
                mask[v_start + i, v_start + i + 1:] = True   # Vision 不能看未来 Vision
                # Vision 可以看所有 Language（无限制）
            
            # Language 因果（Language 之间可以互相看，但 VLA 中 Language 通常在前面）
            # Language 可以看所有 Vision
            
            # Action 完全因果（只看之前的 Action + 全部 Vision/Language）
            for i in range(action_len):
                # Action 不能看自己的未来
                mask[a_start + i, a_start + i + 1:] = True
                # Action 可以看 Vision 和 Language（无限制）
                mask[a_start + i, v_start:v_start + vision_len] = False  # 不mask
            
        elif mode == "wm":
            # ===== WM 模式：帧预测允许非因果 =====
            # Vision tokens 之间：允许双向注意（帧内空间建模需要）
            # （不mask vision-vision 之间）
            
            # Vision 可以看 Language 和 Action
            mask[v_start:v_start + vision_len, l_start:a_start] = False  # Vision → Language
            mask[v_start:v_start + vision_len, a_start:a_start + action_len] = False  # Vision → Action
            
            # Action 因果（只看历史 Action + Vision + Language）
            for i in range(action_len):
                mask[a_start + i, a_start + i + 1:] = True  # 不能看未来 Action
                mask[a_start + i, v_start:v_start + vision_len] = False  # 可以看 Vision
                mask[a_start + i, l_start:a_start] = False  # 可以看 Language
        
        return mask.float().masked_fill(mask, float('-inf'))

    def forward(
        self,
        vision_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        mode: str = "vla"
    ) -> dict:
        """
        双路径前向传播
        """
        # 拼接所有 tokens
        x = torch.cat([vision_tokens, lang_tokens, action_tokens], dim=1)
        
        vision_len = vision_tokens.shape[1]
        lang_len = lang_tokens.shape[1]
        action_len = action_tokens.shape[1]
        
        # 构造掩码
        mask = self.build_dual_mask(vision_len, lang_len, action_len, x.device, mode)
        
        # 逐层 Transformer 前向
        for layer in self.layers:
            x = layer(x, mask)
        
        # 从 action tokens 对应的位置取输出
        a_start = vision_len + lang_len
        action_features = x[:, a_start:, :]  # [B, action_len, D]
        
        if mode == "vla":
            action_logits = self.action_head(action_features)
            return {"action_logits": action_logits}
        elif mode == "wm":
            # WM 模式：取最后一个 action token 之后的位置（预测帧）
            frame_logits = self.frame_head(action_features[:, -1:, :])
            return {"frame_logits": frame_logits}

    @torch.no_grad()
    def interleaved_generation(
        self,
        history_frames: torch.Tensor,
        lang_instruction: torch.Tensor,
        n_steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        交错生成：交替进行 VLA 动作预测和 WM 帧预测
        
        每次迭代：
          1. WM 路径：给定当前状态，预测下一帧（用于想象式规划）
          2. VLA 路径：给定当前状态和语言指令，预测下一动作
          3. 用预测的动作更新状态，进入下一轮
        """
        action_sequence = []
        predicted_frames = []
        
        # 当前状态 = 历史帧
        current_state = history_frames
        
        for step in range(n_steps):
            # Step 1: World Model 预测下一帧（WM 路径，非因果掩码）
            wm_output = self.forward(
                self.encode_vision(current_state),
                lang_instruction,
                torch.zeros(1, 1, self.d_model, device=history_frames.device),
                mode="wm"
            )
            predicted_frame = wm_output["frame_logits"]
            predicted_frames.append(predicted_frame)
            
            # Step 2: VLA 预测下一个动作（VLA 路径，因果掩码）
            # 将预测帧加入状态
            updated_state = torch.cat([current_state, predicted_frame], dim=1)
            
            vla_output = self.forward(
                self.encode_vision(updated_state),
                lang_instruction,
                self.get_action_tokens_so_far(action_sequence),
                mode="vla"
            )
            next_action = vla_output["action_logits"].argmax(dim=-1)
            action_sequence.append(next_action)
            
            # Step 3: 用真实动作更新状态（如果有）或继续
            current_state = updated_state
        
        return torch.cat(action_sequence, dim=1), torch.cat(predicted_frames, dim=1)
```

### S3.4 RynnVLA 类工作的掩码设计核心总结

| 设计维度 | VLA 路径 | WM 路径 |
|---------|---------|---------|
| Vision-Vision | 因果（只看历史帧） | 双向（空间建模需要） |
| Vision-Language | 允许 | 允许 |
| Vision-Action | 允许 | 允许 |
| Action-Action | 块内双向（当前 chunk 内） | 因果（只看历史动作） |
| Action-Frame | 允许 | 允许（帧依赖于动作） |
| 预测性质 | 因果自回归 | 非因果（想象式） |

---

*本文档由 优酱 🍃 补充整理，补充时间：2026-04-13*

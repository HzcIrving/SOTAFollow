# RoPE、3DPE 与 mRoPE 技术详解

> 本文档系统整理位置编码（Positional Encoding）技术：涵盖 NLP 经典方法（Sinusoidal、Learned、ALiBi、RoPE）、计算机视觉（2D PE、3DPE）、以及多模态时序融合（mRoPE）。

---

## 1. Motivation（问题背景）

### 1.1 Transformer 的本质缺陷

Transformer 的核心是 **Self-Attention** 机制，其数学本质是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

**问题**：$QK^{T}$ 的计算是**位置无关**的——对于 Self-Attention，无论 token 出现在序列的哪个位置，它们的交互方式完全相同。

换句话说，Attention 机制是**置换等价（Permutation Equvariant）**的：

$$
\text{Attention}(\text{swap}(x_1, x_2)) = \text{swap}(\text{Attention}(x_1, x_2))
$$

这意味着，如果没有位置信息：
- "狗咬人" 和 "人咬狗" 的语义会被模型认为是相同的
- 图像中 "左上角的猫" 和 "右下角的猫" 会被同等对待
- 视频中 "先抬手后挥手" 和 "先挥后抬手" 无法区分

### 1.2 核心问题

**位置编码的目的**：为序列中的每个位置注入独特的"签名"，让模型能够区分绝对位置和相对位置。

### 1.3 评估维度

评估一种位置编码方案的优劣，主要看以下几个维度：

| 维度 | 含义 | 重要性 |
|------|------|--------|
| **外推性** | 训练时没见过更长序列，推理时能否处理？ | 关键 |
| **相对位置感知** | 能否自然地表达 token 之间的距离？ | 关键 |
| **表达能力** | 位置编码是否足够丰富，能区分任意位置？ | 重要 |
| **参数效率** | 是否需要额外参数？参数越多越容易过拟合 | 一般 |
| **维度支持** | 是否支持 2D/3D 等高维数据？ | 视任务而定 |

### 1.4 Related Works

| 方法 | 核心思想 | 局限性 |
|------|----------|--------|
| **Sinusoidal** | 固定三角函数编码绝对位置 | 无法学习相对位置 |
| **Learned PE** | 可学习位置嵌入 | 外推性差 |
| **Relative PE (Shaw)** | 可学习相对位置偏置 | 参数多，泛化受限 |
| **ALiBi** | 线性偏置，无参数 | 只能表示相对距离 |

**核心问题**：能否用**零参数的旋转矩阵**同时实现绝对位置编码和相对位置解码？

---

## 2. 背景：为什么需要位置编码？

### 0.1 Transformer 的本质缺陷

Transformer 的核心是 **Self-Attention** 机制，其数学本质是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

**问题**：$QK^{T}$ 的计算是**位置无关**的——对于 Self-Attention，无论 token 出现在序列的哪个位置，它们的交互方式完全相同。

换句话说，Attention 机制是**置换等价（Permutation Equvariant）**的：

$$
\text{Attention}(\text{swap}(x_1, x_2)) = \text{swap}(\text{Attention}(x_1, x_2))
$$

这意味着，如果没有位置信息：
- "狗咬人" 和 "人咬狗" 的语义会被模型认为是相同的
- 图像中 "左上角的猫" 和 "右下角的猫" 会被同等对待
- 视频中 "先抬手后挥手" 和 "先挥后抬手" 无法区分

### 0.2 位置编码的目的

**位置编码（Positional Encoding）** 的目的是：为序列中的每个位置注入独特的"签名"，让模型能够区分：

| 维度 | 没有位置编码 | 有位置编码 |
|------|------------|-----------|
| NLP | "I saw a saw" 两个 saw 无法区分 | 能区分 "I [0] saw [1] a [2] saw [3]" |
| Vision | 图像 patch 打乱后仍能处理 | 位置信息丢失，空间关系混乱 |
| Video | 帧顺序打乱后特征相同 | 时序关系丢失，动作无法识别 |

### 0.3 位置编码的评估维度

评估一种位置编码方案的优劣，主要看以下几个维度：

| 维度 | 含义 | 重要性 |
|------|------|--------|
| **外推性** | 训练时没见过更长序列，推理时能否处理？ | 关键 |
| **相对位置感知** | 能否自然地表达 token 之间的距离？ | 关键 |
| **表达能力** | 位置编码是否足够丰富，能区分任意位置？ | 重要 |
| **参数效率** | 是否需要额外参数？参数越多越容易过拟合 | 一般 |
| **维度支持** | 是否支持 2D/3D 等高维数据？ | 视任务而定 |

---

## 1. 经典位置编码方法

### 1.1 Sinusoidal Position Encoding（正弦位置编码）

#### 背景与 Motivation

**来源**：《Attention Is All You Need》(Vaswani et al., 2017)

**问题**：Transformer 的 Attention 是位置无关的，需要注入位置信息。

**核心洞察**：自然语言中，**相对位置往往比绝对位置更重要**。例如：
- "狗咬人" vs "人咬狗"——相对顺序决定语义
- "我今天去了银行，银行很人多"——两个"银行"指代相同的事物，距离很近

三角函数有一个重要性质：**线性组合可以表示相对位置**。

如果 $\text{PE}(pos)$ 是位置 $pos$ 的编码，那么存在线性函数 $f$ 使得：

$$
f(\text{PE}(pos + \delta)) = g(\text{PE}(pos), \text{PE}(\delta))
$$

这意味着正弦位置编码可以**自然地表示相对位置**。

#### 数学公式

对于位置 $pos$ 和维度 $i$：

$$
\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**为什么这样设计？**

| 设计选择 | 原因 |
|---------|------|
| 奇偶交替 sin/cos | 保证相邻维度线性无关 |
| 指数级递减频率 | 低维 = 长周期 = 编码长距离；高维 = 短周期 = 编码短距离 |
| 基数 10000 | 覆盖从 ~6 到 ~60000 的周期范围 |

#### 伪代码

```python
import numpy as np
import torch

def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    生成 Sinusoidal 位置编码

    参数:
        max_len: 最大序列长度
        d_model: 模型隐藏层维度

    返回:
        pe: (max_len, d_model) 位置编码矩阵
    """
    pe = torch.zeros(max_len, d_model)

    # 生成位置索引: (max_len, 1)
    position = torch.arange(0, max_len).unsqueeze(1).float()

    # 生成除数项: (d_model//2,)
    # 推导: 要让维度 i 的周期为 T_i，需要 frequency = 1/T_i
    # 设 base = 10000, 则 freq_i = base^(-2i/d) = 1/T_i
    # 所以 T_i = 10000^(2i/d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        (-np.log(10000.0) / d_model)
    )

    # 偶数维度: sin
    pe[:, 0::2] = torch.sin(position * div_term)

    # 奇数维度: cos
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe  # (max_len, d_model)
```

#### 可视化示意

```
维度 i=0 (低频, 周期 ≈ 6万)
│ sin(0)  ────────────────────────────────────────→ pos
│
维度 i=3 (中频, 周期 ≈ 100)
│ sin(0)  ────●───●───●───●───●───●───●───→ pos
│
维度 i=7 (高频, 周期 ≈ 6.3)
│ sin(0)  ─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─→ pos
```

#### 特点总结

| 优点 | 缺点 |
|------|------|
| **外推性好**：三角函数可自然外推到任意长度 | 无法学习：纯手工设计，参数固定 |
| **零参数**：无需学习参数 | 外推时远距离位置可能不准确 |
| **相对位置表达**：线性组合可表示相对位置 | 低频维度周期过长，短距离区分度差 |

---

### 1.2 Learned Position Encoding（学习型位置编码）

#### 背景与 Motivation

**问题**：Sinusoidal 是手工设计的，无法根据任务学习最优的位置编码方式。

**核心洞察**：如果任务对位置信息有特定规律，那么让模型**自己学习**位置编码可能比手工设计更好。

例如：
- 在语义分割任务中，像素越靠近图像中心可能越重要
- 在代码补全任务中，缩进层级决定了代码结构
- 在情感分析中，句首的"但是"可能决定整个句子的情感

#### 核心思想

将位置编码作为可学习参数，随机初始化后通过训练学习。

```python
class LearnedPE(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # 可学习参数: (max_len, d_model)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        输入: position_ids (batch, seq_len) 位置索引
        输出: (batch, seq_len, d_model) 位置编码
        """
        return self.pe(position_ids)  # 索引查找表
```

#### 特点总结

| 优点 | 缺点 |
|------|------|
| **可端到端学习**：根据任务自适应调整 | **外推性差**：超出训练长度无法处理 |
| **表达能力强**：可以学习任意复杂的位置规律 | 需要额外的可学习参数 |
| **实现简单**：直接用 Embedding | 难以捕捉相对位置关系 |

---

### 1.3 Relative Position Encoding（相对位置编码）

#### 背景与 Motivation

**问题**：无论是 Sinusoidal 还是 Learned，都是**绝对位置编码**——它们编码的是 token 出现在序列的哪个位置，而不是 token 之间相隔多远。

**核心洞察**：在自然语言中，**相对位置往往比绝对位置更重要**。

例如：
- "The cat sat on the mat" 中，"cat" 和 "mat" 隔着 5 个词，但它们是主语和宾语
- "I don't think it's a good idea" 中，"don't" 和 "good" 的关系决定否定范围
- 在代码中，"{" 和 "}" 的距离决定代码块

相对位置编码直接建模 token 之间的距离，而不是它们的绝对位置。

#### 发展脉络

1. **Shaw et al., 2018**：首次提出相对位置编码（RPE），使用可学习的相对位置偏置
2. **Huang et al., 2018**：XLNet 提出双路注意力，同时考虑前向和后向相对位置
3. **Dai et al., 2019**：Transformer-XL 提出更高效的相对位置编码，支持长距离依赖

#### Shaw RPE 公式

$$
\alpha_{ij} = \frac{q_i^T k_j}{\sqrt{d}} + w_{[i-j]}
$$

其中 $w_{[i-j]}$ 是可学习的相对位置偏置，$[\cdot]$ 表示 clipped 裁剪（防止参数爆炸）。

```python
class RelativePositionEncoding(nn.Module):
    def __init__(self, max_relative_position: int, d_model: int):
        super().__init__()
        self.max_rel_pos = max_relative_position
        # 相对位置偏置: (2*max_relative_position+1, num_heads)
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1, 1
        )

    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        """
        生成相对位置偏置矩阵

        返回: (1, 1, query_len, key_len) 或 (1, heads, query_len, key_len)
        """
        # 生成相对位置索引
        position_ids = torch.arange(query_len)  # (query_len,)
        key_position_ids = torch.arange(key_len)  # (key_len,)

        # 计算相对位置矩阵: (query_len, key_len)
        relative_position_matrix = position_ids.unsqueeze(1) - key_position_ids.unsqueeze(0)

        # 裁剪到 [-max_rel_pos, max_rel_pos]
        relative_position_matrix = relative_position_matrix.clamp(
            -self.max_rel_pos, self.max_rel_pos
        )

        # 偏移 +max_rel_pos 以便索引
        relative_position_matrix = relative_position_matrix + self.max_rel_pos

        # 查表得到偏置: (query_len, key_len)
        bias = self.relative_attention_bias(relative_position_matrix).squeeze(-1)

        return bias  # (query_len, key_len)
```

#### 可视化

```
         key positions (j)
         0   1   2   3   4
      ┌────────────────────┐
    0 │  0  -1  -2  -3  -4 │  ← 相对位置 i-j
    i │ +1   0  -1  -2  -3 │
    1 │ +2  +1   0  -1  -2 │
    2 │ +3  +2  +1   0  -1 │
    3 │ +4  +3  +2  +1   0 │
      └────────────────────┘
```

#### 特点总结

| 优点 | 缺点 |
|------|------|
| **直接建模相对位置**：更符合语言直觉 | 需要额外的相对位置参数 |
| **外推性相对较好**：clipped 机制允许一定外推 | 相对位置需要手动设定范围 |
| **可与绝对位置结合**：更灵活 | 计算量略有增加 |

---

### 1.4 ALiBi（Attention with Linear Biases）

#### 背景与 Motivation

**问题**：之前的相对位置编码都需要**可学习参数**（如相对位置 Embedding），这带来两个问题：
1. 参数随相对位置范围线性增长
2. 泛化能力受限：训练时没见过的距离，推理时无法处理

**核心洞察**：如果使用**固定的、非学习的**相对位置偏置，而且这个偏置是**线性的**，那么：
- 天然支持任意长度的外推
- 无需学习参数
- 线性惩罚天然实现"远程衰减"

#### 数学公式

$$
\alpha_{ij} = \frac{q_i^T k_j}{\sqrt{d}} - |i - j| \cdot m
$$

其中 $m$ 是每个注意力头的斜率，$m_{k} = 2^{-8/k}$（第 $k$ 个头）。

**为什么斜率是指数递减的？**

这是为了让不同注意力头有不同的"注意力距离"：
- 首几个头：斜率大 → 注意力更集中于近距离
- 后面的头：斜率小 → 注意力更分散，能看到更远

```python
def alibi_slope(num_heads: int) -> torch.Tensor:
    """生成每个 head 的 ALiBi 斜率"""
    # 首项 1/2，公比 2^(-8/num_heads)
    start = 2 ** (-(8 / num_heads))
    ratio = 2 ** (-(8 / num_heads))
    return start * (ratio ** torch.arange(num_heads))


def alibi_attention_bias(query_len: int, key_len: int,
                          num_heads: int) -> torch.Tensor:
    """生成 ALiBi 偏置矩阵"""
    slopes = alibi_slope(num_heads)  # (num_heads,)

    # 生成距离矩阵: (query_len, key_len)
    q_idx = torch.arange(query_len).unsqueeze(1)  # (query_len, 1)
    k_idx = torch.arange(key_len).unsqueeze(0)    # (1, key_len)
    distance = (q_idx - k_idx).abs().float()     # 绝对距离

    # 扩展 slopes: (num_heads, 1, 1) × (1, query_len, key_len)
    slopes = slopes.view(num_heads, 1, 1)
    bias = - slopes * distance.unsqueeze(0)        # (num_heads, query_len, key_len)

    return bias
```

#### 可视化

```
         key positions (j)
         0    1    2    3    4
      ┌─────────────────────────────┐
      │   0  -1   -2   -3   -4   │  ← head 1: 斜率=1
  i   │  -1   0   -1   -2   -3   │  ← 距离越远, 惩罚越大
      │  -2  -1    0   -1   -2   │
      └─────────────────────────────┘

      ┌─────────────────────────────┐
      │   0  -0.5 -1.0 -1.5 -2.0  │  ← head 2: 斜率=0.5
      │ -0.5   0  -0.5 -1.0 -1.5  │
      │ -1.0 -0.5   0  -0.5 -1.0  │
      └─────────────────────────────┘
```

#### 特点总结

| 优点 | 缺点 |
|------|------|
| **外推性极佳**：线性惩罚天然支持任意长度 | 只能表示相对距离，不能表示绝对位置 |
| **零参数**：纯手工设计的固定偏置 | 需要手动调参斜率 |
| **远程衰减**：距离越远注意力越低 | 对所有任务使用相同的斜率规律 |

---

## 2. RoPE：旋转位置编码

### 2.1 背景与 Motivation

**问题**：之前的相对位置方法都有明显缺陷：
- Shaw RPE：有参数，外推性受限
- ALiBi：零参数，但线性偏置与 Attention 分数直接相加，数学上不够优雅

**核心洞察**（Su et al., 2021）：能否用**旋转矩阵**实现相对位置编码？

旋转有一个重要性质：旋转的角度差等于相对旋转。

如果一个向量旋转了角度 $m\theta$，另一个向量旋转了角度 $n\theta$，那么它们的点积仅与**相对角度**$(m-n)\theta$ 有关：

$$
(\text{Rot}(m\theta) q) \cdot (\text{Rot}(n\theta) k) = q \cdot \text{Rot}((m-n)\theta) k
$$

这意味着：**通过旋转编码绝对位置，attention score 自然变成相对位置的函数**！

### 2.2 核心数学

**对于位置 $m$ 的 query 向量 $q_{m}$**，RoPE 对每对维度 $(2i, 2i+1)$ 做 2D 旋转：

$$
q_m^{(2i)} \rightarrow q_m^{(2i)} \cos(m\theta_i) - q_m^{(2i+1)} \sin(m\theta_i)
$$
$$
q_m^{(2i+1)} \rightarrow q_m^{(2i)} \sin(m\theta_i) + q_m^{(2i+1)} \cos(m\theta_i)
$$

其中旋转角度 $\theta_{i} = 10000^{-2i/d}$。

**为什么能实现相对位置？**

假设两个位置 $m$ 和 $n$ 的 query 和 key 分别经过旋转：
- $\tilde{q}_{m} = \text{Rot}(m\theta) q_{m}$
- $\tilde{k}_{n} = \text{Rot}(n\theta) k_{n}$

它们的点积：
$$
\tilde{q}_m^T \tilde{k}_n = q_m^T \text{Rot}((m-n)\theta) k_n
$$

**仅依赖相对距离 $(m-n)$！**

### 2.3 伪代码

```python
import torch
import math

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    预计算旋转位置编码的频率

    参数:
        dim: 嵌入维度
        max_seq_len: 最大序列长度
        theta: 基础角度（默认10000）

    返回:
        freqs_cis: (max_seq_len, dim//2) 复数形式的旋转角度
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len)
    # 外积得到每个位置每对维度的旋转角度
    freqs = torch.outer(positions, freqs)  # (max_seq_len, dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转为复数
    return freqs_cis


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     freqs_cis: torch.Tensor) -> tuple:
    """
    对 query 和 key 应用 RoPE

    参数:
        q: (batch, num_heads, seq_len, head_dim) query
        k: (batch, num_heads, seq_len, head_dim) key
        freqs_cis: (max_seq_len, head_dim//2) 复数旋转角度

    返回:
        q_rot, k_rot: 应用 RoPE 后的 q, k
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    assert head_dim % 2 == 0

    # Reshape: (batch, num_heads, seq_len, head_dim//2, 2)
    q = q.view(*q.shape[:-1], head_dim // 2, 2)
    k = k.view(*k.shape[:-1], head_dim // 2, 2)

    # 转成复数形式
    q_complex = torch.view_as_complex(q.float())
    k_complex = torch.view_as_complex(k.float())

    # 切片 freqs_cis 到当前序列长度
    freqs = freqs_cis[:seq_len]  # (seq_len, head_dim//2)

    # 复数乘法实现旋转
    q_rot = torch.view_as_real(q_complex * freqs.unsqueeze(0).unsqueeze(0)).flatten(-2)
    k_rot = torch.view_as_real(k_complex * freqs.unsqueeze(0).unsqueeze(0)).flatten(-2)

    return q_rot.type_as(q), k_rot.type_as(k)
```

### 2.4 RoPE 特性总结

```
┌──────────────────────────────────────────────────────────────┐
│                    RoPE 四大核心性质                         │
├──────────────────────────────────────────────────────────────┤
│  1. 绝对位置编码：旋转角度与绝对位置 m 成正比                  │
│  2. 相对位置解码：attention(m,n) ∝ g(q,k,m-n)               │
│  3. 远程衰减：cos(mθ) 随距离增大而振荡衰减                    │
│  4. 零参数：纯三角函数，无需学习参数                          │
├──────────────────────────────────────────────────────────────┤
│  为什么 theta_i = 10000^(-2i/d)?                              │
│  • i 小（低频）：theta≈1，旋转慢 → 捕获长距离依赖             │
│  • i 大（高频）：theta≈0，旋转快 → 捕获短距离模式             │
│  • 类似傅里叶变换，覆盖不同时间尺度                            │
└──────────────────────────────────────────────────────────────┘
```

### 2.5 RoPE 与其他方法的对比

| 特性 | Sinusoidal | Learned | ALiBi | RoPE |
|------|-----------|---------|-------|------|
| 绝对/相对 | 绝对 | 绝对 | 相对 | **绝对+相对** |
| 参数 | 无 | 有 | 无 | **无** |
| 外推性 | 好 | 差 | 极好 | **好** |
| 数学优雅性 | 一般 | 一般 | 一般 | **优雅（旋转群）** |

---

## 3. 2D 图像位置编码

### 3.1 背景与 Motivation

**问题**：NLP 中的位置编码是 1D 的（序列顺序），但图像是 2D 的（空间位置）。

**核心洞察**：将 1D 位置编码自然扩展到 2D，分别对 H 和 W 维度编码，然后拼接。

**两种主要方法**：

#### 方法一：Sinusoidal 2D 扩展

```python
def positional_encoding_2d(height: int, width: int, d_model: int) -> torch.Tensor:
    """
    生成 2D 图像位置编码（Sinusoidal 扩展）

    返回: (H*W, d_model) 展平后的 2D 位置编码
    """
    pe_h = sinusoidal_pe(height, d_model // 2)  # (H, d_model//2)
    pe_w = sinusoidal_pe(width, d_model // 2)   # (W, d_model//2)

    # 拼接: (H, W, d_model)
    pe = torch.zeros(height, width, d_model)
    pe[:, :, :d_model//2] = pe_h.unsqueeze(1).repeat(1, width, 1)
    pe[:, :, d_model//2:] = pe_w.unsqueeze(0).repeat(height, 1, 1)

    # 展平: (H*W, d_model)
    return pe.view(height * width, d_model)
```

#### 方法二：可学习 2D 位置编码

```python
class Learned2DPositionalEncoding(nn.Module):
    def __init__(self, height: int, width: int, d_model: int):
        super().__init__()
        # 可学习的 2D 位置编码
        self.pe = nn.Parameter(
            torch.randn(1, height * width, d_model) * 0.02
        )

    def forward(self) -> torch.Tensor:
        return self.pe  # (1, H*W, d_model)
```

### 3.2 2D 位置编码的问题

**问题**：上述方法编码的是**绝对坐标** (row, col)，但对于图像来说，**相对位置关系**往往更重要：
- "猫在沙发左边" — 需要知道猫和沙发的相对位置
- 物体的空间关系决定了场景布局

**解决方向**：
- 引入相对位置编码
- 使用 2D 版本的 RoPE（但相对复杂）

---

## 4. 3DPE：自动驾驶的 3D 位置编码

### 4.1 背景与 Motivation

**问题**：自动驾驶中，我们需要从多个 2D 相机图像重建 3D 空间信息。

**核心矛盾**：

```
┌────────────────────────────────────────────────────────────────┐
│            多视角 3D 检测的核心矛盾                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  DETR3D:  3D query → 反复投影回 2D 采样 → 投影误差累积         │
│                                                                │
│  BEV类:   2D 图像 → 显式升维到 BEV → Z轴量化误差               │
│                                                                │
│  PETR:    2D 图像 + 3D坐标 → 3DPE融合 → 直接在 3D空间交互       │
│                    ↑                                          │
│              核心洞察：将 3D 几何信息编码进 2D 特征             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**核心洞察**：与其在 2D 图像上做复杂的投影采样，不如直接利用相机的几何参数（内参+外参），将每个像素的 3D 位置信息编码进特征。

这样，注意力机制就能直接在 3D 空间工作，无需降维到 BEV 或反复投影。

### 4.2 PETR 3DPE 详解

#### 流程概述

```
相机图像 (H, W, 3)
    ↓ 图像编码器
2D 特征 (H, W, C)
    ↓ 相机几何变换
3D 坐标 (H, W, D, 3)  ← 每个像素在 3D 空间的射线点
    ↓ MLP 编码
3DPE (H, W, D, C)
    ↓ 逐元素相加
3D 位置感知特征 (H, W, D, C)
```

#### 核心思想

**Step 1**：将相机锥台空间离散化为 3D 网格

对于相机的每个像素 $(u, v)$，沿深度方向采样 $D$ 个点：

$$
(u \cdot d, v \cdot d, d), \quad d \in [d_{min}, d_{max}]
$$

**Step 2**：逆投影到 3D 世界坐标

利用相机内外参，将锥台点投影到世界坐标系：

$$
P_{world} = K^{-1} \cdot [R|t]^{-1} \cdot (u \cdot d, v \cdot d, d, 1)^T
$$

**Step 3**：MLP 编码为位置嵌入

$$
\text{3DPE} = \text{MLP}(\text{normalize}(P_{world}))
$$

### 4.3 PETR 3DPE 伪代码

```python
import torch
import torch.nn as nn
import numpy as np


class PETR_3DPE(nn.Module):
    """
    PETR 的 3D 位置编码器
    核心: 将 3D 世界坐标通过 MLP 编码为位置嵌入
    """

    def __init__(self, embed_dims: int = 256):
        super().__init__()
        self.embed_dims = embed_dims

        # 2层MLP生成3D位置编码
        self.pe_mlp = nn.Sequential(
            nn.Linear(3, embed_dims),    # 输入: (x, y, z)
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )

    def normalize_3d_coords(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        归一化 3D 坐标到 [0, 1]

        输入: points_3d (..., 3)  未归一化的 xyz
        返回: (..., 3) 归一化后的坐标
        """
        # nuScenes 坐标系范围:
        # x: [-50, 50] → (x + 50) / 100
        # y: [-50, 50] → (y + 50) / 100
        # z: [-5,  3]  → (z + 5) / 8
        x_norm = (points_3d[..., 0] + 50.0) / 100.0
        y_norm = (points_3d[..., 1] + 50.0) / 100.0
        z_norm = (points_3d[..., 2] + 5.0) / 8.0

        return torch.stack([x_norm, y_norm, z_norm], dim=-1)

    def forward(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        输入: points_3d (H, W, D, 3)  3D世界坐标
        返回: (H, W, D, embed_dims)   3D位置编码
        """
        # 归一化
        coords_norm = self.normalize_3d_coords(points_3d)  # (H, W, D, 3)

        # MLP编码
        pe_3d = self.pe_mlp(coords_norm)  # (H, W, D, embed_dims)

        return pe_3d


def generate_3d_world_points(
    cam_params: torch.Tensor,
    H: int, W: int, D: int = 64
) -> torch.Tensor:
    """
    生成 3D 世界坐标点 (PETR Step 1-2)

    输入:
        cam_params: (4, 4) 相机投影矩阵 P = K @ RT @ ego2world
        H, W: 特征图尺寸
        D: 深度采样数

    返回:
        points_3d_world: (H, W, D, 3) 3D世界坐标
    """
    # Step 1: 生成像素网格和深度
    u = torch.linspace(0, W-1, W)
    v = torch.linspace(0, H-1, H)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # (H, W)

    d = torch.linspace(1.0, 60.0, D)  # PETR 深度范围 1~60m
    uu = uu.unsqueeze(-1).repeat(1, 1, D)  # (H, W, D)
    vv = vv.unsqueeze(-1).repeat(1, 1, D)
    dd = d.view(1, 1, D).repeat(H, W, 1)    # (H, W, D)

    # Step 2: 锥台点 (u*d, v*d, d, 1)
    points_frustum = torch.stack([
        uu * dd,   # x = u * d
        vv * dd,   # y = v * d
        dd,        # z = d
        torch.ones_like(dd)
    ], dim=-1)  # (H, W, D, 4)

    # Step 3: 逆投影到世界坐标
    P_inv = torch.inverse(cam_params)  # (4, 4)
    points_3d_world = torch.matmul(points_frustum, P_inv.T)  # (H, W, D, 4)
    points_3d_world = points_3d_world[..., :3]  # 去掉齐次项 (H, W, D, 3)

    return points_3d_world
```

### 4.4 PETRv2 改进：FPE（Feature-Guided Position Encoding）

PETR 的 3DPE 是 **data-independent**（仅与几何有关），PETRv2 提出 FPE 使其 **data-dependent**：

```python
class PETRv2_FPE(nn.Module):
    """
    Feature-Guided Position Encoding

    PE(t) = σ(F(t)) ⊙ ψ(P(t))
           ──────    ──────
           数据驱动   几何编码
    """

    def __init__(self, d_model: int):
        super().__init__()

        # 2D特征 → 注意力权重分支
        self.feature_branch = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid()  # 注意力权重
        )

        # 3D坐标 → 基础PE分支 (同PETR)
        self.coord_branch = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, img_feature: torch.Tensor,
                points_3d: torch.Tensor) -> torch.Tensor:
        """
        输入:
            img_feature: (B, C, H, W) 2D图像特征
            points_3d: (B, H, W, D, 3) 3D世界坐标

        返回:
            pe_fpe: (B, H, W, D, C) 数据驱动的3D PE
        """
        B, C, H, W = img_feature.shape
        D = points_3d.shape[3]

        # 特征分支: 1×1 Conv + Sigmoid → 注意力权重
        attention_weight = self.feature_branch(img_feature)
        attention_weight = attention_weight.view(B, C, H, W)

        # 坐标分支: 归一化 + MLP → 基础3DPE
        x_norm = (points_3d[..., 0] + 50.0) / 100.0
        y_norm = (points_3d[..., 1] + 50.0) / 100.0
        z_norm = (points_3d[..., 2] + 5.0) / 8.0
        coords_norm = torch.stack([x_norm, y_norm, z_norm], dim=-1)
        pe_base = self.coord_branch(coords_norm)

        # 逐元素乘法: 数据驱动的3DPE
        attn = attention_weight.permute(0, 2, 3, 1).unsqueeze(3)  # (B, H, W, 1, C)
        pe_fpe = attn * pe_base  # (B, H, W, D, C)

        return pe_fpe
```

### 4.5 PETRv2 时序融合：3D 坐标对齐

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PETRv2 时序融合: 3D坐标对齐                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   帧 t-1                     帧 t                                   │
│   ┌─────────┐               ┌─────────┐                           │
│   │ 3D点    │               │ 3D点    │                           │
│   │ P(t-1)  │ ──对齐变换──→ │ P(t)   │                           │
│   └─────────┘               └─────────┘                           │
│                                                                     │
│   对齐变换矩阵:                                                      │
│   T(t-1→t) = T(e(t)→l(t)) · T(g→e(t)) · T(g→e(t-1))⁻¹ · T(e(t-1)→l(t-1))⁻¹ │
│                                                                     │
│   对齐后: 帧t-1的3D坐标 → 帧t的坐标系 → concat特征                  │
│                                                                     │
│   【关键洞察】                                                       │
│   • 仅对齐3D坐标，不变换特征                                         │
│   • 时序信息由3DPE自动携带                                          │
│   • 无需显式BEV变换                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. mRoPE：多维旋转位置编码

### 5.1 背景与 Motivation

**问题**：RoPE 只能处理 1D 序列位置，但在多模态任务中，位置往往是多维的：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        位置编码维度扩展                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  NLP (1D):      token_1, token_2, token_3, ...                    │
│                  └──────┬──────┘                                   │
│                        位置只用1个数字: pos                         │
│                                                                     │
│  Vision (2D):    pixel(x, y)                                       │
│                  └──┬──┘                                          │
│                     位置需要2个数字: (row, col)                     │
│                                                                     │
│  Video (3D):     pixel(x, y, t)                                    │
│                  └─┬─┘ └──┬──┘                                    │
│                    空间      时间                                    │
│                                                                     │
│  Autonomous:    lidar_point(x, y, z) + temporal                    │
│                  └─────┬─────┘ └─────┬─────┘                       │
│                      空间           时间                            │
│                                                                     │
│  RoPE: 只能处理 1D位置 → mRoPE: 处理多维位置                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**核心洞察**：RoPE 的旋转编码可以**维度分解**——每个维度独立旋转，最后拼接。

### 5.2 mRoPE 核心思想

```python
class mRoPE(nn.Module):
    """
    Multi-dimensional RoPE

    为每个维度分配独立的旋转轴:
    - 空间H: θ_H
    - 空间W: θ_W
    - 时间T: θ_T
    """

    def __init__(self, d_model: int, num_dims: int = 3):
        super().__init__()
        self.num_dims = num_dims
        self.d_model = d_model
        self.dim_per_mode = d_model // num_dims  # 每维度分配的维度

        # 每维度独立的角度基数
        self.theta_H = 10000.0
        self.theta_W = 10000.0
        self.theta_T = 10000.0

    def precompute_freqs_cis(
        self, H: int, W: int, T: int
    ) -> tuple:
        """
        预计算多维旋转角度
        """
        dim = self.dim_per_mode

        # H维度
        freqs_H = torch.arange(0, dim, 2).float() / dim
        pos_H = torch.arange(H)
        freqs_H = torch.outer(pos_H, self.theta_H ** freqs_H)
        freqs_H_cis = torch.polar(torch.ones_like(freqs_H), freqs_H)

        # W维度
        freqs_W = torch.arange(0, dim, 2).float() / dim
        pos_W = torch.arange(W)
        freqs_W = torch.outer(pos_W, self.theta_W ** freqs_W)
        freqs_W_cis = torch.polar(torch.ones_like(freqs_W), freqs_W)

        # T维度
        freqs_T = torch.arange(0, dim, 2).float() / dim
        pos_T = torch.arange(T)
        freqs_T = torch.outer(pos_T, self.theta_T ** freqs_T)
        freqs_T_cis = torch.polar(torch.ones_like(freqs_T), freqs_T)

        return freqs_H_cis, freqs_W_cis, freqs_T_cis

    def apply_mrope(
        self, x: torch.Tensor,           # (B, H, W, T, d_model)
        freqs_H_cis, freqs_W_cis, freqs_T_cis
    ) -> torch.Tensor:
        """
        应用多维RoPE

        三个维度的旋转独立计算后组合
        """
        B, H, W, T, d = x.shape
        dim = self.dim_per_mode

        x_H = x[..., :dim]      # H维度编码
        x_W = x[..., dim:2*dim]  # W维度编码
        x_T = x[..., 2*dim:]     # T维度编码

        # 复数形式
        x_H = torch.view_as_complex(x_H.float().reshape(*x_H.shape[:-1], -1, 2))
        x_W = torch.view_as_complex(x_W.float().reshape(*x_W.shape[:-1], -1, 2))
        x_T = torch.view_as_complex(x_T.float().reshape(*x_T.shape[:-1], -1, 2))

        # 独立旋转
        x_H_rot = torch.view_as_real(x_H * freqs_H_cis.unsqueeze(0))
        x_W_rot = torch.view_as_real(x_W * freqs_W_cis.unsqueeze(0))
        x_T_rot = torch.view_as_real(x_T * freqs_T_cis.unsqueeze(0))

        # 拼接
        x_rot = torch.cat([x_H_rot, x_W_rot, x_T_rot], dim=-1)

        return x_rot.type_as(x).reshape(B, H, W, T, d)
```

### 5.3 mRoPE-I 在 DVGT-2 中的应用

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DVGT-2 滑动窗口流式架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  时间步 t=0          时间步 t=1          时间步 t=2          ...   │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐               │
│  │ Frame 0 │        │ Frame 1 │        │ Frame 2 │               │
│  │   F_0   │        │   F_1   │        │   F_2   │               │
│  └────┬────┘        └────┬────┘        └────┬────┘               │
│       │                  │                  │                      │
│       ▼                  ▼                  ▼                      │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐               │
│  │ 编码器   │   +    │ KV Cache│   +    │ KV Cache│               │
│  │ (含mRoPE)│        │ (FIFO)  │        │ (FIFO)  │               │
│  └────┬────┘        └────┬────┘        └────┬────┘               │
│       │                  │                  │                      │
│       └──────────────────┼──────────────────┘                      │
│                          ▼                                        │
│              Temporal Causal Attention                             │
│              (MRoPE-I 相对时序位置编码)                              │
│                                                                     │
│  【MRoPE-I 关键】                                                   │
│  • Query = 当前帧特征 (F_t)                                         │
│  • Key/Value = 历史缓存 [F_{t-W}, ..., F_{t-1}]                    │
│  • mRoPE-I 保证: 相对时间位置编码正确                                │
│  • 缓存特征可跨时间步复用                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 mRoPE 维度分解可视化

```
┌─────────────────────────────────────────────────────────────────────┐
│                        mRoPE 维度分解                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  假设 d_model = 9, num_dims = 3, 则每维度 = 3                      │
│                                                                     │
│  输入向量 x: [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]         │
│              └─────┘    └─────┘    └─────┘                         │
│                H维度     W维度      T维度                           │
│                                                                     │
│  旋转操作:                                                          │
│  x_H' = x_H × R(θ_H × pos_H)  (仅H维度旋转)                        │
│  x_W' = x_W × R(θ_W × pos_W)  (仅W维度旋转)                        │
│  x_T' = x_T × R(θ_T × pos_T)  (仅T维度旋转)                        │
│                                                                     │
│  输出拼接: x' = concat([x_H', x_W', x_T'])                         │
│                                                                     │
│  【关键性质】                                                        │
│  • 三维度旋转互不干扰                                                │
│  • 相对位置 = pos_H - pos_H' + pos_W - pos_W' + pos_T - pos_T'     │
│  • 远程衰减在每个维度独立生效                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 位置编码全景对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        位置编码方法全面对比                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┬────────┬────────┬────────┬────────┬────────┬────────┐     │
│  │   方法        │ 绝对   │ 相对   │ 外推性 │ 参数   │ 维度   │ 典型   │     │
│  │              │ 位置   │ 位置   │        │        │ 支持   │ 应用   │     │
│  ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────┤     │
│  │ Sinusoidal   │   ✅   │   ❌   │  好    │  无    │  1D    │ Transformer│ │
│  │ Learned      │   ✅   │   ❌   │  差    │  有    │  任意  │ 图像   │     │
│  │ Relative PE  │   ❌   │   ✅   │  一般  │  有    │  1D    │ Shaw'18│     │
│  │ ALiBi        │   ❌   │   ✅   │ 很好   │  有    │  1D    │ 短训练 │     │
│  │ RoPE         │   ✅   │   ✅   │  好    │  无    │  1D    │ LLaMA  │     │
│  │ 2D PE        │   ✅   │   ❌   │  一般  │  有    │  2D    │ ViT    │     │
│  │ 3DPE         │   ✅   │   ❌   │  一般  │  有    │  3D    │ PETR   │     │
│  │ mRoPE        │   ✅   │   ✅   │  好    │  无    │  多维  │ DVGT-2 │     │
│  └──────────────┴────────┴────────┴────────┴────────┴────────┴────────┘     │
│                                                                             │
│  【演进趋势】                                                              │
│  1. 从绝对→相对→混合：兼顾表达能力和泛化                                   │
│  2. 从有参数→零参数：减少过拟合风险                                       │
│  3. 从单维度→多维度：支持视觉、视频、多模态                                 │
│  4. 从外推困难→自然外推：三角函数/线性偏置                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 面试回答模板

### Q1: 为什么 Transformer 需要位置编码？

**答**：因为 Self-Attention 的本质是**置换等价**的——无论 token 出现在序列的哪个位置，它们的交互方式完全相同。如果不注入位置信息，"狗咬人"和"人咬狗"会被模型认为是相同的序列。位置编码让模型能够区分 token 的绝对位置和相对位置。

### Q2: Sinusoidal 和 Learned PE 有什么区别？

**答**：Sinusoidal 用正弦余弦函数编码位置，无需参数，外推性好，但无法学习相对位置；Learned PE 将位置作为可学习参数，表达能力强，但外推性差（超出训练长度无法处理）。实际中常根据任务选择——如果任务对位置规律有特定先验，Sinusoidal 可能更好；如果需要完全自由学习，Learned PE 更合适。

### Q3: RoPE 的核心优势是什么？

**答**：RoPE 解决了三个核心问题：① **绝对位置编码通过旋转实现，无需参数**，避免过拟合；② **attention score 仅依赖相对距离** $(m-n)$，自动捕获相对位置关系；③ **具有远程衰减特性**，距离越远注意力自然衰减；④ **外推性好**，三角函数可自然外推到更长序列。这也是为什么 LLaMA、Qwen、Gemma 等主流大模型都选择 RoPE。

### Q4: 3DPE 在自动驾驶中解决什么问题？

**答**：3DPE 将相机几何（内参+外参+深度）编码进 2D 图像特征，让 DETR 类方法可以直接在 3D 空间做目标检测。具体流程是：相机锥台→逆投影→3D世界坐标→MLP编码→与2D特征融合。相比 BEV 方法，3DPE 避免了 Z 轴量化误差；相比 DETR3D，避免了反复投影采样误差累积。

### Q5: mRoPE 和 RoPE 的区别？为什么要用 mRoPE？

**答**：RoPE 只能处理 1D 序列位置，mRoPE 将其扩展到多维（空间 H×W + 时间 T），每个维度独立分配旋转轴，各维度旋转互不干扰。在 DVGT-2 中，mRoPE-I 通过相对时序位置编码，让 KV Cache 中的历史帧特征可以正确与当前帧交互，同时保持因果性（未来帧不 attend 过去帧）。

### Q6: ALiBi 和 RoPE 都能做相对位置编码，哪个更好？

**答**：各有优劣。RoPE 用旋转矩阵，数学上更优雅（满足 $SO(d)$ 群性质），且零参数；ALiBi 用线性偏置，外推性更好（理论上可处理任意长度），但需要手动设置斜率。实践中 RoPE 应用更广（LLaMA、Qwen、Gemma），因为它更简洁且无需调参；ALiBi 在训练短、推理长的场景有优势（如 StreamingLLM）。

---

## 8. 参考文献

| 论文 | 年份 | 关键贡献 |
|------|------|---------|
| Attention Is All You Need (Vaswani) | 2017 | Sinusoidal PE, Transformer |
| Shaw et al. | 2018 | 相对位置编码 (RPE) |
| ANPE | 2021 | 绝对位置编码的相对位置扩展 |
| ALiBi (Press et al.) | 2021 | 线性偏置，远程衰减，外推性 |
| RoPE (Su et al.) | 2021 | 旋转位置编码，零参数 |
| PETR | 2022 | 3DPE，自动驾驶多视图3D检测 |
| PETRv2 | 2022 | FPE，3D坐标对齐时序融合 |
| DVGT-2 | 2026 | mRoPE-I，时序因果注意力 |

---

*文档整理日期: 2026-04-19*

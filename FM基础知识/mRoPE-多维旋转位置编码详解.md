# m-RoPE 多维旋转位置编码详解

> 本文档系统讲解 m-RoPE (Multi-dimensional Rotary Positional Embedding) 的核心机制、代码实现与数学原理。
> 基于 DVGT-2 / Latent-WAM 等论文中的实际代码。

---

## 1. 核心问题：为什么需要 m-RoPE？

### 1.1 RoPE 的局限

标准 RoPE 只能处理 **1D 序列位置**：

```
RoPE: token_1, token_2, token_3, ...
       └──────┬──────┘
         位置只用1个数字: pos
```

但在多模态视频/自动驾驶中，位置是**多维**的：

```
Video (3D): pixel(x, y, t)
            └─┬─┘ └──┬──┘
             空间     时间

Autonomous Driving: lidar_point(x, y, z) + temporal
                     └─────┬─────┘ └─────┬─────┘
                       空间            时间
```

### 1.2 m-RoPE 的核心洞察

**RoPE 的旋转编码可以维度分解**——每个维度独立旋转，最后拼接。

```
m-RoPE: 为每个维度分配独立的旋转轴
         - 空间H: θ_H
         - 空间W: θ_W
         - 时间T: θ_T
```

---

## 2. 核心概念：逆频率 (Inverse Frequency)

### 2.1 数学定义

逆频率决定了不同特征维度旋转的快慢：

$$
\Theta_i = \text{base}^{-\frac{2i}{d}}
$$

代码实现：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
# 其中 base = 10000, head_dim = 128
```

### 2.2 物理直觉：时钟指针隐喻

将 128 维 Token 特征想象成一个**拥有很多根指针的时钟**：

| 维度 | 逆频率 Θᵢ | 比喻 | 作用 |
|------|-----------|------|------|
| 前排（i 小）| Θᵢ ≈ 1 | 秒针 | 坐标微变 → 角度剧变，捕捉**局部细节** |
| 后排（i 大）| Θᵢ ≈ 0.0001 | 时针 | 跨越长距离才转动，捕捉**全局结构** |

### 2.3 为什么这样设计？

不同频率的旋转角度对位置移动的敏感度不同：

- **高频维度（i 小）**：稍微移动位置（如 pos 1→2），旋转角度变化剧烈 → 对**相邻 token** 敏感
- **低频维度（i 大）**：需要跨越很长距离，角度才明显变化 → 能感知**长距离依赖**

---

## 3. 维度交织 (Interleaving)：时空感知的均衡

### 3.1 问题：为什么要交织？

假设 head_dim = 128（64 对），mrope_section = [24, 20, 20]（T、H、W 各分配多少）：

**如果简单拼接（按顺序塞）**：
```
前24个坑 → T（全占高频）
中间20个 → H
最后20个 → W（只能占低频）
```
**灾难性后果**：T 霸占所有高频维度，对帧间微小变化极其敏感；W 只能捡极低频，对画面左右细节"近视"。

### 3.2 交织的魔法

按照 **T, H, W, T, H, W...** 的顺序**穿插分配**：

```
坑位: [T, H, W, T, H, W, T, H, W, ...]
索引:  0  1  2  3  4  5  6  7  8  ...
```

**结果**：T、H、W 在高频区和低频区都各有份额，保证模型在时空三个维度具有**均衡的感知能力**。

### 3.3 代码实现

```python
def apply_transformation(self, freqs, mrope_section):
    """
    freqs: (3, bs, seq_len, head_dim // 2) - 分别对应 T, H, W 的频率
    mrope_section: (3,) - [T分配的对数, H分配的, W分配的]
    """
    freqs_t = freqs[0].clone()  # 先全部填 T

    for dim, offset in enumerate((1, 2), start=1):  # dim=1 是 H, dim=2 是 W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)  # 步长3，偏移不同
        freqs_t[..., idx] = freqs[dim, ..., idx]  # 用 H/W 替换对应位置

    return freqs_t  # Shape: (bs, seq_len, head_dim // 2)
```

**以 mrope_section = [24, 20, 20] 为例**：

| 步骤 | 操作 | 结果 |
|------|------|------|
| 初始 | 全填 T | `[T, T, T, T, ...]` (64个) |
| dim=1 (H) | slice(1, 60, 3) → 替换奇数位 | `[T, H, T, T, H, T, ...]` |
| dim=2 (W) | slice(2, 60, 3) → 替换偶数位 | `[T, H, W, T, H, W, ...]` |

---

## 4. 代码实现详解

### 4.1 位置索引生成：get_mrope_interleave_index

```python
def get_mrope_interleave_index(video_grid_thw: Tuple[int], device, T_start: int = 0):
    """
    Input:  video_grid_thw (Tuple): Shape [B, T, V, H, W]
    Output: position_ids (Tensor): Shape [3, B*V, Seq_Len]
    """
    b, t, v, h, w = video_grid_thw

    # T 维度: [0, 1, 2, ..., t-1]
    t_index = torch.arange(T_start, T_start+t, device=device).view(-1, 1).expand(-1, h*w).flatten()

    # H 维度: [0, 0, ..., 1, 1, ...]  (h 组，每组重复 w 次)
    h_index = torch.arange(h, device=device).view(1, -1, 1).expand(t, -1, w).flatten()

    # W 维度: [0, 1, ..., 0, 1, ...]  (每组内重复)
    w_index = torch.arange(w, device=device).view(1, 1, -1).expand(t, h, -1).flatten()

    # 堆叠: (3, T*H*W)
    mm_pos_ids = torch.stack([t_index, h_index, w_index])

    # 扩展到 batch 和 view 维度
    position_ids = mm_pos_ids.unsqueeze(1).expand(-1, b * v, -1)

    # reshape: (3, B*T*V, H*W)
    return rearrange(position_ids, 'd (b v) (t h w) -> d (b t v) (h w)', b=b, t=t, v=v, h=h, w=w).clone()
```

### 4.2 旋转_emb 生成：完整 Shape 追踪

假设配置：
- B=2, T=1024, head_dim=128
- mrope_section = [24, 20, 20]

**Step 1**: 计算逆频率
```python
inv_freq = 1.0 / (10000 ** (torch.arange(0, 128, 2) / 128))
# Shape: [64]  (即 D = head_dim // 2)
```

**Step 2**: 扩展逆频率
```python
inv_freq_expanded = inv_freq[None, None, :, None].expand(3, B, -1, 1)
# Shape: [3, 2, 64, 1]
```

**Step 3**: 扩展位置坐标
```python
position_ids_expanded = position_ids[:, :, None, :].float()
# Shape: [3, 2, 1, 1024]
```

**Step 4**: 外积计算旋转角度
```python
freqs = inv_freq_expanded @ position_ids_expanded  # 矩阵外积
freqs = freqs.transpose(2, 3)
# Shape: [3, 2, 1024, 64]
```

**Step 5**: 多维交织
```python
freqs_t = apply_transformation(freqs, mrope_section)
# Shape: [2, 1024, 64]  (外层 3 被消解，T/H/W 交织进 64 维)
```

**Step 6**: 生成 cos/sin
```python
emb = torch.cat((freqs_t, freqs_t), dim=-1)
cos, sin = emb.cos(), emb.sin()
# Shape: [2, 1024, 128]
```

### 4.3 最终公式推导

对于坐标为 $(t, h, w)$ 的 Token，最终 emb 向量为：

$$
emb = \Big[ V_0, V_1, V_2, \dots, V_{63}, \quad V_0, V_1, V_2, \dots, V_{63} \Big]
$$

其中每个 $V_i = C(i) \cdot \Theta_i$：

$$
emb = \Big[ t\Theta_0, h\Theta_1, w\Theta_2, t\Theta_3, h\Theta_4, w\Theta_5, \dots, t\Theta_{60}, t\Theta_{61}, t\Theta_{62}, t\Theta_{63} \Big]
$$

**注意**：由于 slice 机制，H/W 各占 20 个，T 占 24 个（20 + 尾巴 4 个）。

---

## 5. rotate_half 与二维旋转矩阵

### 5.1 二维旋转的数学

对于点 $(x, y)$ 旋转角度 $\theta$：

$$
\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}
$$

展开后：
- $x' = x \cdot \cos\theta - y \cdot \sin\theta$
- $y' = x \cdot \sin\theta + y \cdot \cos\theta$

### 5.2 代码实现

```python
def rotate_half(x):
    """将向量后半部分取负号放到前面"""
    x1 = x[..., :x.shape[-1] // 2]      # 前半: [q_0, ..., q_63]
    x2 = x[..., x.shape[-1] // 2:]       # 后半: [q_64, ..., q_127]
    return torch.cat((-x2, x1), dim=-1)   # [-q_64, ..., -q_127, q_0, ..., q_63]
```

### 5.3 验证

RoPE 的应用公式：

$$
q_{out} = q \odot \cos(emb) + \text{rotate\_half}(q) \odot \sin(emb)
$$

**计算索引 0**（前半部分）：
- $q[0] = q_0$，$emb[0] = \theta_0$，$\text{rotate\_half}(q)[0] = -q_{64}$
- $q_{out}[0] = q_0 \cdot \cos(\theta_0) + (-q_{64}) \cdot \sin(\theta_0)$
- $= q_0 \cos\theta_0 - q_{64} \sin\theta_0$ ✅ 对应 $x'$

**计算索引 64**（后半部分）：
- $q[64] = q_{64}$，$emb[64] = \theta_0$，$\text{rotate\_half}(q)[64] = q_0$
- $q_{out}[64] = q_{64} \cdot \cos(\theta_0) + q_0 \cdot \sin(\theta_0)$
- $= q_0 \sin\theta_0 + q_{64} \cos\theta_0$ ✅ 对应 $y'$

---

## 6. mrope_section 与论文中 "frequency" 的对应

### 6.1 术语对照

| 代码配置 | 论文表述 | 物理含义 |
|---------|---------|---------|
| mrope_section[0] = 24 | Temporal freq 50 | 给时间轴分配多少个"钟表指针" |
| mrope_section[1] = 20 | Camera freq 10 | 给相机索引分配多少个"钟表指针" |
| mrope_section[2] = 20 | Token freq 100 | 给空间 Token 分配多少个"钟表指针" |

### 6.2 反推模型配置

论文说 "temporal freq 50, camera freq 10, token freq 100"：
- 总频率数 = $50 + 10 + 100 = 160$
- head_dim = $160 \times 2 = 320$

这说明该模型使用 **320 维 Attention Head**。

---

## 7. 面试回答模板

### Q1: m-RoPE 和 RoPE 的区别？

**答**：RoPE 只能处理 1D 序列位置；m-RoPE 将其扩展到多维（空间 H×W + 时间 T），每个维度独立分配旋转轴，各维度旋转互不干扰。在 DVGT-2 / Latent-WAM 中，mRoPE-I 通过相对时序位置编码，让 KV Cache 中的历史帧特征可以正确与当前帧交互，同时保持因果性。

### Q2: 为什么要用交织 (interleaving) 而不是简单拼接？

**答**：如果简单拼接（如前 24 个给 T，中间 20 个给 H），会导致 T 霸占所有高频维度，对时间变化极其敏感，而 H/W 只能占低频维度，变成"近视眼"。交织让 T、H、W 在高频区和低频区都各有份额，保证模型在时空三个维度具有均衡的感知能力。

### Q3: 逆频率的物理含义是什么？

**答**：逆频率决定了特征向量每个维度"旋转的快慢"。高频维度（i 小）像秒针，位置微变角度剧变，捕捉局部细节；低频维度（i 大）像时针，需跨越长距离才转动，捕捉全局结构。这种设计让模型能同时关注局部模式和长距离依赖。

### Q4: rotate_half 为什么要这样设计？

**答**：rotate_half 是为了让向量前后两半配对后进行二维旋转。前半 [-q_64, ..., -q_127, q_0, ..., q_63] 和 cos/sin 结合后，恰好实现公式 $q_{out} = q \odot \cos + \text{rotate\_half}(q) \odot \sin$，完美复现二维旋转矩阵的效果。

---

## 8. 参考代码

```python
# 核心模块: MRopeInterleaveEmbedding
class MRopeInterleaveEmbedding(nn.Module):
    def __init__(self, head_dim=128, mrope_section=[24, 20, 20], base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _compute_frequency_components(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_transformation(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    def apply_transformation(self, freqs, mrope_section):
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
```

---

*文档整理日期: 2026-04-22*
*来源: Gemini Chat m-RoPE 讲解对话
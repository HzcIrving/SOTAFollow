# DCT（离散余弦变换）详解

> **整理时间**：2026-04-16
> **本文用途**：为 MINT 论文精读提供频域基础知识支撑

---

## 1. Motivation（问题背景）

### 1.1 什么是 DCT？

**DCT（Discrete Cosine Transform，离散余弦变换）** 是数字信号处理中最常用的正交变换之一，广泛应用于图像/视频压缩（JPEG、MPEG）、音频编码（MP3）、以及本文的核心——**将动作轨迹从时间域变换到频率域**。

DCT 本质上是将一个离散信号分解为不同频率的**余弦函数的叠加**。与傅里叶变换（FFT）不同，DCT 只使用实数余弦函数，不需要复数运算，因此计算效率更高、数值更稳定。

### 1.2 为什么要用 DCT 分析动作轨迹？

在机器人动作表示中，**动作轨迹天然具有频域特性**：

| 动作成分 | 频率特征 | DCT 系数对应 |
|----------|----------|--------------|
| **意图/目标** | 低频 | k=0,1,2（全局趋势） |
| **动作过渡** | 中频 | k=3~10（姿态调整） |
| **精细力控/抖动** | 高频 | k>10（执行细节） |

**核心洞察**：如果低频 DCT 系数就能重建出有意义的轨迹轨迹，说明低频分量捕捉到了**行为意图**。这是 MINT 论文用 DCT 做动作 Tokenization 的核心动机。

### 1.3 Related Works：其他变换方法的问题

| 方法 | 问题 |
|------|------|
| **FFT（傅里叶变换）** | 需要复数运算，数值不稳定；正负频率对称，信息冗余 |
| **小波变换** | 频率分辨率固定，无法自适应调整；难以与现有深度学习框架结合 |
| **直接像素级表示** | 维度过高，无法捕捉动作意图层面的结构 |

DCT 解决了以上问题：全实数运算 + 能量压缩性好 + 固定频率基函数。

---

## 2. 一句话总结

**DCT（Discrete Cosine Transform，离散余弦变换）** 是数字信号处理中最常用的正交变换之一，广泛应用于图像/视频压缩（JPEG、MPEG）、音频编码（MP3）、以及本文的核心——**将动作轨迹从时间域变换到频率域**。

DCT 本质上是将一个离散信号分解为不同频率的**余弦函数的叠加**。与傅里叶变换（FFT）不同，DCT 只使用实数余弦函数，不需要复数运算，因此计算效率更高、数值更稳定。

---

## 二、一维 DCT 定义

给定长度为 $H$ 的离散信号 $\mathbf{a} = (a_{0}, a_{1}, \dots, a_{H-1})$，**一维 DCT-II**（最常用的 DCT 形式）定义为：

$$
F_k = \sum_{h=0}^{H-1} a_h \cos\left[\frac{\pi}{H}\left(h + \frac{1}{2}\right) k\right], \quad k = 0, 1, \dots, H-1
$$

其中：
- $F_{k}$ 是第 $k$ 个 DCT 系数（频率分量的幅度）
- $k=0$ 项称为**直流（DC）分量**，代表信号的均值/低频基线
- $k>0$ 项称为**交流（AC）分量**，$k$ 越大，频率越高

### 逆变换（IDCT）

原始信号可以通过逆变换完美重建：

$$
a_h = \frac{1}{H} F_0 + \frac{2}{H} \sum_{k=1}^{H-1} F_k \cos\left[\frac{\pi}{H}\left(h + \frac{1}{2}\right) k\right], \quad h = 0, 1, \dots, H-1
$$

---

## 三、DCT 的频率物理含义

DCT 的核心洞察是：**低频分量描述信号的全局趋势，高频分量描述局部细节**。

以机械臂关节角度轨迹为例：

| DCT 系数 | 频率 | 物理含义 | 对应 MINT |
|----------|------|----------|-----------|
| $k=0$（DC）| 最低 | 整个轨迹的平均关节角度 | **全局意图/姿态基准** |
| $k=1,2$ | 低频 | 动作的宏观运动方向/目标位置 | **行为意图（Intent）** |
| $k=3\sim10$ | 中频 | 动作的中间过程、姿态调整 | **过渡执行** |
| $k>10$ | 高频 | 抖动、噪声、精细力控 | **执行细节（Execution）** |

> **图示（DCT 频谱分解）：**
> ```
> 幅度
>   │████
>   │  ██  ██      ██                    ██
>   │    ████████████  ████████  █████████████████
>   └──────────────────────────────────────────────→ 频率 k
>   k=0    低频(k=1~5)        中频(k=6~20)     高频(k>20)
>   │        │                    │              │
>   │  Intent/意图             过渡执行       噪声/精细力控
> ```

**这就是 MINT 的核心洞察**：如果强制模型用少量低频 DCT 系数就能重建轨迹，就意味着捕捉到了**意图**——因为意图天然是低频的、全局的。

---

## 四、DCT 与其他变换的对比

| 特性 | DCT-II | DFT（FFT）| 小波变换 |
|------|---------|-----------|---------|
| 变换基函数 | 实数余弦 | 复数指数 | 小波函数 |
| 系数物理含义 | 固定频率 | 正/负频率 | 多尺度时间-频率 |
| 计算复杂度 | $O(H \log H)$ | $O(H \log H)$ | $O(H)$ |
| 边界处理 | 无需填充 | 需要周期延拓 | 依赖小波基 |
| 在 MINT 中的优势 | ✅ 实数计算、数值稳定 | ❌ 复数冗余 | ❌ 频率分辨率固定 |

**DCT 优于 FFT 的关键点**：
1. **全实数运算**：避免复数，数值更稳定
2. **能量压缩性好**：对于平滑信号，DCT 能用更少系数捕捉大部分能量
3. **边界连续性**：DCT 不需要像 FFT 那样做周期延拓，避免了人工边界效应

---

## 五、DCT 在 MINT 中的具体用法

MINT 将长度为 $H$、动作维度为 $D$ 的动作序列 $\mathbf{A} \in \mathbb{R}^{H \times D}$ 的每个维度分别做 DCT：

$$
\mathbf{F}_{k,d} = \sum_{h=0}^{H-1} \hat{\mathbf{A}}_{h,d} \cos\left[\frac{\pi}{H}\left(h + \frac{1}{2}\right) k\right], \quad k=0,\dots,H-1;\; d=1,\dots,D
$$

其中 $\mathbf{F} \in \mathbb{R}^{H \times D}$ 是 DCT 变换后的频域表示。

### 为什么沿时间维度做 DCT？

1. **动作轨迹的时间结构天然适合频域分析**：动作的"意图"（目标位置、宏观路径）和"执行"（精细调整、振动）本身就分别对应低频和高频
2. **多自由度联合分析**：每个动作维度（关节角度/末端位置）单独做 DCT，保留了动作间的时序相关性
3. **可控的压缩粒度**：保留前 $K$ 个 DCT 系数就能重建平滑轨迹，适合设计多尺度量化

### DCT 系数的多尺度分配（对应 MINT 的 S₁~Sₖ）

MINT 中 S₁ 只用低频 DCT 系数（约 1/8 的系数），S₂~Sₖ 则逐步包含更高频系数：

```python
import numpy as np

def dct_decompose(action_chunk, n_scales=4):
    """
    将动作块分解为多尺度 DCT 系数
    action_chunk: [H, D] - H个时间步, D个动作维度
    """
    H, D = action_chunk.shape
    F = dct(action_chunk, type=2, axis=0, norm='ortho')  # 沿时间轴做DCT

    scales = {}
    total_coeffs = H

    for k in range(1, n_scales + 1):
        # 尺度k保留前 n_k 个系数，n_k 按指数增长
        n_k = int(total_coeffs * (k / n_scales) ** 2)
        scales[f'S{k}'] = F[:n_k, :]  # 保留最低频的n_k个系数

    return scales
    # S1: ~1/16系数 → 纯低频 → Intent
    # S2: ~1/4系数  → 中低频
    # S3: ~9/16系数 → 中高频
    # S4: 全系数    → 完整轨迹
```

---

## 六、代码实现

### 纯 Python/NumPy 实现（无依赖）

```python
import numpy as np

def dct(x, type=2, norm='ortho'):
    """
    一维 DCT-II 实现（基于 Numpy）

    Args:
        x: 输入信号 [H,]
        type: DCT 类型（2是最常用的）
        norm: 'ortho' 归一化使变换正交
    Returns:
        X: DCT 系数 [H,]
    """
    H = len(x)
    h = np.arange(H)
    # DCT-II 变换矩阵
    W = np.cos(np.pi * h[:, None] * (2 * h[None, :] + 1) / (2 * H))

    if norm == 'ortho':
        # 正交归一化
        X = np.sqrt(2.0 / H) * W @ x
        X[0] *= 1 / np.sqrt(2)  # DC分量特殊处理
    else:
        X = W @ x

    return X

def idct(X, type=2, norm='ortho'):
    """
    一维 IDCT（逆 DCT-II）实现
    """
    H = len(X)
    h = np.arange(H)

    if norm == 'ortho':
        x_normalized = X / (np.sqrt(2.0 / H))
        x_normalized[0] *= np.sqrt(2)
    else:
        x_normalized = X

    # IDCT 变换矩阵（DCT 矩阵的转置）
    W_inv = np.cos(np.pi * h[:, None] * (2 * h[None, :] + 1) / (2 * H))
    x = W_inv.T @ x_normalized

    return x
```

### 使用 scipy（生产推荐）

```python
from scipy.fft import dct, idct

def decompose_action_dct(action_chunk):
    """
    对动作块做 DCT 分解（生产级实现）

    Args:
        action_chunk: [H, D] numpy array
    Returns:
        F: [H, D] DCT 系数矩阵
    """
    # 沿时间轴（axis=0）对每个动作维度分别做 DCT-II
    F = dct(action_chunk, type=2, axis=0, norm='ortho')
    return F

def reconstruct_from_low_freq(F, keep_ratio=0.25):
    """
    只用最低频的部分系数重建信号（演示低频=意图）
    """
    H = F.shape[0]
    k = int(H * keep_ratio)  # 只保留前 k 个低频系数
    F_low = F.copy()
    F_low[k:] = 0  # 置零高频系数
    return idct(F_low, type=2, axis=0, norm='ortho')
```

---

## 七、DCT 的局限性

| 局限 | 说明 | MINT 中的处理方式 |
|------|------|------------------|
| **固定频率分辨率** | DCT 对所有频率使用相同的窗口大小，无法自适应调整 | 多尺度残差量化自然补偿 |
| **全局变换** | DCT 是全局变换，一个系数变化影响所有时间点 | 短窗口（chunks）缓解这一问题 |
| **正交性依赖窗口** | 需要填充确保信号边界连续 | MINT 使用重叠 chunks（sliding window）|
| **不保留时序顺序** | DCT 只分解频率信息，不保留时间顺序 | 与时域特征 concat 保留时序 |

---

## 八、参考链接

- **IEEE DCT 原始论文**：Ahmed, Natarajan & Rao, "Discrete Cosine Transform", IEEE Trans. on Computers, 1974
- **arXiv 论文引用**：[2] DCT 论文，IEEE Transactions on Computers, 2006（引自 MINT 论文）
- **scipy fft 文档**：https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html

---

*整理 by 优酱 🍃 | 2026-04-16*

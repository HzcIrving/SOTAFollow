# VQVAE 视觉 Tokenizer 详解

> 📅 创建时间：2026-04-13
> 🏷️ 分类：FM基础知识 / WorldModel / VQVAE
> 📌 标签：VQVAE、视觉Tokenizer、Codebook、World Model、视觉表征

---

## 1. Motivation（问题背景）

### 1.1 为什么 World Model 需要视觉 Tokenizer？

World Model 的核心任务是学习环境动态 $p(s_{t+1}|s_{t}, a_{t})$。在实际系统中，**原始像素帧 $s_{t}$ 维度极高**（一张 256×256 的 RGB 图像就有 196,608 维），直接在像素空间学习世界模型面临两大根本问题：

1. **计算不可行**：高维像素空间意味着巨大的计算和存储开销
2. **语义鸿沟**：像素级重建 loss 不利于模型学习高维语义表征

因此需要一个**视觉压缩器**将原始图像压缩到一个**低维、离散、可学习的隐空间**——这正是 **VQVAE（Vector Quantized Variational AutoEncoder）** 的核心使命。

### 1.2 VQVAE 在 World Model 中的位置

```
原始视频帧序列
    ↓
[Encoder] → 高维特征 h = f(x)  （连续向量，维度仍较高）
    ↓
[Quantization] → z = quantize(h)  （离散 token，来自码本）
    ↓
[Decoder] → 重构图像 x' = g(z)
    ↓
隐码序列 z_1, z_2, ..., z_T  → 送入 World Model 学习 p(z_{t+1}|z_t, a_t)
```

**关键洞察：** World Model 实际上是在**离散码本索引空间**而非像素空间进行规划与预测，大幅降低计算复杂度。

### 1.3 Related Works

| 方法 | 表征类型 | 局限性 |
|------|---------|--------|
| **直接像素级建模** | 连续 | 维度极高，计算不可行 |
| **VAE / β-VAE** | 连续 | 潜在空间不离散，难以与语言模型结合 |
| **AE（自动编码器）** | 连续 | 无量化，表征能力有限 |
| **扩散模型解码器** | 连续 | 计算量极高 |

**核心问题**：如何在保证表征质量的前提下，将图像压缩为**离散、低维**的 token 序列？

---

## 2. 背景：从像素到离散 token

### 1.1 为什么 World Model 需要视觉 Tokenizer？

World Model 的核心任务是学习环境动态 $p(s_{t+1}|s_{t}, a_{t})$。在实际系统中，**原始像素帧 $s_{t}$ 维度极高**（一张 256×256 的 RGB 图像就有 196,608 维），直接在像素空间学习世界模型面临两大根本问题：

1. **计算不可行**：高维像素空间意味着巨大的计算和存储开销
2. **语义鸿沟**：像素级重建 loss 不利于模型学习高维语义表征

因此需要一个**视觉压缩器**将原始图像压缩到一个**低维、离散、可学习的隐空间**——这正是 **VQVAE（Vector Quantized Variational AutoEncoder）** 的核心使命。

### 1.2 VQVAE 在 World Model 中的位置

```
原始视频帧序列
    ↓
[Encoder] → 高维特征 h = f(x)  （连续向量，维度仍较高）
    ↓
[Quantization] → z = quantize(h)  （离散 token，来自码本）
    ↓
[Decoder] → 重构图像 x' = g(z)
    ↓
隐码序列 z_1, z_2, ..., z_T  → 送入 World Model 学习 p(z_{t+1}|z_t, a_t)
```

**关键洞察：** World Model 实际上是在**离散码本索引空间**而非像素空间进行规划与预测，大幅降低计算复杂度。

---

## 3. VQVAE 原理详解

### 2.1 核心思想：用码本做离散化

VQVAE 的"灵魂三问"：

- **Encoder**：将图像压缩成什么？ → **连续特征向量** $h = f(x)$
- **Quantizer**：如何离散化？ → **最近邻查码本** $z = \arg\min_{j} \|h - e_{j}\|_{2}$
- **Decoder**：如何重构？ → **从码本向量重建图像** $x' = g(z)$

其中码本（Codebook）$\mathcal{E} = \{e_{1}, e_{2}, ..., e_K\}$ 包含 $K$ 个可学习的 $D$ 维嵌入向量。

### 2.2 量化过程（Quantization）

**本质：把连续向量映射到最近的码本向量**

给定 Encoder 输出 $h_{e}(x) \in \mathbb{R}^{D}$，量化操作：

$$
z_q(x) = e_k, \quad \text{其中} \quad k = \arg\min_j \|h_e(x) - e_j\|_2^2
$$

即在码本 $\mathcal{E}$ 中找到与 $h_{e}(x)$ 欧氏距离最近的码字 $e_{k}$。

**关键性质：**
- $z_{q}(x)$ 是**离散的**（来自有限码本）
- $z_{q}(x)$ 是**不可梯度回传的**（argmin 操作不可导）

### 2.3 梯度估计：Straight-Through Estimator (STE)

量化操作 $z_{q} = \text{quantize}(h_{e})$ 本身不可导（$z_{q}$ 对 $h_{e}$ 的梯度是 0 或 undefined）。

VQVAE 使用 **Straight-Through Estimator（直通估计器）** 解决这个问题：

$$
\frac{\partial \mathcal{L}}{\partial h_e} = \frac{\partial \mathcal{L}}{\partial z_q}
$$

即**前向传播时**用量化后的 $z_{q}$，**反向传播时**将 $z_{q}$ 的梯度直接复制给 $h_{e}$，绕过量化操作。

```python
class VectorQuantize(nn.Module):
    """
    VQVAE 量化层 + Straight-Through Estimator
    """
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.dim = dim           # 每个码字向量维度
        self.codebook_size = codebook_size  # 码本大小 K

        # 可学习的码本：K 个 D 维向量
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, h_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_e: Encoder 输出 [B, H, W, D] 或 [B, N, D]
        Returns:
            z_q: 量化后的隐码 [B, H, W, D]（用于 Decoder 输入）
            indices: 每个位置对应的码本索引 [B, H, W]（用于存储/传输）
            loss: VQ 损失（码本更新用）
        """
        # 将 h_e reshape 成 [B*H*W, D] 方便计算距离
        h_flatten = h_e.view(-1, self.dim)  # [B*H*W, D]

        # 计算每个特征向量到所有码字的欧氏距离
        # [B*H*W, D] - [K, D] → [B*H*W, K]
        distances = (
            torch.sum(h_flatten ** 2, dim=1, keepdim=True)  # [B*H*W, 1]
            + torch.sum(self.codebook.weight ** 2, dim=1)    # [K]
            - 2 * (h_flatten @ self.codebook.weight.T)        # [B*H*W, K]
        )

        # 最近邻索引（不可导，用于前向）
        with torch.no_grad():
            indices = torch.argmin(distances, dim=-1)  # [B*H*W]

        # 从码本中取出对应的码字
        z_q = self.codebook(indices)  # [B*H*W, D]

        # ========== Straight-Through Estimator ==========
        # 前向：用量化后的 z_q
        # 反向：z_q 的梯度直接传给 h_e（跳过量化操作）
        z_q = h_e + (z_q - h_e).detach()

        # 重排回原始形状
        z_q = z_q.view_as(h_e)
        indices = indices.view(h_e.shape[:-1])  # [B, H, W]

        # ========== VQ 损失 ==========
        # 目标：让 Encoder 输出 h_e 尽可能接近码本向量
        # 等价于最小化 ||h_e - sg[z_q]||^2
        # sg[] = stop_gradient（阻止梯度流向码本）
        vq_loss = torch.mean((h_e - z_q.detach()) ** 2)

        return z_q, indices, vq_loss
```

---

## 4. 完整架构与数学推导

### 3.1 VQVAE 完整架构

```
输入图像 x ∈ ℝ^{H×W×3}
    ↓
Encoder f_θ: ℝ^{H×W×3} → ℝ^{h×w×D}
    ↓（Resize/Conv 下采样若干次）
h_e = f_θ(x)  （Encoder 输出，连续特征图）
    ↓
VectorQuantize: ℝ^{h×w×D} → ℝ^{h×w×D}
    ↓
z_q = quantize(h_e)  （离散化特征）
    ↓
Decoder g_φ: ℝ^{h×w×D} → ℝ^{H×H×3}
    ↓
重构图像 x' ∈ ℝ^{H×W×3}
```

**下采样因子（Stride）**：通常 16~32 倍（如 256×256 → 16×16，压缩 256 倍）

### 3.2 损失函数推导

VQVAE 的训练目标由三项组成：

**① 重构损失（Reconstruction Loss）**
$$
\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2 = \|x - g_\phi(z_q)\|^2
$$
通常用 MSE（像素空间）或 Perceptual Loss（感知空间）。

**② VQ 损失（Codebook Loss）**
$$
\mathcal{L}_{\text{vq}} = \|h_e - \text{sg}[z_q]\|^2
$$
其中 $\text{sg}[\cdot]$ 是 stop_gradient 操作。这项损失**只更新码本**，让码本向量向 Encoder 输出靠近。

**③ 承诺损失（Commitment Loss）**
$$
\mathcal{L}_{\text{commit}} = \| \text{sg}[h_e] - z_q \|^2
$$
这项损失**只更新 Encoder**，防止 Encoder 输出偏离码本太远（避免"码本漂移"问题）。

**总损失：**
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{vq}} \mathcal{L}_{\text{vq}} + \lambda_{\text{commit}} \mathcal{L}_{\text{commit}}
$$

通常 $\lambda_{\text{vq}} = \lambda_{\text{commit}} = 1.0$ 或 $\lambda_{\text{vq}} = 0.25, \lambda_{\text{commit}} = 1.0$。

### 3.3 为什么 VQVAE 能学到有意义的表征？

直觉上，最近邻量化天然地做了一件优雅的事：**相似的图像块被映射到相同的码字**。

如果训练数据中频繁出现"猫的眼睛"这一视觉模式，Encoder 会学会把"猫眼"编码成接近 $e_{k}$ 的向量，而码本会学会让 $e_{k}$ 本身成为"猫眼"的良好表征。**码本在训练过程中自动学会了数据中的视觉基本模式（类似于视觉词典）。**

### 3.4 完整伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """残差块，用于 Encoder 和 Decoder"""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Encoder(nn.Module):
    """VQVAE Encoder：将图像压缩为低维特征图"""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # h=128, w=128
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),   # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            # h=64, w=64
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),    # /2
            nn.ReLU(inplace=True),
            # h=32, w=32
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            # h=32, w=32
            nn.Conv2d(hidden_dim, latent_dim, 4, stride=2, padding=1),   # /2 → /16 总计
            nn.ReLU(inplace=True),
            # 最终: h/16 × w/16 × latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """VQVAE Decoder：从离散码本向量重建图像"""
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # 输入: h/16 × w/16 × latent_dim
            nn.ConvTranspose2d(latent_dim, hidden_dim, 4, stride=2, padding=1),  # ×2
            nn.ReLU(inplace=True),
            # h/8 × w/8
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            # h/8 × w/8
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), # ×2
            nn.ReLU(inplace=True),
            # h/4 × w/4
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # ×2
            nn.ReLU(inplace=True),
            # h/2 × w/2
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1), # ×2
            # 最终: h × w × 3
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: [B, latent_dim, h, w]
        return self.net(z_q)


class VQVAE(nn.Module):
    """
    完整的 VQVAE 模型

    使用方式：
        model = VQVAE(codebook_size=8192, latent_dim=256)
        z_q, indices, loss = model(x)
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128,
                 latent_dim: int = 256, codebook_size: int = 8192,
                 commitment_cost: float = 1.0):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, in_channels)
        self.quantize = VectorQuantize(latent_dim, codebook_size)
        self.commitment_cost = commitment_cost

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            x_hat: 重构图像 [B, C, H, W]
            indices: 量化索引 [B, H//16, W//16]
            loss_dict: 各分项损失
        """
        # 1. Encoder
        h_e = self.encoder(x)  # [B, latent_dim, h, w]，h=H/16, w=W/16

        # 2. Vector Quantization（包含 STE）
        z_q, indices, vq_loss = self.quantize(h_e)  # z_q: [B, latent_dim, h, w]

        # 3. Decoder
        x_hat = self.decoder(z_q)  # [B, C, H, W]

        # 4. 计算各分项损失
        recon_loss = F.mse_loss(x_hat, x)
        commitment_loss = torch.mean((h_e.detach() - z_q) ** 2)
        vq_loss = vq_loss + self.commitment_cost * commitment_loss

        total_loss = recon_loss + vq_loss

        loss_dict = {
            "total": total_loss,
            "recon": recon_loss,
            "vq": vq_loss,
            "commitment": commitment_loss,
        }

        return x_hat, indices, loss_dict

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：返回离散码本索引（用于 World Model 输入）"""
        h_e = self.encoder(x)
        z_q, indices, _ = self.quantize(h_e)
        return indices  # [B, h, w] = [B, H//16, W//16]

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """解码：从码本索引还原图像（用于可视化）"""
        # indices: [B, h, w] → 扩展为 [B, h, w, latent_dim]
        B, h, w = indices.shape
        z_q = self.quantize.codebook(indices)  # [B, h, w, latent_dim]
        z_q = z_q.permute(0, 3, 1, 2)           # [B, latent_dim, h, w]
        return self.decoder(z_q)

    @torch.no_grad()
    def decode_from_latents(self, z_q: torch.Tensor) -> torch.Tensor:
        """直接解码隐变量（不经过量化）"""
        return self.decoder(z_q)
```

---

## 5. 为什么选 VQVAE 作为视觉 Tokenizer？

### 4.1 相比其他方案的优劣

| 方案 | 表征类型 | 计算量 | 表征质量 | World Model 适配度 | 代表工作 |
|------|---------|--------|---------|------------------|---------|
| **VQVAE** | 离散 | 中 | 高 | ⭐⭐⭐⭐⭐ | DT算了啥 |
| VAE / β-VAE | 连续 | 中 | 中高 | ⭐⭐⭐ | - |
| AE（自动编码器） | 连续 | 低 | 中 | ⭐⭐ | - |
| 扩散模型解码器 | 连续 | 极高 | 最高 | ⭐⭐ | Stable Diffusion |
| 直接用 ViT | 连续 | 高 | 高 | ⭐⭐⭐ | LLM 编码器 |
| 原始像素 | 无压缩 | 极高 | N/A | ⭐ | Baseline |

### 4.2 离散表征的核心优势

**① 与语言模型天然对齐**

LLM 的 tokens 是离散的。World Model 如果使用离散视觉 token，可以用**统一的自回归建模**：

$$
p(z_{t+1}|z_t, a_t) = \text{Softmax}(W \cdot \text{Transformer}(z_t, a_t))
$$

这让我们可以直接复用 LLM 的预训练成果（如 Attention、MoE 等技术）。

**② 压缩效率高**

一张 256×256 的图像（196,608 维）被压缩为 h×w 的离散 token（h=w=16 时仅 256 个整数），压缩比可达 **768:1**。

**③ 码本容量可控**

码本大小 K 决定视觉词表规模。K 太小→表征能力不足；K 太大→类似连续表征、梯度估计困难。典型值 K=8192~65536。

**④ 想象力生成的天然接口**

离散 token 可以用类似语言模型的方式做"想象式生成"：从当前 $z_{t}$ 自回归地生成 $z_{t+1}, z_{t+2}, ...$，然后用 Decoder 可视化。

### 4.3 VQVAE 的已知局限

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| **码本崩溃（Codebook Collapse）** | 大量码字从未被使用 | EMA 更新策略 / 随机重启 |
| **梯度估计误差** | STE 忽略了量化点的曲率 | 软量化（Soft VQ） |
| **表征能力上限** | 受码本大小限制 | 残差码本（FSQ）、多尺度码本 |
| **训练不稳定** | 码本和 Encoder 同步更新困难 | 分离更新（延迟更新 Encoder） |

---

## 6. 码本原理与优势深度剖析

### 5.1 码本是什么？

**码本（Codebook）** 是一个可学习的查找表 $\mathcal{E} = \{e_{1}, e_{2}, ..., e_K\}, e_{k} \in \mathbb{R}^{D}$。

- $K$ = **码本大小**（视觉词表容量）
- $D$ = **码字维度**（每个码字的向量维度）
- 总参数量：$K \times D$（如 K=8192, D=256 时约 8M 参数）

### 5.2 码本的工作流程

```
训练阶段：
  Encoder(x) → h_e
      ↓
  在码本中找最近邻: k = argmin_j ||h_e - e_j||^2
      ↓
  z_q = e_k  （量化）
      ↓
  Decoder(z_q) → x_hat
      ↓
  计算损失 → 更新 Encoder / Decoder / 码本

推理阶段：
  x → Encoder → h_e → 量化 → indices → 存储/传输
  indices → 从码本取向量 → Decoder → x_hat
```

### 5.3 码本的优势

**① 语义聚类能力**

相似的视觉模式会被量化为同一个码字，码本自动学会将相似的图像块聚类到同一个码字下。这类似于自然语言处理中的 BPE（Byte Pair Encoding）——只不过 BPE 切分文本，VQVAE 切分视觉特征。

**② 高效存储与传输**

存储 indices（整数）而非连续向量，大幅降低存储和带宽需求。256×256 图像只需 16×16=256 个 int16，总计 512 字节 vs 原始像素 196,608 字节。

**③ 天然支持自回归生成**

World Model 输出的是**离散 token ID**（码字索引），可以直接作为下一个 VQVAE Decoder 的输入来可视化生成的图像——无需额外的解码器设计。

**④ 稳定的训练目标**

码本提供的是**固定的、非参数化**的表征锚点，Decoder 始终从已知码本向量重建，训练相对稳定。

### 5.4 码本的高级训练策略

#### 策略一：EMA 更新（Exponential Moving Average）

传统 SGD 更新码本容易导致不稳定。EMA 策略用滑动平均更新码本：

$$
e_k^{new} = \lambda \cdot e_k^{old} + (1-\lambda) \cdot \text{mean}(h_e \text{ assigned to } k)
$$

典型 $\lambda = 0.99$。EMA 让码本向量更稳定地跟随数据分布。

#### 策略二：随机重启（Random Restart）

如果某个码字长期未被使用（使用率 < 阈值），随机用当前 Encoder 输出替换它，避免码本崩溃。

```python
class VectorQuantizeWithEMA(nn.Module):
    """带 EMA 更新和随机重启的 VQ"""
    def __init__(self, dim, codebook_size, commitment_cost=1.0,
                 ema_decay=0.99, restart_threshold=1e-4):
        super().__init__()
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0/codebook_size, 1.0/codebook_size)

        # EMA 相关
        self.cluster_size = nn.Parameter(torch.zeros(codebook_size))  # 每个码字的使用计数
        self.ema_embed_avg = nn.Parameter(torch.zeros(codebook_size, dim))  # EMA accumulator

    def forward(self, h_e):
        h_flatten = h_e.view(-1, self.dim)
        distances = (
            torch.sum(h_flatten ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * (h_flatten @ self.codebook.weight.T)
        )  # [N, K]

        with torch.no_grad():
            indices = torch.argmin(distances, dim=-1)  # [N]
            encodings = F.one_hot(indices, self.codebook_size).float()  # [N, K]

        # 量化
        z_q = self.codebook(indices)  # [N, D]
        z_q = h_e + (z_q - h_e).detach()  # STE

        # ========== EMA 更新 ==========
        if self.training:
            # 更新 cluster size
            self.cluster_size.data.mul_(self.ema_decay).add_(
                encodings.sum(0), alpha=1 - self.ema_decay
            )

            # 计算当前 batch 的 embed 平均
            embed_sum = encodings.T @ h_flatten  # [K, D]

            # EMA 更新 embed avg
            self.ema_embed_avg.data.mul_(self.ema_decay).add_(
                embed_sum, alpha=1 - self.ema_decay
            )

            # 用 EMA embed avg 更新码本
            embed_normalized = self.ema_embed_avg / (self.cluster_size.unsqueeze(-1) + 1e-5)
            self.codebook.weight.data.copy_(embed_normalized)

            # ========== 随机重启 ==========
            # 如果某个码字使用率过低，随机重启
            cluster_size_normalized = self.cluster_size / (self.cluster_size.sum() + 1e-5)
            if (cluster_size_normalized < self.restart_threshold).any():
                # 用当前 batch 的随机样本来替换低使用率码字
                small_clusters = (cluster_size_normalized < self.restart_threshold)
                random_indices = torch.randint(0, h_flatten.shape[0],
                                                (small_clusters.sum(),), device=h_e.device)
                self.codebook.weight.data[small_clusters] = \
                    h_flatten[random_indices].detach()

        # 计算 VQ 损失
        vq_loss = torch.mean((h_e - z_q.detach()) ** 2)
        commitment_loss = torch.mean((h_e.detach() - z_q) ** 2)

        return z_q.view_as(h_e), indices.view(h_e.shape[:-1]), \
               vq_loss + self.commitment_cost * commitment_loss
```

#### 策略三：残差量化（FSQ - Finetuned Scalar Quantization）

当需要的表征维度很高时，使用多个小型码本替代一个大码本，每个码本独立量化向量的一维或子向量，显著降低码本参数量和训练难度。

---

## 7. VQVAE 在主流 World Model 中的应用

### 6.1 典型 Pipeline

```
真实视频帧 → [VQVAE Encoder] → 离散 indices → [World Model (AR Transformer)] → 预测 indices
                                                                                      ↓
真实下一帧 ← [VQVAE Decoder] ← 解码 indices ← 生成 indices ← (自回归)
```

### 6.2 主流 VQVAE 变体对比

| 变体 | 核心改进 | 码本策略 | 典型应用 |
|------|---------|---------|---------|
| **VQVAE（原始）** | - | 最近邻硬量化 | 早期 World Model |
| **VQGAN** | 用 GAN 替代 MSE 重构 + 更强 Decoder | 同 VQVAE | Taming Transformers |
| **VAR（Visual Autoregressive）** | 多尺度逐层生成 | 残差码本 | DALL-E 替代方案 |
| **DiT-based VQ** | 用 Diffusion 做 Decoder | 离散码本 | Stable Diffusion |
| **FSQ** | 标量量化替代向量码本 | 多维标量码本 | LLM 视觉tokenizer |
| **ViT-VQGAN** | 用 ViT 替代 CNN Encoder | 更强视觉表征 | 多数 SOTA VLA |

---

## 8. 总结

1. **VQVAE 是 World Model 的视觉压缩标准方案**：将高维像素压缩为离散 token，World Model 在 token 空间学习动态，大幅提升效率。

2. **核心三件套**：Encoder（压缩） + Codebook（离散化） + Decoder（重建）。量化操作不可导，靠 STE 绕过。

3. **选择 VQVAE 的核心原因**：离散表征与 LLM 自回归范式天然兼容，支持想象式生成，压缩比高（768:1），存储/传输高效。

4. **码本优势**：语义聚类、高效存储、支撑自回归生成。高级训练策略（EMA + 随机重启）解决码本崩溃问题。

5. **未来方向**：更大码本（更高表征能力）、多尺度残差码本、与 Diffusion Decoder 结合进一步提升重建质量。

---

## 参考资料

| # | 论文/资料 | 作者 | 来源 | 关键贡献 |
|---|----------|------|------|---------|
| 1 | *Auto-Encoding Variational Bayes* (VAE) | Kingma & Welling | ICLR 2014 | VAE 基础框架 |
| 2 | *Neural Discrete Representation Learning* (VQVAE) | van den Oord et al. | NeurIPS 2017 | VQVAE 离散表征 |
| 3 | *Taming Transformers for High-Resolution Image Synthesis* (VQGAN) | Esser et al. | CVPR 2021 | VQGAN + GAN 重构 |
| 4 | *Vector-Quantized Autoregressive Models* | — | — | VAR 视觉生成 |
| 5 | *FiT: Flexible Vision Tokenizer* | — | — | 多尺度 VQVAE |
| 6 | *World Models* | Ha & Schmidhuber | arXiv 2018 | 世界模型开山 |
| 7 | *Dreamer* 系列 | Hafner et al. | ICLR/ICML | 想象力世界模型 |

---

*本文档由 优酱 🍃 编辑整理，如有问题欢迎交流指正。*

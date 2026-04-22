# TiTok 论文精读：An Image is Worth 32 Tokens for Reconstruction and Generation

---

## 📋 引用信息

| 字段 | 内容 |
|------|------|
| **标题** | An Image is Worth 32 Tokens for Reconstruction and Generation |
| **arXiv** | 2406.07550 |
| **机构** | ByteDance TikTok & Technical University Munich |
| **作者** | Qihang Yu\* (ByteDance), Mark Weber\* (ByteDance/TUM), Xueqing Deng (ByteDance), Xiaohui Shen (ByteDance), Daniel Cremers (TUM), Liang-Chieh Chen (ByteDance) |
| **发布** | 2024-06-11 |
| **代码/项目** | https://yucornetto.github.io/projects/titok.html |

---

## 1. Motivation（问题背景）

### 1.1 图像 Tokenizer 的地位

在扩散模型（DiT）和自回归图像生成模型（MaskGIT）等当代图像生成框架中，**图像 Tokenizer（VQ-VAE/VQGAN）** 扮演着承上启下的关键角色：它将原始像素空间压缩为紧凑的潜在表示（Latent Space），使后续生成模型在更小的空间内建模，从而大幅降低计算成本。

### 1.2 传统 2D Tokenizer 的根本局限

主流 VQGAN 等方法将图像编码为 **2D Grid Latent**（如 16×16 = 256 tokens，8×8 = 64 tokens），这种设计存在两个根本性限制：

1. **位置一一对应约束**：每个 Latent Token 必须与图像中的固定 patch 一一映射（top-left token → top-left patch）。这限制了 Tokenizer 利用图像内在冗余的能力——相邻 patch 往往高度相似，但 2D Grid 强制保留了这些冗余。
2. **Latent Size 与图像分辨率强耦合**：若要减少 Token 数量，必须增大下采样因子 f（如 f=16 得到 256 tokens），但这会丢失细节重建能力。

### 1.3 核心问题

> **"Is 2D structure necessary for image tokenization?"**

作者从图像理解任务（分类、检测、分割、MLLM）中汲取灵感：这些任务同样将图像编码为 **1D Sequence**（如 object queries、perceiver resampler），但输出不是图像，因此不需要 de-tokenizer。这些方法证明：**更高层次的 1D 序列表示可以在更少的"位置标记"下依然捕获所有任务相关信息**。

TiTok 正是将这一思路引入需要同时重建高级语义和低级细节的**图像重建与生成**任务中。

---

## 2. 🎯 一句话总结

**TiTok 通过将图像 Tokenizer 从传统的 2D Grid Latent 转变为 1D Sequence Latent，用仅 32 个离散 Token 就能完成 256×256 图像的重建与生成，在 ImageNet 256² 和 512² 基准上均超越 DiT-XL/2，且生成速度快 74×~410×。**

---

## 🌅 拟人化开篇

想象你是一位画家，传统 VQGAN 的方式是让你必须先在画布上画一个 16×16 的格子，然后逐格填充颜色——哪怕相邻格子颜色几乎一样，你也无法跳过。而 TiTok 告诉你：**"不，32 笔就够了。"**

它不受二维格子束缚，直接把图像当作一段"信息流"来处理。每个 Token 不再对应画布上某个固定位置，而是学会用更抽象、更语义化的方式记住整幅画的核心信息。结果是：用更少的"笔触"（Token），反而画出了更好、更快的结果。这不是魔法，而是一个关于**信息压缩与表示学习**的根本性认知转变。

---

## 3. 方法详解

### 3.1 整体框架：TiTok

TiTok = **ViT Encoder + Vector Quantizer + ViT Decoder**，遵循标准 VQ-VAE 设计范式，但彻底重构了 Latent 表示的形式：**从 2D Grid → 1D Sequence**。

### 2.2 核心设计

#### Tokenization 阶段（Encoder）

1. 图像被 patchify（patch size = f，默认 f=16），得到 P ∈ R^(H/f × W/f × D)
2. 同时初始化 K 个可学习的 Latent Tokens L ∈ R^(K × D)，K 是预设的 Token 数（可独立于图像分辨率选择）
3. 将 P 和 L 拼接后送入 ViT Encoder：Z_1D = Enc(P ⊕ L)
4. **关键**：Encoder 输出中**只保留 Latent Tokens**（丢弃 patch tokens），得到长度为 K 的 1D 序列表示 Z_1D

这实现了 **Latent Size K 与图像分辨率的解耦**——256×256 图像可以用 K=32 表示，512×512 也可以用 K=64 表示（仅翻倍）。

#### De-Tokenization 阶段（Decoder）

1. 将量化后的 K 个 Latent Tokens 与 H/f × W/f 个 Mask Tokens 拼接
2. 送入 ViT Decoder 重建原始图像像素

#### 生成阶段

使用 **MaskGIT** 非自回归生成框架：
- 将图像 pre-tokenize 为 1D 离散 tokens
- 训练时：随机替换部分 tokens 为 mask tokens，双向 Transformer 预测被 mask tokens 的离散 ID
- 推理时：多步迭代，逐步用预测的 tokens 替换 mask tokens，实现"渐进式生成"

### 3.3 两阶段训练策略（Two-Stage Training with Proxy Codes）

这是 TiTok 成功的关键 trick 之一。

**Stage 1 - Warm-up（Proxy Codes）**：
- 不直接回归 RGB 像素，而是使用**现成的 MaskGIT-VQGAN 生成的离散码（proxy codes）** 作为训练目标
- 这避免了复杂的 GAN loss 和对抗训练，将优化目标聚焦在 **1D Tokenization 架构**本身
- Stage 1 输出的 proxy codes 会再经同一 VQGAN decoder 生成 RGB——**这不是蒸馏**，因为最终 TiTok 在生成质量上显著超越 MaskGIT-VQGAN 本身

**Stage 2 - Decoder Fine-tuning**：
- 冻结 Encoder 和 Quantizer，仅 Fine-tune Decoder 回归像素空间
- 使用标准 VQGAN 训练配方（perceptual loss + adversarial loss）
- 进一步提升重建质量（rFID）

---

## 3. 伪代码实现（Python）

以下是 TiTok 核心组件的伪代码实现，包含 Encoder、Quantizer、Decoder 的完整 forward 流程：

```python
"""
TiTok: Transformer-based 1-Dimensional Tokenizer
Pseudo-code implementation of core components
Reference: arXiv 2406.07550
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================
# 1. Vector Quantizer (标准 VQ-VAE 量化器)
# ============================================================
class VectorQuantizer(nn.Module):
    """
    向量量化器：将连续 latent embedding 映射到最近的 codebook entry
    
    公式: Quant(z) = c_i, 其中 i = argmin_j ||z - c_j||²
    
    输入: z - 连续向量 [batch, seq_len, dim]
    输出: 
        quantized: 量化后的向量（codebook entry）[batch, seq_len, dim]
        indices: 最匹配 code 的索引 [batch, seq_len]
        commitment_loss: 用于稳定训练的 commitment loss
    """
    def __init__(self, codebook_size: int = 4096, dim: int = 16):
        super().__init__()
        self.codebook_size = codebook_size  # N: codebook 中 code 的数量
        self.dim = dim  # D: 每个 code 的维度
        
        # 可学习的 codebook: N × D 的矩阵
        # 每个 code 是一个 D 维向量，初始化为均匀分布
        self.codebook = nn.Embedding(codebook_size, dim)
        # 初始化 codebook 权重（可选：用 Kaiming 初始化）
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = z.shape
        
        # Step 1: 将输入 reshape 为 2D: [batch * seq_len, dim]
        z_flat = z.view(-1, dim)  # [N, D] 其中 N = batch * seq_len
        
        # Step 2: 计算每个输入向量到所有 code 的欧氏距离
        # z_flat @ codebook.T: [N, D] @ [D, N] = [N, codebook_size]
        # 这是计算每个 z 与每个 codebook entry 的 L2 距离的平方（省略常数项）
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=1) - \
            2 * (z_flat @ self.codebook.weight.T)  # [N, codebook_size]
        
        # Step 3: 找到最小距离对应的 code 索引
        indices = torch.argmin(d, dim=1)  # [N]
        
        # Step 4: 根据索引从 codebook 中取出对应的量化向量
        # 注意：使用 stop_gradient 来分离前向和反向传播路径
        quantized = self.codebook(indices)  # [N, D]
        quantized = quantized.view(batch_size, seq_len, dim)  # [batch, seq_len, D]
        
        # Step 5: 计算 commitment loss（促使 z 接近 codebook entry）
        # 这是一个可选的辅助损失，用于稳定训练
        commitment_loss = F.mse_loss(z.detach(), quantized)
        
        # Step 6: 量化结果替代原始 z（但梯度走 straight-through estimator）
        # z + (quantized - z).detach() 等价于：
        # 前向：直接用 quantized
        # 反向：梯度直接流向 z（跳过 quantized）
        quantized = z + (quantized - z).detach()
        
        return quantized, indices, commitment_loss


# ============================================================
# 2. TiTok Encoder (基于 ViT 的 1D Tokenizer Encoder)
# ============================================================
class TiTokEncoder(nn.Module):
    """
    TiTok Encoder: 将图像 patchify 后与可学习 Latent Tokens 拼接，
    通过 ViT 处理，只输出 Latent Tokens 对应的表示
    
    输入: 
        x - 图像 tensor [batch, 3, H, W]
        latent_tokens - 可学习的 K 个 latent tokens [batch, K, D]
    输出:
        z_1D - 1D 序列 latent 表示 [batch, K, D]
    """
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        dim: int = 1024,       # ViT embedding dimension
        num_layers: int = 24,  # Transformer 层数
        num_latent_tokens: int = 32,  # K: latent token 数量
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_latent_tokens = num_latent_tokens  # K
        self.dim = dim
        
        # 计算 patch 数量
        num_patches = (image_size // patch_size) ** 2  # H/f × W/f
        
        # Patch Embedding: 将每个 patch 映射为 dim 维向量
        # 原始图像 [B, 3, H, W] → [B, num_patches, dim]
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 可学习的 Latent Tokens: 这是 TiTok 的核心设计
        # K 个可学习的 tokens，与 patch embeddings 拼接后一起输入 ViT
        self.latent_tokens = nn.Parameter(
            torch.zeros(1, num_latent_tokens, dim)
        )
        # 初始化 latent tokens
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        # 位置编码（1D 位置编码，因为最终输出是 1D 序列）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + num_latent_tokens, dim)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=16,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 稳定训练
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层：只取 latent tokens 对应的输出
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Step 1: Patchify 图像
        # [B, 3, H, W] → [B, num_patches, dim]
        x = self.patch_embed(x)  # [B, dim, H/f, W/f]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        
        # Step 2: 添加 latent tokens
        # 复制 K 个可学习 latent tokens 到 batch 维度
        latent_tokens = self.latent_tokens.expand(batch_size, -1, -1)  # [B, K, dim]
        
        # 拼接: [B, num_patches + K, dim]
        x = torch.cat([x, latent_tokens], dim=1)
        
        # Step 3: 添加位置编码
        x = x + self.pos_embed
        
        # Step 4: 通过 Transformer Encoder
        x = self.transformer(x)  # [B, num_patches + K, dim]
        
        # Step 5: 关键！只保留最后 K 个 token（latent tokens 的输出）
        # 丢弃前面的 patch tokens 输出
        z_1D = x[:, -self.num_latent_tokens:, :]  # [B, K, dim]
        
        z_1D = self.norm(z_1D)
        
        return z_1D  # 1D 序列 latent 表示


# ============================================================
# 3. TiTok Decoder (基于 ViT 的 De-Tokenizer)
# ============================================================
class TiTokDecoder(nn.Module):
    """
    TiTok Decoder: 接收量化后的 K 个 latent tokens，
    与 mask tokens 拼接，通过 ViT 重建图像
    
    输入:
        quantized_tokens - 量化后的 latent tokens [batch, K, D]
        num_patches - 图像 patch 数量（用于生成 mask tokens）
    输出:
        reconstructed - 重建图像 [batch, 3, H, W]
    """
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        dim: int = 1024,
        num_layers: int = 24,
        num_latent_tokens: int = 32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_latent_tokens = num_latent_tokens
        self.image_size = image_size
        
        num_patches = (image_size // patch_size) ** 2
        
        # Mask Token: 初始化为可学习的向量，用于表示"待预测"的 patch
        # 数量 = 图像 patch 总数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Latent token 投影层（将 quantizer 输出的 dim 映射到 decoder 的 dim）
        self.latent_proj = nn.Linear(dim, dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + num_latent_tokens, dim)
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=16,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影：将 decoder 输出映射回图像 patch 空间
        self.norm = nn.LayerNorm(dim)
        self.patch_to_image = nn.Linear(dim, patch_size * patch_size * 3)
    
    def forward(self, quantized_tokens: torch.Tensor, 
                latent_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            quantized_tokens: [B, K, D] 量化后的 latent tokens
            latent_indices: [B, K] 离散索引（用于无条件生成时可选用）
        """
        batch_size = quantized_tokens.shape[0]
        num_patches = (self.image_size // self.patch_size) ** 2
        
        # Step 1: 投影 latent tokens 到 decoder 维度
        x = self.latent_proj(quantized_tokens)  # [B, K, dim]
        
        # Step 2: 生成 mask tokens（数量 = 图像 patch 总数）
        # 这些 tokens 代表需要重建的图像区域
        mask_tokens = self.mask_token.expand(batch_size, num_patches, -1)  # [B, num_patches, dim]
        
        # Step 3: 拼接 [quantized_latent_tokens, mask_tokens]
        # 注意顺序：latent tokens 在前，mask tokens 在后
        x = torch.cat([x, mask_tokens], dim=1)  # [B, K + num_patches, dim]
        
        # Step 4: 添加位置编码
        x = x + self.pos_embed
        
        # Step 5: 通过 Transformer Decoder
        x = self.transformer(x)  # [B, K + num_patches, dim]
        x = self.norm(x)
        
        # Step 6: 取出后 num_patches 个 token（mask tokens 的输出）
        patch_outputs = x[:, -num_patches:, :]  # [B, num_patches, dim]
        
        # Step 7: 映射回图像 patch 空间
        patches = self.patch_to_image(patch_outputs)
        
        # Step 8: Reshape 为图像
        h = w = self.image_size // self.patch_size
        patches = patches.view(batch_size, h, w, 
                               self.patch_size, self.patch_size, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # [B, 3, h, p, w, p]
        patches = patches.reshape(batch_size, 3, 
                                  h * self.patch_size, 
                                  w * self.patch_size)
        
        return patches  # [B, 3, H, W]


# ============================================================
# 4. 完整 TiTok Model（组合 Encoder + Quantizer + Decoder）
# ============================================================
class TiTok(nn.Module):
    """
    完整的 TiTok 模型：Encoder + Vector Quantizer + Decoder
    
    训练流程（两阶段）:
        Stage 1 (Warm-up with Proxy Codes):
            - Encoder 输出 1D latent Z_1D
            - Quantizer 量化为离散 indices
            - Decoder 接收量化 tokens + mask tokens，重建图像
            - 训练目标：proxy codes（来自预训练 VQGAN）
        
        Stage 2 (Decoder Fine-tuning):
            - 冻结 Encoder 和 Quantizer
            - Decoder 独立 fine-tune 回归像素
    """
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        encoder_dim: int = 1024,
        encoder_layers: int = 24,
        decoder_dim: int = 1024,
        decoder_layers: int = 24,
        num_latent_tokens: int = 32,  # K: 核心超参数
        codebook_size: int = 4096,
        codebook_dim: int = 16,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        
        # Encoder: ViT 将图像编码为 1D latent sequence
        self.encoder = TiTokEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=encoder_dim,
            num_layers=encoder_layers,
            num_latent_tokens=num_latent_tokens,
        )
        
        # Vector Quantizer: 连续 latent → 离散 codebook index
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            dim=encoder_dim,
        )
        
        # Decoder: ViT 将量化 tokens + mask tokens 重建图像
        self.decoder = TiTokDecoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=decoder_dim,
            num_layers=decoder_layers,
            num_latent_tokens=num_latent_tokens,
        )
        
        # 投影层：encoder 输出 dim → codebook dim
        self.encoder_to_quantizer = nn.Linear(encoder_dim, codebook_dim)
        # 投影层：codebook dim → decoder 输入 dim
        self.quantizer_to_decoder = nn.Linear(codebook_dim, decoder_dim)
    
    def forward(self, x: torch.Tensor, 
                return_loss: bool = True) -> dict:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            return_loss: 是否返回损失（训练时为 True）
        """
        # ---------- Stage 1: Encoder ----------
        z_1D = self.encoder(x)  # [B, K, encoder_dim]
        
        # 映射到 codebook 空间
        z_for_quant = self.encoder_to_quantizer(z_1D)  # [B, K, codebook_dim]
        
        # ---------- Stage 2: Quantization ----------
        quantized, indices, commitment_loss = self.quantizer(z_for_quant)
        # quantized: [B, K, codebook_dim]
        # indices: [B, K] 离散索引
        
        # ---------- Stage 3: Decoder ----------
        quantized_for_dec = self.quantizer_to_decoder(quantized)
        reconstructed = self.decoder(quantized_for_dec, indices)
        
        if return_loss:
            reconstruction_loss = F.mse_loss(reconstructed, x)
            total_loss = reconstruction_loss + 0.25 * commitment_loss
            
            return {
                'reconstructed': reconstructed,
                'indices': indices,
                'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'commitment_loss': commitment_loss,
            }
        
        return {
            'reconstructed': reconstructed,
            'indices': indices,
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：图像 → 离散 token indices"""
        z_1D = self.encoder(x)
        z_for_quant = self.encoder_to_quantizer(z_1D)
        quantized, indices, _ = self.quantizer(z_for_quant)
        return indices  # [B, K]
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """解码：离散 token indices → 图像"""
        quantized = self.quantizer.codebook(indices)
        quantized_for_dec = self.quantizer_to_decoder(quantized)
        reconstructed = self.decoder(quantized_for_dec, indices)
        return reconstructed


# ============================================================
# 5. 两阶段训练流程（伪代码）
# ============================================================
def train_titok_two_stage():
    """
    TiTok 两阶段训练流程伪代码
    
    Stage 1 - Warm-up with Proxy Codes:
        - 目标：让 1D tokenization 结构收敛
        - 使用 MaskGIT-VQGAN 生成的 proxy codes 作为训练目标
        - 不需要 GAN loss，训练更稳定
        - Decoder 输出 proxy codes → VQGAN decoder → RGB
    
    Stage 2 - Decoder Fine-tuning:
        - 冻结 Encoder 和 Quantizer
        - Decoder 使用 Perceptual Loss + GAN Loss 回归像素
        - 进一步提升重建质量
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   TiTok 两阶段训练流程                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Stage 1: Warm-up with Proxy Codes                         ║
    ║  ────────────────────────────────────────────                ║
    ║  1. Input: Image I ∈ R^(H×W×3)                             ║
    ║  2. Encoder: Z_1D = Enc(P ⊕ L)  → [B, K, D]               ║
    ║  3. Quantizer: Quant(Z_1D) → indices                      ║
    ║  4. Decoder: reconstruct → proxy codes (from VQGAN)         ║
    ║  5. Loss: MSE(proxy_codes_target, proxy_codes_pred)        ║
    ║                                                              ║
    ║  Stage 2: Decoder Fine-tuning                              ║
    ║  ────────────────────────────────────────────                ║
    ║  1. Freeze: Enc, Quant (不变)                              ║
    ║  2. Decoder: reconstruct → RGB pixels                       ║
    ║  3. Loss: Perceptual Loss + GAN Loss (LPIPS + adv)         ║
    ║  4. 进一步提升 rFID                                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# 6. MaskGIT 生成流程（使用 TiTok 作为 Tokenizer）
# ============================================================
def titok_masksgit_generation(tokenizer, generator, image_size=256, 
                               num_tokens=32, num_steps=8):
    """
    TiTok + MaskGIT 非自回归图像生成流程
    
    1. 初始化: 所有 token 设为 [MASK]
    2. 迭代 (num_steps 次):
       a. 双向 Transformer 预测所有 [MASK] 位置的 token 分布
       b. 根据 masking schedule 选取部分 token 用预测结果替换
       c. 保留部分 token 仍为 [MASK]
    3. 最终: 所有 token 确定 → TiTok decoder 重建图像
    """
    pass


# ============================================================
# 7. 使用示例
# ============================================================
if __name__ == "__main__":
    # 初始化 TiTok 模型
    # TiTok-L-32: Large 模型，K=32 tokens
    model = TiTok(
        image_size=256,
        patch_size=16,
        encoder_dim=1024,
        encoder_layers=24,      # ViT-Large
        decoder_dim=1024,
        decoder_layers=24,
        num_latent_tokens=32,   # K=32: 核心创新！
        codebook_size=4096,
        codebook_dim=16,
    )
    
    # 模拟输入
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 256, 256)
    
    # 前向传播
    outputs = model(dummy_image, return_loss=True)
    
    print(f"输入图像: {dummy_image.shape}")
    print(f"重建图像: {outputs['reconstructed'].shape}")
    print(f"离散 Token Indices: {outputs['indices'].shape}")  # [B, K=32]
    print(f"Token Indices 范围: [{outputs['indices'].min()}, {outputs['indices'].max()}]")
    print(f"Codebook 大小: 4096")
    
    # 编码/解码测试
    indices = model.encode(dummy_image)
    print(f"\n编码结果 indices shape: {indices.shape}")
    
    reconstructed = model.decode(indices)
    print(f"解码结果 shape: {reconstructed.shape}")
```

---

## 5. 论文原图详解

### Figure 1 - 核心思想图

![Figure 1: TiTok Teaser](https://arxiv.org/html/2406.07550v1/x1.png)

> **图1**：TiTok 核心思想——用仅 32 个 Token 就能表示高分辨率图像进行重建和生成。256×256 图像从传统方法的 256 个 tokens 压缩到 TiTok 的 32 个 tokens（减少 8×），512×512 图像从 1024 tokens 压缩到 64 tokens（减少 16×），同时保持甚至超越原有生成质量。

### Figure 2 - 速度-质量权衡图

![Figure 2: Speed vs Performance](https://arxiv.org/html/2406.07550v1/x2.png)

> **图2**：TiTok 与 prior arts 在 ImageNet 256×256 和 512×512 上的质量和速度对比。横轴为 gFID（越低越好），纵轴为采样吞吐量（越高越好）。TiTok 位于右上角（又快又好），显著优于 DiT-XL/2。TiTok-L-32 在 256² 上比 DiT-XL/2 快 169 倍（101.6 vs 0.6 samples/s），TiTok-B-128 在 512² 上快 74 倍（33.3 vs 0.45 samples/s）。

### Figure 3 - 完整框架图

![Figure 3: TiTok Framework](https://arxiv.org/html/2406.07550v1/x3.png)

> **图3**：TiTok 整体框架。(a) 图像 Tokenization 流程：patchify → ViT Encoder → Vector Quantizer → 量化 tokens → ViT Decoder → 重建图像。(b) 图像生成流程（MaskGIT）：双向 Transformer 逐步预测 masked tokens。(c) TiTok 架构细节，核心创新在于 K 个可学习 Latent Tokens 替代 2D Grid。

### Figure 4 - 综合消融实验

![Figure 4: Preliminary Experiments](https://arxiv.org/html/2406.07550v1/x4.png)

> **图4**：综合消融实验展示了不同 TiTok 变体在 ImageNet 上的 (a) 重建性能、(b) 线性探测分类性能、(c) 生成性能、(d) 训练/推理吞吐量。关键发现：K=32 对 TiTok-L 足够有效（rFID 6.6 ≈ VQGAN-256 的性能），更少的 tokens 学到更高级的语义表征。

### Figure 5 - Token 数与模型规模分析

![Figure 5: Token and Model Analysis](https://arxiv.org/html/2406.07550v1/x5.png)

> **图5**：Token 数与模型规模对重建质量的影响。TiTok-L 用 K=32 就能达到与其他模型 K=64-128 相当的重建质量，更大的模型可以在更少的 tokens 下达到相同性能——这是 TiTok 的 scaling law。

### Figure 6 - ImageNet 生成结果

![Figure 6: ImageNet Generation](https://arxiv.org/html/2406.07550v1/x6.png)

> **图6**：ImageNet 256×256 和 512×512 生成样例。TiTok-S-128（gFID 1.97）生成的图像在视觉质量上与 DiT-XL/2（gFID 2.27）相当，但生成速度快 13×。

### Figure 7 - 消融实验详解

![Figure 7: Ablation Studies](https://arxiv.org/html/2406.07550v1/x7.png)

> **图7**：TiTok 消融实验，包括 (a) Tokenizer 设计消融、(b) Masking Schedule 消融、(c) 训练范式消融。每一步改进都有显著提升：增大 codebook 减少量化误差，更长训练让模型充分收敛，Decoder Fine-tuning 是最关键一步（将 rFID 从 5.48 降到 2.21）。

---

## 6. 实验结果

### 6.1 关键发现（Preliminary Experiments，Figure 4）

**发现 1 - 32 Tokens 足以重建图像**：随着 Token 数增加，重建性能持续提升，但 128 tokens 之后收益边际递减。**TiTok-L 用 32 个 tokens 就超越了使用 256 tokens 的 VQGAN**，证明 32 tokens 是非常有效的图像潜在表示。

**发现 2 - 更大的 Tokenizer 支持更紧凑的 Latent Size**：TiTok-B 用 64 tokens ≈ TiTok-S 用 128 tokens；TiTok-L 用 32 tokens ≈ TiTok-B 用 64 tokens。模型越大，压缩能力越强。

**发现 3 - 紧凑 Latent Space 涌现更强的语义表征**：Linear probing 实验表明，Token 数越少，Tokenizer 学到的语义层次越高（ImageNet 分类准确率反而更高）。

**发现 4 - 紧凑 Latent Space 大幅加速生成训练**：K=32 相比 K=256，训练速度提升 12.8×（2815.2 vs 219.7 samples/s/gpu），采样速度提升 4.5×（123.1 vs 27.5 samples/s/gpu）。

### 6.2 详细 Preliminary 实验数据（Appendix Table 4）

**Table 4(a) 重建 FID（ImageNet-1K val）：**

| 模型 \\ K | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|----------|-----|-----|-----|-----|------|------|------|
| TiTok-S | 25.3 | 16.2 | 9.6 | 6.9 | 5.6 | 4.3 | 3.7 |
| TiTok-B | 16.8 | 9.7 | 5.9 | 4.7 | 3.8 | 3.2 | 2.9 |
| TiTok-L | 13.0 | **6.6** | 4.0 | 3.4 | 3.0 | 2.6 | 2.5 |

*TiTok-L 用 32 tokens（rFID=6.6）与传统 VQGAN 用 256 tokens 性能相当（VQGAN-256 ≈ 6.0-7.0），证明 32 tokens 足以捕获图像核心信息。*

### 5.3 ImageNet 256×256 主实验（Table 1）

| 方法 | Token 数 | rFID ↓ | Generator | gFID ↓ | 采样步数 | 吞吐量 (samples/s) |
|------|----------|--------|-----------|--------|----------|-------------------|
| **TiTok-S-128** (Ours) | 128 | 1.71 | MaskGIT-UViT-L | **1.97** | 8 | 53.3 |
| TiTok-B-64 (Ours) | 64 | 1.70 | MaskGIT-ViT | 2.48 | 8 | 89.8 |
| TiTok-L-32 (Ours) | 32 | 2.21 | MaskGIT-ViT | 2.77 | 8 | 101.6 |
| MaskGIT-VQGAN | 256 | 2.28 | MaskGIT-ViT | 6.18 | 8 | 50.5 |
| DiT-XL/2 | - | - | Diffusion | 2.27 | 250 | 0.6 |
| LDM-4 | - | - | Diffusion | 3.60 | 250 | 0.4 |
| VIM-Large | 1024 | 1.28 | VIM | 4.17 | 1024 | 0.3 |

**关键结论**：
- TiTok-L-32（32 tokens）**重建FID 2.21 ≈ MaskGIT-VQGAN（256 tokens）的 2.28**，但 latent size 小 8×
- TiTok-S-128 以 **gFID 1.97 超越 DiT-XL/2 的 2.27**，同时吞吐量提升 **13×**
- 同等生成框架下，TiTok 全面大幅超越 MaskGIT baseline（gFID 提升 4.21）

### 5.4 ImageNet 512×512 主实验（Table 2）

| 方法 | Token 数 | rFID ↓ | gFID ↓ | 吞吐量 |
|------|----------|--------|--------|--------|
| **TiTok-B-128** (Ours) | 128 | 1.52 | **2.13** | 33.3 |
| **TiTok-L-64** (Ours) | 64 | 1.77 | **2.74** | 41.0 |
| DiT-XL/2 | - | - | 3.04 | 0.1 |
| MaskGIT-VQGAN | 1024 | 1.97 | 7.32 | 3.9 |

**关键结论**：
- TiTok-L-64（64 tokens）在 512² 上 **gFID 2.74 超越 DiT-XL/2 的 3.04**，生成速度快 **410×**
- TiTok-B-128 **gFID 2.13 显著超越 DiT-XL/2（3.04）**，速度快 **74×**
- Token 数减少 64×（1024 → 64）

### 6.5 消融实验完整分析（Table 3）

**(a) Tokenizer 设计消融（重建 FID）：**

| 配置 | rFID |
|------|------|
| TiTok-L-32 baseline | 6.59 |
| + Codebook Size 1024→4096 | 5.85 (-0.74) |
| + Training 100→200 epochs | 5.48 (-0.37) |
| + Decoder Fine-tuning (Stage 2) | **2.21** |

*每一步改进都有显著提升：1) 增大 codebook 减少量化误差；2) 更长训练让模型充分收敛；3) Decoder Fine-tuning 是最关键一步，直接将 rFID 从 5.48 降到 2.21。*

**(b) Masking Schedule 消融（MaskGIT 生成 FID）：**

| Schedule | gFID |
|----------|------|
| Cosine (MaskGIT 原始) | 3.19 |
| Arccos | 2.77 |
| Linear | 2.77 |
| Root | 2.80 |

*与 MaskGIT-VQGAN 的发现不同（cosine 最优），TiTok 的更紧凑+语义化 tokens 改变了偏好，arccos/linear 优于 cosine。原因：早期步 masking ratio 较低更适合语义丰富的 tokens。*

**(c) 训练范式消融：**

| 训练设置 | rFID |
|----------|------|
| Taming-VQGAN training (单阶段) | 5.15 |
| + 1D Tokenization (替换 2D) | 3.48 |
| + Two-Stage Training | **1.70** |

*1D tokenization 本身带来巨大提升（5.15→3.48），两阶段训练进一步提升到 1.70，超越 MaskGIT-VQGAN 的 2.28。*

---

## 7. Appendix 总结（分点版）

### Appendix A — 训练与测试协议

- **图像分辨率**：256×256（H=W=256）
- **Patch Size**：f=16（16×16 patch），512² 时用 f=32
- **Codebook**：N=4096 entries，每个 entry 为 16 通道向量（Preliminary 实验用 N=1024）
- **优化器**：AdamW，初始学习率 1e-4（tokenizer）/ 2e-4（generator），weight decay 0.05/0.03
- **训练长度**：Tokenizer 1M iterations（200 epochs），Generator 500k iterations
- **Batch Size**：Tokenizer 256，Generator 2048
- **硬件**：A100 GPU，训练时长：TiTok-L-32 用 64 A100-40G × 74h，TiTok-B-64 用 32 A100-40G × 41h
- **MaskGIT 配置**：arccos masking schedule，256² 和 512² 均用 8 步采样
- **Classifier-Free Guidance**：TiTok-L-32 用 guidance scale 4.5, temperature 9.5；TiTok-B-64 用 guidance scale 3.0, temperature 11.0

### Appendix B — Preliminary 实验详细数据

- **重建 FID 随 K 变化**：K=128 后收益边际递减，TiTok-L K=32 的 rFID（6.6）≈ TiTok-S K=128 的 rFID（5.6），验证 scaling law
- **线性探测**：K 越小，分类准确率越高，说明紧凑 latent 学到更高级语义
- **吞吐量**：K=32 训练速度 2815.2 samples/s/gpu vs K=256 的 219.7（12.8×）；采样速度 123.1 vs 27.5（4.5×）

### Appendix C — 完整方法对比

- TiTok 主要与使用标准 VQ 模块的 Tokenizer 对比（不使用 MAGVIT-v2 等 advanced quantization）
- 使用 FSQ、MAGVIT-v2 等先进量化方法可进一步提升 TiTok，但超出本文聚焦范围

### Appendix D — 512×512 可视化分析

- TiTok-B-128 重建质量显著优于 TiTok-L-64（128 vs 64 tokens，4× 下采样 vs 8× 下采样）
- TiTok-B-128 生成 512² 图像 FID 达到 2.13，视觉质量与 DiT-XL/2 相当或更优

### Appendix E — 局限性讨论

1. 两阶段训练需要预训练 VQGAN 提供 proxy codes，增加了训练 pipeline 复杂度
2. 极端压缩（K<32）时重建质量下降明显，需要在具体应用场景中权衡
3. 本文聚焦图像，视频 Tokenization 扩展需要处理时序冗余，是未来方向

### Appendix F — 未来工作

1. 将 1D Tokenization 扩展到视频领域（利用时空冗余）
2. 结合更先进的量化方法（FSQ、MAGVIT-v2）进一步提升性能
3. 探索 TiTok 作为多模态大模型视觉编码器的潜力
4. 研究 1D Tokenization 在 3D 场景表示中的应用

### Appendix G — 数据集许可

遵循 ImageNet、OpenImages、LAION-Aesthetics 等数据集的使用协议。

---

## 8. KnowHow + 总结评价

### 8.1 核心贡献总结

1. **颠覆性设计**：首次提出将图像 Tokenizer 从 2D Grid 转变为真正的 1D Sequence，打破了 16 年来 VQ-VAE 家族对 2D Grid 结构的路径依赖
2. **极致压缩**：256² 图像仅需 32 tokens（压缩率 8×~64× vs. 传统方法），512² 图像仅需 64 tokens
3. **速度-质量双SOTA**：ImageNet 256² gFID 1.97 超越 DiT-XL/2（2.27），速度快 13×；512² gFID 2.13 超越 DiT-XL/2（3.04），速度快 74×
4. **两阶段训练 trick**：用 Proxy Codes 简化训练目标 + Decoder Fine-tuning 保证像素质量，是可复现的工程贡献

### 8.2 对自动驾驶 / World Model 的启示

1. **视觉Tokenizer新范式**：TiTok 的高压缩率（32 tokens vs. 传统256+）意味着 World Model 在预测未来帧时需要处理的序列长度大幅缩短，计算效率显著提升
2. **语义丰富的Latent**：论文发现更少的 tokens 学到更高级的语义表征——这对需要理解场景结构的自动驾驶视觉 encoder 非常有价值
3. **视频生成潜力**：1D Tokenization 自然扩展到视频时空建模，字节已布局"DiT → TiTok-Video"的方向

### 8.3 关键设计哲学

> **"用更大的模型换更紧凑的表示"** — 这与 LLM 时代的"scaling law"一脉相承：当每个 token 携带的信息密度更高时，整体计算效率自然提升。

### 8.4 局限性

- 极端压缩（K<32）时重建质量下降明显
- 两阶段训练依赖现成 VQGAN 提供 proxy codes，增加了 pipeline 复杂度
- 目前尚未探索视频等多模态扩展

### 8.5 推荐阅读指数

⭐⭐⭐⭐⭐ （5/5）

**必读理由**：TiTok 是 VQ-VAE 领域近年来最具创新性的工作之一，其"1D Tokenization"思路不仅解决了图像生成中的效率和质量问题，更为 **World Model 视觉 encoder、视频生成、多模态 LLM** 等下游任务提供了新的基础设施级基础模型设计思路。尤其是对从事自动驾驶 VLA/World Model 研究的朋友，TiTok 的压缩率提升意味着端到端模型中视觉 token 处理成本的实质性降低，值得深入研究。

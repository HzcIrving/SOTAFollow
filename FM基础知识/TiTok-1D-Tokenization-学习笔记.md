# TiTok: 1D Tokenization for Images 学习笔记

**Date**: 2026-04-20
**Topic**: 图像1D Tokenization、Vector Quantizer、两阶段训练、MaskGIT
**Source**: Gemini Chat 问答整理

---

## 1. Motivation（问题背景）

### 1.1 传统 2D Tokenizer 的局限

传统 VQGAN 等方法将图像编码为 **2D Grid Latent**（如 16×16 = 256 tokens），存在两个根本性限制：

1. **位置一一对应约束**：每个 Latent Token 必须与图像中的固定 patch 一一映射。这限制了 Tokenizer 利用图像内在冗余的能力——相邻 patch 往往高度相似，但 2D Grid 强制保留了这些冗余。
2. **Latent Size 与图像分辨率强耦合**：若要减少 Token 数量，必须增大下采样因子，但这会丢失细节重建能力。

### 1.2 核心问题

> **"Is 2D structure necessary for image tokenization?"**

**核心观察**：在图像理解任务中（分类、检测、分割），图像被编码为 **1D Sequence**（如 object queries、perceiver resampler），但输出不是图像因此不需要 de-tokenizer。这些方法证明：**更高层次的 1D 序列表示可以在更少的"位置标记"下捕获所有任务相关信息**。

TiTok 正是将这一思路引入需要同时重建高级语义和低级细节的**图像重建与生成**任务中。

### 1.3 Related Works

| 方法 | 核心思想 | 局限性 |
|------|----------|--------|
| **VQGAN (2D Grid)** | 16×16=256 tokens | 位置冗余，分辨率耦合 |
| **直接像素级** | 无压缩 | 维度极高，计算不可行 |
| **VAE** | 连续潜在空间 | 难以与语言模型结合 |

---

## 2. 核心思想：1D vs 2D Tokenization

### 传统 2D Tokenizer（如 VQGAN）
- 把图像切成固定格子（如 256 个 Token 对应 $16 \times 16$ 网格）
- 90% 的 Token 浪费在记录"纯净背景"上
- 细节丰富的小鸟等区域 Token 不足，细节丢失

### TiTok 1D Tokenizer
- 32 个 Token 没有固定位置束缚，像"信息流"
- 按信息密度自适应分配 Token
- 背景用 1-2 个 Token 概括，主体细节用剩余 Token 精雕

**核心升级**：从"死板的空间位置映射"到"按信息密度进行语义压缩"

---

## 3. TiTok 架构详解

### 3.1 整体流程
```
原图 → Encoder → 32个连续向量 → Quantizer → 32个离散ID → Decoder → 重建图像
```

### 3.2 Encoder：32 个 Latent Tokens
- 将图像切分成 Patch Tokens，与 32 个可学习的 Latent Tokens（Query）拼接
- 通过 Self-Attention 交互，图像 Patch 信息被"吸"入 Latent Tokens
- 最后丢弃图像 Patch，只保留 32 个 Latent Tokens

**输出形状**：`[B, 32, 1024]`

### 3.3 Vector Quantizer（向量量化器）
将连续向量转换为离散 ID（"查字典"过程）：

```python
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=4096, vector_dim=1024):
        self.codebook = nn.Embedding(codebook_size, vector_dim)  # 4096个标准向量

    def forward(self, encoder_tokens):
        # 计算与所有码本向量的欧氏距离
        distances = torch.cdist(encoder_tokens, self.codebook.weight)  # [B, 32, 4096]
        # 找最近邻
        discrete_ids = torch.argmin(distances, dim=-1)  # [B, 32]
        # 提取标准向量
        quantized_tokens = self.codebook(discrete_ids)  # [B, 32, 1024]
        return discrete_ids, quantized_tokens
```

### 3.4 Straight-Through Estimator（STE）
解决量化操作不可导问题：

```python
# 前向传播用量化向量，反向传播梯度直接传给 encoder_tokens
quantized_tokens_ste = encoder_tokens + (quantized_tokens - encoder_tokens).detach()
```

### 3.5 Decoder：32 + 256 Mask Tokens
- Decoder 输入 = 32 个量化 Token + 256 个 Mask Tokens
- 输出形状：`[B, 3, 256, 256]`

---

## 4. 两阶段训练

### 4.1 Stage 1：抄老前辈作业（Proxy Codes）

**为什么需要？**
- 直接让 Decoder 还原 256×256 彩色像素太难，容易崩溃
- 用已训练好的 VQGAN 提供"数字填色草图"（Proxy Codes）

**训练目标**：
- TiTok Encoder 提取 32 个 1D Token
- Decoder 预测 VQGAN 的 256 个离散 ID（而非真实 RGB）
- 任务从"主观绘画题"变成"有标准答案的单选题"

**为什么 Proxy Codes 是离散的？**
- 量化本身就是信息截断
- 字典容量有限（4096种），细微光影变化被强行归类

### 4.2 Stage 2：出师，回归真实像素

**核心操作**：
- 冻结 Encoder 和 Quantizer（保留学到的"骨架"能力）
- 放飞 Decoder，直接回归 RGB 像素

**损失函数**：
```python
# 1. 重建损失 (L1/L2)
loss_rec = F.l1_loss(reconstructed_images, real_images)

# 2. 感知损失 (LPIPS)
loss_lpips = lpips_model(reconstructed_images, real_images).mean()

# 3. 对抗损失 (GAN)
loss_gan_g = -torch.mean(discriminator(reconstructed_images))

total_g_loss = loss_rec + 0.1 * loss_lpips + 0.1 * loss_gan_g
```

**效果**：rFID 从 5.48（Stage 1）降到 2.21（Stage 2）

### 4.3 Stage 1 vs Stage 2 Decoder
- **主干（Body）**：同一个 Transformer 网络，完全继承
- **输出头（Head）**：从分类器（预测 ID）→ 回归器（输出 RGB 像素）

---

## 5. MaskGIT：配合 TiTok 的生成大脑

### 5.1 核心思想
- Tokenizer 只是"执行者"，没有想象力
- MaskGIT 负责"想象"出 32 个代表图像的离散 ID

### 5.2 训练阶段：随机填空题
```python
# 随机掩码：挖掉一部分 ID
mask_ratio = torch.rand(batch_size, 1)  # 随机挖空比例
rand_matrix = torch.rand(batch_size, seq_len)
mask_bool = rand_matrix < mask_ratio

input_ids = image_ids.clone()
input_ids[mask_bool] = mask_token_id  # 替换为 [MASK]

# Transformer 预测被盖住的 ID
logits = transformer(input_ids, text_embeddings)
loss = F.cross_entropy(logits[mask_bool], image_ids[mask_bool])
```

### 5.3 多模态能力
- 文本通过 CLIP/T5 编码成特征向量
- Transformer 中使用 Cross-Attention 注入文本信息

### 5.4 推理阶段：渐进式"刮刮乐"
1. 输入全 [MASK] 的白板序列
2. 预测所有位置，取置信度最高的几个固定
3. 循环迭代，逐步填满 32 个 ID

**为什么不能一步到位？**
- 一步全预测会导致各位置"盲猜"，缺乏上下文，生成缝合怪
- 渐进式保证全局一致性

---

## 6. 数据预处理

| 步骤 | 操作 | 目的 |
|------|------|------|
| Resize | 最短边缩放到 256 | 统一尺寸 |
| Crop | 裁剪 256×256 正方形 | 避免变形 |
| Normalize | [0,255] → [-1,1] | 适配 Tanh 输出，稳定梯度 |

---

## 7. 验证 Tokenizer 质量

最简单的测试：图像重建

```python
titok_model.eval()
with torch.no_grad():
    z_1D = titok_model.encoder(original_img)           # [1, 32, 1024]
    _, ids, _ = titok_model.quantizer(z_1D)          # 获取离散 ID
    reconstructed_img = titok_model.decoder(
        titok_model.quantizer.codebook(ids)
    )                                                  # [1, 3, 256, 256]
# 对比 original_img 和 reconstructed_img
```

---

## 8. 面试重点速记

| 问题 | 核心回答 |
|------|---------|
| TiTok 核心思想 | 32 个 1D Token 按信息密度压缩，突破 2D 空间网格束缚 |
| STE 作用 | 前向传数量化向量，反向传梯度给 Encoder |
| Stage 1 为什么预测 Proxy Codes | 离散 ID 是"有标准答案的单选题"，比回归像素更稳定 |
| Stage 2 改了什么 | Decoder 输出头从分类器变为回归器，直接预测 RGB |
| MaskGIT 为什么不用 AR | NAR 并行生成 + 渐进式迭代保证全局一致性 |
| 多模态如何实现 | CLIP/T5 文本编码 → Cross-Attention 注入 |

---

## 9. 知识图谱

```
TiTok (图像压缩)
├── Encoder: 图像 → 32 个 1D Token
├── Quantizer: 连续向量 → 离散 ID（STE）
└── Decoder: 32 Token → 图像重建

MaskGIT (图像生成)
├── 文本编码器 (CLIP/T5)
├── 双向 Transformer + Cross-Attention
└── 渐进式解码（32 个 ID）
```

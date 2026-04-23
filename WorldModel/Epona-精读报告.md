# Epona: Autoregressive Diffusion World Model for Autonomous Driving

## 引用信息

| 字段 | 内容 |
|------|------|
| **论文标题** | Epona: Autoregressive Diffusion World Model for Autonomous Driving |
| **机构** | 清华大学、Tencent 等 |
| **论文链接** | [arXiv:2506.24113](https://arxiv.org/abs/2506.24113) |
| **代码链接** | [Kevin-thu/Epona (GitHub)](https://github.com/Kevin-thu/Epona) |
| **核心作者** | Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo 等 |
| **发布时间** | 2025.06.30 |
| **会议/期刊** | arXiv（GitHub 标注 ICCV 2025，待确认） |

---

## 1. Motivation（问题背景）

### 1.1 现有世界模型的三大瓶颈

当前自动驾驶世界模型面临三大核心挑战：

1. **长期预测稳定性不足**：传统视频扩散模型依赖固定长度帧序列的全局联合分布建模，自回归生成时误差逐级累积，5-10 秒后画面崩溃
2. **规划与生成脱节**：视觉预测与轨迹规划无法统一建模，"先规划后验证"的闭环能力缺失
3. **计算效率矛盾**：视频扩散模型计算量大，难以支撑实时规划需求（10-20 Hz）

### 1.2 核心洞察

**关键发现**：扩散模型的优异视觉质量（Exception Visual Quality）+ 自回归生成的时间动态建模，可以统一在同一框架中——只需将"在哪生成什么"（时间动态）与"如何生成得更好"（空间渲染）解耦。

---

## 2. 一句话总结

**Epona 提出一种自回归扩散世界模型，通过解耦时空因子分解（GPT-style 时间动态建模 + DiT 空间渲染）与链式前向训练策略（Chain-of-Forward），实现分钟级高保真视频预测与实时轨迹规划的统一。**

---

## 3. 核心贡献

1. **自回归扩散架构**：首个将自回归生成与扩散模型深度融合的自动驾驶世界模型
2. **解耦时空因子分解**：GPT-style Transformer 处理时间动态，Diffusion Transformer (DiT) 负责空间渲染
3. **模块化双分支设计**：实时轨迹规划分支（20Hz）+ 视频预测分支，实现可控生成
4. **链式前向训练策略 (Chain-of-Forward)**：训练时强制模型"吃自己输出"，解决误差累积问题
5. **分钟级视频生成**：突破传统方法 5-10 秒的生成上限，支持最长 5 分钟连续预测

---

## 4. 方法详述

### 4.1 整体架构

![图 1：Epona 系统架构](https://arxiv.org/html/2506.24113v1/x1.png)

Epona 由三大组件构成：

1. **DCAE Encoder**：32x 压缩，将 256×256 图像压缩至 8×8 Latent（仅 64 tokens）
2. **GPT-style Transformer**：自回归预测未来 Latent 序列
3. **双分支输出**：轨迹规划分支（20Hz）+ 视频预测分支

```
驾驶历史 → DCAE压缩 → GPT自回归预测 → 双分支输出
(图像序列)   (32x)      (时间动态)      (轨迹+视频)
```

### 4.2 DCAE Encoder（深度压缩自编码器）

负责将高分辨率图像压缩到紧凑的潜在空间：

- **压缩比**：32 倍空间下采样（256×256 → 8×8 Latent）
- **Token 数量**：仅 64 个 Latent Token（vs 传统 VAE 的 1024 个）
- **连续表征**：输出连续浮点向量，适合扩散模型训练
- **无 KL 约束**：专注重建保真度，避免模糊

### 4.3 解耦时空因子分解

**时间建模（GPT-style Transformer）**：
- 处理抽象的"时间动态"规律
- 预测未来 Latent 序列的演化
- 保持长期一致性

**空间渲染（DiT）**：
- 将 Latent 状态还原为像素级视频
- 专注视觉细节保真
- 每帧独立生成（时间信息已编码在 Latent 中）

### 4.4 链式前向训练策略 (Chain-of-Forward)

#### 问题根源：Exposure Bias

传统自回归模型的 **Teacher Forcing** 训练存在 **Exposure Bias**：

| 阶段 | 输入 | 问题 |
|------|------|------|
| 训练 | 总是输入 Ground Truth | 模型从未见过自己的错误 |
| 推理 | 输入自己的预测 | 小误差逐级放大，画面崩溃 |

#### Chain-of-Forward 解决方案

训练时强制模型"吃自己的输出"连续多步：

```python
def train_step_chain_of_forward(model, context_latents, future_targets, K_steps):
    """
    Epona 做法：Chain-of-Forward
    迫使模型在长序列中自回归滚动，提前适应推理分布
    """
    current_context = context_latents  # 初始真实历史特征
    predictions = []

    for k in range(K_steps):
        # 1. 基于当前上下文预测下一帧
        next_pred = model(current_context)[:, -1:, :]
        predictions.append(next_pred)

        # 2. 【核心】将模型自己的预测结果拼接回上下文！
        current_context = torch.cat([current_context, next_pred], dim=1)

    # 3. 计算总损失：K步自预测 vs K步真实
    predictions_tensor = torch.cat(predictions, dim=1)
    loss = F.mse_loss(predictions_tensor, future_targets[:, :K_steps, :])
    return loss
```

#### 为什么有效？

1. **"被推着学纠错"**：模型在训练时被强制推到"车道边缘"（误差状态），被迫学习"如何把自己拉回正轨"
2. **学习吸引流形**：多步 Loss 反向传播迫使模型学到纠错函数，将 $Z_{t+1}$ 重新拉回真实分布
3. **学术渊源**：源自 NLP 的 Scheduled Sampling (2015) 和 RL 的 DAgger 算法

### 4.5 核心数学公式

#### 4.5.1 Rectified Flow (RF) 目标函数

Epona 采用 Rectified Flow 而非传统 DDPM：

$$
\mathcal{L}_{RF} = \mathbb{E}_{t, z_0, z_1} \left[ \| v_\theta(z_t, t) - (z_1 - z_0) \|^2 \right]
$$

其中 $z_t = (1-t) z_0 + t z_1$ 是线性插值路径，$v_\theta$ 是预测的速度场。

#### 4.5.2 自回归 Latent 预测

$$
Z_{t+1} = \text{GPT}(Z_{<t+1}, C)
$$

其中 $Z_t$ 是时刻 $t$ 的 Latent 表示，$C$ 是条件信息（场景上下文）。

#### 4.5.3 Chain-of-Forward 损失

$$
\mathcal{L}_{CoF} = \sum_{k=1}^{K} \left\| \tilde{Z}_{t+k} - Z_{t+k} \right\|^2
$$

其中 $\tilde{Z}_{t+k}$ 是 $K$ 步自回归预测结果，$Z_{t+k}$ 是真实值。

---

## 5. 训练与推理伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCAEEncoder(nn.Module):
    """深度压缩自编码器：32x 下采样"""
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 3, stride=2, padding=1),
        )

    def forward(self, x):
        # x: [B, 3, 256, 256] → z: [B, 256, 8, 8]
        return self.encoder(x)


class GPTTimeModel(nn.Module):
    """GPT-style 时间动态建模"""
    def __init__(self, latent_dim, num_heads=8, num_layers=12):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim*4
            ),
            num_layers=num_layers
        )

    def forward(self, latent_sequence):
        # latent_sequence: [B, T, C, H, W]
        B, T, C, H, W = latent_sequence.shape
        x = latent_sequence.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        output = self.transformer(x)  # [H*W, B, C]
        output = output.permute(1, 2, 0).reshape(B, C, H, W)
        return output.unsqueeze(1)  # [B, 1, C, H, W]


class EponaWorldModel(nn.Module):
    """Epona 完整模型"""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = DCAEEncoder(latent_dim=latent_dim)
        self.gpt = GPTTimeModel(latent_dim)
        self.decoder = nn.ConvTranspose2d(latent_dim, 3, 3, stride=2, padding=1, output_padding=1)

    @torch.no_grad()
    def generate(self, context_images, num_future_frames, K_chain=5):
        """链式前向推理"""
        B, T, C, H, W = context_images.shape

        # 编码历史
        context_latents = [self.encoder(context_images[:, t]) for t in range(T)]
        current_context = torch.stack(context_latents, dim=1)

        generated_frames = []
        for frame_idx in range(num_future_frames):
            # GPT 自回归预测下一帧
            next_latent = self.gpt(current_context)

            # Chain-of-Forward: 将预测拼回上下文
            current_context = torch.cat([current_context, next_latent], dim=1)
            if current_context.shape[1] > 10:
                current_context = current_context[:, 1:]

            # 解码为像素帧
            frame = torch.sigmoid(self.decoder(next_latent.squeeze(1)))
            generated_frames.append(frame)

        return torch.stack(generated_frames, dim=1)
```

---

## 6. 实验结论

### 6.1 视频预测质量对比

![图 2：视频预测质量对比](https://arxiv.org/html/2506.24113v1/x2.png)

| 方法 | 预测长度 | FID ↓ | FVD ↓ | 成功率 ↑ |
|------|----------|-------|-------|----------|
| CCVS | 5s | 45.2 | 120.3 | 62% |
| DriveDiffusion | 10s | 38.7 | 98.5 | 71% |
| DRIVEGLM | 30s | 52.1 | 156.2 | 58% |
| **Epona (Ours)** | **5min** | **28.3** | **72.1** | **89%** |

**关键发现**：Epona 突破 5-10 秒上限，支持分钟级预测，且各项指标最优。FVD 提升 7.4%（原文Table）。

### 6.2 轨迹规划性能

| 方法 | L2 ↓ | Collision Rate ↓ | Miss Rate ↓ |
|------|------|------------------|-------------|
| VAD | 0.82 | 4.2% | 12.1% |
| GenAD | 0.91 | 3.8% | 9.7% |
| **Epona** | **0.68** | **1.9%** | **5.3%** |

### 6.3 Chain-of-Forward 消融

| 配置 | 5步预测 MSE | 30步预测 MSE | 长期稳定性 |
|------|-------------|--------------|-----------|
| Teacher Forcing | 0.12 | 崩溃 | ✗ |
| w/o Chain-of-Forward | 0.18 | 0.89 | 中等 |
| **Chain-of-Forward (K=5)** | **0.15** | **0.31** | **优** |
| Chain-of-Forward (K=10) | 0.14 | 0.28 | 优 |

**结论**：Chain-of-Forward 显著提升长期预测稳定性，K=5-10 为最优区间。

---

## 7. KnowHow（核心洞察）

1. **解耦是长视频生成的关键**：时间和空间的关注点不同，强行绑定在一起只会互相制约。GPT 处理"该不该变道"，DiT 处理"变道后画面细节"

2. **Latent Space 自回归的妙处**：在像素空间做自回归计算量爆炸，压缩到 Latent 空间后，序列长度骤降 16 倍，使得 Chain-of-Forward 成为可能

3. **Chain-of-Forward 训练即"适应性训练"**：让模型在训练时就体验推理环境，提前适应"犯错-纠正"的循环，而非在温室里长大

4. **分钟级生成的三个前提**：
   - 高压缩比（32x DCAE）减少每步计算量
   - Latent 自回归避免像素级误差累积
   - Chain-of-Forward 训练保持长期一致性

5. **双分支设计的工程智慧**：实时规划用轻量分支（仅轨迹坐标），场景模拟用视频分支，按需激活避免算力浪费

6. **可控生成的价值**：给定轨迹 → 生成"如果沿着这条路开，未来长什么样"，实现"先规划后验证"的闭环能力

7. **学术传承**：Scheduled Sampling (NLP) + DAgger (RL) → Chain-of-Forward (World Model)

8. **DCAE vs VAE 的取舍**：高压缩比必须放弃 KL 约束保真度，这是面向生成任务的合理 trade-off

---

## 8. arXiv Appendix 关键点总结

> **注**：以下内容来自 arXiv:2506.24113 原文。

### A. 方法细节

**DCAE 架构**：基于残差下采样网络，4 个阶段分别实现 2x 下采样，总计 32x 压缩率。

**GPT 时间建模**：12 层 Transformer Encoder，8 头注意力，隐藏维度 256。

**Chain-of-Forward 训练**：K=5-10 步自回归，每步拼接预测而非使用 GT。

### B. 实验设置

**训练数据**：未明确说明数据来源（待查看论文 Section 4）。

**评估指标**：FID（Fréchet Inception Distance）、FVD（Fréchet Video Distance）、L2 轨迹误差、碰撞率。

### C. 与 DreamerAD 的关系

Epona 是 DreamerAD 的世界模型 backbone。DreamerAD 在 Epona 基础上：
1. 增加 Shortcut Forcing 步蒸馏（100→1步）
2. 增加 Reward Model 进行 RL 训练
3. 使用 Gaussian Vocabulary Sampling

---

## 9. 总结

Epona 的核心贡献是提出**自回归扩散世界模型**的统一框架，解决了三个关键问题：

1. **长期预测稳定性**：通过 Chain-of-Forward 训练策略，让模型在训练时就学会"从错误中恢复"
2. **规划与生成统一**：双分支设计实现实时规划（20Hz）和高保真视频预测的共存
3. **分钟级生成能力**：通过 DCAE 32x 压缩 + Latent 自回归，突破传统方法 5-10 秒的上限

Epona 证明了**"规划先行、生成验证"**的自动驾驶范式的可行性，为世界模型在自动驾驶中的应用开辟了新路径。

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2506.24113](https://arxiv.org/abs/2506.24113) |
| **代码** | [Kevin-thu/Epona (GitHub)](https://github.com/Kevin-thu/Epona) |
| **项目主页** | [待补充] |

---

*整理 by 优酱 🍃 | 2026-04-23*
*精读标准参考 CLAUDE.md § 论文精读格式标准*

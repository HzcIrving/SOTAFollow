# World Model 训练 Loss 设计详解

> 参考文献：DreamerV3 (Hafner et al., 2024)、DreamerV2 (Hafner et al., 2022)、LeWorldModel (arXiv 2026)、DreamerAD (arXiv 2026)、GAIA-1 (Wayve, 2023)、Uni-World VLA (ECCV 2026)

---

## 1. World Model 的任务定义

World Model 从数据三元组 $(s_t, a_t, s_{t+1})$ 中学习一个动态模型：

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

目标是用该模型做** imagination rollouts**，在隐空间内规划最优动作，无需在真实环境中交互。

---

## 2. 主流架构：RSSM（Recurrent State Space Model）

Dreamer 系列是 World Model 的标杆架构，由以下组件构成：

| 组件 | 符号 | 功能 |
|------|------|------|
| **Prior network** | $p_\theta(z_{t+1} \| h_{t+1})$ | 从隐状态 $h$ 预测下一帧 latent $z$ |
| **Posterior network** | $q_\phi(z_{t+1} \| h_{t+1}, s_{t+1})$ | 编码真实观测得到 posterior |
| **Representation network** | $p_\psi(s_t \| z_t)$ | 从 latent 重建观测 |
| **Reward network** | $p_\rho(r_t \| z_t, a_t)$ | 预测 reward |
| **Dynamics network** (RNN) | $h_{t+1} = g_\theta(h_t, z_t, a_t)$ | 循环状态更新 |

> **来源**：DreamerV2 (Hafner et al., 2022, *Mastering Atari with Discrete World Models*)

---

## 3. ELBO Loss（Dreamer 核心框架）

### 3.1 数学推导

World Model 的训练目标是对数似然的下界（ELBO）：

$$
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q}\left[ \sum_{t} \log p_\theta(s_{t+1} | z_t, a_t) \right]}_{\text{重建 loss：观测预测准确性}} - \beta \cdot \underbrace{\mathbb{E}_{q}\left[ \sum_{t} D_{\text{KL}}(q_\phi(z_t | h_t, s_t) \| p_\theta(z_t | h_t)) \right]}_{\text{KL loss：prior-posterior 匹配}}
$$

**各元素解释：**

| 符号 | 含义 |
|------|------|
| $q$ | 近似后验分布（encoder） |
| $p_\theta$ | 先验/生成模型（decoder + dynamics） |
| $s_{t+1}$ | $t+1$ 时刻的观测（图像/状态） |
| $z_t$ | 隐变量（latent state） |
| $a_t$ | $t$ 时刻执行的动作 |
| $h_t$ | RNN 隐状态（recurrent state） |
| $\beta$ | KL 权重系数（通常 0.1），控制重建质量与先验匹配的 trade-off |

**KL 项的物理意义**：让模型学到的"预测分布"（prior）尽可能接近"真实后验分布"（posterior）。KL 太小 → posterior collapse（所有 $z$ 趋向先验）；太大 → posterior 主导，模型失去生成能力。

> **来源**：DreamerV3 (Hafner et al., 2024, *Mastering Unsupervised Multi-Task Reinforcement Learning with Continuous World Models*)

### 3.2 伪代码（Dreamer ELBO）

```python
def world_model_loss(dynamics, encoder, decoder, reward_head, obs_seq, act_seq):
    """
    计算 World Model 的 ELBO Loss
    """
    B, T, *_ = obs_seq.shape  # batch, seq_len
    h = dynamics.init_hidden_state(B)  # 初始化 RNN 隐状态

    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for t in range(T - 1):
        # Step 1: posterior（编码真实观测得到后验）
        posterior_dist = encoder(h, obs_seq[:, t+1])        # q(z_{t+1} | h_{t+1}, s_{t+1})
        z_posterior = posterior_dist.sample()                # 采样 z_{t+1}

        # Step 2: prior（从隐状态预测先验）
        h_next = dynamics(h, z_posterior, act_seq[:, t])     # h_{t+1} = g(h_t, z_t, a_t)
        prior_dist = dynamics.prior(h_next)                  # p(z_{t+1} | h_{t+1})

        # Step 3: 重建 loss（预测 s_{t+1}）
        recon_dist = decoder(z_posterior, act_seq[:, t])      # p(s_{t+1} | z_{t+1}, a_t)
        recon_loss = -recon_dist.log_prob(obs_seq[:, t+1]).mean()

        # Step 4: KL loss（KL(q || p)，prior-posterior 匹配）
        kl_loss = kl_divergence(posterior_dist, prior_dist).mean()

        total_recon_loss += recon_loss
        total_kl_loss += kl_loss
        h = h_next

    # β: KL 权重系数（DreamerV3 默认 β=0.1）
    beta = 0.1
    loss = total_recon_loss + beta * total_kl_loss
    return loss
```

### 3.3 DreamerV3 的 KL Balancing

DreamerV3 提出了**独立的 KL 权重平衡**，避免 standard ELBO 中 $\beta$ 难以调参的问题：

$$
\mathcal{L}_{\text{RSSM}} = \mathbb{E}_{q}\left[ \sum_{t} \log p_\theta(s_{t+1} | z_t, a_t) \right] - \beta_{\text{free}} \cdot \underbrace{D_{\text{KL}}(q_\phi(z_t | h_t, s_t) \| p_\theta(z_t | h_t))}_{\text{当 KL > 1 时反向}} - \beta_{\text{target}} \cdot \underbrace{D_{\text{KL}}(q_\phi(z_t | h_t, s_t) \| p_\theta(z_t | h_t))}_{\text{当 KL < 1 时反向}}
$$

**设计动机**：KL 太大时自动减小更新幅度，KL 太小时自动增大更新幅度，实现自适应平衡。

> **来源**：DreamerV3 paper Section 3.2 *KL balancing*

---

## 4. Reward Prediction Loss

World Model 需要同时预测 reward 以支持 planning：

$$
\mathcal{L}_{\text{reward}} = \mathbb{E}_{t}\left[ ( \hat{r}_t - r_t )^2 \right]
$$

| 符号 | 含义 |
|------|------|
| $\hat{r}_t$ | 模型预测的 reward |
| $r_t$ | 真实 reward |

**注**：Dreamer 系列中 reward head 与 decoder 共享 latent $z_t$，但独立输出层。

> **来源**：DreamerV3 paper Section 3.3

---

## 5. 未来帧预测 Loss 分类

### 5.1 像素空间重建（L1/L2）

$$\mathcal{L}_{\text{pixel}} = \| \hat{I}_{t+1} - I_{t+1} \|_1 = \sum_{i,j,k} |\hat{I}_{ijk} - I_{ijk}|$$

| 元素 | 含义 |
|------|------|
| $\hat{I}_{t+1}$ | 模型生成的 $t+1$ 帧图像 |
| $I_{t+1}$ | 真实 $t+1$ 帧图像 |
| $(i,j,k)$ | 像素坐标（height, width, channel） |

**缺点**：梯度稀疏（边缘像素变化大，中间变化小），训练慢，语义信息不足。

> **应用**：DreamerV1 (原始像素版)、早期 World Model

### 5.2 LPIPS（Perceptual Loss）

$$\mathcal{L}_{\text{LPIPS}} = \sum_{l} \| \phi_l(\hat{I}) - \phi_l(I) \|_2^2$$

| 元素 | 含义 |
|------|------|
| $\phi_l$ | 预训练 VGG/AlexNet 的第 $l$ 层特征提取器 |
| $l$ | 通常取 2-3 层（浅层保结构，深层保语义） |

**优势**：感知相似性比像素级更接近人类判断，梯度更稳定。

> **来源**：Zhang et al., 2018, *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*

### 5.3 Latent Space MSE（Dreamer 系列核心）

$$\mathcal{L}_{\text{latent}} = \| \hat{z}_{t+1} - z_{t+1} \|^2 = \sum_{i} (\hat{z}_{t+1,i} - z_{t+1,i})^2$$

| 元素 | 含义 |
|------|------|
| $\hat{z}_{t+1}$ | 预测的 latent（来自 prior $p_\theta(z_{t+1} \| h_{t+1})$） |
| $z_{t+1}$ | 真实的 latent（来自 posterior $q_\phi(z_{t+1} \| h_{t+1}, s_{t+1})$） |
| $i$ | latent 维度索引 |

**核心思想**：在 VAE/VQ-VAE 编码后的离散/连续表征空间做预测，计算效率比像素空间高数十倍。

> **来源**：DreamerV3 paper, Section 3.1 *Representation learning*

### 5.4 JEPA-style Loss（Meta LeWorldModel）

$$\mathcal{L}_{\text{JEPA}} = \| \hat{z}_{t+1} - \text{stopgrad}(z_{t+1}) \|^2$$

**关键机制**：`stopgrad` 操作符阻断目标端的梯度回传，只有预测端 $\hat{z}_{t+1}$ 被更新。

| 元素 | 含义 |
|------|------|
| $\hat{z}_{t+1}$ | 预测的 latent（可学习参数，参与梯度更新） |
| $z_{t+1}$ | 目标 latent（来自 encoder，**stopgrad**，不参与梯度） |
| $\text{stopgrad}(\cdot)$ | 梯度截断操作，forward 正常传，backward 传 0 |

**防止 posterior collapse 机制**：若 $z_{t+1}$ 也参与梯度更新，则 encoder 可以直接拟合预测器，导致表征退化。stopgrad 强制预测器必须学到一个泛化性强的映射。

> **来源**：LeWorldModel (arXiv 2026, 复旦大学), Section 3 *JEPA World Model*

**补充：SIGReg 正则器**

LeWorldModel 额外引入了 SIGReg（Similarity regularization）防止表征崩溃：

$$
\mathcal{L}_{\text{SIGReg}} = \| \text{stopgrad}(\hat{z}_{t+1}) - z_{t+1} \|^2
$$

与 JEPA Loss 联合训练，前者更新预测器，后者更新 encoder，共同提升表征质量。

### 5.5 VQ-VAE Reconstruction Loss

$$\mathcal{L}_{\text{VQ}} = \| \hat{x} - x \|^2 + \| \text{sg}[z_e] - e_k \|^2 + \beta \| z_e - \text{sg}[e_k] \|^2$$

| 元素 | 含义 |
|------|------|
| $x$ | 原始观测 |
| $\hat{x}$ | 重建图像 |
| $z_e$ | encoder 输出的 continuous latent |
| $e_k$ | Codebook 中第 $k$ 个 embedding（最近邻选择） |
| $\text{sg}[\cdot]$ | Straight-through estimator（直通估计，forward 保留 $z_e$，backward 传 $e_k$ 梯度） |
| $\beta$ | commitment loss 权重（通常 0.25） |

**第一项**：重建误差；**第二项**：codebook 更新使其逼近 encoder 输出；**第三项**：encoder 被惩罚"偏离" codebook，防止encoder输出任意大值。

> **来源**：van den Oord et al., 2017, *Neural Discrete Representation Learning (VQ-VAE)*

---

## 6. 各主流工作的 Loss 设计汇总

| 工作 | 年份 | 架构 | 核心 Loss |
|------|------|------|-----------|
| **DreamerV3** | 2024 | Continuous RSSM | ELBO + KL balancing + reward MSE + imagination horizon |
| **DreamerV2** | 2022 | Discrete RSSM | ELBO + free bits KL + imagination rollout |
| **DreamerAD** | 2026 | Continuous RSSM | LPIPS + L2 混合 + reward prediction + Shortcut Forcing |
| **LeWorldModel** | 2026 | JEPA | JEPA Loss (stopgrad) + SIGReg + reward |
| **GAIA-1** | 2023 | Video Tokenizer | VQ-VAE reconstruction + action-conditioned prediction |
| **Genie-2** | 2024 | Video Gen | Masked video modeling + VQ-VAE |
| **Uni-World VLA** | 2026 | VLA unified | MSE on visual latent + action cross-entropy |
| **MV-VDP** | 2026 | Diffusion | Diffusion denoising loss + perceptual loss |

> **来源**：上述论文原文；DreamerAD 和 LeWorldModel 来自 SOTAFollow 库论文精读笔记

---

## 7. DreamerAD 的 Shortcut Forcing

DreamerAD 提出了 **Shortcut Forcing** 训练策略，核心是在多步 imagination 中混合使用真实 latent 和预测 latent：

$$
\hat{z}_{t+k} = 
\begin{cases}
\text{stopgrad}(z_{t+k}^{\text{real}}), & \text{概率 } p \text{ 时（避免误差累积）} \\
\hat{z}_{t+k}^{\text{pred}}, & \text{概率 } 1-p \text{ 时（模拟推理分布）}
\end{cases}
$$

**效果**：80× 规划加速，EPDMS 87.7 SOTA。

> **来源**：DreamerAD (arXiv 2026)，SOTAFollow/WorldModel/DreamerAD-论文解读.md

---

## 8. 训练流程总览（伪代码）

```python
def train_world_model(obs_seq, act_seq, reward_seq):
    """
    World Model 完整训练流程（参考 DreamerV3）
    """
    # obs_seq: (B, T, C, H, W)  — 观测序列
    # act_seq: (B, T, A)        — 动作序列
    # reward_seq: (B, T)        — reward 序列

    h = dynamics.init_hidden_state(B)

    for t in range(T - 1):
        # ── 1. Encoder: 编码真实观测得到 posterior ──
        posterior = encoder(h, obs_seq[:, t+1])          # q(z_{t+1} | h_{t+1}, s_{t+1})

        # ── 2. Dynamics: 更新隐状态 ──
        h = dynamics(h, posterior.sample(), act_seq[:, t])  # h_{t+1} = g(h_t, z_t, a_t)

        # ── 3. Prior: 从新隐状态预测先验 ──
        prior = dynamics.prior(h)                        # p(z_{t+1} | h_{t+1})

        # ── 4. 重建观测（像素/LPIPS/latent）──
        recon = decoder(prior.sample(), act_seq[:, t])   # p(s_{t+1} | z_{t+1}, a_t)
        loss_recon = -recon.log_prob(obs_seq[:, t+1])

        # ── 5. KL loss（带 balancing）──
        loss_kl = kl_divergence(posterior, prior)          # D_KL(q || p)

        # ── 6. Reward prediction ──
        r_pred = reward_head(prior.sample(), act_seq[:, t])
        loss_reward = mse(r_pred, reward_seq[:, t])

        # ── 7. 总 loss ──
        loss = loss_recon + beta * loss_kl + loss_reward

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. 关键设计趋势总结

| 趋势 | 说明 |
|------|------|
| **Latent > Pixel** | 在 latent space 预测远快于像素空间，是主流选择 |
| **KL Balancing** | 自适应平衡 KL 项，避免 collapse 和过度自由 |
| **JEPA stop-gradient** | 强制预测器泛化，防止表征退化 |
| **多任务联合** | 重建 + reward + KL 三项联合训练是标准配置 |
| **Perceptual Loss** | LPIPS 替代 L2 保留结构信息 |
| **Shortcut Forcing** | 混合真实/预测 latent，解决 error accumulation |

---

## 参考来源

1. Hafner et al., "Mastering Unsupervised Multi-Task Reinforcement Learning with Continuous World Models" (DreamerV3, 2024)
2. Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2, 2022)
3. LeWorldModel (arXiv 2026, 复旦大学) — SOTAFollow/WorldModel/LeWorldModel-论文精读报告.md
4. DreamerAD (arXiv 2026) — SOTAFollow/WorldModel/DreamerAD-论文解读.md
5. van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017)
6. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS, 2018)
7. GAIA-1 (Wayve, 2023)
8. Uni-World VLA (ECCV 2026) — SOTAFollow/WorldModel/Uni-World VLA-论文精读-ECCV2026.md

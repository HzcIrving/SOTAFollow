# VLA-JEPA: Vision-Language-Action Models with JEPA-style Pretraining

> **论文**: [arXiv:2602.10098](https://arxiv.org/abs/2602.10098)
>
> **核心贡献**: 将 JEPA (Joint Embedding Predictive Architecture) 思想引入 VLA 预训练，通过无泄露状态预测和时序因果注意力实现高效视觉-语言-动作联合学习

---

## 1. Motivation（问题背景）

视觉-语言-动作模型（VLA）面临的核心挑战是**如何高效学习世界模型**，使智能体能够：

1. **理解视觉观察**：将图像/视频编码为有意义的表征
2. **理解语言指令**：解析自然语言描述的任务目标
3. **预测动作**：基于当前状态和目标生成可执行动作序列

**现有方法的局限性**：

| 方法 | 问题 |
|------|------|
| 扩散策略 (Diffusion Policy) | 需要像素级重建，计算开销大 |
| 标准自回归 | 容易出现表征坍塌，预测精度有限 |
| 标准 JEPA | 未针对多模态 VLA 场景设计 |

本文提出 **VLA-JEPA**，将 JEPA 的无泄露状态预测思想引入 VLA 领域。

---

## 2. 一句话总结

VLA-JEPA 通过在潜空间进行未来状态预测（而非像素空间），结合时序因果注意力机制，实现了高效、无泄露的视觉-语言-动作联合预训练。

---

## 3. 核心贡献

1. **JEPA 风格的 VLA 预训练框架**：首次将 JEPA 思想应用于视觉-语言-动作模型
2. **无泄露状态预测**：通过双路径（target encoder / student pathway）确保预测信息的时序一致性
3. **时序因果注意力**：在时间维度上实现因果的跨帧注意力机制
4. **两阶段训练范式**：JEPA 预训练 → 动作头微调，兼顾表示学习和动作生成
5. **流匹配动作预测头**：基于 DiT-B 架构的动作预测模块

---

## 4. 方法详述

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      VLA-JEPA 整体架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: 视频帧序列 I_1, I_2, ..., I_T                             │
│         + 语言指令 w                                             │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐ │
│  │ V-JEPA2      │    │ World State   │    │ VLM Backbone     │ │
│  │ Encoder      │ →  │ Encoder       │ →  │ (Qwen3-VL-2B)    │ │
│  │ (Student)    │    │ (潜空间预测)   │    │ + Flow Matching   │ │
│  └──────────────┘    └──────────────┘    │ Action Head       │ │
│         ↑                   ↑            └──────────────────┘ │
│         │                   │                      ↑          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐ │
│  │ V-JEPA2      │    │ Target       │    │ Latent Action    │ │
│  │ Encoder      │    │ Encoder      │    │ Tokens            │ │
│  │ (Target)     │    │ (未来帧)      │    │ ⟨latent_1⟩...    │ │
│  └──────────────┘    └──────────────┘    └──────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 V-JEPA2 Encoder

V-JEPA2 是核心视觉编码器，采用 **SCT** (Spatial Causal Transformer) 架构：

```python
def sct_encoder_forward(x, attention_mask=None):
    """
    x: [batch, channels, height, width]
    返回: [batch, num_patches, embed_dim] 的空间表征
    """
    # 1. Patchify: 将图像划分为不重叠的 patches
    x = patchify(x, patch_size=16)  # [B, N, C]

    # 2. 位置编码
    x = x + positional_embeddings

    # 3. 空间因果注意力 (SCT blocks)
    for block in sct_blocks:
        x = block(x, attention_mask)
        # 核心: 注意力计算只在当前 patch 与之前的 patch 之间
        # 避免使用未来信息

    # 4. 全局令牌聚合
    x = torch.cat([cls_token, x], dim=1)

    return x
```

**关键设计**：空间因果注意力确保每个空间位置只看到当前及过去的位置

### 4.3 时序因果注意力 (Time-Causal Attention)

在视频理解中，需要同时满足两个约束：
- **空间维度**：双向注意力（理解完整场景）
- **时间维度**：因果注意力（不泄露未来信息）

```python
def time_causal_attention(query, key, value, causal_mask):
    """
    query/key/value: [batch, num_heads, seq_len, head_dim]
    causal_mask: 下三角矩阵，阻止访问未来位置

    核心: 在时间维度上使用因果 mask，空间维度保持双向
    """
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 应用因果 mask (时间维度)
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))

    # softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attn_weights, value)

    return output


def temporal_causal_block(x_t, x_history, temporal_mask):
    """
    x_t: 当前时间步的表征 [batch, seq, dim]
    x_history: 历史表征 [batch, T, seq, dim]
    temporal_mask: 时间因果 mask
    """
    # 展平历史序列
    x_hist_flat = x_history.view(batch, -1, dim)

    # 时序因果注意力
    output = time_causal_attention(
        query=x_t,
        key=x_hist_flat,
        value=x_hist_flat,
        causal_mask=temporal_mask
    )

    return output
```

### 4.4 World State Encoder（潜空间世界模型）

核心创新：通过在潜空间预测未来状态，而非像素级重建

```python
def world_state_encoder_forward(
    z_t,           # 当前帧的潜表征
    z_target,      # 目标 encoder 编码的未来帧
    mask,          # 预测 mask
    alpha=0.5      # 融合系数
):
    """
    无泄露状态预测的核心实现

    z_t: [batch, seq, dim] - student 路径看到的当前帧
    z_target: [batch, seq, dim] - target encoder 编码的未来帧
    """
    # 1. 目标 encoder 处理未来帧（不梯度回传）
    with torch.no_grad():
        z_future = target_encoder(z_future_frames)
        z_future_masked = mask * z_future

    # 2. Student encoder 处理当前帧
    z_current = student_encoder(z_current_frames)

    # 3. 预测器网络
    z_predicted = predictor_network(
        torch.cat([z_current, z_future_masked], dim=-1)
    )

    # 4. 重建目标（只在 masked 位置计算损失）
    loss = F.mse_loss(z_predicted, z_future_masked, reduction='none')
    loss = (loss * mask).sum() / mask.sum()

    return z_predicted, loss


def elbo_objective(z_t, z_target, mask):
    """
    ELBO 目标函数: Evidence Lower Bound

    L_ELBO = E_{q(z|z_{1:T})} [log p(z_future | z_{1:T})] - β * I(z; x)
    """
    # 重构损失 (预测未来状态)
    z_pred, recon_loss = world_state_encoder_forward(z_t, z_target, mask)

    # 对比正则化 (防止表征坍塌)
    contrastive_loss = contrastive_reg(z_t, z_pred)

    # 总损失
    total_loss = recon_loss + beta * contrastive_loss

    return total_loss, {'recon': recon_loss, 'contrastive': contrastive_loss}
```

### 4.5 Latent Action Model（潜动作模型）

```python
def latent_action_pretraining(
    video_frames,      # 视频帧序列
    actions,           # 对应的动作序列
    language_inst      # 语言指令
):
    """
    潜动作预训练的核心流程

    核心思想: 学习将动作序列压缩到离散的潜动作令牌空间
    """
    # 1. 视频编码
    video_features = v_jepa2_encoder(video_frames)  # [B, T, D]

    # 2. 语言编码
    lang_features = language_encoder(language_inst)   # [B, D]

    # 3. 跨模态融合
    fused = cross_modal_fusion(video_features, lang_features)

    # 4. 生成潜动作令牌
    # ⟨latent_i⟩ 复制 K 次，K = 24 / T（T 是视频长度）
    latent_actions = latent_predictor(fused)  # [B, T, K, D_latent]

    # 5. 动作用损失
    action_loss = flow_matching_loss(latent_actions, actions)

    return action_loss
```

### 4.6 Flow Matching Action Head

动作预测头采用 **DiT-B (Diffusion Transformer)** 架构：

```python
class FlowMatchingActionHead(nn.Module):
    """
    基于流匹配的动作预测头

    核心: 将动作预测建模为从噪声到真实动作的.flow
    """

    def __init__(self, latent_dim, action_dim, num_layers=12):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, latent_dim)
        self.dit_blocks = nn.ModuleList([
            DitBlock(dim=latent_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(latent_dim, action_dim)

    def forward(self, context, noisy_action, timestep):
        """
        context: [batch, seq, latent_dim] - 条件信息
        noisy_action: [batch, seq, action_dim] - 带噪声的动作
        timestep: [batch] - 时间步
        """
        # 1. 时间步嵌入
        t_emb = timestep_embedding(timestep, dim=self.dim)

        # 2. 输入投影
        x = self.input_proj(noisy_action)

        # 3. DiT blocks with adaptive norm
        for block in self.dit_blocks:
            x = block(x, context, t_emb)

        # 4. 输出投影得到速度场
        velocity = self.output_proj(x)

        return velocity

    def training_loss(self, action_true, context):
        """训练时: 计算流匹配损失"""
        # 采样时间步
        t = torch.rand(len(action_true))
        # 采样噪声
        noise = torch.randn_like(action_true)
        # 线性插值得到带噪声的动作
        action_noisy = t.view(-1, 1, 1) * action_true + (1 - t).view(-1, 1, 1) * noise

        # 预测噪声
        noise_pred = self.forward(context, action_noisy, t)

        # MSE 损失
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sampling(self, context, num_steps=50):
        """推理时: 从噪声开始，逐步去噪生成动作"""
        action = torch.randn(context.shape[0], context.shape[1], self.action_dim)

        # 线性时间表
        timesteps = torch.linspace(0, 1, num_steps)

        for t in timesteps:
            # 预测速度场
            v = self.forward(context, action, t)

            # Euler 步更新
            dt = 1.0 / num_steps
            action = action + v * dt

        return action
```

### 4.7 两阶段训练流程

```python
def train_vla_jepa(train_loader, val_loader, config):
    """
    VLA-JEPA 两阶段训练流程
    """
    # ========== 阶段 1: JEPA 预训练 ==========
    print("Stage 1: JEPA Pretraining")
    world_model = WorldStateEncoder(config).cuda()
    optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-4)

    for epoch in range(config.jepa_epochs):
        for batch in train_loader:
            video, lang, actions = batch

            # ELBO 目标
            loss, metrics = elbo_objective(
                z_t=world_model.encode(video[:, :-1]),
                z_target=world_model.encode_target(video[:, 1:]),
                mask=batch.mask
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"JEPA Step {step}: loss={loss:.4f}")

    # ========== 阶段 2: 动作头微调 ==========
    print("Stage 2: Action Head Fine-tuning")
    action_head = FlowMatchingActionHead(config).cuda()
    optimizer = torch.optim.AdamW(action_head.parameters(), lr=1e-5)

    # 冻结 world model
    for param in world_model.parameters():
        param.requires_grad = False

    for epoch in range(config.finetune_epochs):
        for batch in train_loader:
            video, lang, actions = batch

            # 获取条件信息
            context = world_model.encode(video)

            # 流匹配损失
            loss = action_head.training_loss(actions, context)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Action Step {step}: loss={loss:.4f}")

    return world_model, action_head
```

---

## 5. 实验结论

### 5.1 主实验结果

#### 5.1.1 LIBERO 基准

| 模型 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long |
|------|----------------|---------------|-------------|-------------|
| LAPA | 68.3% | 64.1% | 53.2% | 37.8% |
| **VLA-JEPA** | **82.1%** | **75.6%** | **67.4%** | **58.2%** |

#### 5.1.2 SimplerEnv 基准

| 模型 | CALVIN-ABC | CALVIN-Long |
|------|------------|-------------|
| LAPA | 71.2% | 42.3% |
| **VLA-JEPA** | **89.7%** | **76.4%** |

#### 5.1.3 真实机器人实验

| 任务 | 成功率 |
|------|--------|
| 抽屉开关 | 92% |
| 物体抓取 | 87% |
| 衣物折叠 | 71% |

### 5.2 消融实验分析

| 消融项 | 变体 | 性能变化 | 分析 |
|--------|------|----------|------|
| **Target Encoder** | 无 target encoder | -12.3% | 泄露检测导致性能下降 |
| **时间因果注意力** | 标准双向注意力 | -8.7% | 未来信息泄露影响泛化 |
| **流匹配动作头** | MSE 直接回归 | -6.2% | 流匹配提供更好的多模态动作分布 |
| **潜动作令牌数 K** | K=12 (vs K=24/T) | -4.1% | 更多令牌提供更细粒度动作表征 |
| **ELBO β 系数** | β=0 (无对比正则化) | -9.8% | 对比正则化防止表征坍塌 |

### 5.3 关键发现

1. **JEPA 预训练的有效性**：仅使用 10% 的标注数据，VLA-JEPA 即可达到完全监督训练的 95% 性能
2. **长程依赖建模**：在 100+ 步的长程任务中，VLA-JEPA 相比基线提升显著
3. **零样本泛化**：在未见过的物体类别上，VLA-JEPA 展现出良好的零样本能力

---

## 6. KnowHow（核心洞察）

1. **JEPA vs VAE 的本质区别**：JEPA 通过预测未来表征而非重建像素，避免了表征坍塌问题，同时学习到更有意义的时空表征

2. **双路径防泄露设计**：Target encoder（冻结）与 Student encoder（可学习）的分离，确保预测目标不包含当前时刻的信息

3. **时序因果注意力的实现**：通过在注意力计算中引入时间维度的下三角 mask，实现跨帧的因果信息流动

4. **流匹配的优势**：相比 MSE 直接回归，流匹配能够更好地建模多模态动作分布，生成更平滑、更真实的动作序列

5. **潜动作令牌的复制机制**：⟨latent_i⟩ 复制 K 次不是冗余设计，而是提供足够的表征容量来捕捉动作的细粒度变化

6. **两阶段训练的必要**：JEPA 预训练和动作头微调的分离，使得表示学习和动作生成能够各自优化到最优

7. **ELBO 目标的双重作用**：重构损失保证预测精度，对比正则化防止表征空间坍塌，两者缺一不可

8. **V-JEPA2 encoder 的空间因果性**：在图像/视频帧内部使用因果注意力，为后续的时序建模提供干净的空间表征基础

---

## 7. 总结

### 三大核心贡献

1. **JEPA 风格的 VLA 预训练框架**：首次将 JEPA 的无泄露状态预测思想应用于视觉-语言-动作联合学习

2. **时序因果注意力机制**：通过空间因果和时序因果的双重设计，确保模型在学习过程中不会泄露未来信息

3. **流匹配动作预测头**：基于 DiT-B 架构的动作生成模块，提供高质量、多模态的动作预测

### 训练范式创新

- **阶段一**：JEPA 预训练，学习通用视觉-语言表示
- **阶段二**：动作头微调，学习任务相关的动作预测

### 局限性与未来方向

1. **计算成本**：时序因果注意力在长序列上计算复杂度较高
2. **多机器人协同**：当前仅支持单机器人场景
3. **在线学习**：探索如何在部署后持续学习新任务

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2602.10098](https://arxiv.org/abs/2602.10098) |
| **代码** | [GitHub: facebookresearch/VLA-JEPA](https://github.com/facebookresearch/VLA-JEPA) |
| **项目主页** | [Project Page](https://vla-jepa.github.io) |

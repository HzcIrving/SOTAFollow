# Uni-World VLA 论文精读报告

> **论文**: Interleaved World Modeling and Planning for Autonomous Driving
> **arXiv**: [2603.27287](https://arxiv.org/abs/2603.27287)
> **作者**: Qiqi Liu\*, Huan Xu\*, Jingyu Li, Bin Sun†, Zhihui Hao†, Dangen She, Xiatian Zhu, Li Zhang‡
> **机构**: 复旦大学 · 理想汽车（Li Auto）· 萨里大学
> **会议**: ECCV 2026
> **日期**: 2026年3月28日

---

## 1. 一句话总结

Uni-World VLA 提出了一种**交错式世界建模与规划**范式：模型不再先"看完"整个未来再规划，而是每预测一帧未来画面就立即据此调整动作，形成真正的闭环交互。在 NAVSIM 基准上仅用单目前视相机就达到 **PDMS 89.4**，超越了使用多传感器（相机+激光雷达）的方法。

---

## 2. 为什么这个问题值得研究？

### 2.1 现有的两条路，都走不通

做自动驾驶世界模型的人，目前主要有两种范式：

- **"并排跑"（predict-and-plan）**：世界模型和规划模块放在同一个架构里联合训练，但本质上各干各的——世界模型负责生成下一帧，规划模块负责从视觉映射到控制，两者之间没有真正的信息交互。这就像两个人坐在同一辆车里，但各看各的路。

- **"先看再想"（predict-then-plan）**：先生成完整的未来场景视频（比如 4 秒），然后基于生成的场景做规划。这个思路更接近人类的直觉，但有一个致命的隐含假设——环境是静止的。也就是说，它假设周围的车会按照某种固定模式运动，而自车也会一直执行最初的计划。

但真实世界不是这样的。在无保护左转或合流这种复杂场景里，交通态势瞬息万变。如果你在 0.5 秒时轻轻踩了一下刹车或打了一点方向，那么 3 秒时的场景就应该跟没踩刹车时完全不同。但"先看再想"的方法生成的是一条"冻结"的未来——它基于 $t=0$ 的意图生成了完整视频，规划器在 $t=3$ 秒看到的画面，根本没有反映 $t=0.5$ 秒时那一点微小的调整。

### 2.2 "冻结幻觉"——一个被忽视的痛点

论文把上面这个问题命名为 **Frozen Hallucination（冻结幻觉）**。这个名字很形象：世界模型"幻觉"出了一个未来，但这个幻觉在时间维度上是"冻结"的——它不随自车动作的调整而演化。

直觉告诉我们，一个真正智能的驾驶系统，应该像人类一样"走一步看一步"：每往前开一点，就重新审视环境，然后决定下一步怎么走。世界模型和规划之间需要真正的、持续的对话。

现有的三种世界模型范式对比：

| 范式 | 描述 | 核心问题 |
|------|------|----------|
| **(a) Predict-and-Plan** | 世界模型和规划器联合训练但功能解耦 | 两路信号没有真实交互 |
| **(b) Predict-then-Plan** | 先预测完整未来，再做规划 | 存在冻结幻觉，规划与预测脱节 |
| **(c) 本文方法（交错式）** | 交替预测帧+生成动作，闭环交互 | ✅ 解决了冻结幻觉 |

---

## 3. 方法详解：交错生成的闭环之美

### 3.1 核心思想——一步一回头

Uni-World VLA 的核心思想可以用一句话概括：

> **不要一口气看完未来，而是每往前走一步就停下来看看，然后再决定下一步怎么走。**

具体来说，模型在每一步先预测下一帧的未来画面，然后立即基于这个预测画面生成对应的动作。这个动作不是直接执行，而是反馈到模型内部，作为生成下一帧的条件。如此循环，形成一个交错式的预测-规划闭环。

这和传统方法的区别可以用一个比喻理解：

- **传统方法**：你站在山顶，用望远镜一口气看完整条路的 4 秒走势，然后决定怎么走。但你看到的画面是"冻结"的，路上如果突然有车变道，你的计划就废了。

- **Uni-World VLA**：你每走一步就用望远镜看一眼前方，根据看到的实际情况调整下一步。虽然看的更近，但每一步都是"活"的。

### 3.2 整体架构

整个框架由以下几个核心模块组成：

1. **双分支视觉 Tokenizer（MagVIT-v2）**：将历史帧编码为离散视觉 token，分为高分辨率上下文分支（256×448，448 token/帧）和低分辨率动态分支（128×224，28 token/帧），各自拥有独立的 8192 大小 codebook

2. **深度融合模块**：使用 Depth Anything 3 提取单目深度，通过 Cross-Attention 融合几何信息（CDE + DDE 双编码器）

3. **LLM 主干（Show-o / Phi-1.5）**：基于 Phi-1.5 的多模态 LLM，统一自回归生成未来帧 token 和动作 token

4. **解码器**：MagVIT-v2 解码器重建 RGB 帧，MLP 头输出轨迹点

### 3.3 输入与 Token 化

假设历史帧序列为 $\{I_{t-M}, \dots, I_{t-1}\}$（M 帧），每帧 $I \in \mathbb{R}^{H \times W \times 3}$。

模型将历史视频流分成两个互补的模态：

- **Contextual tokens（上下文 token）**：来自高分辨率帧（256×448），提供详细的场景语义和结构信息，每帧产生 448 个 token

- **Dynamic tokens（动态 token）**：以 10Hz 采样低分辨率帧（128×224），捕获细粒度的运动线索，每帧产生 28 个 token

视觉 Token 化过程：

$$
c, d = \text{Encoder}_{\text{MagVIT}}(I)
$$

其中 $c$ 是上下文 token，$d$ 是动态 token。

模型的输入构造为一个对话式上下文序列：

$$
[\text{System Prompt} \mid \text{Dynamic \& Contextual Tokens} \mid \text{User Prompt} \mid \text{Ego Tokens}]
$$

其中 Ego Tokens 拼接了自车当前的速度、加速度和高级驾驶命令。

### 3.4 交错生成——核心数学公式

这是论文最关键的部分。模型生成 $N=8$ 帧未来帧（每帧间隔 0.5 秒，共 4 秒视野）。

**第一步：预测未来帧的动态 token**

$$
\hat{d}_{t+k} \sim p_\theta(d_{t+k} \mid \hat{d}_{\leq t+k-1}, \hat{a}_{\leq t+k-1})
$$

这一步的含义是：在给定所有历史动态 token $\hat{d}_{\leq t+k-1}$ 和所有历史动作 token $\hat{a}_{\leq t+k-1}$ 的条件下，生成第 $t+k$ 步的动态 token。注意这里的因果约束——它只能看到过去，不能看到未来。

**第二步：基于预测帧生成动作**

$$
\hat{a}_{t+k} \sim p_\theta(a_{t+k} \mid \hat{d}_{\leq t+k}, \hat{a}_{\leq t+k-1})
$$

关键来了！动作 token 的条件里，包含了刚刚生成的未来帧 token $\hat{d}_{t+k}$（注意下标是 $\leq t+k$ 而不是 $\leq t+k-1$）。也就是说，规划器是在"看到"了预测的未来画面之后才做决定的。这就是"一步一回头"的数学表达。

**帧解码**

预测的动态 token 通过 MagVIT-v2 解码器还原为 RGB 帧，并使用上下文 token 作为视觉引导：

$$
\hat{I}_{t+k} = \text{Decoder}_{\text{MagVIT}}(\hat{d}_{t+k};\ c_{t+2\lfloor k/2 \rfloor})
$$

上下文 token $c$ 提供每秒级别的视觉引导（上下文帧采样频率较低，通过下标 $t+2\lfloor k/2\rfloor$ 进行时间对齐）。

**轨迹输出**

动作 token 通过 MLP 头回归出对应的自车位置 $\hat{a}_{t+1}, \hat{a}_{t+2}, \dots, \hat{a}_{t+N}$，构成 4 秒视野内的规划轨迹。

### 3.5 训练目标

训练时使用两个损失函数的加权和：

**视觉预测损失：动态焦点损失（Dynamic Focal Loss）**

作者发现，如果直接用交叉熵损失监督帧 token，大量 token 在相邻帧之间会保持不变（静态区域占主导），导致模型倾向于生成"静止"画面。为此引入了空间加权：

$$
\omega(d_{t+k}^i,\ d_{t+k-1}^i) = \alpha \cdot \mathbb{I}(d_{t+k}^i \neq d_{t+k-1}^i) + \beta \cdot \mathbb{I}(d_{t+k}^i = d_{t+k-1}^i), \quad \alpha > \beta
$$

其中 $\mathbb{I}(\cdot)$ 是指示函数，$\alpha$ 和 $\beta$ 是超参数（$\alpha > \beta$）。这个设计的直觉是：让模型更关注发生了变化（动态）的区域，而不是大片不变的背景。

最终视觉损失：

$$
\mathcal{L}_{\text{dyn}} = -\frac{1}{N}\sum_{k=1}^{N}\sum_{i=1}^{L} \omega(d_{t+k}^i,\ d_{t+k-1}^i) \log p_\theta(d_{t+k}^i \mid \hat{d}_{<t+k}, \hat{a}_{<t+k})
$$

**轨迹预测损失：**

$$
\mathcal{L}_{\text{traj}} = \frac{1}{N}\sum_{k=1}^{N} \|\hat{a}_{t+k} - a_{t+k}\|_1
$$

**总损失：**

$$
\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{\text{dyn}} + \lambda_2 \cdot \mathcal{L}_{\text{traj}}
$$

### 3.6 深度信息融合

为了让世界模型有更好的空间感知能力，论文引入了单目深度信息。使用 Depth Anything 3 从输入图像提取深度图：

$$
D = \text{DepthAnything3}(I), \quad I \in \mathbb{R}^{H \times W \times 3},\ D \in \mathbb{R}^{H \times W}
$$

深度图被缩放到两个分辨率（256×448 和 128×224），分别送入两个深度编码器（基于 MagVIT-v2）：

- **CDE（Context-Depth-Encoder）**：编码高分辨率深度特征

- **DDE（Dynamic-Depth-Encoder）**：编码低分辨率深度特征

融合方式是 Cross-Attention——视觉 token embedding 作为 Query，深度特征作为 Key 和 Value：

$$
E_{q,c} = \text{Embed}(c), \quad E_{q,d} = \text{Embed}(d)
$$

$$
E_{\text{fused},c} = \text{CA}(E_{q,c},\ D_{k,c},\ D_{v,c}), \quad E_{\text{fused},d} = \text{CA}(E_{q,d},\ D_{k,d},\ D_{v,d})
$$

融合后的特征送入 LLM 主干。这个设计很轻量——不需要在未来帧生成时显式地建模深度，只需要在历史帧上额外注入几何信息即可。

### 3.7 注意力掩码设计

论文采用了帧内双向 + 帧间因果的注意力掩码（沿用了 PWM 的设计）：

- **帧内**：同一个未来帧的 token 之间可以互相 attend（双向），捕捉空间依赖

- **帧间**：严格因果掩码，未来帧只能 attend 过去帧，保证时序因果性

这个设计让模型既能"看清"同一帧内的空间结构，又能"尊重"时间的流向。

---

## 4. 训练与推理伪代码（带详细注释）

### 4.1 训练阶段

```python
# ===== Uni-World VLA 训练伪代码 =====
# 两阶段渐进式深度融合训练

# --- 阶段 1：深度特征提取模块预训练 ---
# 冻结基础模型（Show-o/Phi-1.5）权重，只训练 CDE 和 DDE
# 采用无动作（action-free）视频预测设置，仅生成 1 秒内 10Hz 未来帧

for epoch in range(5):                      # 共 5 个 epoch
    for batch in dataloader:
        # 1. 提取单目深度图（Depth Anything 3，推理时冻结）
        depth_maps = DepthAnything3(batch.images)

        # 2. 双分支 Token 化（MagVIT-v2，双 codebook 各 8192）
        ctx_tokens, dyn_tokens = MagVIT_Encoder(batch.images)
        #   ctx_tokens: [B, 448, D]  ← 256×448 高分辨率，448 token/帧
        #   dyn_tokens: [B, 28, D]   ← 128×224 低分辨率，28 token/帧

        # 3. 深度特征编码（两个分辨率分别进入 CDE/DDE）
        ctx_depth_feat = CDE(depth_maps @ 256x448)  # 上下文深度
        dyn_depth_feat = DDE(depth_maps @ 128x224)  # 动态深度

        # 4. 视觉 token embedding 化，作为 Cross-Attention 的 Query
        E_q_c = Embed(ctx_tokens)  # [B, 448, D]
        E_q_d = Embed(dyn_tokens)  # [B, 28, D]

        # 5. Cross-Attention 融合：视觉 Query attend 深度 KV
        E_fused_c = CrossAttn(Q=E_q_c, K=ctx_depth_feat, V=ctx_depth_feat)
        E_fused_d = CrossAttn(Q=E_q_d, K=dyn_depth_feat, V=dyn_depth_feat)

        # 6. 拼接输入序列，送入 LLM 自回归生成
        input_seq = concat([SysPrompt, E_fused_d, E_fused_c, UserPrompt, EgoTokens])
        pred_tokens = LLM.autoregressive(input_seq, targets=future_frame_tokens)

        # 7. 动态焦点损失（Dynamic Focal Loss）
        #    只训练 CDE/DDE 参数，LLM 权重冻结
        loss = DynamicFocalLoss(pred_tokens, future_frame_tokens)
        loss.backward()
        optimizer.step(lr=3e-5)  # 小学习率，稳定收敛


# --- 阶段 2：多模态联合训练 ---
# 冻结 CDE/DDE（保持已学到的深度特征提取能力）
# 解冻融合模块 + LLM 主干，进行交错式联合训练

for epoch in range(16):                     # 共 16 个 epoch
    for batch in dataloader:
        # === 步骤 1-5：与阶段 1 完全相同的前向过程（CDE/DDE 冻结前向）===
        # Step 1: 提取单目深度
        depth_maps = DepthAnything3(batch.images)              # 冻结权重

        # Step 2: 双分支 Token 化 → ctx_tokens, dyn_tokens
        ctx_tokens, dyn_tokens = MagVIT_Encoder(batch.images)  # 冻结权重

        # Step 3: 深度编码 → ctx_depth_feat, dyn_depth_feat
        ctx_depth_feat = CDE(depth_maps @ 256x448)             # 冻结权重
        dyn_depth_feat = DDE(depth_maps @ 128x224)             # 冻结权重

        # Step 4: Embedding → E_q_c, E_q_d
        E_q_c = Embed(ctx_tokens)
        E_q_d = Embed(dyn_tokens)

        # Step 5: Cross-Attention 融合 → E_fused_c, E_fused_d
        E_fused_c = CrossAttn(Q=E_q_c, K=ctx_depth_feat, V=ctx_depth_feat)
        E_fused_d = CrossAttn(Q=E_q_d, K=dyn_depth_feat, V=dyn_depth_feat)

        # === 步骤 6：构造交错式 token 序列 ===
        # 输入序列构造（Scheme E，严格 2Hz F→A 交替）：
        # [SysPrompt | E_fused_d | E_fused_c | UserPrompt | EgoTokens |
        #  [Dyn_t+1×28 | Act_t+1 | Dyn_t+2×28 | Act_t+2 | ... | Dyn_t+8×28 | Act_t+8]]
        input_seq = concat([
            SysPrompt,
            E_fused_d,              # 动态 token（来自历史帧）
            E_fused_c,              # 上下文 token（来自历史帧）
            UserPrompt,
            EgoTokens,               # 自车状态：速度 + 加速度 + 高级命令
            # 以下为 LLM 要生成的目标序列（自回归监督信号）：
            gt_dyn_tokens_k1,        # t+0.5s 未来帧动态 token（28 个）
            gt_action_token_k1,      # t+0.5s 动作 token（1 个）
            gt_dyn_tokens_k2,        # t+1.0s 未来帧动态 token（28 个）
            gt_action_token_k2,      # t+1.0s 动作 token（1 个）
            gt_dyn_tokens_k3,        # t+1.5s ...
            gt_action_token_k3,
            gt_dyn_tokens_k4,
            gt_action_token_k4,
            gt_dyn_tokens_k5,
            gt_action_token_k5,
            gt_dyn_tokens_k6,
            gt_action_token_k6,
            gt_dyn_tokens_k7,
            gt_action_token_k7,
            gt_dyn_tokens_k8,
            gt_action_token_k8,
        ])

        # === 步骤 7：LLM 自回归前向 → 获得预测 token ===
        # 帧内双向注意力（同一时间步内），帧间因果注意力（跨时间步）
        # LLM 根据 input_seq 自回归生成对应的预测 token 序列
        all_pred_tokens = LLM.autoregressive(input_seq)
        # 切分输出：
        #   pred_dyn_k1~k8 = all_pred_tokens 中对应位置（共 8×28 = 224 个）
        #   pred_act_k1~k8 = all_pred_tokens 中对应位置（共 8 个）

        # === 步骤 8：解码 ===
        # 动态 token → 解码为 RGB 帧序列
        pred_frames = MagVIT_Decoder(pred_dyn_tokens)          # [8 帧未来帧]
        # 动作 token → MLP 解码为轨迹点
        pred_trajectory = MLP_Head(pred_action_tokens)         # [8 个轨迹点]

        # === 步骤 9：计算双重损失 ===
        loss_dyn  = DynamicFocalLoss(pred_dyn_tokens,  gt_dyn_tokens)    # 视觉损失
        loss_traj = L1Loss(pred_trajectory, gt_trajectory)              # 轨迹损失

        # === 步骤 10：加权总损失 + 反向传播 ===
        total_loss = λ₁ * loss_dyn + λ₂ * loss_traj
        total_loss.backward()
        optimizer.step(lr=2e-5)  # cosine annealing schedule
```

### 4.2 推理阶段

```python
# ===== Uni-World VLA 推理伪代码 =====

def inference(history_frames, ego_state):
    # 输入：2 秒历史帧 + 自车当前状态（速度、加速度、驾驶命令）
    # 输出：8 帧未来帧（4 秒视野）+ 对应轨迹点

    # 1. 提取单目深度
    depth_maps = DepthAnything3(history_frames)

    # 2. 双分支 Token 化 + 深度 Cross-Attention 融合
    ctx_tokens, dyn_tokens = MagVIT_Encoder(history_frames)
    E_fused = DepthCrossAttention(ctx_tokens, dyn_tokens, depth_maps)

    # 3. 构造输入序列并初始化 KV-Cache
    input_seq = concat([SysPrompt, E_fused, UserPrompt, EgoTokens])
    kv_cache = LLM.init_kv_cache(input_seq)

    planned_trajectory = []

    for k in range(1, 9):  # 生成 8 帧，每帧间隔 0.5 秒
        # Step A: 生成第 t+k 帧的动态 token（28 个）
        dyn_tokens_k = LLM.generate_step(
            prompt=f"[预测 t+{k*0.5}s 的未来帧]",
            kv_cache=kv_cache,
            num_tokens=28
        )
        kv_cache.update(dyn_tokens_k)

        # Step B: 解码为 RGB 帧（用上下文 token 做条件引导）
        ctx_guide = ctx_tokens[t + 2*(k//2)]  # 线性插值对齐
        frame_k = MagVIT_Decoder(dyn_tokens_k, condition=ctx_guide)

        # Step C: 基于刚预测的画面，生成同时间步的动作
        #   条件中包含了 d̂_{t+k} → "看到了未来才决策"
        action_token_k = LLM.generate_step(
            prompt=f"[基于刚预测的画面，规划 t+{k*0.5}s 的动作]",
            kv_cache=kv_cache,
            num_tokens=1
        )
        kv_cache.update(action_token_k)

        # Step D: MLP 解码轨迹点
        ego_position_k = MLP_Head(action_token_k)
        planned_trajectory.append(ego_position_k)

        # Step E: 新 token 追加到 KV-Cache → 闭环关键！

    return {
        "future_frames": [frame_1, ..., frame_8],  # 4 秒未来帧序列
        "trajectory": planned_trajectory            # 规划轨迹
    }
```

推理效率优化：KV-Cache 复用之前步骤的 key-value 表示，LLM 每步只需为新生成的 token 计算 attention，无需重新处理整个序列。

---

## 5. 实验结果

### 5.1 闭环规划性能（NAVSIM Test Split）

| 方法 | 输入 | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | 舒适度 ↑ | PDMS ↑ |
|------|------|------|-------|------|-------|----------|--------|
| VADv2 | 多目相机 | 97.2 | 89.1 | 76.0 | 91.6 | 100.0 | 80.9 |
| UniAD | 多目相机 | 97.8 | 91.9 | 78.8 | 92.9 | 100.0 | 83.4 |
| DiffusionDrive | 相机+LiDAR | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| DriveVLA-W0 | 单目相机 | 98.4 | 95.3 | 80.9 | 95.4 | 100.0 | 87.2 |
| PWM | 单目相机 | 98.6 | 95.9 | 81.8 | 95.4 | 100.0 | 88.1 |
| ResWorld | 相机+LiDAR | 98.9 | 96.5 | 83.1 | 95.6 | 100.0 | 89.0 |
| **Uni-World VLA（Ours）** | **单目相机** | **98.7** | **96.7** | **83.2** | **96.1** | **100.0** | **89.4** |

几个值得关注的点：

1. **单目吊打多传感器**：Uni-World VLA 只用前视单目相机，PDMS 89.4 超过了 ResWorld（相机+激光雷达，89.0）。交错式闭环本身的价值可能比多传感器融合更大。

2. **EP 和 TTC 全场最优**：EP（ego progress，自车前进效率）83.2 和 TTC（time-to-collision，碰撞安全）96.1 都是最高——既跑得快又最安全。

3. **舒适度满分**：100.0，规划的轨迹非常平滑。

4. **PDMS 的五个子指标中**，NC 略低于 ResWorld（98.7 vs 98.9），但综合得分更高，说明各维度平衡更优。

### 5.2 视频生成质量（FVD）

在 NAVSIM 4s/2Hz 协议下，Uni-World VLA 的 FVD 为 141.8，优于 DrivingGPT、SVD、GenAD 等基线方法。更重要的是，它是所有对比方法中唯一一个同时实现顶级规划性能和顶级视频生成质量的。

---

## 6. 消融实验——每个设计选择到底值多少？

### 6.1 三大组件的分别贡献

| 配置 | NC | DAC | EP | TTC | 舒适度 | PDMS | FVD |
|------|-----|-----|-----|-----|--------|------|-----|
| 无预训练 | 95.7 | 91.5 | 76.3 | 91.3 | 100.0 | 82.1 | — |
| 无未来帧生成 | 98.6 | 95.3 | 80.5 | 95.1 | 100.0 | 88.2 | — |
| 无深度融合 | 98.8 | 96.3 | 81.8 | 95.8 | 100.0 | 89.2 | 164.2 |
| **完整模型** | **98.7** | **96.7** | **83.2** | **96.1** | **100.0** | **89.4** | **141.8** |

消融结论：
- **深度融合**：去除后 PDMS 从 89.4 降到 89.2，FVD 从 141.8 恶化到 164.2——深度信息主要影响视频生成质量，对规划性能贡献有限
- **未来帧生成**：去除后 PDMS 降到 88.2，是最大的单一贡献来源
- **预训练阶段**：去除后 PDMS 大幅降到 82.1，说明两阶段训练策略至关重要

### 6.2 交错排列方案对比（Scheme E vs Others）

论文测试了 5 种不同的帧-动作排列方案，最终 Scheme E（严格 2Hz F→A 交替）取得最优性能：

$$
[\text{Prompt}] \to [0.5\text{s F}] \to [0.5\text{s A}] \to [1.0\text{s F}] \to [1.0\text{s A}] \to \dots \to [4.0\text{s A}]
$$

Scheme E 相比其他方案的关键优势：动作 token 只在对应帧之后出现，保证规划决策严格基于已生成的画面。

### 6.3 动态焦点损失的作用

消融实验显示，使用标准交叉熵损失时，模型生成的视频存在明显的"运动停滞"问题（大量 token 趋于生成相同的背景区域）。引入动态焦点损失后：

- 动态区域 token 的梯度权重提升（$\alpha > \beta$）
- 视频生成 FVD 从 164.2 降至 141.8
- 弯道场景的运动预测更加连续稳定

---

## 7. 附录关键内容

### 7.1 实现细节

| 项目 | 配置 |
|------|------|
| **Tokenizer** | MagVIT-v2，双分支 codebook（各 8192 tokens） |
| **上下文分支** | 256×448 分辨率，每帧 448 tokens |
| **动态分支** | 128×224 分辨率，每帧 28 tokens |
| **LLM 主干** | Show-o / Phi-1.5 |
| **深度模型** | Depth Anything 3（推理时冻结） |
| **训练阶段 1** | 5 epochs，冻结 LLM，lr = $3 \times 10^{-5}$ |
| **训练阶段 2** | 16 epochs，解冻 LLM，lr = $2 \times 10^{-5}$（cosine annealing） |
| **规划视野** | 8 帧 × 0.5 秒 = 4 秒 |
| **注意力掩码** | 帧内双向 + 帧间因果（PWM 风格） |

### 7.2 数据与硬件

- **数据**：理想汽车内部量产数据
- **深度监督**：Depth Anything 3 伪标签
- **训练硬件**：8× NVIDIA A100
- **开源状态**：代码和模型权重即将开源（截至论文发表时）

### 7.3 更多可视化分析

论文在附录中提供了丰富的可视化对比，包括：

- **弯道场景**：Uni-World VLA 预测的运动更稳定，规划轨迹更贴合可行驾驶行为
- **加减速场景**：闭环更新后，自车行为对周围车辆的"拉动效应"清晰可见
- **对比基线**：与 PWM（无闭环）、无深度融合版本的可视化对比

---

## 8. 总结

Uni-World VLA 提出了一种简洁但深刻的范式转换：**从"先看完整再规划"到"走一步看一步"**。通过交错式生成未来帧和动作 tokens，它首次明确提出并解决了"冻结幻觉"问题，实现了真正的闭环世界建模。

核心贡献：
1. **命名并解决了 Frozen Hallucination 问题**，为世界模型研究提供了新的视角
2. **交错式闭环范式**：动作 token 的条件包含刚预测的未来帧——规划决策"看到"了未来
3. **深度 Cross-Attention 融合**：轻量注入几何信息，不增加推理负担
4. **单目相机的强大性能**：PDMS 89.4 超过多传感器方案，证明了算法本身的价值

**局限性**：依赖大量真实驾驶数据训练（理想内部数据），泛化性待验证；交错生成相比开环推理有额外的计算开销。

---

## 附录：数学符号汇总表

| 符号 | 含义 |
|------|------|
| $I_t$ | $t$ 时刻的输入图像 |
| $c_t$ | $t$ 时刻的上下文 token（高分辨率，448 维） |
| $d_t$ | $t$ 时刻的动态 token（低分辨率，28 维） |
| $\hat{a}_{t+k}$ | 预测的 $t+k$ 时刻自车动作/轨迹点 |
| $\hat{d}_{t+k}$ | 预测的 $t+k$ 时刻动态 token |
| $\hat{I}_{t+k}$ | 预测的 $t+k$ 时刻 RGB 帧 |
| $D$ | Depth Anything 3 提取的单目深度图 |
| $\mathcal{L}_{\text{dyn}}$ | 动态焦点损失（视觉预测损失） |
| $\mathcal{L}_{\text{traj}}$ | L1 轨迹回归损失 |
| $\omega(\cdot)$ | 动态焦点加权函数（$\alpha > \beta$） |
| CDE / DDE | 上下文/动态深度编码器 |
| CA | Cross-Attention（视觉 Query × 深度 KV） |

---

*精读日期: 2026-04-04 | 最后同步: 2026-04-06*

# Gated Memory Policy (GMP) 精读报告

> **论文**：Gated Memory Policy: Learning When and What to Recall for Visuomotor Manipulation
> **arXiv**：2604.18933
> **作者**：Yihuai Gao, Jinyun Liu, Shuang Li, Shuran Song
> **机构**：Stanford University, Carnegie Mellon University
> **顶会/顶刊**：arXiv 2026 (cs.RO)
> **发布日期**：2026-04-21
> **代码/项目**：[gated-memory-policy.github.io](https://gated-memory-policy.github.io/)

---

## 一句话总结

GMP 通过**学习何时召回记忆**（Memory Gate）+ **学习召回什么**（Cross-Attention History Conditioning）+ **扩散噪声增强鲁棒性**三重设计，让视觉运动策略在无需人工设计规则的情况下，自适应地在 Markov 任务和 Non-Markov 任务之间切换，在 MemMimic 基准上相对长历史基线提升 **30.1%** 成功率。

---

## 核心贡献

1. **Gated Memory Policy**：首个同时解决"何时召回"和"召回什么"的视觉运动策略，在 Markov 和 Non-Markov 任务上均保持 SOTA 性能
2. **自监督 Gate 校准**：通过对比"有/无记忆"两策略的动作预测误差生成二值监督信号，避免端到端训练的梯度冲突，无需手工调参
3. **Cross-Attention 时序融合**：替代自注意力，复杂度从 \(\mathcal{O}(H^{2})\) 降至 \(\mathcal{O}(h \times H)\)，配合 KV Cache 实现线性历史检索
4. **扩散噪声一致性**：训练/推理均对历史动作加噪，避免过度依赖干净历史，显著提升鲁棒性
5. **MemMimic 基准**：首个覆盖 In-Trial 和 Cross-Trial 记忆评估的机器人操作基准

---

## 对自动驾驶时序融合的启发与借鉴点

### 1. 时序编码机制（Positional Encoding）

GMP 的历史存储机制对自动驾驶感知的时间编码有直接借鉴意义：

**GMP 的时序信息隐式编码在 KV Cache 中**，每个历史时间步的 token 包含：
- 图像 token（ViT 编码的视觉特征）
- 动作 token（历史动作序列）
- 时间步索引（通过序列位置隐式区分）

**自动驾驶的时序编码设计**：

| 组件 | GMP 机器人 | 自动驾驶 |
|------|-----------|---------|
| **历史内容** | 图像 + 动作 tokens | BEV 特征 + 轨迹 + 雷达点云 |
| **时间编码** | 隐式（位置索引） | 显式（时间戳 + 帧间隔） |
| **存储结构** | KV Cache（滑动窗口） | 轨迹队列 + 特征缓冲区 |

```python
class TemporalEncoder:
    """
    自动驾驶时序编码器设计
    借鉴 GMP 的隐式时序编码思路
    """
    def __init__(self, feature_dim, history_len=30, time_encoding="sinusoidal"):
        self.feature_dim = feature_dim
        self.history_len = history_len
        self.time_encoding = time_encoding

        if time_encoding == "sinusoidal":
            # 方案1: Sinusoidal PE（GMP 隐式思路）
            self.pe = self._build_sinusoidal_pe(history_len, feature_dim)
        elif time_encoding == "learned":
            # 方案2: 可学习的时间嵌入
            self.pe = nn.Embedding(history_len, feature_dim)
        elif time_encoding == "relative":
            # 方案3: 相对时间编码（Δt 编码）
            self.time_delta_net = MLP(input_dim=1, hidden=64, output=feature_dim)

    def _build_sinusoidal_pe(self, max_len, d_model):
        """Sinusoidal PE: 不同频率的正弦波编码时间步"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [max_len, d_model]

    def encode_frame(self, features, frame_idx, time_delta=None):
        """
        对单帧进行时序编码
        features: [B, C, H, W] 当前帧特征
        frame_idx: 当前帧在历史中的位置
        time_delta: 距离上一帧的时间间隔（秒）
        """
        B = features.shape[0]

        # 空间特征压缩
        spatial_tokens = self.vit_encoder(features)  # [B, N, C]

        # 时间编码融合
        if self.time_encoding == "sinusoidal":
            # 方案1: 全局位置编码
            time_emb = self.pe[frame_idx % self.history_len]  # [C]
            time_emb = time_emb.unsqueeze(0).expand(B, -1)  # [B, C]
            enhanced_tokens = spatial_tokens + time_emb.unsqueeze(1)

        elif self.time_encoding == "relative":
            # 方案3: 相对时间编码（更适合变帧率场景）
            delta_emb = self.time_delta_net(time_delta)  # [B, C]
            enhanced_tokens = spatial_tokens + delta_emb.unsqueeze(1)

        return enhanced_tokens  # [B, N, C]
```

### 2. 时空融合机制 / 时序融合机制

GMP 的 Cross-Attention 历史融合机制是自动驾驶感知融合的核心参考：

**GMP 的时序融合架构**：
- **输入**：当前帧 DiT hidden states（Query）vs 历史 KV Cache
- **机制**：Cross-Attention，非自注意力（历史之间不交互）
- **复杂度**：$\mathcal{O}(h \times H)$ 而非 $\mathcal{O}(H^2)$

**自动驾驶的时空融合设计**：

```python
class SpatiotemporalFusion:
    """
    自动驾驶时空融合模块
    借鉴 GMP Cross-Attention 线性复杂度设计
    """
    def __init__(self, query_dim, kv_dim, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 历史 KV 缓存
        self.kv_cache = []  # List of [B, N_kv, D]
        self.max_history = 30

    def fuse(self, current_query, historical_features):
        """
        时序融合前向传播
        current_query: [B, N_q, D] 当前帧 Query
        historical_features: [B, H*N_kv, D] 历史帧级联特征
        """
        # 方法1: 直接拼接融合（GMP 原始方式）
        fused = torch.cat([current_query, historical_features], dim=1)
        # 通过交叉注意力让 Query 与历史交互
        attn_output, _ = self.cross_attn(
            query=current_query,
            key=historical_features,
            value=historical_features
        )
        return attn_output

    def fuse_with_temporal_weight(self, current_query, historical_features, temporal_weights):
        """
        方法2: 加权融合（融合权重随时间衰减）
        temporal_weights: [B, H] 各历史帧的权重
        """
        B, H, N_kv, D = historical_features.shape
        # 调整权重形状: [B, H, 1, 1]
        weights = temporal_weights.view(B, H, 1, 1)

        # 加权平均历史特征
        weighted_history = (historical_features * weights).sum(dim=1)  # [B, N_kv, D]

        # 与当前 Query 拼接
        fused = torch.cat([current_query, weighted_history], dim=1)

        attn_output, _ = self.cross_attn(
            query=current_query,
            key=fused,
            value=fused
        )
        return attn_output

    def update_cache(self, new_features):
        """更新历史 KV 缓存（滑动窗口）"""
        self.kv_cache.append(new_features)
        if len(self.kv_cache) > self.max_history:
            self.kv_cache.pop(0)  # 移除最老的帧

    def get_historical_features(self):
        """获取历史级联特征"""
        if len(self.kv_cache) == 0:
            return None
        return torch.cat(self.kv_cache, dim=1)  # [B, H*N_kv, D]
```

### 3. Gated 机制设计

GMP 的 Memory Gate 是决定"何时使用历史"的核心机制：

**GMP 的门控设计**：
- **结构**：MLP 二分类器（sigmoid 输出）
- **输入**：当前帧图像 + 本体感受
- **输出**：二值决策 $\mu_t \in \{0, 1\}$（或软门控版本）
- **校准方式**：两阶段对比学习

**自动驾驶的门控机制设计**：

```python
class TemporalGate:
    """
    时序融合门控模块
    借鉴 GMP Memory Gate 的两阶段校准思路
    """
    def __init__(self, perception_dim, hidden_dim=64):
        # 门控网络：判断当前是否需要融合历史
        self.gate_mlp = nn.Sequential(
            nn.Linear(perception_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 备选：可学习的二值阈值
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def compute_gate(self, current_perception, ego_motion=None):
        """
        计算门控值
        current_perception: [B, C] 当前帧 BEV 特征或感知状态
        ego_motion: [B, 6] 自车运动（平移+旋转），用于判断场景变化程度
        """
        if ego_motion is not None:
            # 融合自车运动信息（速度大、转弯急 → 更需要历史）
            gate_input = torch.cat([current_perception, ego_motion], dim=-1)
        else:
            gate_input = current_perception

        # 预测门控概率
        gate_prob = self.gate_mlp(gate_input)  # [B, 1]

        # 软门控版本
        return gate_prob

    def compute_binary_gate(self, current_perception, ego_motion=None):
        """二值门控版本（GMP 方式）"""
        gate_prob = self.compute_gate(current_perception, ego_motion)
        return (gate_prob > self.threshold).float()  # {0, 1}


class GatedTemporalFusion:
    """
    完整的门控时序融合模块
    """
    def __init__(self, feature_dim):
        self.temporal_encoder = TemporalEncoder(feature_dim)
        self.spatial_fusion = SpatiotemporalFusion(feature_dim)
        self.gate = TemporalGate(feature_dim)

    def forward(self, current_frame, historical_cache, ego_motion=None):
        """
        前向传播
        current_frame: [B, C, H, W] 当前帧
        historical_cache: List of 历史帧特征
        ego_motion: [B, 6] 自车运动
        """
        # 1. 时序编码
        current_tokens = self.temporal_encoder.encode_frame(current_frame, frame_idx=0)

        # 2. 获取历史特征
        if historical_cache is not None and len(historical_cache) > 0:
            historical_tokens = torch.cat(historical_cache, dim=1)
        else:
            historical_tokens = None

        # 3. 计算门控
        gate_value = self.gate.compute_gate(
            current_perception=current_tokens.mean(dim=1),  # [B, C]
            ego_motion=ego_motion
        )

        # 4. 门控融合
        if historical_tokens is not None:
            # 当 gate_value > 0.5 时融合历史
            fused = gate_value * self.spatial_fusion.fuse(current_tokens, historical_tokens)
            fused = fused + (1 - gate_value) * current_tokens
        else:
            fused = current_tokens

        return fused, gate_value
```

### 4. When / How / What 三问

#### 4.1 什么时候触发时序融合？（When）

| 触发条件 | GMP 机器人实现 | 自动驾驶实现 |
|----------|--------------|-------------|
| **基于误差对比** | 校准阶段计算 $\delta_t^{mem}$ vs $\delta_t$ | 感知不确定性估计 |
| **基于场景检测** | — | 换道、交叉路口、紧急制动 |
| **基于自车运动** | — | 速度变化率、转弯角度 |
| **基于感知差异** | — | 当前帧 vs 历史帧的 BEV 特征差异 |

```python
def should_activate_temporal(perception_diff, ego_motion_change, scene_type):
    """
    判断是否触发时序融合
    perception_diff: 当前帧与历史帧的感知差异
    ego_motion_change: 自车运动变化（速度，加速度，转角）
    scene_type: 场景类型（高速/城市/交叉路口）
    """
    # 触发条件组合
    triggers = []

    # 条件1: 感知差异大（新增目标、目标消失）
    if perception_diff > 0.3:
        triggers.append(1.0)

    # 条件2: 自车运动变化剧烈（急刹、急转）
    motion_change_norm = torch.norm(ego_motion_change, dim=-1)
    if motion_change_norm > 0.5:
        triggers.append(1.0)

    # 条件3: 关键场景（换道、路口）
    if scene_type in ["lane_change", "intersection", "emergency"]:
        triggers.append(1.0)

    # 满足任一条件则触发
    return torch.any(torch.stack(triggers) > 0, dim=0)
```

#### 4.2 怎么设计时序模块？（How）

| 设计维度 | GMP 方案 | 自动驾驶借鉴 |
|----------|---------|-------------|
| **存储结构** | KV Cache（滑动窗口） | 轨迹队列 + 特征缓冲区 |
| **融合机制** | Cross-Attention | Cross-Attention 或门控加法 |
| **复杂度控制** | $O(h \times H)$ | $O(N \times H)$ 目标数 × 历史帧数 |
| **模块位置** | DiT 内部条件 | BEV 编码器后 / 感知头前 |

#### 4.3 时序存什么历史？（What）

| 历史内容 | GMP 存储 | 自动驾驶存储 |
|----------|---------|-------------|
| **视觉信息** | 图像 token（64 个/帧） | BEV 特征图（H × W grid） |
| **运动信息** | 历史动作序列 | 自车轨迹 + 他车轨迹 |
| **感知状态** | — | 检测框、跟踪 ID、语义地图 |
| **时间信息** | 隐式位置 | 显式时间戳 + Δt |

```python
class HistoryBuffer:
    """
    自动驾驶时序历史缓冲区
    """
    def __init__(self, max_frames=30):
        self.max_frames = max_frames

        # 历史数据结构
        self.bev_features = []      # BEV 特征图
        self.ego_trajectory = []      # 自车轨迹 [x, y, theta, v]
        self.obj_trajectories = []   # 他车轨迹 Dict[track_id -> [x, y, theta]]
        self.timestamps = []         # 时间戳
        self.semantic_map = []       # 语义地图（可选）

    def update(self, bev, ego_state, obj_detections, timestamp, sem_map=None):
        """更新历史缓冲区"""
        self.bev_features.append(bev.detach().clone())
        self.ego_trajectory.append(ego_state)
        self.obj_trajectories.append(obj_detections)
        self.timestamps.append(timestamp)
        if sem_map is not None:
            self.semantic_map.append(sem_map)

        # 滑动窗口
        if len(self.bev_features) > self.max_frames:
            self.bev_features.pop(0)
            self.ego_trajectory.pop(0)
            self.obj_trajectories.pop(0)
            self.timestamps.pop(0)
            if self.semantic_map:
                self.semantic_map.pop(0)

    def get_temporal_features(self, frame_indices=None):
        """获取指定帧的历史特征"""
        if frame_indices is None:
            frame_indices = list(range(len(self.bev_features))

        # 采样历史帧
        bev_hist = torch.stack([self.bev_features[i] for i in frame_indices], dim=1)  # [B, H, C, H, W]

        # 计算相对时间间隔
        current_time = self.timestamps[-1]
        time_deltas = [current_time - self.timestamps[i] for i in frame_indices]

        return {
            'bev_features': bev_hist,                    # [B, H, C, H, W]
            'ego_trajectory': self.ego_trajectory[-1],    # 最新自车状态
            'obj_trajectories': self.obj_trajectories,     # 所有跟踪目标
            'time_deltas': time_deltas,                   # 相对时间
        }
```

### 5. 扩散噪声一致性的工程价值

GMP 在训练和推理**均使用扩散噪声**，避免了 Diffusion Forcing 的训练-推理不一致问题。

**借鉴思路**：在自动驾驶的轨迹预测或行为预测中，若使用扩散模型，应保持训练和推理的噪声调度一致，而非在推理时去除噪声。

---

## 方法详述

### 问题定义

机器人操作任务按记忆需求分为三层：

| 任务类型 | 记忆跨度 | 示例 |
|---------|---------|------|
| Markov 任务 | 无需历史 | 简单抓取，单帧视觉+本体感受即可 |
| In-Trial Memory | 单次执行内 | Match Color 需记住箱子初始颜色 |
| Cross-Trial Memory | 跨试次总结 | Iterative Pushing 从推距推断摩擦力 |

**核心挑战**：一个好的记忆增强策略必须同时回答两个问题：
- **何时召回？**（When to recall）：在大多数时刻，记忆是噪音，策略应忽略它
- **召回什么？**（What to recall）：无法靠人工规则指定，必须端到端学习

### 整体 Pipeline

```
当前时间步 t
    │
    ├──→ ViT Encoder ─→ 聚合图像 token
    │
    ├──→ Memory Gate MLP(φ) ─→ μt ∈ {0,1}
    │
    └──→ DiT Hidden States (Query)
              │
              ▼
         ┌─────────────────────────┐
         │  Cross-Attention        │
         │  Query ← 历史 KV Cache  │
         └─────────────────────────┘
              │
              ▼
         历史上下文 ht:t+h
              │
              ▼
         z̄t:t+h = μt · ht:t+h + zt:t+h  (门控融合)
              │
              ▼
         DiT 去噪预测
```

### 核心数学公式

**1. Diffusion Policy 损失函数**（基础）：

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{A^0_{t:t+h}, \epsilon, k} \left[ \left\| A^0_{t:t+h} - \varphi_\theta\left(A^k_{t:t+h}, I_t, P_t, k\right) \right\|_2^2 \right]$$

**2. Memory Gate 二值门控**：

$$\mu_t = \mathbf{1}\{ \sigma(\phi(I_t, P_t)) > 0.5 \} \in \{0, 1\}$$

**3. 门控融合输出**：

$$\bar{\mathbf{z}}_{t:t+h} = \mu_t \cdot \mathbf{h}_{t:t+h} + \mathbf{z}_{t:t+h}$$

**4. Gate 校准阈值判断**：

$$\mu_t = \begin{cases} 1 & \text{if } \delta_t \geq \theta \cdot \delta_t^{\text{mem}} \\ 0 & \text{otherwise} \end{cases}$$

> **图 1：机器人操作任务的三层记忆需求**（对应论文 Figure 1）
>
> ![GMP 任务概览](https://arxiv.org/html/2604.18933v1/figures/jpg/teaser.jpg)
>
> - **(a) Markov 任务**：如简单抓取，无需历史，当前感知即可完成
> - **(b) In-Trial Memory**：单次执行内需要记忆上下文，例如 Match Color 任务需记住箱子初始颜色
> - **(c) Cross-Trial Memory**：跨 Trial 总结物理属性（如摩擦力、质量），通过试错迭代调整动作

> **图 2：GMP 网络架构**（对应论文 Figure 2）
>
> ![GMP 方法架构](https://arxiv.org/html/2604.18933v1/figures/jpg/network.jpg)
>
> - **左 (a)**：基于 DiT 的完整 GMP 框架，额外增加了 Gated Attention 模块
> - **右 (b)**：Gated Attention 模块三要素：① 二值门控 \(\mu_{t}\) 决定是否执行历史交叉注意力；② 加噪历史动作条件提升鲁棒性；③ KV Cache 缓存历史 token 降低计算成本

---

## 训练与推理伪代码

### 阶段一：Memory Gate 校准（Calibration）

```python
# ============================================================
# 阶段一：Memory Gate 校准
# 输入：数据集 D_split = D_train ∪ D_val
# ============================================================

# Step 1: 划分数据集
D_train, D_val = split_dataset(D, ratio=0.5)

# Step 2: 在 D_train 上分别训练两个策略
pi_always_off = train_policy(D_train, gate_mode="off")     # π, μt=0 始终
pi_always_on  = train_policy(D_train, gate_mode="on")     # π_mem, μt=1 始终

# Step 3: 在 D_val 上采样，计算各时刻动作预测误差
trajectories_off = sample(pi_always_off, D_val, N_rounds=N)
trajectories_on  = sample(pi_always_on, D_val, N_rounds=N)

errors_off = compute_errors(trajectories_off)   # δ_t: 无记忆策略在时刻 t 的误差
errors_on  = compute_errors(trajectories_on)    # δ_t^mem: 有记忆策略在时刻 t 的误差

# Step 4: 生成 Gate 标签
theta = 1.5  # 比率阈值（人工设定）
gate_labels = (errors_off >= theta * errors_on).float()  # 1=需要记忆, 0=不需要

# Step 5: 单独训练 Memory Gate MLP
gate_mlp = MLP(input_dim=dim(I_t)+dim(P_t), hidden=64, output=1, activation="sigmoid")
optimizer = Adam(gate_mlp.parameters(), lr=1e-3)

for epoch in range(num_gate_epochs):
    for batch in D_train:
        I_t, P_t = batch["image"], batch["proprio"]
        mu_pred  = gate_mlp(I_t, P_t)           # 预测概率
        mu_label = gate_labels[t]                # 真实标签
        loss_gate = BCE(mu_pred, mu_label)
        optimizer.zero_grad()
        loss_gate.backward()
        optimizer.step()

# 校准完成：冻结 Gate 权重
freeze(gate_mlp)
```

### 阶段二：完整策略微调（Final Policy Training）

```python
# ============================================================
# 阶段二：Gated Memory Policy 完整训练
# 输入：完整数据集 D = D_train ∪ D_val，标定好的 Gate MLP
# ============================================================

pi_gated = train_policy(D_full, gate_mode="calibrated", gate_mlp=gate_mlp)

# 训练目标
for batch in D_full:
    I_t, P_t, actions, k = batch["image"], batch["proprio"], batch["actions"], batch["diffusion_step"]

    # Memory Gate 前向
    mu_t = gate_mlp(I_t, P_t)   # 二值门: 0 或 1

    # 历史 KV Cache（滑动窗口）
    history_kvcache = build_kvcache(batch["history"], window_size=n)

    # 加噪历史动作条件（扩散调度）
    A_history_noised = add_diffusion_noise(A_history, k-1)

    # Cross-Attention 历史查询
    h_t = cross_attention(query=z_t, kv=history_kvcache)   # 历史上下文

    # 门控融合
    if mu_t == 1:
        z_bar = h_t + z_t   # 融合历史
    else:
        z_bar = z_t          # 跳过历史

    # 去噪预测
    A_pred = denoiser(z_bar, A_noisy, k)
    loss = MSE(A_pred, A_gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 输出：π_gated（既能在 Markov 任务上关闭记忆，又能在 Non-Markov 任务上按需召回）
```

---

## 实验结论

### 5.1 主实验结果：MemMimic 基准

GMP 提出了 MemMimic，一个专门评估视觉运动策略记忆能力的基准，包含 6 个任务：

| 任务 | 类型 | 描述 |
|------|------|------|
| T1: Match Color | In-Trial | 记住初始箱子颜色，放置立方体 |
| T2: Discrete Place Back | In-Trial | 记住原位置，悬空 2 秒后放回 |
| T3: Continuous Place Back | In-Trial (Real) | 真实机器人，杯子放回原位 |
| T4: Iterative Pushing | Cross-Trial | 从推距推断摩擦力，动态调整推力 |
| T5: Iterative Flinging | Cross-Trial | 从甩布结果推断质量，调整甩力 |
| T6: Iterative Casting | Cross-Trial (Real) | 从滑行距离推断摩擦系数 |

**关键结果**：GMP 相对长历史基线提升 **30.1%** 平均成功率。

### 5.2 消融实验

| 消融项 | 发现 |
|--------|------|
| **Calibration vs STE** | Calibration 显著优于 Straight-Through Estimator；STE 倾向于让门值趋向 1（过度使用记忆） |
| **Calibration vs 正则项** | 正则项方法对权重敏感，难以在 Markov/Non-Markov 两类任务上同时 work |
| **扩散噪声方案** | No Noise / Random Level / Diffusion Forcing 三种方案均不如 GMP 的 Diffusion Noising 方案 |

### 5.3 鲁棒性分析

- **计算效率**：120 步历史自注意力需要 ~0.5s/步，GMP (Gate Off) 仅需 ~0.05s/步，与无历史策略相当
- **长记忆测试**：T1' Match Color with Random Delay（5-600 秒随机延迟），GMP 在 6000 帧记忆 buffer 下达到 99.0%±1.0% 成功率，推理仅需 0.16 秒

### 5.4 定性分析

> **图 3：Cross-Attention 注意力可视化**（对应论文 Figure 4）
>
> ![GMP 注意力可视化](https://arxiv.org/html/2604.18933v1/figures/jpg/exp1_match_color.jpg)
>
> 上图展示了 Match Color 任务（t=80 时刻放置立方体，注意力落在 t=48 首次观察箱子颜色）和 Iterative Pushing 任务（第 4 次推动注意力集中在第 2、3 次推结果）。蓝色 = \(\mu_{t}=1\)（门开），灰色 = \(\mu_{t}=0\)（门关）

**关键发现**：
- Match Color 任务中，t=80 时刻放置立方体时，注意力权重最高点落在 t=48（首次观察箱子颜色的时刻）
- Iterative Pushing 中，第 4 次推的时刻，注意力集中在第 2、3 次推的结果（过推/欠推）

---

## KnowHow（核心洞察）

1. **Memory Gate 校准的 insight**：用"动作预测误差对比"这一自监督信号替代端到端训练中的正则项，避免了手工调参，优雅地解决了"何时用记忆"的问题。核心洞察是：**让数据本身告诉我们哪些时刻真正需要记忆**。

2. **Cross-Attention 替代自注意力的 insight**：历史 token 不需要与当前 token 做完整自注意力——历史信息只需要被"查询"，而不需要彼此交互。这将复杂度从 \(\mathcal{O}(H^{2})\) 降为 \(\mathcal{O}(h \times H)\)，对 \(H\) 是**线性**而非二次。

3. **门控的运行特点**：即使在 Non-Markov 任务上，门也**大部分时间关闭**（Match Color 中 73% 时刻关闭，Iterative Pushing 中 58% 关闭），仅在关键时刻（如需要回忆初始颜色、回顾上次推力结果时）才打开。这说明记忆应当是"稀缺资源"而非"默认选项"。

4. **扩散噪声一致性的 insight**：Diffusion Forcing 在训练时加随机噪声，但推理时不加噪声，导致训练-推理不一致。GMP 在训练和推理**都使用扩散噪声**，保证了两者的一致性。

5. **为什么端到端训练门控不可行？**：如果把门控和策略一起端到端训练，无正则时策略倾向于尽可能多地使用历史（Markov 任务严重过拟合）；加正则时如果权重过大，门值趋向于始终为 0（在 Non-Markov 任务上完全丧失记忆能力）。很难找到一个正则权重在两类任务上都 work。

6. **KV Cache 的关键优势**：推理时，如果 Gate 关闭，则完全跳过历史注意力，推理时间与无历史策略相同。零额外计算负担使得 GMP 可以处理任意长度历史而不影响推理延迟。

7. **视觉特征压缩的设计**：用 Multi-Head Attention Pooling (MAP) 将所有 patch token 聚合为**单一 token**，大幅压缩视觉 token 数量，每个历史时间点只对应一个聚合图像 token + \(h\) 个动作 token。

8. **校准流程的两阶段设计**：先固定 Gate 训练两个极端策略（常开/常闭），再基于误差对比生成监督信号训练 Gate，最后冻结 Gate 重新训练策略。这个流程避免了梯度冲突，让每个模块各自优化到最优。

---

## arXiv Appendix 关键点总结

**A. 超长记忆测试（T1' Match Color with Random Delay）**
- 5-600 秒随机延迟测试记忆长度
- GMP 在 6000 帧记忆 buffer 下达到 99.0%±1.0% 成功率
- 推理仅需 0.16 秒（8 步去噪，5090 GPU）

**B. Gate 校准消融（Finding 4）**
- 验证了 Calibration 策略优于 STE（Straight-Through Estimator）和正则项两种替代方案
- 正则项方法对权重敏感，难以找到通用权重
- STE 倾向于让门值趋向 1（过度使用记忆）

**C. 推理时间对比（Finding 5）**
- 在 RTX 3080 上，120 步历史自注意力需要 ~0.5s/步
- GMP (Gate Off) 仅需 ~0.05s/步，与无历史策略相当
- Gate 关闭时完全跳过历史注意力，零额外计算

**D. 噪声注入消融（Finding 6）**
- No Noise / Random Level / Diffusion Forcing 三种方案对比
- Diffusion Noising（GMP 方案）在 Iterative Pushing 上显著优于所有基线
- 验证了训练-推理噪声一致性的重要性

**E. 更多实验任务**
- exp2_discrete_place_back: 悬空 2 秒后放回原位
- exp3_continuous_place_back: 真实机器人实验，杯子放回原位
- exp7_robomimic: Markov 任务验证，GMP 与无历史策略性能相当

**F. 补充材料图**
- supp_in_the_wild_data.jpg: 真实场景数据采集
- supp_casting_pos_control.jpg: 位置控制可视化
- supp_gate_label_statistics.jpg: Gate 标签统计（73%/58% 门关闭比例验证）

---

## 总结

**3 大核心贡献**：
1. **Gated Memory Policy 架构**：首次同时解决"何时召回"和"召回什么"两个子问题，在 Markov 和 Non-Markov 任务上均保持 SOTA 性能
2. **自监督 Gate 校准**：通过两阶段误差对比生成监督信号，避免端到端训练的梯度冲突，无需手工调参
3. **Cross-Attention + 扩散噪声一致性**：复杂度从 \(\mathcal{O}(H^{2})\) 降至 \(\mathcal{O}(h \times H)\)，训练/推理噪声一致保证鲁棒性

**最重要洞察**：
GMP 的核心贡献不是某个新架构，而是**对"记忆"这件事的精准问题分解**：When/What/How Robust 三个子问题分别用 Gate/Cross-Attention/Diffusion Noise 解决，干净利落。尤其是 Gate 校准流程，用两个固定策略的误差对比来生成监督信号，避免了端到端训练的梯度冲突——这个思路非常值得借鉴到其他需要"选择性使用信息"的场景（如 attention 机制、skip connection、 gating network 等）。

---


## 参考链接

- **论文**：https://arxiv.org/abs/2604.18933
- **arXiv HTML**：https://arxiv.org/html/2604.18933v1
- **项目主页**：https://gated-memory-policy.github.io/
- **HuggingFace 模型**：https://huggingface.co/yihuai-gao/gated-memory-policy
- **HuggingFace 数据集**：https://huggingface.co/datasets/yihuai-gao/gated-memory-policy
# MoE 做 RL：训练-推理不一致问题

> 本文档结合 Routing Replay 机制与 GSPO 算法解析
>
> **视频来源**：丁师兄大模型 - [面试官：MoE模型做RL，训推不一致怎么办？](https://www.bilibili.com/video/BV1khQbBLE4u/)

---

## 1. 核心问题：训练时和推理时 MoE 行为不一致

### 1.1 现象描述

MoE（Mixture of Experts）模型中，训练阶段和推理阶段的 token-expert 分配路径不同：

| 阶段 | Expert 路由行为 |
|------|----------------|
| **推理** | 稀疏激活，每个 token 仅激活 Top-K 个 expert（如 Top-2/8），但这个决定不是确定性的——同一个输入跑两遍可能激活的专家不一样 |
| **训练（PPO/GRPO）** | 需要对所有 expert 计算 value function、优势函数，反向传播需全 expert 参与计算 |
| **核心矛盾** | RL 训练时的 expert 利用分布 ≠ 推理时的 expert 利用分布 |

### 1.2 根因：Expert 路由的随机性

PPO/GRPO 类算法依赖旧策略 $\pi_{\text{old}}$ 采样。当你在推理引擎做 rollout 采样后，这些样本去训练引擎做梯度更新时，即使是同一条 response，你在前向一遍激活的专家可能变了——算出来的 logit 也变了。这就导致重要性权重（importance weight）的方差会非常大，训练直接崩掉。

---

## 2. 解决方案一：Routing Replay（路由回放）

### 2.1 核心思想

训练时**记录每个 token 在推理阶段的 routing 决策**（即 expert 分配结果），在后续训练中使用这些记录来保持一致性。

该方法已形式化，主要有两个对齐目标：

| 方法 | 描述 |
|------|------|
| **R1** | 让训练时候算 reward 分布的那次前向复用推理时候存下来的路由决策 |
| **R2** | 在算 reward 分子的时候也做复用 |
| **R3** | 在算 reward 分子和分母时都使用存储的路由 |

### 2.2 训练流程

1. **推理阶段**（rollout）：记录每个 token 的 expert 分配
2. **训练阶段**：强制要求训练时的 expert 分配与记录的分配保持一致
3. **KL 约束**：对 deviation 施加惩罚

$$
\mathcal{L}_{\text{RL}} = \mathcal{L}_{\text{RL}}^{\text{original}} + \lambda \cdot D_{\text{KL}}(\pi_{\theta}^{\text{routing}} \| \pi_{\text{replay}}^{\text{routing}})
$$

### 2.3 优势与局限

| 优点 | 局限 |
|------|------|
| 训练与推理 routing 一致性提升 | 需要额外存储 routing 历史 |
| 减少因 distribution shift 导致的梯度方差 | 记录频率和存储开销 |
| 实现相对简单 | **锁死路由后，模型的探索空间被限制**——同一组输入输出对理论上可以对应很多种不同的专家激活模式，强行重放等于砍掉了这部分多样性 |

---

## 3. 解决方案二：多次采样降低方差

### 3.1 核心思想

既然专家路由引入了随机性，那能不能通过**多次采样**来降低这个方差？

具体做法：对同一个输入输出 pair 在推理引擎多跑几次前向，得到不同的专家路由，然后去平均。这样估计出来的概率就比单次前向稳定得多。

### 3.2 效果

快手的 CostCo 团队在 2023 年提出这个方法，效果比 RunningMan 更好，而且**不会限制模型的探索能力**。

---

## 4. 解决方案三：GSPO（Group Sampling Policy Optimization）

### 4.1 背景：GRPO 的问题

DeepSeek GRPO 核心思想是对同一 prompt 采样多个回复，用 group 内 baseline 计算优势：

$$
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$

**MoE 场景下的问题**：
1. GRPO 的重要性权重本身设计有问题——GRPO 是在 sequence-level 算 reward 的，但每个 token 只有一个采样样本
2. 这**违背了重要性采样需要多样本的基本前提**
3. 单个 token 的 routing 质量无法从 episode-level reward 直接监督

### 4.2 GSPO 核心思想

GSPO = **Group Sampling + Token-level Routing Alignment**

**关键改进**：

1. **Group 内共享 Routing Pattern**：同一 prompt 的多个样本共享 expert 分配策略，减少 variance
2. **Token-level Advantage Estimation**：不仅估计 sequence-level reward，还估计每个 token 对应的 advantage
3. **Adaptive Routing Regularization**：在 advantage 估计中引入 routing 稳定性项

数学表述：

$$
\mathcal{L}_{\text{GSPO}} = \mathbb{E}_{g \sim \mathcal{G}} \left[ \sum_{i \in g} \hat{A}_i^{\text{token}} \cdot \log \pi_\theta(a_i | s_i) \right] - \beta \cdot \text{RoutingReg}(\pi_\theta)
$$

其中 $\hat{A}_i^{\text{token}}$ 是 token-level 优势函数，RoutingReg 约束训练时的 routing 分布接近推理分布：

$$
\text{RoutingReg} = D_{\text{KL}}\left( \text{RoutingDist}_{\text{train}} \| \text{RoutingDist}_{\text{inference}} \right)
$$

### 4.3 GSPO vs GRPO 对比

| 维度 | GRPO | GSPO |
|------|------|------|
| 优势估计粒度 | Sequence-level（整个回复） | Token-level + Sequence-level |
| Expert 利用 | 无约束，可能偏斜 | 通过 Group Sampling 约束 routing |
| 训练稳定性 | 较高 variance | 通过 Group 内共享降低 variance |
| MoE 适配性 | 一般 | **专门针对 MoE 设计** |

---

## 5. 综合方案对比

| 方法 | 核心思路 | 优点 | 缺点 |
|------|----------|------|------|
| **Routing Replay** | 存储推理时的路由，训练时重放 | 训练稳定 | 限制探索空间 |
| **多次采样平均** | 同一输入多次前向取平均 | 不限制探索 | 计算开销大 |
| **GSPO** | Token-level advantage + Group Sampling | 根本性解决高方差问题 | 实现复杂 |

---

## 6. 面试回答模板

**先点名问题本质**：专家路由的随机性导致体度估计方差大

**然后说最直接的解法**：Routing replay——把推理时的路由存下来，训练时重放，但会限制探索空间

**接着说更优的方案**：
- 一个是多次前向去平均（来自快手 CostCo 团队）
- 另一个是从算法层面改进，用 GSPO 做 token-level advantage estimation

**核心记住**：MoE RL 不稳定的根源是路由随机性导致的高方差。解法要么是对齐推理的路由决策，要么是通过采样或算法设计来降低方差。

---

## 7. 追问储备

| 问题 | 回答要点 |
|------|----------|
| GSPO 的 Group Sampling 是什么？ | 对同一 prompt 采样多条回复构成 group，用 group 内 baseline 计算 advantage，减少 reward variance |
| 为什么 GRPO 在 MoE 上不够用？ | GRPO 只做 sequence-level advantage，没有约束 token-level routing 分布，MoE expert 容易偏斜；且违背了重要性采样需要多样本的基本前提 |
| 为什么 Routing Replay 会限制探索？ | 把路由锁死后，模型的探索空间被限制。同一输入输出对理论上可以对应多种不同的专家激活模式，强行重放等于砍掉了多样性 |
| 多次采样具体怎么做？ | 对同一个输入输出 pair 在推理引擎多跑几次前向得到不同的专家路由，然后去平均估计概率 |

---

## 8. 参考文献

- DeepSeek GRPO: Group Relative Policy Optimization
- 快手 CostCo Team: 多专家采样平均方法 (2023)
- Sanmu Team: GSPO (Group Sampling Policy Optimization)

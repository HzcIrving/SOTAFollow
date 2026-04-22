# PPO 精读报告

**Date**: 2026-04-22
**Topic**: Proximal Policy Optimization (PPO) 详解
**Author**: SOTAFollow 机器人助手

---

## 核心贡献

1. **信赖域策略优化**：通过 KL 散度约束，限制每次策略更新幅度
2. **Clipped Surrogate Objective**：避免灾难性策略退化
3. **自适应 KL 惩罚**：自动调节更新步长
4. **GAE 优势估计**：平衡偏差与方差

---

## 1. 背景：策略梯度基础

### 1.1 策略梯度定理

策略梯度方法直接优化策略 $\pi_\theta(a|s)$：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a) \right]$$

### 1.2 梯度估计形式

- **REINFORCE**：使用采样轨迹的回报 $G_t$ 代替 Q 值
$$\nabla_\theta J \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot G_{i,t}$$

- **Actor-Critic**：使用 bootstrap 的 V 函数减少方差
$$\nabla_\theta J \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$$

### 1.3 策略更新的核心问题

- **步长过大**：策略剧烈变化，灾难性退化
- **步长过小**：收敛太慢

**TRPO 解决方案**：限制更新前后的 KL 散度
$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a)\right] \quad \text{s.t.} \quad KL(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$

**PPO 解决方案**：将约束加入目标函数，实现简单、效果稳定

---

## 2. PPO 核心公式

### 2.1 Clipped Surrogate Objective

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比率
- $\epsilon$ 通常取 0.2
- $\text{clip}(x, 1-\epsilon, 1+\epsilon) = \begin{cases} 1-\epsilon & \text{if } x < 1-\epsilon \\ 1+\epsilon & \text{if } x > 1+\epsilon \\ x & \text{otherwise} \end{cases}$

### 2.2 为什么要 clip？

当 $A_t > 0$（动作好）时：
- 若 $r_t > 1+\epsilon$，则 $r_t A_t$ 被 clip 到 $(1+\epsilon)A_t$
- 防止过度提高该动作的概率

当 $A_t < 0$（动作差）时：
- 若 $r_t < 1-\epsilon$，则 $r_t A_t$ 被 clip 到 $(1-\epsilon)A_t$
- 防止过度降低该动作的概率

### 2.3 为什么用 min？

- 正常情况下：$\min$ 不起作用，等同原始目标
- clip 触发时：$\min$ 限制不利方向的过度优化

**核心思想**：不去强制约束，而是让目标函数在极端时变得"平坦"

---

## 3. GAE：优势函数估计

### 3.1 从 TD(λ) 到 GAE

#### n-step Return

定义 n-step 回报：
$$G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

对应的 n-step 优势估计：
$$A_t^{(n)} = G_t^{(n)} - V(s_t)$$

#### TD(λ) 回报

TD(λ) 将不同 n-step 回报加权平均：
$$G_t^{(\lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

#### λ 的几何分布解释

λ 实际上是几何分布的参数，决定了指数加权的衰减速度。

### 3.2 TD(λ) 的一般化：GAE

GAE (Generalized Advantage Estimation) 使用 TD(λ) 思想平衡偏差与方差：

$$\hat{A}_t^{GAE(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^{l} \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

**推导过程**：

从 n-step 优势展开：
$$A_t^{(n)} = \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{n-1} \delta_{t+n-1}$$

对 n-step 优势加权求和：
$$\sum_{n=1}^{\infty} w_n A_t^{(n)} = \sum_{n=1}^{\infty} w_n \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$

交换求和顺序：
$$= \sum_{l=0}^{\infty} \delta_{t+l} \sum_{n=l+1}^{\infty} w_n \gamma^l$$

令 $w_n = (1-\lambda) \lambda^{n-1}$（TD(λ) 权重），得：
$$\sum_{n=l+1}^{\infty} (1-\lambda) \lambda^{n-1} = (1-\lambda) \cdot \frac{\lambda^l}{1-\lambda} = \lambda^l$$

所以：
$$\sum_{n=1}^{\infty} w_n A_t^{(n)} = \sum_{l=0}^{\infty} \gamma^l \lambda^l \delta_{t+l} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

这正是 GAE 的定义。

### 3.3 截断版本（实际使用）

由于 MDP 序列有限，实际使用截断版本：

$$\hat{A}_t^{GAE(\lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots + (\gamma \lambda)^{T-t-1} \delta_{T-1}$$

**递归形式**（更常用）：
$$\hat{A}_t = \delta_t + \gamma \lambda \cdot \hat{A}_{t+1}$$

其中 $\hat{A}_{T} = 0$（最后一步无后续优势）。

### 3.4 λ 参数的影响

| λ | 特性 | 等价于 |
|---|------|--------|
| λ = 0 | 仅TD误差，高偏差低方差 | TD(0) |
| λ → 1 | Monte Carlo，无偏但高方差 | REINFORCE |
| λ ∈ (0,1) | 平衡偏差与方差 | 介于两者之间 |

**λ 的有效范围分析**：

当 λ ∈ (0, 1) 时：
- 有效视角数 ≈ 1 / (1 - λ)
- λ = 0.95 → 有效平均 20 步
- λ = 0.99 → 有效平均 100 步

### 3.5 GAE 与优势函数的关系

**目标**：估计 $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$

**GAE 提供的估计**：

$$\hat{A}_t^{GAE(\lambda)} \approx A^\pi(s_t, a_t)$$

当 λ = 1 时，GAE 退化为 Monte Carlo 优势估计：
$$\hat{A}_t^{(1)} = \sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l} = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l} - V(s_t)$$

这正是 $G_t - V(s_t)$，即 Q 值减去 baseline。

### 3.6 为什么 GAE 有效？

**核心洞察**：GAE 将不同 horizon 的优势估计指数加权组合

1. **短期（l=0）**：TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
   - 偏差：依赖当前价值函数估计
   - 方差：低，因为只考虑一步

2. **中期（l=1,2,...）**：累积 TD 误差
   - 偏差：逐渐减小（bootstrap 次数减少）
   - 方差：逐渐增大（累积随机性增加）

3. **λ 控制偏差-方差权衡**：
   - 小 λ：更依赖 bootstrap，高偏差低方差
   - 大 λ：更接近 Monte Carlo，低偏差高方差

### 3.7 GAE 计算流程图

```
                    ┌─────────────────────────┐
                    │     轨迹序列            │
                    │  s₀, r₀, s₁, r₁, ..., s_T │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  预计算所有 V(s)         │
                    │  (通过 Critic 网络)       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  计算 TD 误差            │
                    │  δₜ = rₜ + γV(s_{t+1}) - V(sₜ) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  从后向前递推 GAE         │
                    │  Aₜ = δₜ + γλ(1-dₜ)A_{t+1} │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  优势归一化               │
                    │  A = (A - μ) / (σ + ε)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  用于 PPO 更新           │
                    └─────────────────────────┘
```

### 3.8 Python 实现（详细注释）

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算 GAE 优势估计

    Args:
        rewards: [T] 奖励序列
        values: [T+1] 价值序列 (包括最后状态)
        dones: [T] 是否终止
        gamma: 折扣因子
        lam: GAE lambda

    Returns:
        advantages: [T] 优势估计
        returns: [T] 用于价值函数训练的返回目标
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)

    # 从后向前计算（利用递归关系）
    gae = 0
    for t in reversed(range(T)):
        # ========== TD 误差 ==========
        # δₜ = rₜ + γV(s_{t+1}) - V(sₜ)
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # ========== 指数加权递推 ==========
        # Aₜ = δₜ + γλ(1-dₜ)A_{t+1}
        # (1 - dones[t]) 在 terminated 时截断后续优势
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

        # ========== 返回值（优势 + baseline）==========
        returns[t] = gae + values[t]

    return advantages, returns


def compute_gae_vectorized(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    向量化 GAE 实现（更高效）
    """
    T = len(rewards)

    # 计算所有 TD 误差
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    # 指数衰减权重
    weights = torch.tensor([(gamma * lam) ** i for i in range(T)])

    # 从后向前累积
    advantages = torch.zeros(T)
    running_sum = torch.zeros(T)

    for t in reversed(range(T)):
        running_sum[:t+1] += weights[t] * deltas[t]

    # 归一化（可选）
    advantages = running_sum

    returns = advantages + values[:-1]

    return advantages, returns
```

### 3.9 GAE 在 PPO 中的使用

```python
def ppo_update_with_gae(self, trajectory):
    # 1. 计算 GAE
    with torch.no_grad():
        rewards = torch.tensor(trajectory['rewards'])
        dones = torch.tensor(trajectory['dones'])
        values = torch.stack(trajectory['values'])

        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=self.gamma,
            lam=self.lam
        )

        # 2. 优势归一化（重要！）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. 使用 GAE 优势进行 PPO 更新
    # ... (见 4.1 完整伪代码)
```

---

## 4. PPO 算法流程

### 4.1 完整伪代码

```python
class PPO:
    def __init__(self, actor, critic, clip_eps=0.2, lr=3e-4,
                 gamma=0.99, lam=0.95, k_epochs=4, mini_batch=64):
        self.actor = actor          # 策略网络
        self.critic = critic        # 价值网络
        self.clip_eps = clip_eps    # clip 范围
        self.gamma = gamma          # 折扣因子
        self.lam = lam              # GAE lambda
        self.k_epochs = k_epochs    # 每次更新轮数
        self.mini_batch = mini_batch

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    def update(self, trajectory):
        """
        Args:
            trajectory: 包含 (states, actions, rewards, dones, values, logps) 的列表
        """
        states = torch.stack(trajectory['states'])
        actions = torch.stack(trajectory['actions'])
        old_values = torch.stack(trajectory['values'])
        old_logps = torch.stack(trajectory['logps'])

        # ========== 1. 计算 GAE 优势 ==========
        with torch.no_grad():
            rewards = torch.tensor(trajectory['rewards'])
            dones = torch.tensor(trajectory['dones'])
            advantages, returns = compute_gae(rewards, old_values, dones,
                                              self.gamma, self.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ========== 2. 多次 epoch 更新 ==========
        dataset_size = len(states)
        indices = torch.randperm(dataset_size)

        for _ in range(self.k_epochs):
            for start in range(0, dataset_size, self.mini_batch):
                end = start + self.mini_batch
                idx = indices[start:end]

                # Mini batch
                s = states[idx]
                a = actions[idx]
                old_v = old_values[idx]
                old_logp = old_logps[idx]
                adv = advantages[idx]

                # ========== 3. 计算新策略 ==========
                dist = self.actor(s)
                new_logp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                # ========== 4. Importance Sampling ==========
                ratio = torch.exp(new_logp - old_logp)

                # ========== 5. Clipped Surrogate ==========
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # ========== 6. Value Loss ==========
                new_values = self.critic(s).squeeze()
                critic_loss = F.mse_loss(new_values, returns[idx])

                # ========== 7. 总损失 ==========
                # 可选：加入熵正则促进探索
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # ========== 8. 反向传播 ==========
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.step()

    def select_action(self, state):
        """带噪声的探索动作"""
        dist = self.actor(state)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(state)
        return action, logp, value
```

### 4.2 训练流程图

```
┌─────────────────────────────────────────────────────────┐
│                     PPO Training Loop                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                        │
│  │  Collect N  │ ← 环境交互收集轨迹数据                  │
│  │  trajectories│                                        │
│  └──────┬──────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐                                        │
│  │   Compute   │ ← GAE 计算优势函数                      │
│  │ GAE + Returns│                                        │
│  └──────┬──────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐    ┌─────────────┐                     │
│  │    for k    │ →  │ Update Actor│ ← Clipped Objective │
│  │   epochs    │    │  + Critic   │                     │
│  └──────┬──────┘    └──────┬──────┘                     │
│         │                   │                            │
│         └───────┬───────────┘                            │
│                 │                                        │
│         (repeat until convergence)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 5. PPO 与其他算法对比

### 5.1 算法对照表

| 特性 | PPO | TRPO | SAC | DDPG | DQN |
|------|-----|------|-----|------|-----|
| 策略类型 | on-policy | on-policy | off-policy | off-policy | off-policy |
| 动作空间 | 离散/连续 | 离散/连续 | 连续 | 连续 | 离散 |
| 更新方式 | 多次小步 | 一次大步 | 多次小步 | 多次小步 | 多次小步 |
| 目标约束 | soft clip | hard KL | soft KL | - | - |
| 探索方式 | 随机 | 随机 | 最大熵 | 确定性+噪声 | ε-greedy |
| 样本效率 | 中等 | 中等 | 高 | 高 | 高 |

### 5.2 PPO vs TRPO

**TRPO**：
- 使用共轭梯度法求解约束优化问题
- 计算量大（需计算 Hessian）
- 实现复杂

**PPO**：
- 将约束转化为惩罚项
- 可以使用 Adam 等一阶优化器
- 实现简单，效果相近

**为什么 PPO 更好**：
1. 实现简单，超参数少
2. 允许更大批次更新
3. 对大规模并行训练友好

### 5.3 PPO vs SAC

**PPO** (on-policy)：
- 使用 importance sampling 纠正分布偏移
- 样本利用率较低（每个样本只用一次）
- 训练稳定，不易崩溃

**SAC** (off-policy)：
- 使用经验回放缓冲区
- 样本利用率高
- 最大熵目标鼓励探索

**选择建议**：
- 样本昂贵（真实机器人）：SAC
- 样本便宜（仿真）：PPO
- 需要稳定训练：PPO

### 5.4 PPO vs DQN

| 维度 | PPO | DQN |
|------|-----|-----|
| 策略 | 随机策略 | 确定性策略 |
| 动作空间 | 连续/离散 | 离散 |
| Q网络 | 直接优化 | 使用 target network |
| 优势函数 | 使用 | 不使用 |

---

## 6. 变体与进阶

### 6.1 PPO-Penalty（自适应 KL 惩罚）

替代 clip 的另一种方式：

$$\mathcal{L}^{KLPEN}(\theta) = \mathbb{E}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a) - \beta \ KL(\pi_{\theta_{old}} || \pi_\theta) \right]$$

其中 $\beta$ 自适应调整：
- KL 过大 → 增大 $\beta$
- KL 过小 → 减小 $\beta$

### 6.2 PPO with 熵正则

$$\mathcal{L}^{ENT}(\theta) = \mathcal{L}^{CLIP}(\theta) + c \cdot H(\pi_\theta)$$

- $c$ 通常取 0.01
- 防止策略过早收敛到确定性

### 6.3 多 Agent PPO (MAPPO)

- 中心化训练，分散执行
-  Critic 使用全局信息
- Actor 仅使用局部观测

---

## 7. 面试常见问题

### Q1: PPO 的核心思想是什么？

**参考答案**：
PPO 的核心是通过限制策略更新的步长来保证训练的稳定性。具体通过 clip 机制，防止策略在单次更新中变化过大。

当优势函数为正时，限制策略概率比上升超过 $1+\epsilon$；当优势函数为负时，限制策略概率比下降超过 $1-\epsilon$。

### Q2: 为什么 PPO 要用 importance sampling？

**参考答案**：
PPO 是 on-policy 算法，但使用 importance sampling 实现 off-policy 更新，允许用旧策略采集的样本训练当前策略，提高样本效率。

概率比 $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 修正了分布偏移。

### Q3: clip 机制如何防止策略退化？

**参考答案**：
假设某动作优势为正 ($A>0$)：
- 旧策略概率：$\pi_{old}(a|s) = 0.1$
- 新策略概率：$\pi_\theta(a|s) = 0.4$ → $r = 4$

正常目标：$4 \times A$（过度提升）
Clipped 目标：$(1+\epsilon) \times A$（限制提升幅度）

这防止了在新动作空间区域过度乐观。

### Q4: GAE 中 λ 参数的作用？

**参考答案**：
λ 控制偏差-方差权衡：
- λ=0：仅用 TD 误差，高偏差低方差，等价 TD(0)
- λ=1：Monte Carlo 回报，无偏但高方差
- λ∈(0,1)：GAE 提供的平衡方案

实践中最常用 λ=0.95~0.99。

### Q5: PPO 和 TRPO 的区别？

**参考答案**：
| | PPO | TRPO |
|---|---|---|
| 约束形式 | 软约束（clip） | 硬约束（KL） |
| 优化方法 | 一阶（Adam） | 二阶（CG+Hessian） |
| 实现复杂度 | 简单 | 复杂 |
| 超参数量 | 少 | 多 |

PPO 本质上是 TRPO 的一阶近似，效果相近但实现简单得多。

### Q6: PPO 为什么需要多次 epoch 更新？

**参考答案**：
1. **数据效率**：每个采样样本可以用多次
2. **稳定收敛**：小批量多次更新比大批量单次更新更稳定
3. **减少方差**：多次平均减少梯度方差

但 epoch 数不宜过多（通常 4~10），否则会过度拟合采样数据。

### Q7: value clipping 的作用？

**参考答案**：
部分 PPO 实现对价值函数也进行 clip：
$$\mathcal{L}^{VF} = \mathbb{E}[\max((V_\theta(s) - V^{clip})^2, (V_\theta(s) - V_{targ})^2)]$$

这防止价值函数在优势函数波动时过度修正，但可能引入偏差，实际效果取决于任务。

### Q8: PPO 的三个关键超参数？

**参考答案**：
1. **clip ε**：通常 0.2，控制更新幅度
2. **GAE λ**：通常 0.95~0.99，控制优势估计偏差-方差
3. **学习率**：通常 3e-4，需配合学习率衰减

### Q9: PPO 在连续动作空间如何计算 logp？

**参考答案**：
假设动作服从对角高斯分布：
```python
# 策略网络输出均值和标准差
mu, log_std = actor(states)
std = log_std.exp()

# 多元高斯分布 log 概率
dist = torch.distributions.Normal(mu, std)
log_prob = dist.log_prob(actions).sum(dim=-1)  # 多维动作求和
```

### Q10: PPO 的优势和局限？

**参考答案**：
**优势**：
- 实现简单，超参数友好
- 训练稳定，收敛性好
- 适合大规模并行训练

**局限**：
- on-policy 限制了样本效率
- 连续动作空间需要高斯假设
- 对稀疏奖励任务效果一般

---

## 8. 工程实现要点

### 8.1 关键技术细节

1. **GAE 计算**：从后向前递推，数值稳定
2. **优势函数归一化**：减去均值除以标准差
3. **熵正则项**：促进探索，防止过早收敛
4. **梯度裁剪**：防止梯度爆炸
5. **价值函数 early stopping**：当 KL 过大时跳过 critic 更新

### 8.2 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 策略崩溃 | 学习率过大 | 降低学习率，增加 clip |
| 价值函数发散 | bootstrap 误差累积 | 减小 γ，增加 critic 更新次数 |
| 探索不足 | 熵太小 | 增加熵系数 |
| 训练不稳定 | batch 太小的 | 增加 batch size |

---

## 9. 参考资料

- Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016
- 智元RL技术凉经

---

## 附录：完整 Python 示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Actor: 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim * 2)  # mu + log_std
        )
        # Critic: 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def get_action(self, state):
        """用于采样"""
        output = self.actor(state)
        mu, log_std = output.chunk(2, dim=-1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state).squeeze()
        return action, log_prob, value

    def get_action_deterministic(self, state):
        """用于评估（无噪声）"""
        output = self.actor(state)
        mu, _ = output.chunk(2, dim=-1)
        return mu
```

---

*文档生成时间: 2026-04-22*
*来源: 和航天 飞书消息请求*
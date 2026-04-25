# 强化学习面试题 & 思考题

**Date**: 2026-04-25
**来源**: 基于仓库 RL 论文（PPO、FlowGRPO）

---

## 一、基础概念题

### 1. PPO 中 Clip Surrogate Objective 的数学公式是什么？为什么要用 min 操作？

**答案提示**：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**思考方向**：min 操作在正常情况和 clip 触发时分别起什么作用？

---

### 2. GAE (Generalized Advantage Estimation) 的公式是什么？λ 参数如何影响偏差-方差权衡？

**答案提示**：

$$\hat{A}_t^{GAE(\lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

| λ 值 | 特性 |
|------|------|
| λ = 0 | 仅 TD 误差，高偏差低方差 |
| λ → 1 | Monte Carlo，无偏但高方差 |

---

### 3. FlowGRPO 为什么需要 ODE-to-SDE 转换？

**答案提示**：
- Flow Matching 的概率流 ODE 是**确定性**采样，无法计算 importance sampling ratio
- 需要随机性来提供 exploration diversity
- 关键：marginal-preserving reverse-time SDE 保证与原 ODE 的边缘分布一致

---

## 二、深度思考题

### 4. 为什么 PPO 用 clip 而不是直接 KL 散度约束？（与 TRPO 对比）

**思考方向**：
- TRPO 用硬约束 $\text{KL}(\pi_{\theta_{old}} \mid\mid \pi_\theta) \leq \delta$，需要额外求解约束优化问题
- PPO 用软约束（clip 操作），实现更简单
- clip 的几何直觉是什么？为什么它能防止灾难性策略退化？

---

### 5. FlowGRPO 中 Score Function 的闭式推导利用了什么数学性质？

**答案提示**：
- Rectified Flow 使用线性插值路径：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$
- 这个线性结构使得 score function 可以解析推导，无需额外网络

---

### 6. GAE 中为什么要用 TD 误差 δ 而非直接用 reward？

**思考方向**：
- Monte Carlo（λ=1）直接用折扣回报，偏差为 0 但方差大
- TD(0)（λ=0）只用一步 TD 误差，方差小但偏差大
- GAE 如何在这两者之间插值？不同 λ 值对应的"有效平均步数"是多少？

---

## 三、优缺点对比题

### 7. PPO vs TRPO：为什么 PPO 成为最流行的在线策略算法？

| 方面 | TRPO | PPO |
|------|------|-----|
| 约束方式 | 硬 KL 约束，二阶优化 | 软约束（clip），一阶优化 |
| 实现复杂度 | 高（需要共轭梯度） | 低 |
| 采样效率 | 相当 | 相当 |
| 效果稳定性 | 好 | 更好 |

**思考**：PPO 的 clip 机制本质上是在目标函数层面近似 KL 约束，这种近似的误差在什么情况下会成为问题？

---

### 8. FlowGRPO vs DDPO：两者都将 RL 应用于生成模型，核心区别是什么？

**答案提示**：
- DDPO 依赖额外的 score function 估计网络
- FlowGRPO 利用 Rectified Flow 的线性结构，score function 可闭式推导
- FlowGRPO 还提出 Denoising Reduction，训练时用 10 步而非 40 步

---

## 四、实践应用题

### 9. 如果让你在 PPO 中调整 GAE 的 λ 参数，会如何选择？考虑不同场景。

**答案提示**：
- 稀疏奖励环境 → 偏向高 λ（接近 1），减少偏差
- 稠密奖励环境 → 可以用较低 λ，依赖 bootstrap
- 训练不稳定时 → 降低 λ 减小方差

---

### 10. FlowGRPO 的 Denoising Reduction 策略：训练用 10 步采样，推理用 40 步，为什么这样做是合理的？

**思考方向**：
- 训练阶段的目标是学习去噪方向，而不是生成质量
- 少量步数的采样已经包含足够的梯度信号
- 这种策略在 RL 中被称为 "truncated rollouts"，是效率和性能之间的权衡

---

## 五、代码实现题

### 11. 实现 GAE 的计算（给定 δ 序列和参数 γ、λ）

```python
def compute_gae(delta_seq, gamma=0.99, lambda_=0.95):
    """
    delta_seq: TD误差序列 [t0, t1, t2, ..., tT-1]
    return: 优势函数估计序列
    """
    # 实现你的代码
    pass
```

**答案**：从后向前递归计算：
```python
adv = 0
for delta in reversed(delta_seq):
    adv = delta + gamma * lambda_ * adv
```

---

### 12. PPO 中如何计算 clipped surrogate loss？

```python
def ppo_clip_loss(theta_old_log_probs, theta_new_log_probs, 
                  advantages, epsilon=0.2):
    """
    计算 PPO clipped surrogate objective
    返回每个时间步的 loss
    """
    # 实现你的代码
    pass
```

**思考提示**：
- ratio = exp(new_log_prob - old_log_prob)
- 当 A > 0 时，ratio 被 clip 到 1+ε 上方
- 当 A < 0 时，ratio 被 clip 到 1-ε 下方

---

## 六、开放性思考题

### 13. PPO 的 clip 机制有一个著名的"陷阱"：当优势函数很大时，clip 可能失效。为什么？

**提示**：如果 $A_t \gg 0$ 且 $r_t(\theta)$ 远远大于 $1+\epsilon$，clip 后的损失函数值仍然会很大，策略仍然会大幅更新。

---

### 14. FlowGRPO 解决了 Flow Matching 的确定性采样问题。如果将其思想扩展到 Diffusion Transformer (DiT)，需要考虑什么？

**思考方向**：
- DDPM/DDIM 的随机采样天然支持 importance sampling
- 关键挑战在于如何平衡生成质量和 RL 训练效率

---

## 参考来源

| 论文 | 核心考点 |
|------|---------|
| PPO (Schulman et al., 2017) | Clip Objective, GAE, TRPO 对比 |
| FlowGRPO (2025) | ODE-to-SDE, Score Function, Denoising Reduction |

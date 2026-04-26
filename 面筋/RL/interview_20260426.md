## 日期：20260426

### RL 领域

**论文：《PPO: Proximal Policy Optimization》**

1. **[基础概念题]** PPO 的 Clipped Surrogate Objective 的核心公式是什么？clip 操作在 $A_t > 0$ 和 $A_t < 0$ 两种情况下分别防止了什么问题？
   - 答案/思考提示：公式为 $\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$。当 $A_t > 0$ 时 clip 防止过度提高该动作概率（$r_t > 1+\epsilon$ 时截断）；当 $A_t < 0$ 时防止过度降低概率（$r_t < 1-\epsilon$ 时截断）。核心思想是让目标函数在极端时变"平坦"而非强制约束。

2. **[深度思考题]** PPO 使用 $\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$ 而不是 $\max$，为什么？请从优化方向解释。
   - 答案/思考提示：$\min$ 的作用是"选择更保守的方向"。当 clip 触发时（策略变化过大），$\min$ 会选择被 clip 后的值而非原始值，从而限制不利方向的更新幅度。如果用 $\max$，则在 clip 触发时反而会增大更新幅度，与目标相悖。本质上 $\min$ 是在"正常更新"和"受限更新"中取较小者，保证策略不会因为一次大更新而崩溃。

3. **[优缺点对比题]** PPO 和 TRPO 都试图限制策略更新幅度，但实现方式有何本质不同？PPO 为什么在工程上更受欢迎？
   - 答案/思考提示：TRPO 使用 hard KL 约束 $\text{s.t. } KL(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$，需要用共轭梯度法求解约束优化问题（计算 Hessian）；PPO 将约束转化为目标函数中的 clip 项，可用标准一阶优化器（Adam）求解。PPO 实现简单、无需计算二阶导数、允许更大批次更新、对大规模并行训练友好。

4. **[实践应用题]** GAE (Generalized Advantage Estimation) 中参数 $\lambda$ 如何控制偏差-方差权衡？请给出 $\lambda=0$ 和 $\lambda \to 1$ 时的极端情况。
   - 答案/思考提示：GAE 递归形式为 $\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}$。$\lambda=0$ 时 $\hat{A}_t = \delta_t$（仅 TD 误差），高偏差低方差；$\lambda \to 1$ 时退化为 Monte Carlo 估计 $\hat{A}_t = \sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l}$，无偏但高方差。有效平均 horizon 为 $1/(1-\lambda)$，如 $\lambda=0.95$ 对应约 20 步。

5. **[基础概念题]** 在 PPO 的 Actor-Critic 架构中，Actor 和 Critic 分别负责什么？为什么需要两个网络？
   - 答案/思考提示：Actor（策略网络 $\pi_\theta(a|s)$）负责选择动作并计算策略梯度 $\nabla_\theta J \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$；Critic（价值网络 $V_\phi(s)$）负责估计状态价值 $V(s)$，提供优势函数中的 baseline 来减少方差。Baseline 不改变策略梯度的期望，但能显著降低方差、加速训练。

---

**通用 RL 领域思考题**

6. **[深度思考题]** On-policy 算法（如 PPO）和 Off-policy 算法（如 SAC）各有何优劣？在什么场景下你会选择其中一种？
   - 答案/思考提示：On-policy 每次更新只能用当前策略采集的样本（样本利用率低），但训练稳定不易崩溃；Off-policy 可用经验回放池复用历史样本（效率高），但引入 distribution shift 需要 IS 矫正。选择场景：样本便宜（仿真环境）选 PPO；样本昂贵（真实机器人）选 SAC；需要稳定训练选 PPO。

7. **[实践应用题]** 如果你在训练 PPO 时发现策略突然崩溃（loss 变 NaN），你会从哪些方向排查问题？
   - 答案/思考提示：① 优势函数归一化——GAE 输出的 advantage 可能数值不稳定，需要归一化；② Value Loss 爆炸——价值网络预测范围过大，考虑用 reward scaling 或 value clipping；③ 学习率过大——尝试降低或使用学习率调度；④ 梯度裁剪——确认 `clip_grad_norm_(0.5)` 是否生效；⑤ 观测/动作空间异常——检查输入是否包含 NaN/Inf。

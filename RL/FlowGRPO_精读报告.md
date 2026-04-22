# FlowGRPO 精读报告

## 引用信息

| 字段 | 内容 |
|------|------|
| **论文标题** | Flow-GRPO: Training Flow Matching Models via Online RL |
| **arXiv** | https://arxiv.org/abs/2505.05470 |
| **代码** | https://github.com/yifan123/flow_grpo |
| **作者** | Jie Liu\*（港中文 MMLab）、Gongye Liu\*（清华）、Jiajun Liang（快手 Kling）等 |
| **机构** | 港中文 MMLab · 清华大学 · 快手 Kling Team · 南京大学 · 上海 AI Lab |
| **核心贡献** | 首个将 GRPO 引入 Flow Matching 模型：通过 ODE-to-SDE 转换实现随机采样 + Denoising Reduction 加速训练 |
| **实验任务** | 文生图：Compositional GenEval、Visual Text Rendering、Human Preference Alignment |
| **关键结果** | SD3.5-M GenEval: 63% → 95%，OCR: 59% → 92%，几乎无 reward hacking |

---

## 一句话总结

FlowGRPO 通过**构造 marginal-preserving reverse-time SDE**（保证与原 ODE marginal distribution 一致），将 Flow Matching 的确定性采样转换为随机采样，从而为 GRPO 提供 exploration diversity 和 importance sampling ratio 的计算基础；同时提出 Denoising Reduction 用少量步数采样训练、完整步数推理，显著加速训练。

---

## 核心贡献

1. **首个将在线 RL（GRPO）引入 Flow Matching 模型** — 解决了 Flow Matching 确定性采样无法直接应用 RL 的根本问题

2. **ODE-to-SDE 转换** — 通过构造 marginal-preserving reverse-time SDE，在保持原有生成分布的前提下注入可控随机性，使得 importance sampling ratio 可计算

3. **Score Function 闭式推导** — 对于 Rectified Flow，利用 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ 的线性结构，推导出 score function 的解析形式，无需额外网络

4. **Denoising Reduction** — 训练时用 10 步采样（而非推理的 40 步），大幅加速数据收集，对性能几乎无影响

5. **KL 散度闭式形式** — 两个高斯策略之间的 KL 散度可直接求值，用于 PPO-style clipped objective

---

## 方法详述

### 1. 问题定义

#### 1.1 Flow Matching 框架

Flow Matching 将图像生成建模为从纯噪声到数据的传输过程。Rectified Flow 使用线性插值路径：

$$
\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1] \tag{1}
$$

模型通过最小化 Flow Matching 目标函数来学习速度场 $\mathbf{v}_\theta(\mathbf{x}_t, t)$：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \left\| \mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

其中 $\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$。
$$

#### 1.2 确定性 ODE 采样的根本问题

Flow Matching 的生成过程是确定性的概率流 ODE：

$$
d\mathbf{x}_t = \mathbf{v}_t(\mathbf{x}_t) \, dt \tag{6}
$$

Euler 离散化后：给定初始噪声 $\mathbf{x}_T$ 和 prompt $\mathbf{c}$，**只能产生唯一一条轨迹**。

**GRPO 的两个要求因此都无法满足：**

1. **无法计算概率** $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c})$ → importance sampling ratio 算不了
2. **无探索多样性** → 同一个 prompt 只能得到完全相同的输出

### 2. ODE-to-SDE 转换：核心推导

#### 2.1 Marginal-Preserving Reverse-Time SDE

根据 Score-Based SDE 理论，对任意前向 SDE，其 reverse-time SDE 保持 marginal distribution 不变的通式为：

$$
d\mathbf{x}_t = \left( \mathbf{v}_t(\mathbf{x}_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) \right) dt + \sigma_t \, d\mathbf{w} \tag{7}
$$

**三项的物理含义：**

| 项 | 作用 |
|----|------|
| $\mathbf{v}_t(\mathbf{x}_t) dt$ | 确定性漂移，沿原 ODE 速度场推进 |
| $-\frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) dt$ | Itô 修正项，保证 marginal 不变 |
| $\sigma_t d\mathbf{w}$ | Wiener 扩散项，引入随机探索噪声 |

#### 2.2 Rectified Flow 的 Score Function 闭式推导

对于 Rectified Flow，score function 与速度场存在闭式关系。关键洞察：给定 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，当 $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 固定时，$\mathbf{x}_t$ 服从高斯分布，其协方差为 $t^2 \mathbf{I}$。

利用高斯分布的 score function 解析形式，可得：

$$
\nabla \log p_t(\mathbf{x}_t) = -\frac{\mathbf{x}_t}{t} - \frac{1-t}{t} \mathbf{v}_t(\mathbf{x}_t) \tag{A1}
$$

#### 2.3 代入得到 Rectified Flow 特有的 Reverse-Time SDE

将公式 (A1) 代入公式 (7)，整理得到：

$$
d\mathbf{x}_t = \left[ \mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t)\mathbf{v}_t(\mathbf{x}_t) \right) \right] dt + \sigma_t \, d\mathbf{w} \tag{8}
$$

**直观理解：**
- 第一项：沿速度场向前推进
- 第二项（Langevin correction）：补偿扩散项对 marginal 的漂移影响
- 第三项：高斯随机噪声

#### 2.4 Euler-Maruyama 离散化

对连续 SDE 应用 Euler-Maruyama 离散化（时间步长 $\Delta t$）：

$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \left[ \mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t)\mathbf{v}_\theta(\mathbf{x}_t, t) \right) \right] \Delta t + \sigma_t \sqrt{\Delta t} \, \epsilon \tag{9}
$$

其中 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

> **注**：论文中 $t$ 从 0 到 1，推理时 $t$ 递减（$t+\Delta t$ 在反向时间中代表更小的 $t$ 值）。

#### 2.5 噪声系数的参数化

论文设定噪声系数为：

$$
\sigma_t = a \sqrt{\frac{t}{1-t}}
$$

其中 $a$ 是超参数，控制整体噪声水平。

- 当 $t \to 0$（低噪声阶段）：$\sigma_t \to 0$（几乎无随机性，保证最终生成质量）
- 当 $t \to 1$（高噪声阶段）：$\sigma_t \to \infty$（更多探索噪声）

#### 2.6 策略分布的高斯形式

由公式 (9) 可知，**每步的策略分布是各向同性高斯分布**：

$$
\pi_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) = \mathcal{N}\left( \mathbf{x}_{t-1}; \, \boldsymbol{\mu}_\theta, \, \sigma_t^2 \Delta t \cdot \mathbf{I} \right)
$$

其中均值 $\boldsymbol{\mu}_\theta$ 为公式 (9) 中除随机项外的所有确定性项。

**三个重要推论：**

1. ✅ **可以计算概率**：高斯分布概率密度函数可直接求值
2. ✅ **可以采样多样性**：每次采样 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 得到不同轨迹
3. ✅ **marginal 不变**：Itô 修正项保证与原 ODE 的分布一致性

### 3. 算法框架

**Figure 2: Overview of Flow-GRPO**

![FlowGRPO Algorithm Overview](https://arxiv.org/html/2505.05470v5/x2.png)

算法框架包含三个主要模块：
- **左侧**：Prompt Set → ODE-to-SDE 转换 → 多条随机轨迹
- **中间**：GRPO Loss 计算（Group-normalized advantage + Clipped PPO + KL penalty）
- **右侧**：策略更新 → Aligned Policy → 高质量图像
- **底部**：Denoising Reduction（训练 T=10，推理 T=40）

### 4. KL 散度的闭式推导

两个均值不同、方差相同的高斯分布之间的 KL 散度为：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\| \boldsymbol{\mu}_\theta - \boldsymbol{\mu}_{\text{ref}} \|^2}{2\sigma^2}
$$

代入 $\boldsymbol{\mu}$ 的具体形式，经过化简得到：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\Delta t}{2} \left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}_{\text{ref}}(\mathbf{x}_t, t) \right\|^2 \tag{10}
$$

其中 $\frac{\Delta t}{2} \left( \frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t} \right)^2$ 为系数项。

**简化分析：** 代入 $\sigma_t = a\sqrt{t/(1-t)}$ 后，KL 散度在 $t$ 方向上由 $\frac{1}{t(1-t)}$ 类项主导，在 $t \to 0$ 和 $t \to 1$ 时会显著放大。这是后续 SAGE-GRPO 重点解决的问题。

### 5. GRPO 目标函数与 Flow Matching 的衔接

#### 5.1 Group-Normalized Advantage

给定 prompt $\mathbf{c}$，采样 $G$ 张图 $\{\mathbf{x}_0^i\}_{i=1}^G$，每张图的 advantage 为：

$$
\hat{A}_t^i = \frac{R(\mathbf{x}_0^i, \mathbf{c}) - \text{mean}(\{R(\mathbf{x}_0^j, \mathbf{c})\}_{j=1}^G)}{\text{std}(\{R(\mathbf{x}_0^j, \mathbf{c})\}_{j=1}^G)} \tag{4}
$$

Reward $R$ 仅在最终步 $t=0$ 给出。

#### 5.2 Importance Sampling Ratio

$$
r_t^i(\theta) = \frac{p_\theta(\mathbf{x}_{t-1}^i | \mathbf{x}_t^i, \mathbf{c})}{p_{\theta_{\text{old}}}(\mathbf{x}_{t-1}^i | \mathbf{x}_t^i, \mathbf{c})} \tag{5}
$$

分子分母都是 ODE-to-SDE 转换后每步的高斯概率密度，由公式 (9) 给出。

#### 5.3 完整 FlowGRPO Loss

将 PPO-style clipped objective 与 KL penalty 结合：

$$
\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \min\left(r_t^i(\theta) \hat{A}_t^i, \text{clip}(r_t^i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t^i\right) - \beta \, D_{\text{KL}} \right) \right] \tag{5}
$$

其中 $D_{\text{KL}} = D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$。

### 6. Denoising Reduction

#### 6.1 核心发现

在线 RL 需要大量采样来收集训练数据，而 Flow Matching 生成一张图通常需要 $T=30$–$50$ 步，成本极高。

**关键发现**：训练时不需要用这么多步数。

#### 6.2 具体策略

| 阶段 | 去噪步数 | 目的 |
|------|---------|------|
| 训练采样 | $T = 10$ 步 | 快速收集多样化轨迹 |
| 推理生成 | $T = 40$ 步 | 保证最终生成质量 |

推理时用完整步数保证质量，训练时用少量步数大幅加速数据收集。实验证明这种策略对最终性能几乎无影响。

---

## 训练与推理伪代码

```python
"""
FlowGRPO Algorithm Pseudocode
"""

def flowgrpo_train(model, ref_model, prompts, G=8, T_train=10, T_infer=40,
                   beta=0.1, epsilon=0.2, a=1.0):
    """
    Args:
        model: 可学习的 Flow Matching 模型 (参数化为速度场 v_theta)
        ref_model: 参考模型（通常是 SFT 后的模型）
        prompts: 文本 prompt 列表
        G: 每个 prompt 采样的图片数量（group size）
        T_train: 训练时去噪步数
        T_infer: 推理时去噪步数
        beta: KL penalty 系数
        epsilon: PPO clip 范围
        a: 噪声水平超参数
    """

    for iteration in range(N_iterations):
        all_rewards = []
        all_log_probs = []

        for prompt in prompts:
            # Step 1: 用 SDE 采样 G 张图片
            trajectories = []
            for g in range(G):
                # 初始化: x_1 ~ N(0, I) [反向时间从 t=1 开始]
                x_t = sample_noise()
                prompt_emb = encode(prompt)

                # 反向去噪（训练时用少量步）
                for t in reversed(range(T_train)):
                    t_norm = t / T_train  # 归一化到 [0, 1]
                    sigma_t = a * sqrt(t_norm / (1 - t_norm))

                    # Euler-Maruyama 离散化 (公式 9)
                    v = model(x_t, t_norm, prompt_emb)
                    drift = v + (sigma_t**2 / (2 * t_norm)) * (x_t + (1 - t_norm) * v)
                    diffusion = sigma_t * sqrt(1/T_train) * epsilon  # epsilon ~ N(0, I)
                    x_t = x_t + drift * (1/T_train) + diffusion

                x_0 = x_t  # 最终生成的图片
                reward = reward_model(x_0, prompt)
                trajectories.append({
                    'x_0': x_0,
                    'reward': reward,
                    'trajectory': [x_t]  # 可记录完整轨迹用于分析
                })

            # Step 2: 计算 Group-Normalized Advantage (公式 4)
            rewards = [t['reward'] for t in trajectories]
            mean_reward = mean(rewards)
            std_reward = std(rewards)
            advantages = [(r - mean_reward) / std_reward for r in rewards]

            # Step 3: 计算 FlowGRPO Loss (公式 5)
            policy_loss = 0.0
            for g, traj in enumerate(trajectories):
                for t in range(T_train):
                    # 计算 importance sampling ratio (公式 5)
                    # 这里需要计算 p_theta 和 p_theta_old 的比值
                    # 由于策略是高斯分布，可以闭式计算
                    log_ratio = compute_log_ratio(traj['x_t'], model, ref_model, t)
                    ratio = exp(log_ratio)

                    # Clipped PPO objective
                    clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
                    surrogate = min(ratio * advantages[g], clipped_ratio * advantages[g])

                    # KL penalty
                    kl_penalty = beta * compute_kl(model, ref_model, t)

                    policy_loss -= (surrogate - kl_penalty)

            # Step 4: 梯度更新
            model.update(policy_loss)

    return model


def flowgrpo_sample(model, prompt, T=40):
    """
    推理时使用完整步数保证质量
    """
    x_t = sample_noise()
    prompt_emb = encode(prompt)

    for t in reversed(range(T)):
        t_norm = t / T
        sigma_t = a * sqrt(t_norm / (1 - t_norm))  # 噪声水平

        v = model(x_t, t_norm, prompt_emb)
        # 确定性 ODE（无随机项），与训练时的 SDE 对应
        x_t = x_t + v * (1/T)

    return x_t
```

---

## 实验结论

### 1. 主实验结果

#### 1.1 GenEval 数值结果

**Table 1: GenEval Benchmark Results**

| 模型 | Overall | Single Obj. | Two Obj. | Counting | Colors | Position | Attr. Binding |
|------|---------|-------------|----------|----------|--------|----------|---------------|
| SD3.5-M（基线） | 0.63 | 0.98 | 0.78 | 0.50 | 0.81 | 0.24 | 0.52 |
| **+ FlowGRPO** | **0.95** | **1.00** | **0.99** | **0.95** | **0.92** | **0.99** | **0.86** |
| 提升幅度 | +32pp | +2pp | +21pp | +45pp | +11pp | +75pp | +34pp |

**关键洞察**：
- Position 提升最显著（+75pp）：FlowGRPO 有效解决了空间关系推理难题
- Counting 提升 +45pp：从 50% 到 95%，接近完美
- Two Objects 提升 +21pp：复合对象场景大幅改善

#### 1.2 GenEval 训练曲线

![Figure 1: GenEval Training Curves](https://arxiv.org/html/2505.05470v5/x1.png)

**Figure 1 解析**：
- GenEval 准确率从基线 0.63 稳定上升到 **0.95**，超过 GPT-4o 的 0.84
- 训练过程中 **Aesthetic Score** 和 **PickScore** 基本保持稳定
- 说明 FlowGRPO 实现了 "reward 提升 + 质量稳定 + 偏好对齐" 三赢局面

#### 1.3 定性对比

![Figure 3: Qualitative Comparison on the GenEval Benchmark](https://arxiv.org/html/2505.05470v5/x3.png)

**Figure 3 解析**：Grid 布局展示多组 GenEval 任务中的生成质量对比：
- **Counting**：Prompt 要求生成精确数量，FlowGRPO 准确渲染
- **Colors**：多对象颜色绑定正确
- **Position**：空间关系（左边、右边）准确
- **Attribute Binding**：复合属性（戴红色帽子的猫）正确绑定

#### 1.4 多任务综合性能

**Table 2: Performance on Compositional Image Generation, Visual Text Rendering, and Human Preference**

| 配置 | Task Metric | Image Quality | Preference |
|------|------------|---------------|------------|
| SD3.5-M 基线 | — | Aesthetic: 21.72 | PickScore: 0.87 |
| FlowGRPO (w/o KL) | GenEval 0.95 | Aesthetic: 4.93 ❌ | PickScore: 0.44 ❌ |
| **FlowGRPO (w/ KL)** | **GenEval 0.95** | **Aesthetic: 5.25** ✅ | **PickScore: 1.03** ✅ |

**核心结论**：KL 约束是防止 reward hacking 的关键，不是 early stopping 的替代品。

#### 1.5 Visual Text Rendering (OCR)

| 模型 | OCR Accuracy |
|------|-------------|
| SD3.5-M（基线） | 59% |
| **+ FlowGRPO** | **92%** |

文本渲染能力大幅提升（+33pp）。

### 2. 消融实验

#### 2.1 Denoising Reduction 的影响

![Figure 7a: Effect of Denoising Reduction on GenEval](https://arxiv.org/html/2505.05470v5/x7.png)

实验证明：训练步数从 40 降到 10，性能几乎不变，但训练速度提升 4 倍。

#### 2.2 Group Size 的影响

![Figure 5: Ablation Studies on Different Group Size G](https://arxiv.org/html/2505.05470v5/x5.png)

Group Size 越大，性能越好。推荐使用 $G \geq 8$。

#### 2.3 KL Regularization 的影响

![Figure 6: Effect of KL Regularization](https://arxiv.org/html/2505.05470v5/x6.png)

KL 约束有效抑制 reward hacking，保证图像质量和偏好分数不下降。

#### 2.4 噪声水平的影响

![Figure 7b: Effect of Noise Level Ablation on the OCR](https://arxiv.org/html/2505.05470v5/x8.png)

噪声系数 $a$ 需调参，过高或过低都会影响性能。

---

## KnowHow

### 核心洞察

1. **为什么 ODE-to-SDE 转换是可行的？**
   - 关键在于 marginal-preserving 性质：reverse-time SDE 与原 forward ODE 具有完全相同的 marginal distributions $\{p_t\}$
   - 这保证了 SDE 采样的图片分布与原 ODE 一致，不会引入分布偏移

2. **Score Function 为什么可以闭式推导？**
   - Rectified Flow 的线性插值路径 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ 决定了 $\mathbf{x}_t$ 是高斯分布
   - 高斯分布的 score function 有解析形式，无需学习额外网络

3. **Denoising Reduction 为什么有效？**
   - 低噪声阶段（$t \to 0$）$\sigma_t \to 0$，此时 SDE 退化为 ODE
   - 高噪声阶段（$t \to 1$）的探索多样性在早期训练中已经足够
   - 10 步足以捕捉策略更新的方向，完整步数主要用于推理精度

4. **KL Penalty 为什么能防止 reward hacking？**
   - KL 散度衡量当前策略与参考策略的偏离程度
   - 通过惩罚大的 KL 散度，强制策略更新幅度受限，避免一次性过拟合到 reward

5. **Group Size $G$ 的选择**
   - $G$ 越大，advantage 估计越准确，但内存开销也越大
   - 实验表明 $G \geq 8$ 是较好的平衡点

6. **噪声系数 $a$ 的影响**
   - $a$ 控制整体探索水平：过大导致训练不稳定，过小则探索不足
   - 需要针对具体任务调参

7. **训练稳定性技巧**
   - 使用 gradient clipping 防止梯度爆炸
   - KL penalty 系数 $\beta$ 需要足够大以约束策略变化

---

## arXiv Appendix 关键点总结

### Appendix A: Marginal-Preserving SDE 证明

**目标：** 证明公式 (7) 的 reverse-time SDE 与原 forward ODE 具有完全相同的 marginal distributions。

**证明思路（基于 Fokker-Planck 方程）：**

1. **Forward ODE** $\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_t(\mathbf{x}_t)$ 对应的 Fokker-Planck 方程描述了 marginal density $p_t$ 的演化

2. **Reverse SDE**（公式 7）的 Fokker-Planck 方程中，漂移项包含 $-\frac{\sigma_t^2}{2}\nabla\log p_t$（Itô correction），这恰好抵消了扩散项对 marginal 的影响

3. 联立两个 Fokker-Planck 方程，可以证明 reverse SDE 的 marginal density 演化与 forward ODE 完全一致

### Appendix B: SAGE-GRPO（后续工作）

针对 FlowGRPO 在高噪声区（$t \to 1$）KL 散度爆炸的问题，SAGE-GRPO 提出了更稳定的改进方案。

---

## 总结

FlowGRPO 首次将在线 RL（GRPO）成功应用于 Flow Matching 模型，核心贡献包括：

1. **ODE-to-SDE 转换机制**：通过 marginal-preserving reverse-time SDE，在保持生成分布的前提下注入可控随机性，解决了 Flow Matching 确定性采样无法直接应用 RL 的根本问题

2. **Score Function 闭式推导**：利用 Rectified Flow 的线性结构，推导出 score function 的解析形式，无需额外网络估计

3. **Denoising Reduction 训练策略**：训练时用 10 步采样（而非推理的 40 步），大幅加速数据收集，对性能几乎无影响

**最重要洞察**：Flow Matching 的确定性 ODE 采样并非 RL 的障碍——通过精心设计的 SDE 转换，可以保留原分布的同时引入随机性，这为所有 Flow Matching + RL 的后续工作奠定了基础。

---

## 参考链接

- **arXiv**: https://arxiv.org/abs/2505.05470
- **代码**: https://github.com/yifan123/flow_grpo
- **项目主页**: https://gongyeliu.github.io/Flow-GRPO/
- **相关工作 SAGE-GRPO**（解决 FlowGRPO 高噪声区问题）: arXiv:2603.21872

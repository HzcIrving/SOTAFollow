# FlowGRPO 精读报告：首个将在线 RL 引入 Flow Matching 的 GRPO 方法

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

## 拟人化开篇

想象你训练一个 Flow Matching 模型（比如 SD3.5）来做文生图。它已经看过海量图片，现在你想让它更听指令——精确数清"图里有几只猫"、准确渲染"STOP"这样的文字。标准做法是强化学习 GRPO：让模型对同一个 prompt 生成多张图，比较 reward 高低来更新策略。

但问题来了——**Flow Matching 的采样是确定性的 ODE**。给定同一个 prompt 和同一份初始噪声，你永远只能得到同一张图。没有多样性，GRPO 的 advantage 没法算，在线学习根本无法进行。

FlowGRPO 要解决的就是这个核心矛盾：如何在不破坏模型原有分布的前提下，给 ODE 采样过程注入可控的随机性？

---

## 背景与问题

### 2.1 Flow Matching 框架

Flow Matching 将图像生成建模为从纯噪声到数据的传输过程。Rectified Flow 使用线性插值路径：

**（公式 1）**

$$
\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1]
$$

模型通过最小化 Flow Matching 目标函数来学习速度场 $\mathbf{v}_\theta(\mathbf{x}_t, t)$：

**（公式 2）**

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \left\| \mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

其中目标速度场 $\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$。

### 2.2 确定性 ODE 采样的根本问题

Flow Matching 的生成过程是确定性的概率流 ODE：

**（公式 7）**

$$
d\mathbf{x}_t = \mathbf{v}_t \, dt
$$

Euler 离散化后得到：

**（公式 8）**

$$
\mathbf{x}_{t-1} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c})
$$

给定初始噪声 $\mathbf{x}_T$ 和 prompt $\mathbf{c}$，**只能产生唯一一条轨迹**。

**GRPO 的两个要求因此都无法满足：**

1. **无法计算概率** $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c})$ → importance sampling ratio 算不了
2. **无探索多样性** → 同一个 prompt 只能得到完全相同的输出

### 2.3 解决方案：ODE-to-SDE 转换

将确定性 ODE 转换为一个等价 SDE，要求：
- 每步产生**随机**输出（支持 GRPO 多样性采样）
- **marginal distribution** 与原 ODE 完全一致（保证采样质量）
- 可以计算**显式概率分布**

---

## 三、ODE-to-SDE 转换：核心推导

### 3.1 构造 Marginal-Preserving Reverse-Time SDE

根据 Score-Based SDE 理论，对任意前向 SDE，其 reverse-time SDE 保持 marginal distribution 不变的通式为：

**（公式 9）**

$$
d\mathbf{x}_t = \left( \mathbf{v}_t(\mathbf{x}_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) \right) dt + \sigma_t \, d\mathbf{w}
$$

**三项的物理含义：**

| 项 | 作用 |
|----|------|
| $\mathbf{v}_t(\mathbf{x}_t) dt$ | 确定性漂移，沿原 ODE 速度场推进 |
| $-\frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) dt$ | Itô 修正项，保证 marginal 不变 |
| $\sigma_t d\mathbf{w}$ | Wiener 扩散项，引入随机探索噪声 |

### 3.2 Rectified Flow 的 Score Function 闭式推导

对于 Rectified Flow，score function 与速度场存在闭式关系。利用 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，可得：

**（公式 10）**

$$
\nabla \log p_t(\mathbf{x}) = -\frac{\mathbf{x}}{t} - \frac{1-t}{t} \mathbf{v}_t(\mathbf{x})
$$

**推导思路：** 当 $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 固定时，$\mathbf{x}_t$ 服从高斯分布，其协方差为 $t^2 \mathbf{I}$，score function 有解析形式即上式。

### 3.3 代入得到 Reverse-Time SDE

将公式 (10) 代入公式 (9)，整理得到 Rectified Flow 特有的 reverse-time SDE：

**（公式 11）**

$$
d\mathbf{x}_t = \left[ \mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t)\mathbf{v}_t(\mathbf{x}_t) \right) \right] dt + \sigma_t \, d\mathbf{w}
$$

**直观理解：**

- 第一项：沿速度场向前推进
- 第二项（Langevin correction）：补偿扩散项对 marginal 的漂移影响
- 第三项：高斯随机噪声

### 3.4 Euler-Maruyama 离散化

对连续 SDE 应用 Euler-Maruyama 离散化（时间步长 $\Delta t$）：

**（公式 12）**

$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \left[ \mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t)\mathbf{v}_\theta(\mathbf{x}_t, t) \right) \right] \Delta t + \sigma_t \sqrt{\Delta t} \, \epsilon
$$

其中 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### 3.5 噪声系数的参数化

论文设定噪声系数为：

$$
\sigma_t = a \sqrt{\frac{t}{1-t}}
$$

其中 $a$ 是超参数，控制整体噪声水平。

- 当 $t \to 0$（低噪声阶段）：$\sigma_t \to 0$（几乎无随机性，保证最终生成质量）
- 当 $t \to 1$（高噪声阶段）：$\sigma_t \to \infty$（更多探索噪声）

### 3.6 策略分布的高斯形式

由公式 (12) 可知，**每步的策略分布是各向同性高斯分布**：

$$
\pi_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) = \mathcal{N}\left( \mathbf{x}_{t-1}; \, \boldsymbol{\mu}_\theta, \, \sigma_t^2 \Delta t \cdot \mathbf{I} \right)
$$

其中均值 $\boldsymbol{\mu}_\theta$ 为公式 (12) 中除随机项外的所有确定性项。

**三个重要推论：**

1. ✅ **可以计算概率**：高斯分布概率密度函数可直接求值
2. ✅ **可以采样多样性**：每次采样 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 得到不同轨迹
3. ✅ **marginal 不变**：Itô 修正项保证与原 ODE 的分布一致性

---

## 四、KL 散度的闭式推导

有了高斯策略分布，可以计算 $\pi_\theta$ 与参考策略 $\pi_{\text{ref}}$ 之间的 KL 散度闭式解。

两个均值不同、方差相同的高斯分布之间的 KL 散度为：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\| \boldsymbol{\mu}_\theta - \boldsymbol{\mu}_{\text{ref}} \|^2}{2\sigma^2}
$$

代入 $\boldsymbol{\mu}$ 的具体形式，经过化简得到：

**（公式 13）**

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\Delta t}{2} \left( \frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t} \right)^2 \left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}_{\text{ref}}(\mathbf{x}_t, t) \right\|^2
$$

**简化分析：** 代入 $\sigma_t = a\sqrt{t/(1-t)}$ 后，KL 散度在 $t$ 方向上由 $\frac{1}{t(1-t)}$ 类项主导，在 $t \to 0$ 和 $t \to 1$ 时会显著放大。这是后续 SAGE-GRPO 重点解决的问题。

---

## 五、GRPO 目标函数与 Flow Matching 的衔接

### 5.1 Group-Normalized Advantage

给定 prompt $\mathbf{c}$，采样 $G$ 张图 $\{\mathbf{x}_0^i\}_{i=1}^G$，每张图的 advantage 为：

**（公式 4）**

$$
\hat{A}_t^i = \frac{R(\mathbf{x}_0^i, \mathbf{c}) - \text{mean}(\{R(\mathbf{x}_0^j, \mathbf{c})\}_{j=1}^G)}{\text{std}(\{R(\mathbf{x}_0^j, \mathbf{c})\}_{j=1}^G)}
$$

Reward $R$ 仅在最终步 $t=0$ 给出。

### 5.2 Importance Sampling Ratio

**（公式 6）**

$$
r_t^i(\theta) = \frac{p_\theta(\mathbf{x}_{t-1}^i | \mathbf{x}_t^i, \mathbf{c})}{p_{\theta_{\text{old}}}(\mathbf{x}_{t-1}^i | \mathbf{x}_t^i, \mathbf{c})}
$$

分子分母都是 ODE-to-SDE 转换后每步的高斯概率密度，由公式 (12) 给出。**没有 SDE 转换，$p_\theta$ 不存在，$r_t^i$ 无法计算。**

### 5.3 完整 FlowGRPO Loss

将 PPO-style clipped objective 与 KL penalty 结合：

**（公式 5）**

$$
\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \min\left(r_t^i(\theta) \hat{A}_t^i, \text{clip}(r_t^i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t^i\right) - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]
$$

其中 $D_{\text{KL}}$ 由公式 (13) 的闭式公式高效计算。

---

## 六、Denoising Reduction

### 6.1 核心发现

在线 RL 需要大量采样来收集训练数据，而 Flow Matching 生成一张图通常需要 $T=30$–$50$ 步，成本极高。

**关键发现**：训练时不需要用这么多步数。

### 6.2 具体策略

| 阶段 | 去噪步数 | 目的 |
|------|---------|------|
| 训练采样 | $T = 10$ 步 | 快速收集多样化轨迹 |
| 推理生成 | $T = 40$ 步 | 保证最终生成质量 |

推理时用完整步数保证质量，训练时用少量步数大幅加速数据收集。实验证明这种策略对最终性能几乎无影响。

### 6.3 直观解释

少量步数采样得到的轨迹"质量较低"，但 **reward signal 仍然有效**——轨迹末端的图像内容是否匹配 prompt，这个判断在 $T=10$ 和 $T=40$ 下基本一致，因此低步数采样的 reward 仍然可以指导策略更新。

---

## 七、实验结果

### 7.1 GenEval 基准（Compositional Image Generation）

| 模型 | Overall | Single Obj. | Two Obj. | Counting | Colors | Position | Attr. Binding |
|------|---------|-------------|----------|----------|--------|----------|---------------|
| DALLE-3 | 0.67 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 |
| Janus-Pro-7B | 0.80 | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 |
| GPT-4o | 0.84 | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 |
| **SD3.5-M + FlowGRPO** | **0.95** | **1.00** | **0.99** | **0.95** | **0.92** | **0.99** | **0.86** |

SD3.5-M 从 63% → **95%**，超越 GPT-4o (84%)，尤其在 Counting 和 Position 上提升显著。

### 7.2 Visual Text Rendering

| 模型 | OCR Accuracy |
|------|-------------|
| SD3.5-M（基线） | 59% |
| **SD3.5-M + FlowGRPO** | **92%** |

文字渲染能力从 59% → 92%，提升 33pp。

### 7.3 Reward Hacking 分析

| 配置 | Task Reward | Aesthetic Score | PickScore | DeQA Score |
|------|-------------|-----------------|-----------|------------|
| SD3.5-M（基线） | — | 21.72 | 0.87 | 5.39 |
| FlowGRPO (w/o KL) | 高 | 4.93 | 0.44 | 2.77 |
| **FlowGRPO (w/ KL)** | **高** | **5.25** | **1.03** | **4.01** |

**关键发现**：不加 KL 约束时，task reward 很高但 Aesthetic 和 PickScore 严重下降（reward hacking）。加了 KL 约束后，task reward 不变，但所有质量指标都维持甚至超过基线水平。

---

## 八、算法框架图

```
┌──────────────────────────────────────────────────────────────┐
│                   FlowGRPO 整体框架                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  【Step 1: ODE → SDE 转换（核心创新）】                        │
│                                                              │
│  Rectified Flow ODE:                                         │
│  dxt = vt · dt                                              │
│         │                                                    │
│         ▼                                                    │
│  Marginal-Preserving Reverse-Time SDE:                        │
│  dxt = [vt - (σt²/2)∇log pt] dt + σt dwt                   │
│         │                                                    │
│         ▼                                                    │
│  Score Function 闭式（Rectified Flow 特有）:                  │
│  ∇log pt = -x/t - (1-t)/t · vt                             │
│         │                                                    │
│         ▼                                                    │
│  Reverse-Time SDE（代入后）:                                  │
│  dxt = [vt + (σt²/2t)(xt + (1-t)vt)] dt + σt dwt           │
│         │                                                    │
│         ▼                                                    │
│  Euler-Maruyama 离散化:                                      │
│  xt+Δt = xt + [vθ + (σt²/2t)(xt+(1-t)vθ)]Δt               │
│          + σt√Δt · ε                                        │
│         │                                                    │
│         ▼                                                    │
│  策略分布: πθ(xt-1|xt,c) = N(μθ, σt²Δt · I)               │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  【Step 2: GRPO 目标函数】                                   │
│                                                              │
│  采样 G 个 rollouts → 计算 group-normalized advantage       │
│  Âti = [Ri - mean(R)] / std(R)                             │
│                                                              │
│  Importance Sampling Ratio:                                   │
│  rti(θ) = pθ(xt-1|xt,c) / pθold(xt-1|xt,c)               │
│                                                              │
│  Loss = min(clipped rt · Â, rt · Â) - β·DKL(πθ||πref)    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  【Step 3: Denoising Reduction】                            │
│                                                              │
│  训练: T=10 步（快速采样，低成本）                           │
│  推理: T=40 步（全步数保质量）                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 九、方法对比与局限性

### 9.1 与 Diffusion Model RL 的对比

| 维度 | Diffusion Model RL | Flow Matching + FlowGRPO |
|------|-------------------|-------------------------|
| 采样天然随机性 | ✅ DDPM 前向过程自带随机 | ❌ 需 ODE-to-SDE 转换 |
| 推理步数 | 通常 50-100 步 | 少量步即可（10-40 步） |
| Score Function | 需要额外网络估计 | ✅ Rectified Flow 有闭式推导 |
| RL 训练成本 | 高 | 较低（更少步数） |

### 9.2 局限性

1. **KL 散度在边界处行为不稳定**：当 $t \to 0$ 或 $t \to 1$ 时，KL 散度中的 $\frac{1}{t(1-t)}$ 项导致训练不稳定。SAGE-GRPO 正是针对此问题的改进工作。

2. **噪声系数 $\sigma_t$ 需要手工调参**：论文设定 $\sigma_t = a\sqrt{t/(1-t)}$，$a$ 的最优值对不同任务可能不同。

3. **Denoising Reduction 的理论保证缺失**：为什么 $T=10$ 步训练就能 work，论文只提供了实验观察，缺乏理论分析。

4. **视频生成未验证**：论文聚焦图像（SD3.5-M），视频生成模型的 ODE-to-SDE 行为是否相同有待研究。

---

## 十、个人点评

FlowGRPO 的核心贡献是**证明了 Flow Matching 模型可以被 GRPO 有效训练**，这本身就是一个重要的里程碑。

**最值得关注的工程发现**是 Denoising Reduction——训练时用 10 步采样居然能达到甚至超过 40 步的效果。这个发现对所有 Flow Matching + RL 的工作都有直接的实用价值，因为它直接解决了在线 RL 的采样效率瓶颈。

**最重要的理论贡献**是公式 (10) 的 score function 闭式推导。Rectified Flow 的 marginal $p_t$ 有解析形式，使得 score function 完全由 $\mathbf{x}_t$ 和 $\mathbf{v}_t$ 表示，无需额外网络——这是 ODE-to-SDE 转换在 Flow Matching 中可以精确执行的根本原因。

---

## 附录 A：Marginal-Preserving SDE 证明摘要

论文 Appendix A 给出了完整的数学证明，核心逻辑：

**目标：** 证明公式 (9) 的 reverse-time SDE 与原 forward ODE（公式 7）具有完全相同的 marginal distributions $\{p_t\}$。

**证明思路（基于 Fokker-Planck 方程）：**

1. **Forward ODE** $\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_t(\mathbf{x}_t)$ 对应的 Fokker-Planck 方程描述了 marginal density $p_t$ 的演化

2. **Reverse SDE**（公式 9）的 Fokker-Planck 方程中，漂移项包含 $-\frac{\sigma_t^2}{2}\nabla\log p_t$（Itô correction），这恰好抵消了扩散项对 marginal 的影响

3. 联立两个 Fokker-Planck 方程，可以证明 reverse SDE 的 marginal density 演化与 forward ODE 完全一致

**关键条件：** $\sigma_t$ 的选择必须满足特定约束，使得 Itô correction 在数学上与 marginal 演化自洽。

---

## 参考链接

- **arXiv**: https://arxiv.org/abs/2505.05470
- **代码**: https://github.com/yifan123/flow_grpo
- **相关工作 SAGE-GRPO**（解决 FlowGRPO 高噪声区问题）: arXiv:2603.21872

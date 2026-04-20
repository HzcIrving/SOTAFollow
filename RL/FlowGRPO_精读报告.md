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

$$
\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1] \tag{1}
$$

模型通过最小化 Flow Matching 目标函数来学习速度场 $\mathbf{v}_\theta(\mathbf{x}_t, t)$：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \left\| \mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_t, t) \right\|^2 \right] \tag{2}
$$

其中目标速度场 $\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$。

### 2.2 确定性 ODE 采样的根本问题

Flow Matching 的生成过程是确定性的概率流 ODE：

$$
d\mathbf{x}_t = \mathbf{v}_t \, dt \tag{7}
$$

Euler 离散化后得到：

$$
\mathbf{x}_{t-1} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}) \tag{8}
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

$$
d\mathbf{x}_t = \left( \mathbf{v}_t(\mathbf{x}_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) \right) dt + \sigma_t \, d\mathbf{w} \tag{9}
$$

**三项的物理含义：**

| 项 | 作用 |
|----|------|
| $\mathbf{v}_t(\mathbf{x}_t) dt$ | 确定性漂移，沿原 ODE 速度场推进 |
| $-\frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) dt$ | Itô 修正项，保证 marginal 不变 |
| $\sigma_t d\mathbf{w}$ | Wiener 扩散项，引入随机探索噪声 |

### 3.2 Rectified Flow 的 Score Function 闭式推导

对于 Rectified Flow，score function 与速度场存在闭式关系。利用 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，可得：

$$
\nabla \log p_t(\mathbf{x}) = -\frac{\mathbf{x}}{t} - \frac{1-t}{t} \mathbf{v}_t(\mathbf{x}) \tag{10}
$$

**推导思路：** 当 $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 固定时，$\mathbf{x}_t$ 服从高斯分布，其协方差为 $t^2 \mathbf{I}$，score function 有解析形式即上式。

### 3.3 代入得到 Reverse-Time SDE

将公式 (10) 代入公式 (9)，整理得到 Rectified Flow 特有的 reverse-time SDE：

$$
d\mathbf{x}_t = \left[ \mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t)\mathbf{v}_t(\mathbf{x}_t) \right) \right] dt + \sigma_t \, d\mathbf{w} \tag{11}
$$

**直观理解：**
- 第一项：沿速度场向前推进
- 第二项（Langevin correction）：补偿扩散项对 marginal 的漂移影响
- 第三项：高斯随机噪声

### 3.4 Euler-Maruyama 离散化

对连续 SDE 应用 Euler-Maruyama 离散化（时间步长 $\Delta t$）：

$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \left[ \mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t)\mathbf{v}_\theta(\mathbf{x}_t, t) \right) \right] \Delta t + \sigma_t \sqrt{\Delta t} \, \epsilon \tag{12}
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

两个均值不同、方差相同的高斯分布之间的 KL 散度为：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\| \boldsymbol{\mu}_\theta - \boldsymbol{\mu}_{\text{ref}} \|^2}{2\sigma^2}
$$

代入 $\boldsymbol{\mu}$ 的具体形式，经过化简得到：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\Delta t}{2} \left( \frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t} \right)^2 \left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}_{\text{ref}}(\mathbf{x}_t, t) \right\|^2 \tag{13}
$$

**简化分析：** 代入 $\sigma_t = a\sqrt{t/(1-t)}$ 后，KL 散度在 $t$ 方向上由 $\frac{1}{t(1-t)}$ 类项主导，在 $t \to 0$ 和 $t \to 1$ 时会显著放大。这是后续 SAGE-GRPO 重点解决的问题。

---

## 五、GRPO 目标函数与 Flow Matching 的衔接

### 5.1 Group-Normalized Advantage

给定 prompt $\mathbf{c}$，采样 $G$ 张图 $\{\mathbf{x}_0^i\}_{i=1}^G$，每张图的 advantage 为：

$$
\hat{A}_t^i = \frac{R(\mathbf{x}_0^i, \mathbf{c}) - \text{mean}(\{R(\mathbf{x}_0^j, \mathbf{c})\}_{j=1}^G)}{\text{std}(\{R(\mathbf{x}_0^j, \mathbf{c})\}_{j=1}^G)} \tag{4}
$$

Reward $R$ 仅在最终步 $t=0$ 给出。

### 5.2 Importance Sampling Ratio

$$
r_t^i(\theta) = \frac{p_\theta(\mathbf{x}_{t-1}^i | \mathbf{x}_t^i, \mathbf{c})}{p_{\theta_{\text{old}}}(\mathbf{x}_{t-1}^i | \mathbf{x}_t^i, \mathbf{c})} \tag{6}
$$

分子分母都是 ODE-to-SDE 转换后每步的高斯概率密度，由公式 (12) 给出。

### 5.3 完整 FlowGRPO Loss

将 PPO-style clipped objective 与 KL penalty 结合：

$$
\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \min\left(r_t^i(\theta) \hat{A}_t^i, \text{clip}(r_t^i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t^i\right) - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right] \tag{5}
$$

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

---

## 七、论文原图解析（图文并茂）

### 图 1：GenEval 训练曲线与质量指标

**Figure 1(a) GenEval Performance**

> **图片描述**：折线图展示 FlowGRPO 训练过程中 GenEval 准确率随训练 step 的变化。
>
> - **X 轴**：Training Steps（从 0 到 3000+）
> - **Y 轴**：GenEval Overall Accuracy（0.0 → 0.95）
> - **曲线特征**：从基线 0.63 开始，经过 FlowGRPO 训练后稳定上升到 **0.95**，超过 GPT-4o 的 0.84 基线（虚线）
> - **关键观察**：曲线呈现**稳定上升趋势**，无明显震荡，说明训练过程平稳

**Figure 1(b) Image Quality Metrics on DrawBench**

> - **X 轴**：同 (a)
> - **Y 轴**：CLIP Score / Aesthetic Score 等质量指标
> - **曲线特征**：所有质量指标在训练过程中**基本保持水平**（轻微波动），说明 reward 提升没有以牺牲图像质量为代价
> - **关键观察**：几乎没有 reward hacking 现象

**Figure 1(c) Human Preference Scores on DrawBench**

> - **X 轴**：同 (a)
> - **Y 轴**：PickScore / UnifiedReward 等人类偏好分数
> - **曲线特征**：偏好分数在训练后**明显提升**，说明 FlowGRPO 有效对齐了人类审美

**Figure 1 总结**：三个子图共同说明 FlowGRPO 实现了 "reward 提升 + 质量稳定 + 偏好对齐" 三赢局面。

---

### 图 2：FlowGRPO 算法框架（核心框架图）

> **图片描述**：流程图展示 FlowGRPO 的完整 pipeline，包含三个主要模块。
>
> **左侧：Prompt Set & ODE-to-SDE 转换**
> - 顶部：Prompt Set（多个文本 prompt）
> - 向下：进入 "ODE-to-SDE" 转换模块（虚线框标注"Key Strategy 1"）
> - 输出：多条随机轨迹（Sampling with SDE）
>
> **中间：GRPO Loss 计算**
> - 输入：随机轨迹集合
> - 关键节点：GRPO Loss（Group-normalized advantage + Clipped PPO objective + KL penalty）
> - 计算 reward $R(\mathbf{x}_0, \mathbf{c})$
>
> **右侧：策略更新**
> - GRPO Loss → Online Update → 对齐的策略（Aligned Policy）
> - 输出：高质量图像（High-Quality Images）
>
> **底部注释**：Denoising Reduction（Key Strategy 2）：训练步数 T=10（快速），推理步数 T=40（保质量）
>
> **图片元素**：
> - 圆角矩形：表示模块
> - 箭头：数据流方向
> - 虚线框：标注关键策略
> - 底部：T=10 / T=40 对比标注

---

### 图 3：GenEval 定性对比（Qualitative Comparison）

> **图片描述**：Grid 布局展示多组 GenEval 任务中的生成质量对比。
>
> 每行包含：
> - **左侧**：Prompt 文本（描述生成要求）
> - **中间**：SD3.5-M 基线生成结果
> - **右侧**：SD3.5-M + FlowGRPO 生成结果
>
> **典型行示例：**
> - **Counting（数数）**：Prompt 要求生成 "3 只猫" → 基线可能生成错误数量，FlowGRPO 生成精确数量
> - **Colors（颜色）**：Prompt 要求 "红色球体 + 蓝色方块" → 基线颜色混淆，FlowGRPO 准确渲染
> - **Position（位置）**：Prompt 要求 "左边的狗" → 基线空间关系错误，FlowGRPO 准确定位
> - **Attribute Binding（属性绑定）**：Prompt 要求 "戴红色帽子的猫" → 基线属性错配，FlowGRPO 正确绑定
>
> **关键发现**：FlowGRPO 在所有类别上都明显优于基线，尤其在复合约束（multiple attributes + spatial relations）场景。

---

### 图 4 / 表 1：GenEval 数值结果对比

**Table 1：GenEval Benchmark Results**

| 模型 | Overall | Single Obj. | Two Obj. | Counting | Colors | Position | Attr. Binding |
|------|---------|-------------|----------|----------|--------|----------|---------------|
| SD3.5-M（基线） | 0.63 | 0.98 | 0.78 | 0.50 | 0.81 | 0.24 | 0.52 |
| **+ FlowGRPO** | **0.95** | **1.00** | **0.99** | **0.95** | **0.92** | **0.99** | **0.86** |
| 提升幅度 | +32pp | +2pp | +21pp | +45pp | +11pp | +75pp | +34pp |

**关键洞察**：
- Position 提升最显著（+75pp）：FlowGRPO 有效解决了空间关系推理难题
- Counting 提升 +45pp：从 50% 到 95%，接近完美
- Two Objects 提升 +21pp：复合对象场景大幅改善

---

### 表 2：多任务综合性能

| 配置 | Task Metric | Image Quality | Preference |
|------|------------|---------------|------------|
| SD3.5-M 基线 | — | Aesthetic: 21.72 | PickScore: 0.87 |
| FlowGRPO (w/o KL) | GenEval 0.95 | Aesthetic: 4.93 ❌ | PickScore: 0.44 ❌ |
| **FlowGRPO (w/ KL)** | **GenEval 0.95** | **Aesthetic: 5.25** ✅ | **PickScore: 1.03** ✅ |

**核心结论**：KL 约束是防止 reward hacking 的关键，不是 early stopping 的替代品。

---

## 八、算法框架图（手绘简化版）

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FlowGRPO 整体框架                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Prompt Set                                                        │
│   { c₁, c₂, ..., cₙ }                                               │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │  Step 1: ODE → SDE 转换（核心创新）                        │      │
│   │                                                          │      │
│   │  Rectified Flow ODE:                                     │      │
│   │  dxt = vt · dt                                           │      │
│   │       │                                                 │      │
│   │       ▼                                                 │      │
│   │  Marginal-Preserving Reverse-Time SDE:                    │      │
│   │  dxt = [vt + (σt²/2t)(xt + (1-t)vt)]dt + σt dwt         │      │
│   │       │                                                 │      │
│   │       ▼                                                 │      │
│   │  Euler-Maruyama 离散化:                                   │      │
│   │  xt+Δt = xt + [vθ + (σt²/2t)(xt+(1-t)vθ)]Δt             │      │
│   │          + σt√Δt · ε,  where ε~N(0,I)                     │      │
│   │       │                                                 │      │
│   │       ▼                                                 │      │
│   │  策略分布: πθ(xt-1|xt,c) = N(μθ, σt²Δt·I)               │      │
│   └─────────────────────────────────────────────────────────┘      │
│         │                                                           │
│         ▼                                                           │
│   Sampling with SDE → 多条随机轨迹 {x⁽¹⁾, x⁽²⁾, ..., x⁽G⁾}         │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │  Step 2: GRPO Loss                                      │      │
│   │                                                          │      │
│   │  Reward: R(x₀, c)  [仅在 t=0 给出]                      │      │
│   │  Advantage: Âi = [Ri - mean(R)] / std(R)               │      │
│   │  Importance Ratio: rti(θ) = pθ/pθ_old                   │      │
│   │                                                          │      │
│   │  Loss = min(clipped rt·Â, rt·Â) - β·DKL(πθ||πref)      │      │
│   └─────────────────────────────────────────────────────────┘      │
│         │                                                           │
│         ▼                                                           │
│   Online Update → Aligned Policy                                   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   Step 3: Denoising Reduction                                      │
│   训练: T=10 步 (快速采样)                                          │
│   推理: T=40 步 (全步数保质量)                                        │
└──────────────────────────────────────────────────────────────────────┘
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

**目标：** 证明公式 (9) 的 reverse-time SDE 与原 forward ODE（公式 7）具有完全相同的 marginal distributions $\{p_t\}$。

**证明思路（基于 Fokker-Planck 方程）：**

1. **Forward ODE** $\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_t(\mathbf{x}_t)$ 对应的 Fokker-Planck 方程描述了 marginal density $p_t$ 的演化

2. **Reverse SDE**（公式 9）的 Fokker-Planck 方程中，漂移项包含 $-\frac{\sigma_t^2}{2}\nabla\log p_t$（Itô correction），这恰好抵消了扩散项对 marginal 的影响

3. 联立两个 Fokker-Planck 方程，可以证明 reverse SDE 的 marginal density 演化与 forward ODE 完全一致

---

## 参考链接

- **arXiv**: https://arxiv.org/abs/2505.05470
- **代码**: https://github.com/yifan123/flow_grpo
- **项目主页**: https://gongyeliu.github.io/Flow-GRPO/
- **相关工作 SAGE-GRPO**（解决 FlowGRPO 高噪声区问题）: arXiv:2603.21872
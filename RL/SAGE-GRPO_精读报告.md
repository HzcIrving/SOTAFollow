# SAGE-GRPO 精读报告：流模型 RL 中的 ODE-to-SDE 探索

## 引用信息

- **论文**：Manifold-Aware Exploration for Reinforcement Learning in Video Generation
- **arXiv**：https://arxiv.org/abs/2603.21872
- **作者**：Mingzhe Zheng*, Weijie Kong*, et al.（*共同一作）
- **机构**：上海 AI Lab、清华大学、港中文（深圳）等
- **顶会**：ICML 2026（submitted Mar 2026）

---

## 一句话总结

提出 **SAGE-GRPO**，通过流匹配模型的**流形感知 SDE 精确离散化**（含对数曲率校正 + 梯度范数均衡器）和**双信任域机制**（位置-速度双重控制），解决视频生成 GRPO 中 ODE-to-SDE 转换导致的过噪声、off-manifold 漂移和奖励估计不稳定问题。

---

## 拟人化开篇

训练视频生成模型对齐人类偏好，本质上是让一个已经"见过世间万物的老模型"学会按特定风格或指令微调自己。GRPO 是这个场景下的标准强化学习方法论——它通过比较一群 rollouts 的 reward 相对大小来估算 advantage，无需 critic 网络，简洁高效。

然而，当 GRPO 遇上视频生成，一切变得棘手。视频解空间庞大、时序连贯性要求极高，现有的 DanceGRPO、FlowGRPO 做法是：先把确定性的 ODE（常微分方程）采样器转换成 SDE（随机微分方程）来引入探索噪声——这一步叫做 **ODE-to-SDE 转换**。

问题恰好出在这里：**这个转换太粗糙了**。Euler 风格的离散化加一阶近似，会在高价噪声区域（high-noise steps）注入多余的噪声能量。这些多余噪声把采样出来的视频帧"推离"了合法的视频流形（manifold），产生时间抖动、artifacts，最终 reward 估计变得不可靠，整个对齐过程就崩了。

这正是 SAGE-GRPO 要解决的问题。

---

## 背景与动机

### 2.1 Flow Matching 与 Rectified Flow

视频生成任务中，Flow Matching 将生成过程建模为在概率路径 $p_t(\mathbf{x})$ 上的传输过程，由 ODE 描述：

$$
\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t) \tag{1}
$$

其中 $\mathbf{v}_\theta$ 是神经网络预测的速度场。Rectified Flow 使用**线性插值路径**：

$$
\mathbf{x}_t = (1 - \sigma_t)\mathbf{x}_0 + \sigma_t \mathbf{z}_1 \tag{2}
$$

其中 $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是纯噪声，$\sigma_t$ 是噪声调度，$\sigma_0 = 0$（纯净数据），$\sigma_1 = 1$（纯噪声）。

对 (2) 求导，可得隐含的速度场：

$$
\mathbf{v}_\theta(\mathbf{x}_t, t) = \frac{d\mathbf{x}_t}{dt} = -\frac{d\sigma_t}{dt}(\mathbf{x}_0 - \mathbf{z}_1) = \frac{1}{1 - \sigma_t}(\mathbf{x}_t - \mathbf{x}_0) \tag{3}
$$

这意味着给定任意中间状态 $\mathbf{x}_t$，模型可以计算指向数据分布的"速度向量"，从而通过 ODE 积分从噪声走向数据。

### 2.2 GRPO 核心公式

给定 prompt $\mathbf{c}$，GRPO 采样 $G$ 个 rollouts，用 group-normalized advantage 优化策略：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} A_i \cdot \sum_{t=1}^{T} \log \pi_\theta(\mathbf{x}_{t-1}^{(i)} | \mathbf{x}_t^{(i)}, \mathbf{c}) \tag{4}
$$

其中 $A_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}$ 是 group 归一化 advantage，$r_i = R(\mathbf{x}_0^{(i)})$。

### 2.3 ODE-to-SDE 转换的必要性

GRPO 需要**多样性探索**：同一个 prompt，要通过随机采样得到多段不同风格/内容的视频，用来比较 reward 高低从而估算 advantage。确定性 ODE 只能产生一条轨迹，无法提供多样性——所以必须把 ODE 转换成 SDE，在采样过程中注入随机噪声。

这是 ODE-to-SDE 转换的核心动机。

### 2.4 现有方法的问题：Euler 风格离散化 + 一阶近似

DanceGRPO 和 FlowGRPO 在做 ODE-to-SDE 转换时，采用的是 **Euler-Maruyama 离散化 + 一阶近似**：

**传统方法**（以 FlowGRPO 为代表）使用的 SDE 噪声方差是一阶近似：

$$
\Sigma_t^{\text{linear}} \approx \eta^2 \cdot (\sigma_t - \sigma_{t+1}) \cdot \frac{\sigma_t}{1 - \sigma_t} \tag{前人常用}
$$

这种近似在 $\sigma_t$ 接近 1（高噪声区域）时，**严重低估分母 $(1-\sigma_t)$**，导致注入的噪声方差远大于正确值。

**危害**：
1. 高噪声步骤中噪声过多，视频帧被"炸离"合法视频流形 $\mathcal{M}$，产生 temporal jitter 和 artifacts
2. rollout 质量下降，reward 模型评估不可靠
3. 策略梯度方向被噪声主导，对齐不稳定

### 2.5 核心问题定义

> **核心问题**：如何约束探索范围在视频数据流形 $\mathcal{M}$ 附近，使每个探索步骤产生的 rollout 仍然有效，reward 评估可信？

---

## 三、SAGE-GRPO 方法论：Micro-Level（ODE-to-SDE 精确化）

### 3.1 目标：精确的 Marginal-Preserving SDE

我们需要在离散化的 SDE 中保持 marginal distribution 不变。具体来说，给定 marginal-preserving SDE：

$$
d\mathbf{z}_t = \left( \mathbf{v}_\theta(\mathbf{x}_t, t) - \frac{1}{2} \varepsilon_t^2 \mathbf{s}_\theta(\mathbf{x}_t) \right) dt + \varepsilon_t d\mathbf{w}_t \tag{16}
$$

其中：
- $\varepsilon_t$ 是扩散系数（时间 $t$ 的函数）
- $\mathbf{w}_t$ 是布朗运动
- $\mathbf{s}_\theta(\mathbf{x}_t) \approx -(\mathbf{x}_t - \hat{\mathbf{x}}_0) / \sigma_t^2$ 是 score function 估计
- $-\frac{1}{2}\varepsilon_t^2 \mathbf{s}_\theta(\mathbf{x}_t)$ 是 **Itô 修正项**，保证 marginal 分布与原 ODE 一致

### 3.2 扩散系数的设定

论文设定扩散系数为：

$$
\varepsilon_t = \eta \sqrt{\frac{\sigma_t}{1 - \sigma_t}} \tag{扩散系数}
$$

其中 $\eta$ 是探索规模超参数。这个形式保证了在 $\sigma_t \to 0$（低噪声）时 $\varepsilon_t \to 0$，在 $\sigma_t \to 1$（高噪声）时 $\varepsilon_t \to \infty$。

### 3.3 【核心】ODE-to-SDE：精确方差积分推导

**关键问题**：在离散化 SDE 时，噪声标准差 $\Sigma_t^{1/2}$ 应该取多少？

#### 步骤 1：积分方差

不是用一阶近似 $\varepsilon_t^2 \Delta t$，而是对扩散系数在 $[ \sigma_{t+1}, \sigma_t ]$ 区间上**精确积分**：

$$
\Sigma_t = \int_{\sigma_{t+1}}^{\sigma_t} \varepsilon_s^2 \, ds = \eta^2 \int_{\sigma_{t+1}}^{\sigma_t} \frac{\sigma_s}{1 - \sigma_s} \, ds \tag{积分方差}
$$

对被积函数做变量替换，分式分解：

$$
\int \frac{\sigma}{1-\sigma} d\sigma = \int \left( -1 + \frac{1}{1-\sigma} \right) d\sigma = -\sigma - \log(1-\sigma) + C
$$

代入上下限得到**精确方差**：

$$
\Sigma_t = \eta^2 \left[ -(\sigma_t - \sigma_{t+1}) + \log\left(\frac{1-\sigma_{t+1}}{1-\sigma_t}\right) \right] \tag{5}
$$

#### 步骤 2：对数曲率校正项（Logarithmic Curvature Correction）

式 (5) 中的对数项：

$$
\log\left(\frac{1-\sigma_{t+1}}{1-\sigma_t}\right) = -\log\left(\frac{1-\sigma_t}{1-\sigma_{t+1}}\right)
$$

这就是**对数曲率校正项**——它精确捕捉了信号系数 $(1-\sigma_t)$ 的几何收缩效应。一阶线性近似完全忽略了这个曲率，而精确积分天然包含了它。

> **几何直觉**：在 $(\sigma_t, 1-\sigma_t)$ 参数空间中，被积函数 $\sigma/(1-\sigma)$ 并不是线性函数，其曲线下面积不能简单用矩形近似。对数项正是对这一弯曲几何的精确补偿。

#### 步骤 3：噪声标准差

对 $\Sigma_t$ 开方：

$$
\Sigma_t^{1/2} = \eta \sqrt{ -(\sigma_t - \sigma_{t+1}) + \log\left(\frac{1-\sigma_{t+1}}{1-\sigma_t}\right) } \tag{6}
$$

#### 步骤 4：Euler-Maruyama 离散化

设时间步长 $\Delta t = \sigma_t - \sigma_{t+1}$，应用 Euler-Maruyama 离散化：

$$
\boxed{
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{v}_\theta(\mathbf{x}_t, t) \Delta t + \frac{\Sigma_t}{2} \mathbf{s}_\theta(\mathbf{x}_t) + \Sigma_t^{1/2} \bm{\epsilon}
} \tag{7}
$$

其中 $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

**三项的物理含义**：
1. $\mathbf{v}_\theta(\mathbf{x}_t, t) \Delta t$：**确定性漂移**，沿速度场方向推进
2. $\frac{\Sigma_t}{2} \mathbf{s}_\theta(\mathbf{x}_t)$：**Itô 修正项**，保证 marginal 与原 Rectified Flow 一致
3. $\Sigma_t^{1/2} \bm{\epsilon}$：**探索噪声**，从标准正态采样 $\times$ 精确标准差

> **注意**：由于 $\Sigma_t$ 是对 $[ \sigma_{t+1}, \sigma_t ]$ 积分的结果（已经包含时间尺度），随机项直接用 $\Sigma_t^{1/2}$ 而**不需要再乘 $\sqrt{\Delta t}$**。这是和一阶近似的另一关键区别。

#### 对比：FlowGRPO（Euler 一阶）vs SAGE-GRPO（精确积分）

| 方法 | 噪声方差 | 问题 |
|------|----------|------|
| FlowGRPO | $\eta^2 (\sigma_t - \sigma_{t+1}) \cdot \frac{\sigma_t}{1-\sigma_t}$ | 高估噪声，off-manifold 漂移 |
| **SAGE-GRPO** | $\eta^2[ -(\sigma_t - \sigma_{t+1}) + \log\frac{1-\sigma_{t+1}}{1-\sigma_t} ]$ | 精确积分，更小的探索区域 |

### 3.4 梯度范数均衡器（Gradient Norm Equalizer）

即使做了精确的 SDE 离散化，扩散过程本身还存在**内禀的 SNR 不平衡问题**：

对于高斯转移 $\pi(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\bm{\mu}_\theta, \Sigma_t \mathbf{I})$：

$$
\| \nabla_{\bm{\mu}} \log \pi \| \propto \frac{1}{\Sigma_t^{1/2}} \tag{8}
$$

**后果**：
- 当 $\sigma_t \to 1$（高噪声，$t \to 1$）：$\Sigma_t$ 大，梯度 $\to 0$（消失）
- 当 $\sigma_t \to 0$（低噪声，$t \to 0$）：$\Sigma_t$ 小，梯度爆炸

经验验证（论文 Figure 4）：观测到的梯度范数随 $\sigma$ 增大急剧下降，与 $1/\Sigma_t^{1/2}$ 预测一致。

**均衡器公式**：

$$
\boxed{
S_t = \frac{\text{Median}(\{\mathcal{N}_\tau\}_{\tau=1}^{T})}{\mathcal{N}_t + \epsilon}
} \tag{9}
$$

其中 $\mathcal{N}_t$ 是从 SDE 参数估算的每步梯度尺度。用中位数归一化使所有时间步的优化压力在同一量级，防止某些 phase 主导学习。

> **直观理解**：相当于在优化目标中引入了一个与时间相关的加权系数——在噪声大的步骤（原本梯度小）加大权重，在噪声小的步骤（原本梯度大）缩小权重，使每步对最终 loss 的贡献"均等化"。

---

## 四、Macro-Level：双信任域（Dual Trust Region）

### 4.1 KL 散度作为动态锚定机制

将 KL 散度理解为策略空间中的距离度量。对于 Gaussian 策略：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{\mathbf{x}_t \sim \pi_\theta} \left[ \frac{(\bm{\mu}_\theta - \bm{\mu}_{\text{ref}})^2}{2\Sigma_t^2} \right] \approx \frac{(\bm{\mu}_\theta - \bm{\mu}_{\text{ref}})^2}{2\Sigma_t^2} \tag{11}
$$

参考策略的选择决定约束性质：固定参考 = 硬约束，移动参考 = 自适应探索。

### 4.2 固定 KL：位置约束但限制了最优性

传统做法：$\pi_{\text{ref}} = \pi_0$（初始预训练模型）

问题：随着训练进行，最优策略 $\pi^*$ 可能远离 $\pi_0$。强制 $D_{\text{KL}}(\pi_\theta \| \pi_0)$ 小，会导致**欠拟合**——策略无法充分探索远离初始分布的区域。

### 4.3 步级 KL：速度约束但无法防止累积漂移

改进做法：$\pi_{\text{ref}} = \pi_{k-1}$（上一步策略）

这作为速度限制器，限制每步参数更新幅度：

$$
\| \nabla_\theta D_{\text{KL}}(\pi_\theta \| \pi_{k-1}) \| \propto \|\bm{\mu}_\theta - \bm{\mu}_{k-1}\| / \Sigma_t \tag{12}
$$

但它只约束瞬时更新方向 $\nabla_\theta$，**不约束累积位移** $\|\theta_k - \theta_0\|$。即便每步很小，策略也可以缓慢漂移，最终偏离流形，产生 reward hacking 或崩溃。

### 4.4 周期性移动锚（Periodical Moving Anchor）

每 $N$ 步更新一次参考策略：$\pi_{\text{ref}} \leftarrow \pi_\theta$，创建动态信任域：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}_N}) = \frac{(\bm{\mu}_\theta - \bm{\mu}_{\text{ref}_N})^2}{2\Sigma_t^2} \tag{13}
$$

$N$ 步内做局部探索，然后以当前策略为锚重新建立安全区——类似 TRPO 的多阶段松弛版本，兼顾塑性（plasticity）和稳定性（stability）。

### 4.5 双 KL：位置-速度控制器

$$
\boxed{
\mathcal{L}_{\text{KL}} = \beta_{\text{pos}} \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}_N}) + \beta_{\text{vel}} \cdot D_{\text{KL}}(\pi_\theta \| \pi_{k-1})
} \tag{14}
$$

- **位置项** $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}_N})$：主方向锚，防止长期漂移
- **速度项** $D_{\text{KL}}(\pi_\theta \| \pi_{k-1})$：阻尼因子，平滑瞬时更新

### 4.6 完整 SAGE-GRPO 目标函数

将 (4)(9)(14) 结合：

$$
\mathcal{L}_{\text{SAGE-GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} A_i \cdot \sum_{t=1}^{T} S_t \cdot \log \pi_\theta(\mathbf{x}_{t-1}^{(i)} | \mathbf{x}_t^{(i)}, \mathbf{c}) + \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{KL}} \tag{完整目标}
$$

其中 $S_t$ 是梯度均衡系数。

---

## 五、实验结果

### 5.1 实验设置

- **模型**：HunyuanVideo 1.5
- **Reward 模型**：VideoAlign（冻结，不微调）
- **评估指标**：VQ（视觉质量）、MQ（运动质量）、TA（文本对齐）、CLIPScore、PickScore
- **对比基线**：DanceGRPO、FlowGRPO、CPS
- **配置**：Setting A（均衡奖励）、Setting B（对齐聚焦，$w_{vq}=0.5, w_{mq}=0.5, w_{ta}=1.0$）

### 5.2 主要结果（Table 2）

**Setting B（对齐聚焦）下，SAGE-GRPO + Dual Moving KL 达到最佳**：

| 方法 | Overall | VQ | MQ | TA | CLIPScore |
|------|---------|----|----|----| ----------|
| FlowGRPO w/ Fixed KL | 0.2103 | -0.6654 | -0.5506 | 1.4263 | 0.5427 |
| CPS w/o KL | 0.3694 | -0.6650 | -0.5325 | 1.5669 | 0.5479 |
| **SAGE-GRPO w/ Dual Mov KL** | **0.8066** | **-0.4765** | **-0.2384** | 1.5216 | **0.5484** |

- Overall 从 0.37 → **0.81**，提升了 2.2×，显著优于所有基线
- VQ 从 -0.67 → **-0.48**，MQ 从 -0.53 → **-0.24**，TA 基本持平

### 5.3 KL 策略消融（Figure 8）

- **Dual Moving KL**：最高且最稳定的 reward，收敛最快
- **Moving KL**：早期探索充分，但后期探索水平下降
- **Dual Moving KL** 在整个训练过程中保持更高的探索水平，验证了位置-速度双重控制的有效性

### 5.4 梯度均衡器消融（Figure 3）

- **无均衡器**：低噪声时间步主导优化，reward 曲线不稳定或 plateau
- **有均衡器**：reward 曲线平滑、持续提升，梯度规模差异从"超过一个数量级"压缩到"小常数因子内"

---

## 六、算法框架图

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAGE-GRPO 整体框架                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【Micro Level：精确 SDE 采样】                                   │
│                                                                 │
│  Rectified Flow ODE                                             │
│  dxt/dt = vθ(xt, t)                                             │
│           │                                                     │
│           ▼                                                     │
│  Marginal-Preserving SDE                                        │
│  dzt = [vθ(xt,t) - ½·εt²·sθ(xt)]dt + εt·dwt                  │
│           │                                                     │
│           ▼                                                     │
│  精确方差积分（核心创新）                                         │
│  Σt = η²[-(σt-σt+1) + log((1-σt+1)/(1-σt))]                   │
│           │                                                     │
│           ▼                                                     │
│  Euler-Maruyama 离散化：                                         │
│  xt+Δt = xt + vθ(xt,t)Δt + (Σt/2)·sθ(xt) + Σt^½·ε           │
│           │                                                     │
│           ▼                                                     │
│  梯度范数均衡器                                                  │
│  St = Median({Nτ}) / (Nt + ε)                                   │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────┐                       │
│  │ GRPO Loss:                           │                       │
│  │ -1/G Σ Ai · Σ St · log πθ(xt-1|xt) │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【Macro Level：双信任域优化】                                    │
│                                                                 │
│  位置控制：Periodical Moving Anchor（N步重置）                    │
│  DKL(πθ || πref_N)  →  防止长期漂移                             │
│           +                                                         │
│  速度控制：Step-wise KL（每步约束）                               │
│  DKL(πθ || πk-1)   →  平滑瞬时更新                              │
│           │                                                       │
│           ▼                                                       │
│  Dual KL Objective                                               │
│  LKL = βpos·DKL(πθ||πref_N) + βvel·DKL(πθ||πk-1)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、附录要点总结

### A.1 SDE 方差推导细节

论文 Appendix 提供了完整的 SDE 离散化推导，包含：
- Marginal-preserving SDE 的一般形式推导
- Euler-Maruyama 离散化的 Itô 修正项来源
- 为什么随机项直接用 $\Sigma_t^{1/2}$ 而不需要额外乘 $\sqrt{\Delta t}$（因为 $\Sigma_t$ 已经对时间积分）

### A.4 奖励组成与 Group-Normalized Advantage

$$
A_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}, \quad r_i = R(\mathbf{x}_0^{(i)})
$$

其中 $R$ 是 VideoAlign 提供的复合 reward：$R = w_{vq} S_{vq} + w_{mq} S_{mq} + w_{ta} S_{ta}$。

### A.5 梯度尺度估算

论文从 SDE 参数推导了梯度尺度 $\mathcal{N}_t$ 的理论形式，与经验观测的 $1/\Sigma_t^{1/2}$ 关系一致。

### A.6 两阶段 KL 权重调度

实验中采用两阶段 $\lambda_{\text{KL}}$ 调度：

$$
\lambda_{\text{KL}}: 10^{-7} \rightarrow 10^{-5}
$$

- 早期（$10^{-7}$）：宽松约束，允许强探索
- 后期（$10^{-5}$）：收紧信任域，稳定收敛

---

## 八、局限性

1. **计算开销**：精确积分和对数项的计算略增成本，但属于可接受范围
2. **流形假设依赖预训练模型质量**：如果预训练模型本身偏差大，流形 $\mathcal{M}$ 本身可能不够好
3. **视频之外的拓展**：论文聚焦视频，对图像生成（如 Stable Diffusion 族）是否同样有效需要验证
4. **$N$ 和 $\beta$ 的手工设定**：周期性移动锚的间隔 $N$ 和双 KL 的权重 $\beta_{\text{pos}}, \beta_{\text{vel}}$ 需要调参

---

## 九、个人点评

SAGE-GRPO 是一篇工程扎实、理论清晰的工作。核心贡献在于**把 ODE-to-SDE 转换从"Euler 一阶近似"推进到"精确积分 + 对数曲率校正"**，这个改进看似简单，但物理直觉非常清晰——Rectified Flow 的信号衰减有几何曲率，一阶近似忽略了这个曲率就会在高噪声区过量噪声。

**最有意思的观察**是：即使你正确地做了 SDE 离散化，扩散过程本身的 SNR 特性（梯度随 $\sigma$ 变化横跨数个量级）仍然会破坏均衡学习。梯度范数均衡器用中位数归一化来平衡各时间步的优化压力，这个方法非常实用。

宏观的双信任域（位置控制 + 速度控制）也是工程直觉和理论分析的很好结合——从优化角度看，这是用周期性重置的 anchor 防止 KL 约束在大步数上产生 underfitting。

**对 VLA 研究的启示**：视频生成模型的后训练正在成为 VLA 系统的重要组成部分（参考 $\pi_0.7$ 中 VLA+GRPO 的结合）。这类 manifold-aware exploration 思想，对未来 VLA 的后训练优化具有直接参考价值。

---

## 参考链接

- **arXiv**：https://arxiv.org/abs/2603.21872
- **代码/主页**：https://dungeonmassster.github.io/SAGE-GRPO-Page/
- **FlowGRPO 基线**：Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL", arXiv:2505.05470
- **DanceGRPO 基线**：Xue et al., "DanceGRPO: Unleashing GRPO on Visual Generation", arXiv:2505.07818
- **CPS 基线**：Wang & Yu, "Coefficients-Preserving Sampling for RL with Flow Matching", arXiv:2509.05952

# Near-Future Policy Optimization 精读报告

**论文**：Near-Future Policy Optimization (NPO)
**arXiv**：2604.20733
**作者**：Chuanyu Qin, Chenxu Yang, Qingyi Si, Naibin Gu, Dingyu Yao, Zheng Lin, Peng Fu, Nan Duan, Jiaqi Wang
**机构**：Institute of Information Engineering, CAS · School of Cyber Security, UCAS · JD.COM
**来源**：arXiv (2026-04-22), cs.LG
**代码/链接**：待补充

---

## 一句话总结

提出 **Near-Future Policy Optimization (NPO)**，利用同一训练过程中更靠后的 checkpoint 作为近未来策略，为当前策略提供高质量（高 Q）且低方差（低 V）的辅助轨迹，直接最大化有效学习信号 **𝒮 = Q/V**，突破 RLVR 的收敛速度和性能上限。

---

## 拟人化开篇

试想你在解一道数学题。早期训练就像一个刚学解题的学生——几乎所有尝试都是错的，根本没有正确的轨迹可以学习。而到了训练后期，你的解题方法已经高度固化，换 100 种方式都是同一个套路，rollout 多样性枯竭，性能停止增长。

**两个问题的根源相同**：你只能从自己生成的轨迹中学习。

有没有一种方法，让"未来的自己"来教"现在的自己"？毕竟，未来的你解出了你现在解不出的题，而且你们的解题思路几乎一致——这不比请个外部家教（外部 teacher 轨迹）更靠谱吗？

**NPO 的核心洞察正是如此。**

---

## 背景与问题动机

### RLVR 的两大局限

Reinforcement Learning with Verifiable Rewards (RLVR) 已成为推理模型 post-training 的核心范式（DeepSeek-R1、Qwen3 等均基于此），但纯 on-policy 探索面临结构性限制：

1. **早期：稀疏正确轨迹** —— 大多数 rollout 都是错的，梯度信号几乎为零
2. **后期：rollout 分布坍缩** —— 策略趋于固化，pass@1 提升主要来自对已有解空间的重新分配，而非真正的能力扩展

### 现有混合策略方法的困境

| 方法 | Q（信号质量） | V（方差代价） | 问题 |
|------|-------------|-------------|------|
| **外部教师**（LUFFY） | 高 | 高 | 分布差距大，信号被方差淹没 |
| **历史回放**（ExGRPO） | 中 | 中 | 轨迹质量被早期 checkpoint 上限束缚 |
| **远未来回放**（RLEP） | 高 | 极高 | 完全训练好的模型轨迹强但差距爆炸 |
| **纯 On-policy**（GRPO） | =当前策略 | =0 | 无法突破自身探索边界 |

**核心问题**：如何找到**足够强**（能给出当前策略做不到的正确轨迹）且**足够近**（分布差距小，方差可控）的辅助轨迹源？

---

## 方法详解

### 核心洞察：𝒮(Δ) = Q(Δ) / V(Δ)

论文将有效学习信号形式化为：

$$\mathcal{S}(\Delta) = \frac{Q(\Delta)}{V(\Delta)}$$

- **Q(Δ)**：信号质量 —— 当前策略失败的 prompt 中，近未来 checkpoint（Δ 步之后）能产生验证正确答案的比例。Δ 越大，checkpoint 越强，Q 越高（但会饱和）。
- **V(Δ)**：方差代价 —— 由于 importance weighting πθ/π(t+Δ)，引入 off-policy 轨迹带来的梯度方差。Δ 越大，参数漂移越多，V 呈**指数增长**。

𝒮 随 Δ 呈**凹型**：先升后降，存在唯一内点最优 Δ*。

> **图 2(b)(c) 实证验证**：在 Qwen3-VL-8B-Instruct 上，Q concave 上升，V 指数增长，𝒮 确实有清晰的内点最大值（Δ* ≈ 20 步对 base policy；Δ* ≈ 70 步对 mid-training policy）。

### NPO 机制

**核心操作**（Section 3.1）：

对每个 prompt，若当前策略 group 准确率低（p̂ ≤ τgate），则将 rollout group 的第 n 个 slot 替换为近未来 checkpoint π(t+Δ) 生成的**已验证正确轨迹**：

$$\mathcal{G}_{\text{NPO}}(x) = \{o_1, \ldots, o_{n-1}, \tilde{o}_n\}$$

$$\tilde{o}_n = \begin{cases} o_x' & \text{if } \hat{p}(x) \leq \tau_{\text{gate}} \text{ and } o_x' \text{ exists} \\ o_n & \text{otherwise} \end{cases}$$

**重要性采样修正**（可选）：由于 π(t+Δ) 是近未来 checkpoint，πθ/π(t+Δ) ≈ 1，IS 修正几乎不需要。消融实验证明完全省略 IS 几乎不掉点。

**完整 NPO 目标**（与 GRPO 完全一致的结构，仅改动了 rollout group 的组成）：

$$\mathcal{L}_{\text{NPO}}(\theta) = \mathbb{E}_{x, \mathcal{G}_{\text{NPO}}(x)} \left[ \frac{1}{n} \sum_{i=1}^{n} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \rho_{i,t}^q(\theta) A_i, \text{clip}(\rho_{i,t}^q(\theta), 1-\epsilon, 1+\epsilon) A_i \right) \right]$$

其中：
- on-policy slots：q_i = π(t)
- guidance slot：q_i = π(t+Δ)
- ρ^q_{i,t}(θ) = πθ(o_{i,t}|x,o_{i,<t}) / q_i(o_{i,t}|x,o_{i,<t})

### 两种手动干预（Section 3.2）

#### 1️⃣ 早期 Bootstrap（Early-Stage Bootstrapping）

- **动机**：冷启动阶段，几乎没有正确轨迹
- **做法**：先跑一个短 scout run，取其最终 checkpoint 作为 π(t+Δ)，然后从同一起点重启训练，用 scout checkpoint 的轨迹作为辅助
- **效果**：加速约 **2.1×** 收敛（Figure 1(a)）

#### 2️⃣ 晚期 Plateau 突破（Late-Stage Plateau Breakthrough）

- **动机**：训练曲线走平，group 准确率停滞
- **做法**：继续训练穿过 plateau，取 plateau 之后的更强 checkpoint，回滚到 plateau 起点重放
- **效果**：突破 on-policy 上限，抬升最终性能（Figure 1(a)）

### AutoNPO：自适应版本（Section 3.3）

将两个手动选择自动化：

**触发条件**：监控 EMA 训练 reward 停滞 + policy entropy 下降（探索坍缩信号），确认阶段在 mistake pool ℬ 上做一次 rollout 探测。

**回滚距离选择**：最大化经验有效信号

$$\Delta^* = \arg\max_{\Delta \in \mathcal{D}} \frac{\hat{Q}(\Delta)}{\hat{V}(\Delta)}$$

其中 Q̂(Δ) = pass-rate(π(t); ℬ_Δ)，V̂(Δ) 来自 per-token KL 散度的指数形式估计（所有值从同一次确认 rollout 中读取，无额外推理开销）。

**执行**：仅在 ℬ_Δ*（错误池的 Δ* 切片）上准备 guidance 轨迹并注入，而非整个 segment，节省缓存开销。

---

## 算法框架图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NPO 机制（Section 3.1）                       │
│                                                                     │
│  当前策略 π(t)  ──rollout──▶  {o₁,...,oₙ}                          │
│       ▲                                        │                     │
│       │                                         │ 若 p̂(x) ≤ τgate   │
│       │                                         ▼                     │
│   梯度更新                            替换第 n slot → o'ₓ（已验证）   │
│   π(t+Δ) ──offline rollout──▶ o'ₓ（近未来 checkpoint 产生）          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  AutoNPO 控制流程（Section 3.3）                                    │
│                                                                     │
│  训练日志 → EMA reward停滞 + entropy下降 ──▶ Warning Stage          │
│       │                                              │              │
│       ▼                                              ▼              │
│  Rollback距离搜索                              Confirmation Stage   │
│  Δ* = argmax Q̂(Δ)/V̂(Δ)                       探测pass-rate       │
│       │                                              │              │
│       ▼                                              ▼              │
│  在 ℬ_Δ* 上准备 guidance 轨迹，回滚到 t-Δ* 开始 NPO replay          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 实验结果

### 主实验（Table 1）

**Base**：Qwen3-VL-8B-Instruct，训练数据 MMFineReason-123K，评测 8 个多模态推理基准。

| Method | Avg. | MMMU-Pro | MathVista | MathVision | ZeroBench | WeMath | MMBench | MM-Star | MathVerse |
|--------|------|----------|-----------|------------|-----------|--------|---------|---------|-----------|
| **Base** | 57.88 | 51.75 | 73.80 | 47.37 | 19.76 | 54.10 | 89.79 | 71.83 | 54.61 |
| LUFFY | 58.68 | 54.23 | 73.80 | 54.00 | 20.51 | 52.38↓ | 89.49 | 69.47 | 55.58 |
| GRPO | 60.25 | 55.78 | 76.20 | 48.82 | 22.60 | 56.57 | 90.29 | 72.20 | 59.52 |
| ExGRPO | 61.16 | 55.49 | 77.30 | 55.46 | 19.01 | 62.67 | 90.44 | 72.00 | 56.89 |
| RLEP | 61.48 | 55.38 | 78.50 | 54.23 | 19.61 | 62.48 | 90.45 | 72.27 | 58.91 |
| **NPO (early)** | 62.12 | 56.85 | 76.60 | 54.31 | **26.35** | 62.76 | 90.41 | 70.30 | 59.38 |
| **NPO (early+late)** | 62.84 | 57.07 | 76.30 | 54.61 | 24.85 | **66.95** | 90.30 | 72.20 | 60.00 |
| **AutoNPO** | **63.15** | **57.24** | **79.20** | **55.72** | 24.70 | 66.00 | **90.63** | **72.63** | 59.11 |

**关键结论**：
- NPO 早期干预 → ZeroBench 最强（稀缺正确轨迹场景）
- NPO 晚期干预 → WeMath 和 MathVerse 最大增益（推理深度 + plateau 突破）
- AutoNPO 全面最优，5/8 任务领先

### 训练动态（Figure 4）

- **图 4(a)**：AutoNPO 全程高于 GRPO，每次 intervention 后 gap 扩大
- **图 4(b)**：GRPO 持续 entropy 坍缩；AutoNPO 通过 intervention 重新扩展 entropy，避免过早收敛
- **图 4(c)**：IS 修正消融——有无 IS 修正几乎无差别，确认近策略假设

### 消融实验（IS Correction）

LUFFY 去掉 IS → 训练崩溃（外部 teacher 分布太远）；NPO 去掉 IS → **几乎不掉点**。这是 NPO 独有的优势：近未来 checkpoint 本来就跟当前策略足够近。

---

## 核心公式推导

### 1. 有效学习信号定义

$$\mathcal{S}(\Delta) = \frac{Q(\Delta)}{V(\Delta)}$$

- Q(Δ)：近未来 checkpoint 在当前策略失败 prompt 上的 pass rate
- V(Δ)：IS 权重引入的梯度方差上界

### 2. 方差上界（Appendix B）

参数距离随 Δ 线性增长，但 IS 方差随参数距离**指数**增长：

$$V(\Delta) \lesssim \exp(\beta \cdot \Delta), \quad \beta > 0$$

因此 𝒮(Δ) 必有内点最大值。

### 3. NPO 目标函数

$$\mathcal{L}_{\text{NPO}}(\theta) = \mathbb{E}_{x, \mathcal{G}_{\text{NPO}}(x)} \left[ \frac{1}{n} \sum_{i=1}^{n} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_\theta(o_{i,t}|x,o_{i,<t})}{q_i(o_{i,t}|x,o_{i,<t})} A_i, \text{clip}(\cdots) A_i \right) \right]$$

- on-policy slots：q_i = π(t)
- guidance slot：q_i = π(t+Δ)

---

## KnowHow + 总结评价

### 亮点

1. **问题形式化漂亮**：将"够强且够近"的直观洞察形式化为 𝒮 = Q/V，并通过理论 + 实验双验证，是少有的从数学出发的 RL 工作
2. **工程设计简洁**：只改动了 rollout group 的一个 slot，不改目标函数，不改 verifier，复用性极强
3. **AutoNPO 的工程价值**：将人工干预自动化，且干预逻辑（reward stagnation + entropy decline → 探测 ℬ → 选 Δ*）完全来自 online 信号，工程实用性强
4. **IS 修正可省略的洞察**：近未来 checkpoint → 低分布差距 → IS ≈ 1，这是 NPO 独有的 safe simplification

### 局限

1. **需要保存 checkpoint 快照**：Δ 步的 future checkpoint 需要先跑出来再回滚，增加了 memory 开销
2. **AutoNPO 的 trigger 依赖 entropy 下降**：在某些任务上 entropy 不一定持续下降，可能需要 task-specific 适配
3. **验证器依赖**：需要可验证的 reward（RLVR 范式），纯偏好任务不直接适用
4. **Δ* 的搜索空间受限于已保存的 checkpoint**：无法做分数级精细搜索

### 个人点评

NPO 是 **Self-Taught RLVR** 框架的第二篇（继 Self-Distilled RLVR 之后），核心贡献在于发现"近未来"这个天然介于"够强"和"够近"的甜点区域。形式化 𝒮 = Q/V 并通过实验验证 U-shape 曲线是本文最硬的部分——这比大多数 RL 论文中拍脑袋设计目标函数要严谨得多。

AutoNPO 让我印象深刻的地方在于：它把"什么时候干预 + 干预多远"两个原本需要人工看曲线的问题，变成了完全自动的 online 信号驱动流程，且效果不亚于手动调参。这对实际训练场景非常有价值。

**对齐小何同学研究方向的价值**：NPO 本质上是一种 off-policy correction 思想在 RLVR 后训练场景的具体应用，其 Δ-空间分析的思维方式对 VLA 的后训练同样有参考价值——尤其是当 VLA 策略需要在长周期训练中避免 exploration collapse 时。

---

## 参考链接

- **arXiv**：https://arxiv.org/abs/2604.20733
- **PDF**：https://arxiv.org/pdf/2604.20733.pdf
- **HTML**：https://arxiv.org/html/2604.20733v1
- **代码**：待补充

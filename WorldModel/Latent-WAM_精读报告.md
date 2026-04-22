# Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving

## 引用信息

| 字段 | 内容 |
|------|------|
| **标题** | Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving |
| **arXiv** | [2603.24581](https://arxiv.org/abs/2603.24581) [cs.CV] |
| **作者** | Linbo Wang, Yupeng Zheng, Qiang Chen, Shiwei Li, Yichen Zhang, Zebin Xing, Qichao Zhang, Xiang Li, Deheng Qian, Pengxuan Yang, Yihang Dong, Ce Hao, Xiaoqing Ye, Junyu Han, Yifeng Pan, Dongbin Zhao |
| **机构** | 中科院自动化所 · 长安汽车 · 中科院大学人工智能学院 · 清华大学 · 中关村研究院 |
| **顶会/顶刊** | arXiv 2026 (CS.CV) |
| **发布日期** | 2026-03-25 |

---

## 一句话总结

Latent-WAM 通过**空间感知压缩世界编码器（SCWE）** 和**动态潜在世界模型（DLWM）** 两大模块，以 104M 参数在感知自由（perception-free）设置下于 NAVSIM v2 达到 **89.3 EPDMS**，刷新 SOTA，训练数据量与参数量均显著低于竞品。

---

## 拟人化开篇

想象你坐在一辆自动驾驶汽车的驾驶位。窗外是复杂的城市道路——行人、骑手、前车急刹，红绿灯变化。当前所有端到端自动驾驶系统面临一个共同困境：**如何让模型真正"理解"这个 3D 世界，而不只是"记住"像素？**

传统方案要么依赖海量感知标注，要么在隐空间里建模得不够精准，导致规划轨迹总差那么一点。长安汽车与中科院的团队提出了一个新思路：**与其重建整个像素世界，不如让模型学会"压缩感知"——只保留对开车真正有用的信息。**

这就是 Latent-WAM 的核心出发点。

---

## 背景与问题动机

### 端到端自动驾驶的演进

端到端自动驾驶直接优化轨迹规划，其**中间场景表征**是关键。早期的 `UniAD`、`VAD` 采用 BEV 稠密布局，需要语义地图和占用标签的监督；随后 `SparseDrive` 等转向轻量级向量化表征；近期 `Drive-VLA`、`World4Drive` 等引入视觉-语言模型（VLM）或世界模型做自监督表征学习。

### 世界模型的两条路线

| 路线 | 代表工作 | 问题 |
|------|---------|------|
| **显式视频生成** | `Epona`、`DriveVLA-W0` | 像素级重建计算开销大，表征偏向视觉细节而非规划 |
| **隐式未来预测** | `LAW`、`World4Drive` | 表征压缩不足、缺乏空间理解、时序动态利用不充分 |

### 三大核心问题

1. **表征压缩不足**：隐空间表征仍过于冗余
2. **空间理解缺失**：缺乏 3D 几何感知，依赖外部深度估计模型，引入额外延迟
3. **时序动态利用不充分**：仅预测 T+1 帧，忽视长期时序依赖

这些问题导致在有限数据和计算预算下，规划性能次优。

![图 1：NAVSIM v2 性能 vs 训练数据量](https://arxiv.org/html/2603.24581v1/x1.png)

> **图 1**：NAVSIM v2 上性能 vs 训练数据量。气泡大小代表模型参数量。Latent-WAM 以显著更少训练数据和更小模型规模达到最高 EPDMS，彰显极高数据效率。World4Drive 单独标注因其使用了额外的 ViT-L 深度估计器。

---

## 方法详解

### 3.1 整体架构

Latent-WAM 由三个核心模块组成：SCWE（场景压缩 + 几何蒸馏）、DLWM（因果世界建模）、Trajectory Decoder（多候选轨迹解码）。

![图 2：Latent-WAM 整体架构图](https://arxiv.org/html/2603.24581v1/x2.png)

> **图 2**：Latent-WAM 整体架构图。输入为 T×M×H×W×3 的多视角图像序列，经 SCWE 压缩为紧凑 scene tokens，再经 DLWM 因果预测未来世界状态，最终由 Trajectory Decoder 输出 4 秒规划轨迹。

### 3.2 SCWE：空间感知压缩世界编码器

#### 3.2.1 场景压缩

**输入**：时序多视角图像 $I \in \mathbb{R}^{\mathbf{T} \times \mathbf{M} \times H \times W \times 3}$

1. 图像嵌入为 patch tokens：$X \in \mathbb{R}^{\mathbf{T} \times \mathbf{M} \times \mathbf{S} \times D_e}$
   - $\mathbf{T}$：时序帧数
   - $\mathbf{M}$：相机数量
   - $\mathbf{S}$：每张图的 patch 数
   - $D_e = 768$

2. 随机初始化场景查询 $Q_{\text{scene}} \in \mathbb{R}^{\mathbf{T} \times \mathbf{M} \times \mathbf{N} \times D_e}$，与 $X$ 拼接后送入 DINO encoder $\mathcal{E}$（含 MLP 投影到 $D_l = 256$ 维潜在空间）：

$$
\hat{Q}_{\text{scene}}, \hat{X} = \mathcal{E}([Q_{\text{scene}}; X])
$$

其中 $\mathbf{N} = 16$ 是 scene query token 数量。相比原始 patch tokens（数百甚至上千），仅用 16 个 token 大幅压缩。

**关键洞察**：将丰富的视觉信息压缩到 compact token 集合中，大幅降低后续世界模型训练和轨迹规划的计算开销。

#### 3.2.2 几何对齐（Geometric Alignment）

**动机**：增强 DINO encoder 的空间理解能力

**方法**：使用几何基础模型 **WorldMirror**（基于 VGGT）作为教师，将几何感知能力蒸馏到视觉 backbone。

1. 多视角图像输入几何基础模型 $f_g$，生成 patch 级几何特征 $f_g(I) \in \mathbb{R}^{\mathbf{T} \times \mathbf{M} \times \mathbf{S} \times D_g}$，其中 $D_g = 2048$

2. DINO backbone 输出的 $\hat{X}$ 通过几何投影器 $\phi$ 投影到同一空间：$\phi(\hat{X}) \in \mathbb{R}^{\mathbf{T} \times \mathbf{M} \times \mathbf{S} \times D_g}$

3. **对齐损失**（cosine 相似度）：

$$
\mathcal{L}_{\text{align}} = 1 - \cos\left(\text{LN}(\phi(\hat{X})), \text{LN}(f_g(I))\right)
$$

**工程技巧**：$f_g$ 在训练中**保持冻结**，几何特征可**离线预计算并缓存**，避免重复推理开销和 GPU 显存占用。

### 3.3 DLWM：动态潜在世界模型

#### 3.3.1 世界潜在状态聚合

**问题**：每相机的 scene tokens 仅捕获孤立的视图特定信息，不足以做全局世界建模。

**解决方案**：

1. **Ego 状态编码器**（单层 MLP）：将驾驶命令、速度、加速度编码为 $S_{\text{ego}} \in \mathbb{R}^{\mathbf{T} \times D_l}$

2. **Scene-Ego 融合**：多相机 scene tokens 聚合为全局感知表示，拼接 $S_{\text{ego}}$，得到统一的世界状态：

$$
S_{\text{world}} \in \mathbb{R}^{\mathbf{T} \times (\mathbf{M} \times \mathbf{N} + 1) \times D_l}
$$

#### 3.3.2 因果世界模型预测

**核心思路**：将世界状态转移建模为**自回归预测问题**，用标准 next-token prediction 训练。

1. 随机初始化未来世界状态查询 $Q_{\text{future}} \in \mathbb{R}^{(\mathbf{T}-1) \times (\mathbf{M} \times \mathbf{N} + 1) \times D_l}$

2. 历史世界状态 $S_{\text{world}}^{i}, i \in \{1, \ldots, T-1\}$ 构成 KV cache

3. DLWM（带 RoPE 的标准 Transformer 解码器）预测未来状态：

$$
S_{\text{future}} = \text{DLWM}(Q_{\text{future}}, \text{KV}_{\text{future}})
$$

**GT 生成**：使用 EMA 更新的目标编码器（SCWE 的冻结副本）提供稳定监督信号。

#### 3.3.3 Teacher Forcing Attention Mask

**设计**：在 Scene-Ego 交错序列中，每个未来 token 可 attend 到所有历史 token，但保持时序因果性。

- **帧内**：token 双向 attend（信息充分交互）
- **帧间**：每个 token 只能 attend 到序列中更早的 token

这一帧级 attention 设计使得所有未来世界状态可以**并行预测**，同时保持因果一致性，显著提升训练速度和效率。

![图 3：Teacher Forcing Attention Mask](https://arxiv.org/html/2603.24581v1/x3.png)

> **图 3**：Teacher Forcing Attention Mask。帧内双向 attend，帧间严格因果。所有未来世界状态可并行预测，同时保持时序一致性。

#### 3.3.4 3D-RoPE

**问题**：$Q_{\text{future}}$ 和 $\text{KV}_{\text{future}}$ 未携带明确的时序/空间位置信息。

**解决方案**：将头维度 $D_h$ 拆分为三部分，分别编码：
- **时间坐标 $t$**（频率 50）
- **相机索引 $m$**（频率 10）
- **Token 索引 $n$**（频率 100）

这使得模型能够区分长序列中的时序关系和空间关系。

#### 3.3.5 Ego 状态监督

**动机**：精确的自我状态预测对建模世界动态至关重要。

从预测的未来世界状态中提取 ego embedding，通过三个独立 MLP 预测：
- 驾驶命令：$\hat{C} = \text{Softmax}(D_{\text{cmd}}(S_{\text{ego}}^{i'}))$
- 速度：$\hat{V} = D_v(S_{\text{ego}}^{i'})$
- 加速度：$\hat{A} = D_a(S_{\text{ego}}^{i'})$

损失：$\mathcal{L}_{\text{ego}} = \mathcal{L}_{\text{cmd}} + \mathcal{L}_v + \mathcal{L}_a$

### 3.4 轨迹规划

**输入**：可学习的轨迹查询 $Q_\tau \in \mathbb{R}^{K \times n_p \times D_l}$ + 当前世界状态 $S_{\text{world}}^t$ + 当前驾驶命令 $C$

**输出**：$K$ 条候选轨迹，每条 $n_p$ 个 pose $(x, y, \theta)$，根据 $C$ 选择对应候选作为最终轨迹 $\tau$：

$$
\tau = D_\tau(Q_\tau, S_{\text{world}}^t, C)
$$

轨迹以 ego 车辆局部坐标系表示，预测 4 秒未来（8 个 pose，0.5 秒间隔）。

### 3.5 训练目标

完整损失函数：

$$
\mathcal{L} = \mathcal{L}_{\text{traj}} + \alpha \mathcal{L}_{\text{align}} + \beta \mathcal{L}_{\text{wm}} + \gamma \mathcal{L}_{\text{ego}}
$$

其中 $\alpha = 0.1, \beta = 0.2, \gamma = 0.1$。

- $\mathcal{L}_{\text{traj}}$：L1 损失（模仿专家轨迹）
- $\mathcal{L}_{\text{align}}$：cosine 相似度损失（几何蒸馏）
- $\mathcal{L}_{\text{wm}}$：MSE 损失（未来世界状态预测）
- $\mathcal{L}_{\text{ego}}$：交叉熵 + MSE（Ego 状态监督）

**推理时仅需 SCWE + Trajectory Decoder**，无额外推理延迟模块。

---

## 实验结果

### 4.1 主结果

#### NAVSIM v2（EPDMS，感知自由设置下 SOTA）

| 方法 | EPDMS ↑ | 备注 |
|------|---------|------|
| **Latent-WAM（Ours）** | **89.3** | 感知自由，104M |
| Drive-JEPA | 87.8 | 需感知标注 |
| WorldRFT | 86.7 | 感知基础 |
| DiffusionDrive | 84.5 | 感知基础 |
| DriveVLA-W0 | 86.1 | 感知自由 |
| Epona | 85.1 | 感知自由 |
| World4Drive | 84.8 | 感知自由 |

**EPDMS 指标分解**：

| 指标 | Latent-WAM | DriveVLA-W0 | Epona |
|------|------------|-------------|-------|
| NC | 98.1 | 98.5 | 97.1 |
| DAC | 97.3 | 99.1 | 95.7 |
| DDC | 99.6 | 98.0 | 99.3 |
| TLC | 99.8 | 99.7 | 99.7 |
| EP | 87.7 | 86.4 | 88.6 |
| **EC（Extended Comfort）** | **87.3** | 58.9 | 67.8 |

**关键洞察**：Latent-WAM 在 EC（Extended Comfort）上远超所有感知自由方法（87.3 vs 58.9/67.8），说明其世界表征更关注规划相关动态，舒适性显著提升。

#### HUGSIM（零样本跨数据集泛化）

| 方法 | RC ↑ | HD-Score ↑ |
|------|------|------------|
| **Latent-WAM（Ours）** | **45.9** | **28.9** |
| DriveVLA-W0 | 44.7 | 28.9 |
| Epona | 40.5 | 25.5 |

在仅用 NAVSIM v2 训练的情况下零样本到 HUGSIM，RC 和 HD-Score 均达第一。

### 4.2 消融实验

#### 各模块贡献（逐步添加）

| 配置 | EPDMS | Δ |
|------|-------|---|
| Baseline（直接送入轨迹解码器） | 87.9 | — |
| + 场景压缩（16 queries） | 87.7 | -0.2 |
| + 几何信息注入 | 88.6 | +0.9 |
| + 动态世界建模 | 88.0 | — |
| + Ego 状态 | 88.3 | — |
| + 几何信息（完整） | 89.0 | — |
| **Full Model** | **89.3** | +1.4（相对基线 +1.4）|

**消融洞察**：
- 几何注入带来最大收益（+0.9 EPDMS），证明 3D 空间感知对轨迹规划至关重要
- 压缩后性能几乎不降（-0.2），说明 16 个 token 足以编码规划相关信息

#### 几何注入方式对比

| 方法 | EPDMS |
|------|-------|
| 无几何注入 | 88.3 |
| 直接拼接冻结几何特征 | 88.0（下降！引入冲突信号）|
| **蒸馏到 backbone（ours）** | **89.3** |

**洞察**：直接拼接引入冲突信号；端到端蒸馏使 backbone 自适应学习与规划对齐的空间表征。

#### Vision Backbone 规模与微调策略

| 配置 | EPDMS |
|------|-------|
| DINO-Small | 86.3 |
| DINO-Base-Full FT | **89.3** |
| DINO-Small-LoRA | 84.7 |
| DINO-Base-LoRA | 68.5（灾难性下降）|

**洞察**：高维几何特征蒸馏需要**全参数微调**，LoRA 的低秩约束不足以建模 2048 维几何空间与规划目标之间的对齐。

#### 世界模型预测时序跨度

| 配置 | EPDMS |
|------|-------|
| 仅预测最终帧（stride: 0→8） | 88.4 |
| **stride 4（-3→0→4→8）** | **89.3** |
| stride 2（-3→-2→...→8） | 89.1 |

**洞察**：stride 4 平衡监督有效性；stride 2 无额外收益，因驾驶场景相邻帧高度相似且密集预测增加优化负担。

### 4.3 定性分析

#### 轨迹可视化

![图 4：轨迹规划对比](https://arxiv.org/html/2603.24581v1/x4.png)

> **图 4**：轨迹规划可视化。绿线为人驾轨迹，黄线为各方法预测轨迹。Latent-WAM 与人驾轨迹对齐最好，与其他车辆保持更安全距离；Epona 轨迹相对次优，World4Drive 可接受但非最优。

#### Attention Map 可视化

![图 5：Scene Tokens 与 Image Patches 的 Cross-Attention 可视化](https://arxiv.org/html/2603.24581v1/x5.png)

> **图 5**：Scene Tokens 与 Image Patches 的 Cross-Attention 可视化。从上到下分别为直行、右转、左转三种驾驶意图。无几何蒸馏的 Baseline attention 分散在天空、建筑等无关背景区域；Latent-WAM 的 attention 高度聚焦于车道线、场景几何结构和可行驶区域，且 attention 分布与驾驶意图强相关。

**关键发现**：attention 分布与驾驶意图强相关——直行时关注前方，右转时聚焦右侧，这与人类的注意力分配模式高度一致。

---

## 核心洞察与总结评价

### 技术亮点

1. **SCWE 的 Query-based 压缩**：仅用 16 个 scene tokens 从数百个 patch tokens 中提取信息，压缩率极高且信息保留充分，为后续长时域世界建模奠定基础。

2. **几何蒸馏而非特征拼接**：不引入额外推理模块，而是将几何基础模型（WorldMirror/VGGT）的空间感知能力通过蒸馏内化到 Vision Backbone，避免了直接拼接带来的特征不对齐问题。

3. **Ego 状态自监督+有监督联合**：自监督未来视觉预测 + 有监督运动状态预测（速度/加速度/命令），双重监督信号确保世界状态既感知未来又理解自我运动。

4. **3D-RoPE 显式建模时空位置**：区分时间、相机、token 三维位置，长序列下不混淆。

5. **Teacher Forcing Mask**：帧内双向 attend + 帧间因果 attention，并行训练 + 因果推理兼得。

### 局限性

1. **感知自由 vs 感知基础的差距**：NC（No-collision）指标仍略低于有感知方法（98.1 vs 98.4），安全关键场景仍有提升空间。
2. **几何基础模型依赖**：需要预训练几何模型 WorldMirror（基于 VGGT），当没有可用几何基础模型时方案受限。
3. **仅用 3 个相机**：前视/左视/右视，缺少后视和环视相机，对后方信息利用不足。
4. **无强化学习后训练**：纯模仿学习，规划能力受专家数据分布限制。

### 个人点评

Latent-WAM 展现了**极致的压缩效率**：仅 104M 参数、显著更少训练数据，即在感知自由设置下达到 SOTA。这说明当前端到端自动驾驶的主要瓶颈不是模型规模，而是**表征质量**——如何让模型学到对规划真正有用的表征，而非堆砌像素细节。

论文的核心贡献在于**几何蒸馏**的设计：通过 teacher forcing 让规划导向的几何感知内化到 backbone，而非在推理时外挂深度估计器。这在工程上也非常优雅——冻结几何特征 + 离线缓存，零额外推理开销。

**论文评分**：⭐⭐⭐⭐☆（4/5）
- 创新性：SCWE + DLWM 的组合设计有明确创新，几何蒸馏思路清晰
- 完整性：实验充分，消融覆盖全面，定性可视化详实
- 可复现性：架构细节充分披露，但代码未开源（arXiv 时）

---

## Appendix 补充

### A. 指标详解

#### NAVSIM v2 EPDMS 公式

$$
\text{EPDMS} = \text{NC} \times \text{DAC} \times \text{DDC} \times \text{TLC} \times \frac{5 \times (\text{EP} + \text{TTC}) + 2 \times (\text{LK} + \text{HC} + \text{EC})}{16}
$$

新增 DDC（驾驶方向合规）、TLC（交通灯合规）、LK（车道保持）、HC（历史舒适性）、EC（扩展舒适性）指标。

#### HUGSIM HD-Score 公式

$$
\text{HD-Score}_t = \text{NC} \times \text{DAC} \times \frac{5 \times \text{TTC} + 2 \times \text{COM}}{7}
$$

最终 HD-Score = $R_c \times \frac{1}{T}\sum_{t=0}^{T} \text{HD-Score}_t$

### B. 数据处理管线

#### 图像预处理（两阶段）
1. **Resize**：1920×1080 → 455×256
2. **Center Crop**：→ 448×224（2:1 宽高比）

#### 相机内参调整

当图像 resize/crop 后，内参矩阵需同步调整：

$$
f_x' = f_x \cdot s_x, \quad f_y' = f_y \cdot s_y, \quad c_x' = c_x \cdot s_x - \Delta_x, \quad c_y' = c_y \cdot s_y - \Delta_y
$$

### C. 几何特征提取

**Camera Prior 提取**：几何基础模型接收图像 + 内参 + 外参，输出空间-几何先验 $\mathcal{P}$，封装相机 pose、深度估计和内参信息。

**Geometric Feature Encoding**：

$$
f_g(I) = \text{WorldMirror}(\mathbf{I}, \mathcal{P}; \mathbf{c})
$$

其中条件标志 $\mathbf{c} = [1, 0, 1]$（使用相机 pose 和内参，不使用深度监督）。

### D. 更多可视化（Appendix D）

论文提供了大量 attention map 可视化（Appendix D.2），涵盖：
- 直行、左右转弯、换道等多种场景
- 连续帧间的 attention 演变
- HUGSIM 上 nuScenes/KITTI-360/Waymo/PandaSet 的轨迹可视化

**共性结论**：几何蒸馏后的 attention 始终聚焦于几何边界（车道线、障碍物边缘、可行驶空间），且 attention 分布随驾驶意图动态调整，验证了 intent-aware 表征学习目标。

---

## 参考信息

- **arXiv**：[https://arxiv.org/abs/2603.24581](https://arxiv.org/abs/2603.24581)
- **PDF**：[https://arxiv.org/pdf/2603.24581](https://arxiv.org/pdf/2603.24581)
- **HTML**：[https://arxiv.org/html/2603.24581v1](https://arxiv.org/html/2603.24581v1)

---

*精读日期：2026-04-22*

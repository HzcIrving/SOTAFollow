# Vega: Learning to Drive with Natural Language Instructions

## 论文精读报告

---

## 1. 引用信息表

| 字段 | 内容 |
|------|------|
| **论文标题** | Vega: Learning to Drive with Natural Language Instructions |
| **arXiv ID** | [2603.25741](https://arxiv.org/abs/2603.25741) |
| **代码链接** | [https://github.com/zuosc19/Vega](https://github.com/zuosc19/Vega) |
| **作者** | Sicheng Zuo¹∗, Yuxuan Li¹∗, Wenzhao Zheng¹∗†, Zheng Zhu², Jie Zhou¹, Jiwen Lu¹  <br/>¹ Tsinghua University  ² GigaAI |
| **发表来源** | arXiv (2026) |
| **关键词** | VLA, 指令跟随驾驶, 世界模型, 扩散模型, 自动驾驶 |

---

## 2. 一句话总结

**Vega** 提出了一个**统一的多模态生成-规划框架**，通过将**自然语言指令**引入自动驾驶决策过程，并利用**未来帧图像生成**作为密集监督信号，弥补传统 VLA 模型稀疏动作监督的缺陷，实现了 SOTA 的指令跟随驾驶能力。

---

## 3. 拟人化开篇

![图 1：Vega 指令跟随驾驶可视化](https://arxiv.org/html/2603.25741v1/x1.png)

> **图 1**：Vega 论文 Teaser — 同一场景多条轨迹跟随不同指令。给定不同指令（"Pull up to the side"、"Follow the car straight through the intersection"、"Goal along the road and pass the orange traffic barrier"、"Stop at the crosswalk, wait for the light to turn green"），Vega 能预测出完全不同的多条轨迹。


想象你坐在副驾驶，对司机说："**前面那辆车开得太慢了，超它然后赶下一个绿灯**"。一个普通司机听完会心领神会，果断变道加速——而老司机甚至能预判超车后前方路况会变成什么样。

但大多数现有自动驾驶模型呢？它们只听得懂"左转"、"直行"这样封闭的导航指令，面对"超车赶绿灯"这种**灵活的自然语言指令**就束手无策了。它们要么模仿一个保守的专家策略，要么压根不理会你的个性化需求。

**Vega** 就是来解决这个问题的——它不仅能听懂你的自然语言指令，还能据此规划出**多条不同的轨迹**，并"想象"出遵循这条指令驾驶后，未来会看到什么样的画面。

---

## 4. 背景与问题动机

### 4.1 自动驾驶技术演进路线

自动驾驶技术栈经历了三次范式转变：

| 范式 | 代表工作 | 核心问题 |
|------|----------|----------|
| **模块化Pipeline** | BEVFormer, UniAD | 依赖昂贵3D标注，难以扩展 |
| **VLA (Vision-Language-Action)** | GPT-Driver, AutoVLA | 语言仅用于场景描述或推理，缺乏灵活指令跟随 |
| **世界模型增强VLA** | DriveVLA-W0 | 引入未来预测增强规划，但无法做指令跟随 |

### 4.2 核心问题：稀疏动作监督

现有 VLA 模型的致命缺陷：**高维视觉-语言输入 → 低维动作输出** 的映射，缺乏足够的监督信号。

```
传统VLA: 图像序列 + 历史动作 → 预测动作 (稀疏监督，仅1个动作标签)
         ↑
     高维视觉输入　　　　　低维动作输出
     信息密度差　　　　　　监督信号弱
```

论文做了一个 VLA Baseline 实验验证这一问题——直接用 Qwen2.5-VL 加一个 planning head，在相同数据集上训练，**仅达到 ~60 PDMS**（远低于 SOTA 的 90+），且频繁生成不符合指令的轨迹。

### 4.3 从模仿驾驶到指令驾驶

Vega 将这一转变定义为：

> **Imitation Driving → Instructional Driving**

- **Imitation Driving**：模型只能模仿训练数据中专家的平均策略
- **Instructional Driving**：模型能根据用户自然语言指令，生成**多样化的**、符合指令的轨迹

---

## 5. 方法详解

### 5.1 整体架构

Vega 是一个**统一的自回归-扩散混合架构**，命名为 **Vision-Language-World-Action Model**。

核心设计哲学：
- **AR (Autoregressive)** 范式 → 处理视觉 + 语言理解
- **Diffusion** 范式 → 生成未来图像（世界建模）+ 轨迹（动作规划）
- **Joint Attention** → 跨模态深度交互


![图 2：Vega 模型总览](https://arxiv.org/html/2603.25741v1/x2.png)

> **图 2**：Vega 的整体框架图，左侧展示传统模仿驾驶模型（单一专家轨迹 + 导航命令），右侧展示 Vega（多模态指令 + 个性化轨迹 + 世界建模）。Vega 同时输出动作规划和未来图像预测，实现真正的指令跟随。

### 5.2 输入编码

**文本**：Qwen2.5 tokenizer

**视觉输入**（仅用前视相机）：
1. **VAE encoder** → 压缩为 latent $F_t^V$（用于 diffusion 生成）
2. **SigLIP2 ViT encoder** → 提取语义特征，拼接至 VAE latents（增强视觉上下文）

**动作编码**：
- 轨迹从绝对坐标 $(x, y, \theta)$ 转为**相邻步之间的相对运动** $(\Delta x, \Delta y, \Delta \theta)$
- 目的：不同步的动作共享分布，易于归一化
- 经 Linear head 投影至模型 latent 维度

### 5.3 构建输入序列（Interleaving Design）

序列结构（**严格因果**）：

```
S = [I_{t-T}, ..., I_t,  L_t,  A_t^{noisy}]
    ↑                    ↑      ↑
 历史图像序列           指令    噪声动作（待去噪）
```

训练时两个任务**联合优化**：
- **动作规划任务**：预测 $A_t^{(N)} = [A_t, ..., A_{t+N-1}]$
- **世界建模任务**：预测 $I_{t+K}$（未来图像）

关键设计：**级联条件化**（Cascaded Conditioning）：
- 推理阶段：先 denoise 动作，再以完全 denoise 的动作 condition 去噪未来图像
- 训练阶段：解决"后续 token 会 attend to 噪声 token"的 mismatch 问题

**解决方案**：对同时作为预测目标和后续条件化的 latent，做**两份拷贝**：
- **Copy 1** ($F_t^{noisy}$)：添加噪声，用于去噪监督
- **Copy 2** ($F_t^{clean}$)：保持干净，作为条件输入
- 从所有后续 token 遮蔽 Copy 1，确保它们只 attend to clean latents

### 5.4 Mixture-of-Transformers (MoT) 架构

![图 3：Vega 统一 VLA 框架](https://arxiv.org/html/2603.25741v1/x3.png)

> **图 3**：Vega 的核心框架图，展示了多模态输入（Vision → VAE、Text → Qwen2.5 Tokenizer、Action → Linear）如何通过 Causal Attention 机制与 Understanding Transformer、Generation Transformer、Action Expert 三类模块交互，最终输出 Planning（动作规划）和 Generation（未来图像生成）。



与 MoE（仅 FFN 分离）不同，MoT 对**所有可学习参数**（attention + FFN）都为每个模组复制一套：

| 模块 | Hidden Size | Depth | 初始化来源 |
|------|------------|-------|------------|
| **Understanding Transformer** | 3584 | 28 | Qwen2.5 LLM / Bagel-7B |
| **Generation Transformer** | 3584 | 28 | Bagel-7B |
| **Action Expert** | 256 | - | 从上述模块改编 |

**为什么 Action Expert 用更小的 hidden size (256)？**
→ 动作空间维度低，降低计算开销，同时不显著损伤性能

### 5.5 训练目标

两个 diffusion 任务的联合优化：

**动作损失**（Eq. 7）：
$$\mathcal{L}_A = \mathbb{E}_{A_t^{(N)}, \epsilon, m} \left[ \|\epsilon - \hat{\epsilon}(A_t^{(N)}, \epsilon, m, I_t^{(-T)}, L_t)\|^2 \right]$$

其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 为高斯噪声，$m$ 为随机时间步。

**图像损失**（Eq. 8）：
$$\mathcal{L}_V = \mathbb{E}_{F_{t+N}^V, \epsilon, n} \left[ \|\epsilon - \hat{\epsilon}(F_{t+N}^V, \epsilon, n, I_t^{(-T)}, L_t, A_t^{(N)})\|^2 \right]$$

**总损失**：
$$\mathcal{L} = \lambda_A \cdot \mathcal{L}_A + \lambda_V \cdot \mathcal{L}_V, \quad \lambda_A = \lambda_V = 1.0$$

### 5.6 数据集：InstructScene

基于 **NAVSIM** 构建，约 **100,000 scenes**，两阶段自动标注：

**Stage 1 - Scene Understanding**：
- 输入：前视相机 14 帧（2Hz，1920×1080），前4帧=过去/当前，后10帧=未来
- Prompt Qwen2.5-VL-72B-Instruct 描述场景和驾驶行为

**Stage 2 - Instruction Formulation**：
- 将 Stage 1 输出组合，Prompt VLM 生成简洁驾驶指令
- 补充规则-based 指令作为辅助

最终：**85,109 scenes (train) + 12,144 scenes (test)**

---

## 6. 实验结果

### 6.1 NAVSIM v2 benchmark（表 1）

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | **EPDMS ↑** |
|--------|------|-------|-------|-------|------|-------|------|------|------|-------------|
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | - | 83.1 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **Vega** | **98.9** | 95.3 | 99.4 | 99.9 | 87.0 | 98.4 | 96.5 | 98.3 | 76.3 | 86.9 |
| **Vega†** | **99.2** | 96.6 | 99.5 | 99.9 | 87.5 | 98.7 | 97.4 | 98.4 | 84.5 | **89.4** |

> **†**: Best-of-N (N=6) strategy
>
> **NC**: No at-fault Collision | **DAC**: Drivable Area Compliance | **DDC**: Driving Direction Compliance
> **TLC**: Traffic Light Compliance | **EP**: Ego Progress | **TTC**: Time to Collision
> **LK**: Lane Keeping | **HC**: History Comfort | **EC**: Extended Comfort | **EPDMS**: Extended PDMS

**关键发现**：
- Vega 在 **NC（无责碰撞）** 上达到 **99.2%**（全场最高）
- 使用 Best-of-N 后，EPDMS **89.4**（NAVSIM v2 **SOTA**）
- 仅用**单目前视相机**，不需要多相机或 LiDAR

### 6.2 NAVSIM v1 benchmark（表 2）

| Method | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | **PDMS ↑** |
|--------|---------|------|-------|-------|------|------|-------------|
| Hydra-MDP | 3x Cam + L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| DiffusionDrive | 3x Cam + L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| AutoVLA† | 3x Cam | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 |
| DriveVLA-W0† | 1x Cam | 99.3 | 97.4 | 97.0 | 99.9 | 88.3 | 93.0 |
| **Vega** | **1x Cam** | 98.9 | 95.3 | 96.1 | 100.0 | 81.6 | 87.9 |
| **Vega†** | **1x Cam** | 99.2 | 96.6 | 96.9 | 100.0 | 83.4 | **89.8** |

**关键发现**：
- 仅用**单目前视相机**（1x Cam），达到接近多相机 BEV 方法的水平
- Best-of-N 后 PDMS **89.8**

---

## 7. 可视化与消融实验

### 7.1 指令跟随可视化

![图 5：指令跟随规划示例](https://arxiv.org/html/2603.25741v1/x5.png)

> **图 5**：展示了两个场景（Scene 1 和 Scene 2）中，不同指令对车辆规划轨迹的影响。每个场景包含前视相机图像（上排）和对应的 BEV 地图（下排）。三条指令分别产生三条不同轨迹，验证了 Vega 对自然语言指令的精确理解和执行能力。


### 7.2 未来图像生成可视化

![图 6：未来图像生成](https://arxiv.org/html/2603.25741v1/x6.png)

> **图 6**：同一场景下，给定三组不同指令，Vega 规划出三条不同的动作序列，并生成对应的高保真未来图像。所有动作序列都严格遵循各自指令，所有生成的未来图像都与对应动作保持时序一致性。


### 7.3 未来帧预测的消融（表 3）

| 配置 | PDMS ↑ | EPDMS ↑ |
|------|--------|---------|
| **Next Frame**（预测最近未来帧） | **77.9** | **76.0** |
| Random Frame（随机采样帧） | 77.3 | 75.2 |
| Action Only（移除未来帧预测） | 51.8 | 48.9 |

**结论**：未来帧预测任务确实能显著提升规划能力（+26 PDMS），但具体选哪帧影响不大——**任务的存在本身**比具体选择更重要。

### 7.4 交错图像-动作序列的消融

![图 4：交错序列消融实验](https://arxiv.org/html/2603.25741v1/x4.png)

> **图 4**：交错图像-动作序列设计的消融实验。(a) 展示不同交错长度（2/4/6）对收敛速度和最终性能的影响；(b) 验证交错设计对模型学习动态关系的重要性。
- **收敛阶段**：交错序列模型**收敛更快且 loss 更低**
- 更长交错的模型最终 loss 更低

**结论**：交错图像和动作帮助模型学习它们之间的动态关系，加速收敛。

### 7.5 独立 Action Expert 的消融（表 4）

| Action Planner 来源 | PDMS ↑ | EPDMS ↑ |
|---------------------|--------|---------|
| Use Diffusion Module | 19.7 | 19.6 |
| Use VLM Module | 77.6 | 75.7 |
| **Action Expert（独立小模块）** | **77.9** | **76.0** |

**结论**：独立 Action Expert 显著优于使用 VLM 或 Diffusion 模块处理动作规划，确认了低维动作空间适合独立小模块的架构决策。

---

## 8. KnowHow + 总结评价

### 8.1 技术亮点

1. **密集监督信号**：用未来图像生成弥补稀疏动作监督的缺陷（核心创新）
2. **级联条件化设计**：双拷贝机制解决 diffusion 在 AR 框架中的条件 attend mismatch
3. **MoT 架构**：每层参数按模态分离（attention + FFN）而非仅 FFN
4. **单目相机 SOTA**：仅用前视相机达到与多相机 BEV 方法相当的成绩

### 8.2 局限性

1. NAVSIM v1 分数偏低（指标权重偏向保守策略，Vega 学到的多样化策略在 v1 上反而吃亏）
2. 无 CoT 推理能力
3. Best-of-N 策略计算成本翻倍
4. 数据标注依赖 VLM（Qwen2.5-VL-72B），存在幻觉风险

### 8.3 个人点评

Vega 的核心贡献是**证明了世界模型可以作为一种有效的密集监督信号**，来解决 VLA 中稀疏动作监督的根本问题。这个思路其实很直觉——光预测"方向盘转10°"这一个动作，模型很难学到指令和动作之间的因果关系；但如果同时要"想象"执行这个动作后未来会看到什么画面，模型就必须理解**为什么**这个指令会导致这个动作。

这个洞察其实和 LLM 的 Chain-of-Thought 有异曲同工之妙——让模型进行**中间推理步骤**（这里是未来图像生成），能显著增强最终输出的质量。

---

## 9. Appendix 分点总结

论文 Appendix 包含以下内容：

1. **VLA Baseline 详细分析**：Qwen2.5-VL + planning head 仅达 ~60 PDMS，验证了稀疏监督问题的严重性
2. **交错序列设计细节**：图像-动作交错的长度（2/4/6）对收敛速度和最终性能的影响分析
3. **CFG 消融实验**：验证 text guidance 和 image guidance 的有效性
4. **多指令多样性分析**：证明 InstructScene 数据集中指令的多样性和覆盖度
5. **计算开销分析**：Vega 的 FLOPs 和推理延迟与同期工作的对比

---

## 10. 参考链接

| 类型 | 链接 |
|------|------|
| **论文** | [https://arxiv.org/abs/2603.25741](https://arxiv.org/abs/2603.25741) |
| **arXiv HTML** | [https://arxiv.org/html/2603.25741](https://arxiv.org/html/2603.25741) |
| **代码** | [https://github.com/zuosc19/Vega](https://github.com/zuosc19/Vega) |
| **Project Page** | [https://zuosc19.github.io/Vega](https://zuosc19.github.io/Vega) |

---

_精读日期：2026-04-14_
_分类：VLA / 指令跟随驾驶 / 世界模型_

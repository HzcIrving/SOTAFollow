# VISTA: Scaling World Model for Hierarchical Manipulation Policies

## 引用信息

| 字段 | 内容 |
|------|------|
| **论文标题** | Scaling World Model for Hierarchical Manipulation Policies |
| **arXiv** | https://arxiv.org/abs/2602.10983 |
| **代码** | https://github.com/VISTA-WM/VISTA |
| **作者** | Qian Long, Yueze Wang, Jiaxi Song, Junbo Zhang, Peiyan Li, Wenxuan Wang, Yuqi Wang, Haoyang Li, Shaoxuan Xie, Guocai Yao, Hanbo Zhang, Xinlong Wang, Zhongyuan Wang, Xuguang Lan, Huaping Liu, Xinghang Li |
| **机构** | 西安交通大学 · 北京人工智能研究院(BAAI) · 清华大学 · 新加坡国立大学(NUS) · 中国科学院自动化研究所 |
| **发布时间** | 2026年2月12日 |
| **核心贡献** | VISTA：利用大规模预训练世界模型进行视觉子目标任务分解，解决 VLA 在分布外场景的泛化瓶颈 |

---

## 一句话总结

**VISTA 提出层次化 VLA 框架：由世界模型作为高层规划器生成视觉子目标图像（Visual Subgoal），由 VLA 作为低层执行器跟随文本和视觉引导生成动作序列，在未见场景下将同一结构 VLA 的成功率从 14% 提升至 69%。**

---

## 核心贡献

1. **视觉子目标任务分解（VISTA）**：利用世界模型生成目标图像作为子任务，比纯文本目标提供更丰富的视觉和物理约束

2. **层次化架构**：世界模型（高层规划器）+ VLA（低层执行器），将长程任务分解为可执行的子任务序列

3. **分布外泛化突破**：同一结构 VLA 在未见场景下成功率从 14% 提升至 69%

4. **跨本体迁移**：基于提示中的机械臂类型，生成不同本体的 rollout

5. **大规模具身数据集**：构建包含多样化场景的具身数据集，验证视觉目标合成的有效性

---

## 方法详述

### 1. 问题定义

#### 1.1 VLA 的成就与泛化瓶颈

视觉-语言-动作（VLA）模型在机器人操控领域展现出作为通用策略的潜力。然而，在分布外（OOD）场景中，特别是数据有限的真实机器人场景下，VLA 仍面临严重的泛化瓶颈。

**核心问题**：
- 端到端 VLA 难以处理长程任务（需要多个中间步骤）
- 纯文本目标缺乏表达力，无法强制执行严格的视觉和物理约束
- 缺乏中间表示导致规划不鲁棒

#### 1.2 核心洞察

**视觉目标比文本目标更具泛化性**。相比 raw 文本目标规范，由世界模型合成的目标图像为低层策略提供了视觉和物理上的一致细节，使得跨未见物体和新型场景的泛化变得可行。

### 2. 整体架构

VISTA 采用**层次化 Vision-Language-Action 框架**：

```
高层规划器（世界模型 W）
    ↓ 生成 (l_i, g_i) 子目标对
低层执行器（GoalVLA π_θ）
    ↓ 预测动作序列 a
机器人执行
```

**输入**：
- 指令 $L$（自然语言任务描述）
- 历史 $\boldsymbol{h} = (l_0, g_0, ..., l_{i-1}, g_{i-1})$
- 当前观察 $I_t$

**输出**：
- 子目标对 $(l_i, g_i)$：语言子任务 + 视觉目标图像
- 动作序列 $\boldsymbol{a}$

![Figure 1: VISTA Overview](https://arxiv.org/html/2602.10983v2/x1.png)

VISTA 框架图：世界模型作为高层规划器分解任务为子目标序列，GoalVLA 作为低层控制器执行动作。

### 3. 核心方法：视觉子目标任务分解

#### 3.1 世界模型作为高层规划器

世界模型 $W$ 根据指令 $L$ 和历史 $\boldsymbol{h}$ 预测下一个所需的子目标：

$$
g_i, l_i = W(L, \boldsymbol{h})
$$

其中：
- $l_i$：语言子任务（如 "pick up the red block"）
- $g_i$：对应的视觉目标图像（由世界模型生成）

#### 3.2 GoalVLA 作为低层执行器

低层策略 $\pi_\theta$ 接收当前观察 $I_t$ 和当前子目标 $(l_i, g_i)$，推断动作序列：

$$
\boldsymbol{a} = \pi_\theta(I_t, l_i, g_i)
$$

#### 3.3 统一序列建模

VISTA 将所有元素（图像 tokens、语言 tokens、动作 tokens）建模为统一序列：

$$
S = (I_0, L, a_0, g_0, l_0, I_1, a_1, g_1, l_1, ..., I_t)
$$

通过标准自回归建模优化交叉熵损失：

$$
\mathcal{L} = -\sum_{t} \log p(s_t \mid s_{<t})
$$

#### 3.4 关键设计：子目标不变性

与生成密集未来帧的视频生成框架不同，VISTA 将连续任务抽象为离散关键子目标 $(l_i, g_i)$。这种方法利用了**子目标不变性**的启发式，避免了视频预测的随机性歧义。

### 4. 子任务切换器

当低层执行器完成当前子目标后，子任务切换器（Subtask Switcher）决定：
1. 切换到下一个子目标
2. 重新生成更细粒度的子目标

切换条件由世界模型判断：当前子目标完成且任务未结束时，生成下一个 $(l_{i+1}, g_{i+1})$。

### 5. 数据集构建

VISTA 构建了大规模具身数据集，包含多样化场景：
- 不同的物体类别
- 不同的空间布局
- 不同的背景

![Figure 2: Embodied Dataset Samples](https://arxiv.org/html/2602.10983v2/imgs/robot_dataset_v3.001.jpeg)

---

## 训练与推理伪代码

```python
"""
VISTA Inference Algorithm Pseudocode
"""

def vista_inference(world_model, goal_vla, instruction, initial_obs):
    """
    VISTA 推理流程

    Args:
        world_model: 世界模型 W（高层规划器）
        goal_vla: GoalVLA π_θ（低层执行器）
        instruction: 自然语言指令 L
        initial_obs: 初始观察 I_0
    """
    # 初始化历史
    history = []  # (l_0, g_0), (l_1, g_1), ...
    current_obs = initial_obs
    subtask_idx = 0

    # 主循环
    while not task_complete():
        # Step 1: 世界模型生成子目标（高层规划）
        l_i, g_i = world_model.predict_subgoal(
            instruction=instruction,
            history=history
        )

        # Step 2: GoalVLA 执行子目标（低层控制）
        actions = goal_vla.predict_actions(
            observation=current_obs,
            subtask=l_i,
            subgoal_image=g_i
        )

        # Step 3: 执行动作
        for action in actions:
            execute(action)
            current_obs = get_observation()

            # 检查子目标是否完成
            if goal_vla.check_subgoal_complete(current_obs, g_i):
                # 记录完成的子目标
                history.append((l_i, g_i))
                subtask_idx += 1
                break

    return trajectory


def world_model_subgoal_generation(world_model, instruction, history):
    """
    世界模型子目标生成
    """
    # 编码指令和历史
    emb = world_model.encode(instruction, history)

    # 解码生成子目标对
    l_i = world_model.decode_language(emb)      # 语言子任务
    g_i = world_model.decode_image(emb)          # 视觉目标图像

    return l_i, g_i


def goal_vla_action_prediction(goal_vla, observation, subtask, subgoal_image):
    """
    GoalVLA 动作预测
    """
    # 融合观察、语言子任务、视觉目标
    fused = goal_vla.fusion模块(
        visual=observation,
        language=subtask,
        subgoal=subgoal_image
    )

    # 自回归生成动作序列
    actions = goal_vla.autoregressive_generate(fused)

    return actions
```

---

## 实验结论

### 1. 主实验结果

#### 1.1 分布外泛化性能

| 配置 | 基础任务 | 未见干扰物 | 未见目标物体 |
|------|---------|-----------|-------------|
| **App** (方法成功率) | - | - | - |
| **Suc** (执行成功率) | - | - | - |

**核心结果**：同一结构的 VLA 在未见场景下，成功率从 **14% 提升至 69%**（+55pp）。

#### 1.2 真实机器人实验

![Figure 3: Real Robot Experiment Setups](https://arxiv.org/html/2602.10983v2/x2.png)

在真实机器人场景（包括领域内和新型场景）验证了方法的有效性。

### 2. 视觉目标合成质量

![Figure 4: Subtask and Goal Image Sequences](https://arxiv.org/html/2602.10983v2/imgs/bench_gen_short.001.jpeg)

VISTA 生成的子任务和目标图像序列，展示了多步规划的能力。

### 3. 多视角目标图像生成

![Figure 5: Multi-view Goal Images in Novel Scenarios](https://arxiv.org/html/2602.10983v2/x3.png)

在未见布局、干扰物、目标物体和背景的新型场景中，VISTA 生成的多视角目标图像。

### 4. 跨本体迁移

![Figure 7: Cross-Embodiment Transfer](https://arxiv.org/html/2602.10983v2/x5.png)

基于提示中的机械臂类型，VISTA 能为不同本体生成高质量 rollout，验证了跨本体迁移能力。

---

## VISTA 与 π0.7 对比分析

### 1. 核心思想对比

| 维度 | VISTA | π0.7 |
|------|-------|------|
| **论文全称** | Scaling World Model for Hierarchical Manipulation Policies | π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities |
| **机构** | 西安交通大学、BAAI、清华大学、NUS、中科院 | Physical Intelligence (PI) |
| **核心方法** | 世界模型生成视觉子目标 + VLA 执行 | 多样化上下文条件化（DCC） |
| **层次化** | 显式双层：世界模型（规划）+ VLA（执行） | 隐式单层：统一 VLM + Action Expert |
| **arXiv** | 2602.10983 | 无（仅 PI 官网） |

### 2. 技术路线对比

#### VISTA：层次化视觉子目标

```
指令 L → 世界模型 W → (l_i, g_i) 子目标对 → GoalVLA π_θ → 动作序列
```

- **高层规划**：世界模型基于历史和指令生成 $(l_i, g_i)$ 对
- **低层执行**：GoalVLA 接收视觉目标图像，生成可执行动作
- **关键创新**：视觉目标图像提供物理一致的约束信息

#### π0.7：多样化上下文条件化

```
Context C_t = {ℓ_t, ℓ̂_t, g_t, m, c} → 统一 VLA → 动作 chunk
```

- **Context 组成**：语言指令 + 子任务 + 子目标图像 + Episode 元数据 + 控制模式
- **关键创新**：Episode Metadata 消歧异构数据质量
- **核心洞察**：将"如何做"的策略信息从数据中分离

### 3. 相同点：视觉子目标的核心地位

两者都将**视觉子目标图像**作为核心中间表示：

| 方面 | VISTA | π0.7 |
|------|-------|------|
| **子目标图像来源** | 世界模型生成 | 世界模型（BAGEL 初始化）生成 |
| **作用** | 高层规划的具体目标 | Context 中的视觉引导 |
| **本质** | 显式层次化接口 | 统一模型的上下文条件 |

**共性洞察**：视觉目标图像比纯文本目标更具表达力，能编码细粒度的空间和物理约束。

### 4. 不同点

| 方面 | VISTA | π0.7 |
|------|-------|------|
| **架构** | 显式双层（世界模型 + VLA） | 统一单层（VLM + Action Expert） |
| **数据消歧** | 无 | Episode Metadata (Quality/Speed/Mistake) |
| **跨本体迁移** | 依赖文本提示指定本体 | 零样本自主适应（+80% vs 人类 80.6%） |
| **Coaching 范式** | 无 | 语言 step-by-step 指导 |
| **Action Expert** | 未知 | 860M Flow Matching Transformer |

### 5. 总结对比

| 维度 | VISTA | π0.7 |
|------|-------|------|
| **核心问题** | VLA 泛化瓶颈（14%→69%） | 专用 RL 微调的通用性不足 |
| **解决思路** | 层次化视觉子目标分解 | 上下文条件化消歧异构数据 |
| **通用性** | 依赖视觉目标合成 | 无需任务特定微调 |
| **世界模型** | 核心组件（生成子目标） | 辅助组件（异步生成子目标图像） |

**两者共同趋势**：都认识到视觉子目标的重要性，但采用不同架构实现。VISTA 用显式层次化，π0.7 用隐式上下文条件化。

---

## KnowHow

### 核心洞察

1. **为什么视觉子目标比文本目标更具泛化性？**
   - 图像提供丰富的视觉和物理约束细节
   - 文本描述难以精确指定空间关系和物体姿态
   - 视觉目标使得跨未见物体和新型场景的迁移更可行

2. **为什么选择层次化架构？**
   - 长程任务需要分解为多个可执行的子目标
   - 高层规划（世界模型）处理语义理解
   - 低层执行（VLA）处理精细动作控制
   - 子目标不变性避免视频预测的随机性歧义

3. **世界模型在机器人操控中的双重作用**
   - 作为高层规划器：生成子目标序列
   - 作为视觉目标合成器：提供物理一致的中间表示

4. **跨本体迁移的关键**
   - VISTA 通过文本提示指定机械臂类型
   - π0.7 通过零样本自主发现适应新本体的策略
   - 本质都是将"意图"与"执行"解耦

5. **层次化 vs 统一架构的权衡**
   - 层次化：更可解释、可组合，但需要额外的世界模型
   - 统一：端到端优化，但需要更多数据和上下文条件化

6. **VISTA 的局限**
   - 依赖大规模预训练世界模型
   - 子目标质量受世界模型能力限制
   - 尚未达到 π0.7 的跨本体零样本迁移水平

---

## 总结

VISTA 提出了利用大规模预训练世界模型进行层次化视觉子目标任务分解的核心思想，主要贡献包括：

1. **视觉子目标任务分解（VISTA）**：用世界模型生成 $(l_i, g_i)$ 子目标对，比纯文本目标提供更丰富的视觉和物理约束

2. **层次化 VLA 架构**：世界模型（高层规划器）+ GoalVLA（低层执行器），实现长程任务的鲁棒规划

3. **显著的泛化提升**：同一结构 VLA 在未见场景下成功率从 14% 提升至 69%

4. **跨本体迁移能力**：基于提示生成不同本体的 rollout

**与 π0.7 的核心差异**：VISTA 采用显式层次化架构（世界模型 + VLA），而 π0.7 采用统一 VLM + Action Expert + 多样化上下文条件化。两者都认识到视觉子目标的重要性，但实现路径不同。VISTA 更依赖世界模型的能力，π0.7 更强调数据层面的消歧和上下文条件化。

---

## 参考链接

- **arXiv**: https://arxiv.org/abs/2602.10983
- **项目主页**: https://vista-wm.github.io/
- **代码**: https://github.com/VISTA-WM/VISTA

---

*整理 by 优酱 🍃 | 2026-04-22*

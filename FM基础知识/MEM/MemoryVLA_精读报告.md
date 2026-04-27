# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

## 引用信息

| 字段 | 内容 |
|------|------|
| **论文标题** | MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation |
| **机构** | 清华大学（Tsinghua）、Moss Robotics |
| **论文链接** | [arXiv:2508.19236](https://arxiv.org/abs/2508.19236) |
| **代码链接** | [待补充] |
| **项目主页** | [MemoryVLA Project Page](https://shihao1895.github.io/MemoryVLA) |
| **核心作者** | Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, Gao Huang |
| **发布时间** | 2025.08.26 |

---

## 1. Motivation（问题背景）

### 1.1 VLA 的时序建模缺陷

当前主流 VLA 模型存在根本性缺陷：**忽视时间上下文**。机器人操控任务本质上是**非马尔可夫过程**——当前动作的决策需要依赖历史观察和动作序列，而非仅依赖当前帧。然而现有 VLA 模型（如 π0、OpenVLA）：

- 仅处理单帧或有限历史窗口
- 无法建模跨长时域的时序依赖关系
- 在长时域任务中性能显著下降

### 1.2 认知科学的启示

人类执行操控任务时依赖两套互补的记忆系统：

| 记忆类型 | 功能 | 时间尺度 |
|----------|------|----------|
| **工作记忆（Working Memory）** | 缓冲短期表征，用于即时控制决策 | 秒级 |
| **情景记忆（Episodic Memory）** | 海马系统保存过去经历的原始细节和语义要点 | 分钟~长时 |

### 1.3 核心洞察

**关键问题**：如何将认知科学的双记忆机制引入 VLA，使其具备处理长时域、非马尔可夫任务的能力？

---

## 2. 一句话总结

**MemoryVLA 提出感知-认知记忆框架，借鉴人类工作记忆和情景记忆机制，通过感知-认知记忆银行实现跨时域信息检索与融合，使 VLA 能够在长时域操控任务中实现 84% 成功率。**

---

## 3. 核心贡献

1. **Cognition-Memory-Action 框架**：首次将认知科学双记忆机制引入 VLA
2. **感知-认知记忆银行（Perceptual-Cognitive Memory Bank）**：同时存储低层细节和高层语义
3. **自适应记忆融合（Memory Gate Fusion）**：动态融合历史记忆与当前感知
4. **记忆条件扩散动作专家（Memory-Conditioned Diffusion Action Expert）**：时序感知的动作生成
5. **跨平台优越性能**：在 Bridge/Fractal/LIBERO-5/Mikasa-Robo 四项基准均超越 SOTA

---


## 10. 对自动驾驶时序融合的启发与借鉴点

### 10.1 双记忆系统的时间尺度解耦

MemoryVLA 的感知-认知双记忆系统对自动驾驶时序融合的启发：

| MemoryVLA 机器人场景 | 自动驾驶对应 | 时间尺度 |
|---------------------|-------------|---------|
| **工作记忆** | 当前帧 BEV 特征 + 短期轨迹跟踪 | ~1 秒 |
| **情景记忆银行** | 历史轨迹库 + 场景语义地图 | ~10 秒以上 |

**借鉴思路**：自动驾驶感知可采用类似解耦——短期历史（如最近 1 秒）直接拼接或注意力融合；长期历史（如过去 10 秒的轨迹）存入记忆银行，通过检索获取相关记忆。

### 10.2 感知-认知分离编码

MemoryVLA 将 VLM 编码分离为感知 tokens 和认知 tokens：

- **感知 tokens**：低层视觉特征（边缘、纹理、颜色）
- **认知 tokens**：高层语义特征（物体类别、空间关系、行为意图）

**借鉴思路**：自动驾驶中，BEV 特征可以类似分解：
- **感知 tokens**：占据栅格、车道线几何、目标检测框
- **认知 tokens**：场景图（物体关系）、意图预测（换道/让行）、交通规则

分别处理后再融合，可能比单一 BEV 特征更高效。

### 10.3 记忆检索的语义匹配

MemoryVLA 通过 query-key 匹配从记忆银行检索相关记忆：

$$
\text{Retrieval}(\mathbf{Q}) = \text{TopK}(\mathbf{Q} \cdot \mathbf{M}^\top)
$$

**借鉴思路**：自动驾驶场景中，可以将历史驾驶上下文编码为记忆 bank's keys，当前驾驶场景作为 query，检索最相似的历史场景，辅助决策（如"这个路口上次遇到行人时我是怎么做的"）。

### 10.4 记忆门融合的软选择

MemoryVLA 使用 sigmoid 门控实现软融合：

$$
\mathbf{F} = \text{sigmoid}(\mathbf{W}_g \cdot [\mathbf{C}; \mathbf{R}]) \cdot \mathbf{R} + (1 - \text{sigmoid}(\mathbf{W}_g \cdot [\mathbf{C}; \mathbf{R}])) \cdot \mathbf{C}
$$

**借鉴思路**：在自动驾驶感知融合中，可以类似使用门控网络决定"当前感知"和"历史记忆"的融合比例，而非简单拼接或加权平均。

### 10.5 记忆整合与冗余合并

MemoryVLA 的 Memory Consolidation 机制将新信息整合到记忆银行，同时合并冗余。

**借鉴思路**：自动驾驶中，相似场景的记忆应当合并（如连续几帧的同一场景），避免记忆银行膨胀。这对长期运行的自动驾驶系统尤为重要。

---

## 4. 方法详述

### 4.1 整体架构

![图 1：MemoryVLA 整体架构](https://arxiv.org/html/2508.19236v1/x2.png)

MemoryVLA 由三大模块组成：

```
观察 o_t
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│          Vision-Language Cognition Module                │
│  (预训练 VLM 编码为感知 tokens + 认知 tokens)            │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐    ┌─────────────────────────────┐
│     Working Memory   │    │  Perceptual-Cognitive       │
│     (工作记忆)        │    │  Memory Bank               │
│  - 短期缓冲          │    │  (感知-认知记忆银行)          │
│  - 即时控制决策      │    │  - 低层细节存储              │
│                     │    │  - 高层语义抽象              │
└─────────┬───────────┘    └──────────────┬──────────────┘
          │                    ▲           │
          │                    │           │
          │            ┌───────┴───────────┘
          │            │  Memory Gate Fusion
          │            │  (自适应融合)
          │            ▼
          │    ┌─────────────────────────┐
          └───►│ Memory-Conditioned      │──► 动作序列 a_t:t+H
               │ Diffusion Action Expert │
               │ (记忆条件扩散动作专家)   │
               └─────────────────────────┘
```

### 4.2 Vision-Language Cognition Module

预训练 VLM 将观察编码为两类 tokens：

| Token 类型 | 来源 | 作用 |
|------------|------|------|
| **感知 tokens** | 视觉编码器的低层特征 | 保留原始视觉细节 |
| **认知 tokens** | VLM 的高层语义表征 | 提供任务相关的语义理解 |

### 4.3 Perceptual-Cognitive Memory Module

#### 4.3.1 记忆检索（Memory Retrieval）

给定当前工作记忆中的 queries，从记忆银行中检索相关条目：

$$
\text{Retrieval}(\mathbf{Q}) = \text{TopK}(\mathbf{Q} \cdot \mathbf{M}^\top)
$$

其中 $\mathbf{Q}$ 是 query 矩阵，$\mathbf{M}$ 是记忆银行中的记忆矩阵。

#### 4.3.2 记忆门融合（Memory Gate Fusion）

自适应融合检索到的历史记忆与当前 tokens：

$$
\mathbf{F} = \text{sigmoid}(\mathbf{W}_g \cdot [\mathbf{C}; \mathbf{R}]) \cdot \mathbf{R} + (1 - \text{sigmoid}(\mathbf{W}_g \cdot [\mathbf{C}; \mathbf{R}])) \cdot \mathbf{C}
$$

其中 $\mathbf{C}$ 是认知 tokens，$\mathbf{R}$ 是检索到的记忆，$\mathbf{F}$ 是融合后的表征。

#### 4.3.3 记忆整合（Memory Consolidation）

工作记忆将新的决策相关信息整合到记忆银行，同时合并冗余：

$$
\mathbf{M}_{\text{new}} = \text{Consolidate}(\mathbf{M}_{\text{old}}, \mathbf{W})
$$

### 4.4 Memory-Conditioned Diffusion Action Expert

基于扩散模型的动作生成器，以记忆融合表征为条件：

```python
def memory_conditioned_diffusion(model, memory_features, current_obs, num_steps=50):
    """
    MemoryVLA 动作生成
    memory_features: 融合后的记忆表征
    current_obs: 当前观察
    """
    # 初始化噪声动作序列
    a_T = torch.randn(B, H, action_dim)

    # 扩散去噪
    for t in reversed(range(num_steps)):
        # 条件融合
        condition = torch.cat([memory_features, current_obs], dim=-1)

        # 预测噪声并去噪
        noise_pred = model(a_t, t, condition)
        a_{t-1} = a_t - noise_pred * (1 / num_steps)

    return a_0  # 预测的动作序列
```

---

## 5. 训练与推理伪代码

```python
class MemoryVLA:
    def __init__(self, vlm, memory_bank, action_expert):
        self.vlm = vlm  # 预训练 VLM
        self.memory_bank = memory_bank
        self.action_expert = action_expert

    def forward(self, obs, task_description, memory_bank_state=None):
        """
        MemoryVLA 前向传播
        obs: 当前观察 (图像序列)
        task_description: 语言指令
        """
        # 1. VLM 编码
        perceptual_tokens, cognitive_tokens = self.vlm.encode(obs)

        # 2. 工作记忆初始化
        working_memory = self.initialize_working_memory(
            perceptual_tokens, cognitive_tokens
        )

        # 3. 记忆检索
        retrieved_memories = self.memory_bank.retrieve(working_memory.query)

        # 4. 自适应融合
        fused_features = self.memory_gate_fusion(
            cognitive_tokens, retrieved_memories
        )

        # 5. 记忆整合
        self.memory_bank.consolidate(working_memory, fused_features)

        # 6. 记忆条件动作生成
        action_sequence = self.action_expert.generate(
            memory_features=fused_features,
            current_obs=perceptual_tokens
        )

        return action_sequence

    def memory_gate_fusion(self, cognitive, retrieved):
        """自适应记忆融合"""
        concat = torch.cat([cognitive, retrieved], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(concat))
        return gate * retrieved + (1 - gate) * cognitive
```

---

## 6. 实验结论

### 6.1 仿真基准测试

![图 2：SimplerEnv-Bridge 实验结果](https://arxiv.org/html/2508.19236v1/x5.png)

| 方法 | SimplerEnv-Bridge ↑ | Fractal ↑ | LIBERO-5 ↑ | Mikasa-Robo ↑ |
|------|---------------------|-----------|------------|---------------|
| CogACT | 57.3% | 63.7% | 95.0% | 29.4% |
| π-0 | 56.1% | 60.9% | 93.2% | 30.5% |
| **MemoryVLA** | **71.9%** | **72.7%** | **96.5%** | **41.2%** |
| **Δ** | **+14.6** | **+8.8** | **+1.5** | **+11.8** |

### 6.2 真实世界任务

| 任务类型 | 任务数 | MemoryVLA | SOTA | Δ |
|----------|--------|-----------|------|---|
| **通用技能** | 8 | 92.5% | 83.0% | +9.5 |
| **长时域依赖** | 4 | 67.5% | 41.5% | **+26.0** |
| **整体** | 12 | **84.0%** | 68.5% | +15.5 |

### 6.3 消融实验

| 配置 | Bridge | Mikasa-Robo | 说明 |
|------|--------|-------------|------|
| w/o Memory Bank | 58.2% | 28.7% | 记忆银行至关重要 |
| w/o Perceptual Tokens | 65.4% | 35.2% | 感知 tokens 提供细节 |
| w/o Cognitive Tokens | 63.1% | 33.8% | 认知 tokens 提供语义 |
| **Full Model** | **71.9%** | **41.2%** | 完整模型最优 |

---

## 7. KnowHow（核心洞察）

1. **为什么双记忆机制有效？**
   - 工作记忆缓冲即时决策所需信息
   - 情景记忆银行存储跨episode的长期知识
   - 两者协同实现"既见树木又见森林"

2. **为什么需要感知+认知两类 tokens？**
   - 感知 tokens 保留低层视觉细节（如抓取角度、物体颜色）
   - 认知 tokens 提供高层语义理解（如任务目标、中间状态）
   - 分离编码避免信息压缩损失

3. **记忆门融合的工程智慧**
   - sigmoid 门控实现软选择，决定多少信息来自记忆、多少来自当前
   - 避免硬切换导致的信息断裂

4. **扩散动作专家的优势**
   - 相比回归输出，扩散模型更好捕捉动作分布的多模态性
   - 记忆条件提供时序一致性约束

5. **长时域任务提升 26% 的原因**
   - 记忆银行累积了任务相关的时序上下文
   - 检索机制能快速定位相关历史状态
   - 避免重复学习相同技能

---

## 8. arXiv Appendix 关键点总结

> **注**：以下内容来自 arXiv:2508.19236 原文。

### A. 方法细节

**VLM Backbone**：使用 InternVL3 作为视觉-语言编码器（待确认）。

**记忆银行容量**：未明确说明具体容量参数。

**动作空间**：连续动作空间，输出 H=8 步动作 chunk。

### B. 训练策略

**预训练阶段**：VLM 冻结，仅训练记忆模块和动作专家。

**微调阶段**：可选端到端微调 VLM。

### C. 与 π0.7 的对比

| 维度 | MemoryVLA | π0.7 |
|------|------------|------|
| **记忆机制** | 外显记忆银行 | 隐式 MEM |
| **时序建模** | 显式检索+融合 | 隐式历史编码 |
| **动作生成** | 扩散模型 | Flow Matching |
| **长时域任务** | +26% 提升 | 依赖上下文 |

### D. 局限性与未来方向

- 记忆银行需要额外的存储和检索计算
- 当前未探索多智能体协作场景
- 真实世界部署的实时性待验证

---

## 9. 总结

MemoryVLA 的核心贡献是**首次将认知科学的双记忆机制引入 VLA**，解决了三个关键问题：

1. **时序建模不足**：通过感知-认知记忆银行，显式建模跨时域的决策依赖
2. **长时域任务失败**：记忆检索机制快速定位相关历史，避免重复学习
3. **多模态动作分布**：扩散动作专家结合记忆条件，生成时序一致的动作序列

在 Bridge 基准上提升 +14.6%，Mikasa-Robo 真实机器人上提升 +11.8%，真实世界长时域任务提升 +26%，全面超越 π0 和 CogACT。

**最重要洞察**：外显记忆银行相比隐式历史编码，在长时域任务上具有显著优势——显式存储和检索使模型能够"选择性回忆"相关经验，而非被动受限于固定窗口。

---

---


---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2508.19236](https://arxiv.org/abs/2508.19236) |
| **代码** | [待补充] |
| **项目主页** | [MemoryVLA Project Page](https://shihao1895.github.io/MemoryVLA) |

---

*整理 by 优酱 🍃 | 2026-04-26*
*精读标准参考 CLAUDE.md § 论文精读格式标准*

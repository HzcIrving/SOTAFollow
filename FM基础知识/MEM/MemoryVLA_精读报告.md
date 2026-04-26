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

## 10. 对自动驾驶时序融合的启发与借鉴点

### 10.1 时序编码机制（Positional Encoding）

MemoryVLA 的感知-认知双 tokens 机制对自动驾驶感知编码有直接借鉴意义：

**MemoryVLA 的时序编码思路**：
- **感知 tokens**：来自 ViT 的低层特征（保留细节）
- **认知 tokens**：来自 VLM 高层的语义表征（抽象理解）
- 两者分离使检索和融合更高效

**自动驾驶的时序编码设计**：

| 组件 | MemoryVLA 机器人 | 自动驾驶 |
|------|------------------|---------|
| **感知 tokens** | ViT 低层特征 | 占据栅格、边缘、纹理 |
| **认知 tokens** | VLM 语义表征 | 场景图、意图、关系 |
| **融合方式** | Gate Fusion | 门控或注意力 |

```python
class PerceptualCognitiveEncoder:
    """
    自动驾驶感知-认知双编码器
    借鉴 MemoryVLA 的双 tokens 分离设计
    """
    def __init__(self, feature_dim, perceptual_dim, cognitive_dim):
        # 感知编码器：低层视觉特征
        self.perceptual_encoder = CNNBackbone(
            input_channels=3,  # RGB 图像
            output_dim=perceptual_dim
        )

        # 认知编码器：高层语义理解
        self.cognitive_encoder = TransformerEncoder(
            input_dim=perceptual_dim,
            hidden_dim=cognitive_dim,
            num_layers=6
        )

        # 场景理解网络
        self.scene_understanding = SceneGraphNetwork(
            num_obj_types=10,
            relation_dim=64
        )

    def encode(self, bev_image, obj_detections, road_graph):
        """
        双编码前向传播
        bev_image: [B, 3, H, W] BEV 图像
        obj_detections: [B, N, 6] 检测框 (x, y, z, w, h, l)
        road_graph: 道路拓扑图
        """
        # 1. 感知 tokens：低层视觉特征
        perceptual_tokens = self.perceptual_encoder(bev_image)  # [B, N_p, C_p]

        # 2. 认知 tokens：高层语义特征
        # 融合目标检测和道路结构
        obj_context = self._encode_objects(obj_detections)  # [B, N_o, C_c]
        road_context = self._encode_road(road_graph)        # [B, N_r, C_c]
        cognitive_tokens = self.cognitive_encoder(
            torch.cat([perceptual_tokens, obj_context, road_context], dim=1)
        )  # [B, N_c, C_c]

        # 3. 分离返回
        return {
            'perceptual': perceptual_tokens,  # 用于细节检索
            'cognitive': cognitive_tokens,    # 用于语义匹配
        }

    def _encode_objects(self, detections):
        """编码目标检测（位置、类别、运动）"""
        # 空间位置编码
        pos_encoding = self._positional_encode(detections[..., :3])

        # 运动特征编码
        motion_features = detections[..., 3:]  # 速度、大小等

        return torch.cat([pos_encoding, motion_features], dim=-1)

    def _positional_encode(self, coords):
        """3D 位置编码"""
        # Sinusoidal 3D PE
        x_emb = self._sin_encode(coords[..., 0], dim=64)
        y_emb = self._sin_encode(coords[..., 1], dim=64)
        z_emb = self._sin_encode(coords[..., 2], dim=64)
        return torch.stack([x_emb, y_emb, z_emb], dim=-2)

    def _sin_encode(self, x, dim):
        """Sinusoidal 编码"""
        pe = torch.zeros(*x.shape, dim, device=x.device)
        position = x.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=x.device) * (-math.log(10000.0) / dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe
```

### 10.2 时空融合机制 / 时序融合机制

MemoryVLA 的记忆检索-融合-整合流程是自动驾驶时序融合的核心参考：

**MemoryVLA 的时序融合架构**：
1. **记忆检索**：Query-Key 匹配从记忆银行取相关记忆
2. **记忆门融合**：sigmoid 门控软选择
3. **记忆整合**：新记忆合并到银行

**自动驾驶的时空融合设计**：

```python
class MemoryBank:
    """
    自动驾驶记忆银行
    借鉴 MemoryVLA 的感知-认知双记忆设计
    """
    def __init__(self, bank_capacity=1000, feature_dim=256):
        self.capacity = bank_capacity
        self.feature_dim = feature_dim

        # 记忆银行：存储历史时刻的感知-认知特征
        self.perceptual_keys = []   # 感知特征作为检索 key
        self.cognitive_keys = []    # 认知特征作为检索 key
        self.timestamps = []       # 时间戳
        self.metadata = []         # 元数据（场景类型、天气等）

        # 记忆 value：完整的历史状态
        self.values = []

    def retrieve(self, query_perceptual, query_cognitive, top_k=5):
        """
        记忆检索：找到与当前 query 最相关的历史记忆
        """
        if len(self.perceptual_keys) == 0:
            return None

        # 感知层检索：基于低层特征相似度
        perc_scores = torch.matmul(
            query_perceptual,
            torch.stack(self.perceptual_keys).transpose(0, 1)
        )  # [B, N_bank]

        # 认知层检索：基于语义相似度
        cogn_scores = torch.matmul(
            query_cognitive,
            torch.stack(self.cognitive_keys).transpose(0, 1)
        )  # [B, N_bank]

        # 加权融合
        combined_scores = 0.3 * perc_scores + 0.7 * cogn_scores

        # 取 Top-K
        topk_scores, topk_indices = torch.topk(combined_scores, k=min(top_k, len(self.perceptual_keys)))

        retrieved = {
            'perceptual': torch.stack(self.perceptual_keys)[topk_indices],
            'cognitive': torch.stack(self.cognitive_keys)[topk_indices],
            'values': [self.values[i] for i in topk_indices.cpu().tolist()],
            'timestamps': [self.timestamps[i] for i in topk_indices.cpu().tolist()],
            'scores': topk_scores
        }
        return retrieved

    def consolidate(self, perceptual, cognitive, value, metadata):
        """
        记忆整合：将新记忆加入银行，合并相似记忆
        """
        # 计算与现有记忆的相似度
        if len(self.perceptual_keys) > 0:
            sim = torch.matmul(
                perceptual,
                torch.stack(self.perceptual_keys).transpose(0, 1)
            ).max()

            # 相似度高于阈值 → 合并
            if sim > 0.95:
                # 更新已有记忆（指数移动平均）
                idx = torch.argmax(sim)
                self.values[idx] = 0.7 * self.values[idx] + 0.3 * value
                return

        # 否则新增记忆
        self.perceptual_keys.append(perceptual.detach().cpu())
        self.cognitive_keys.append(cognitive.detach().cpu())
        self.values.append(value)
        self.timestamps.append(time.time())
        self.metadata.append(metadata)

        # 容量超限 → 删除最老记忆
        if len(self.perceptual_keys) > self.capacity:
            self.perceptual_keys.pop(0)
            self.cognitive_keys.pop(0)
            self.values.pop(0)
            self.timestamps.pop(0)
            self.metadata.pop(0)


class SpatiotemporalFusion:
    """
    时空融合模块
    结合记忆检索 + 门控融合
    """
    def __init__(self, feature_dim):
        self.memory_bank = MemoryBank(bank_capacity=1000, feature_dim=feature_dim)
        self.gate_fusion = GateFusionModule(feature_dim)

    def fuse(self, current_perceptual, current_cognitive, ego_motion):
        """
        时空融合主流程
        """
        # 1. 记忆检索
        retrieved = self.memory_bank.retrieve(
            query_perceptual=current_perceptual,
            query_cognitive=current_cognitive,
            top_k=5
        )

        if retrieved is None:
            # 无历史 → 直接使用当前感知
            return current_perceptual, current_cognitive, None

        # 2. 门控融合
        fused_perceptual, fused_cognitive, gate_values = self.gate_fusion(
            current_perceptual=current_perceptual,
            current_cognitive=current_cognitive,
            retrieved_perceptual=retrieved['perceptual'],
            retrieved_cognitive=retrieved['cognitive'],
            ego_motion=ego_motion
        )

        # 3. 更新记忆银行
        self.memory_bank.consolidate(
            perceptual=current_perceptual,
            cognitive=current_cognitive,
            value={'perceptual': current_perceptual, 'cognitive': current_cognitive},
            metadata={'ego_motion': ego_motion}
        )

        return fused_perceptual, fused_cognitive, gate_values
```

### 10.3 Gated 机制设计

MemoryVLA 的记忆门融合是"如何融合"的核心机制：

**MemoryVLA 的门控设计**：
- **输入**：认知 tokens + 检索到的记忆 tokens
- **机制**：sigmoid 门控，输出 [0, 1] 软权重
- **公式**：$F = \sigma(W_g \cdot [C; R]) \cdot R + (1 - \sigma(W_g \cdot [C; R])) \cdot C$

**自动驾驶的门控设计**：

```python
class GateFusionModule:
    """
    门控融合模块
    借鉴 MemoryVLA 的 sigmoid 软门控
    """
    def __init__(self, feature_dim, hidden_dim=128):
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 情境编码器（判断当前场景是否需要记忆）
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim + 6, hidden_dim),  # 特征 + 自车运动
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, current_perceptual, current_cognitive,
                retrieved_perceptual, retrieved_cognitive, ego_motion):
        """
        门控融合前向传播
        """
        # 1. 情境门控：判断是否需要检索到的记忆
        context_input = torch.cat([current_cognitive.mean(1), ego_motion], dim=-1)
        context_gate = self.context_encoder(context_input)  # [B, 1]

        # 2. 特征门控：决定感知和认知的融合比例
        # 感知层门控
        perc_input = torch.cat([current_perceptual, retrieved_perceptual.mean(1)], dim=-1)
        perc_gate = self.gate_net(perc_input)  # [B, 1]

        # 认知层门控
        cogn_input = torch.cat([current_cognitive, retrieved_cognitive.mean(1)], dim=-1)
        cogn_gate = self.gate_net(cogn_input)  # [B, 1]

        # 3. 软融合
        fused_perceptual = (
            perc_gate * retrieved_perceptual.mean(1) +
            (1 - perc_gate) * current_perceptual.mean(1)
        ).unsqueeze(1)

        fused_cognitive = (
            cogn_gate * retrieved_cognitive.mean(1) +
            (1 - cogn_gate) * current_cognitive.mean(1)
        ).unsqueeze(1)

        # 4. 情境门控调整：场景变化大时更依赖记忆
        gate_values = {
            'context_gate': context_gate,
            'perceptual_gate': perc_gate,
            'cognitive_gate': cogn_gate
        }

        return fused_perceptual, fused_cognitive, gate_values
```

### 10.4 When / How / What 三问

#### 10.4.1 什么时候触发时序融合？（When）

| 触发条件 | MemoryVLA 机器人实现 | 自动驾驶实现 |
|----------|---------------------|-------------|
| **任务相关性** | 检索记忆与当前子任务的相关度 | 当前场景与历史场景的语义相似度 |
| **场景变化** | — | 天气变化、光照变化、道路结构变化 |
| **决策关键点** | 子任务切换时 | 换道、汇入、让行、紧急制动 |

```python
def should_activate_temporal(context_similarity, scene_change, decision_type):
    """
    判断是否触发时序融合
    context_similarity: 当前与检索记忆的语义相似度
    scene_change: 场景变化程度（天气、光照、道路）
    decision_type: 当前决策类型
    """
    # 触发阈值
    SIMILARITY_THRESHOLD = 0.3
    SCENE_CHANGE_THRESHOLD = 0.5

    triggers = 0

    # 条件1: 检索记忆相关性低 → 需要更多历史信息
    if context_similarity < SIMILARITY_THRESHOLD:
        triggers += 1

    # 条件2: 场景变化大 → 需要参考历史相似场景
    if scene_change > SCENE_CHANGE_THRESHOLD:
        triggers += 1

    # 条件3: 关键决策点
    if decision_type in ['lane_change', 'merging', 'yielding', 'emergency']:
        triggers += 1

    # 满足至少一个条件
    return triggers >= 1
```

#### 10.4.2 怎么设计时序模块？（How）

| 设计维度 | MemoryVLA 方案 | 自动驾驶借鉴 |
|----------|---------------|-------------|
| **存储结构** | 双记忆银行（感知+认知 keys） | 分层记忆银行（短期+长期） |
| **检索机制** | Query-Key 相似度匹配 | 语义相似度 + 空间邻近度 |
| **融合机制** | sigmoid 软门控 | 门控 + 注意力加权 |
| **更新策略** | 相似记忆合并 | 滑动窗口 + 重要性重加权 |

#### 10.4.3 时序存什么历史？（What）

| 历史内容 | MemoryVLA 存储 | 自动驾驶存储 |
|----------|---------------|-------------|
| **感知层** | 图像 token 低层特征 | BEV 占据栅格、雷达点云特征 |
| **认知层** | VLM 语义表征 | 场景图、目标关系、意图 |
| **动作层** | 机器人动作序列 | 自车轨迹、他车轨迹 |
| **元数据** | 任务描述、子目标 | 场景类型、天气、时间 |

```python
class AutonomousDrivingMemory:
    """
    自动驾驶时序历史存储
    借鉴 MemoryVLA 的双层记忆设计
    """
    def __init__(self, short_term_len=10, long_term_capacity=500):
        # 短期记忆：滑动窗口，直接存储
        self.short_term = {
            'perceptual': [],      # BEV 特征
            'cognitive': [],       # 场景理解
            'ego_trajectory': [],   # 自车轨迹
            'obj_trajectories': [], # 他车轨迹
            'timestamps': []
        }
        self.short_term_len = short_term_len

        # 长期记忆：记忆银行，支持检索
        self.long_term_bank = MemoryBank(
            bank_capacity=long_term_capacity,
            feature_dim=256
        )

    def store_short_term(self, perceptual, cognitive, ego_state, obj_states, timestamp):
        """存储短期记忆"""
        self.short_term['perceptual'].append(perceptual)
        self.short_term['cognitive'].append(cognitive)
        self.short_term['ego_trajectory'].append(ego_state)
        self.short_term['obj_trajectories'].append(obj_states)
        self.short_term['timestamps'].append(timestamp)

        # 滑动窗口
        if len(self.short_term['perceptual']) > self.short_term_len:
            for key in self.short_term:
                self.short_term[key].pop(0)

    def store_long_term(self, perceptual, cognitive, ego_state, scene_type):
        """存储长期记忆到银行"""
        self.long_term_bank.consolidate(
            perceptual=perceptual,
            cognitive=cognitive,
            value={
                'ego_state': ego_state,
                'scene_type': scene_type
            },
            metadata={'scene_type': scene_type}
        )

    def retrieve_long_term(self, current_perceptual, current_cognitive, top_k=5):
        """检索长期记忆"""
        return self.long_term_bank.retrieve(
            query_perceptual=current_perceptual,
            query_cognitive=current_cognitive,
            top_k=top_k
        )
```

### 10.5 双记忆系统的时间尺度解耦

MemoryVLA 的感知-认知双记忆系统对自动驾驶时序融合的启发：

| MemoryVLA 机器人场景 | 自动驾驶对应 | 时间尺度 |
|---------------------|-------------|---------|
| **工作记忆** | 当前帧 BEV 特征 + 短期轨迹跟踪 | ~1 秒 |
| **情景记忆银行** | 历史轨迹库 + 场景语义地图 | ~10 秒以上 |

**借鉴思路**：自动驾驶感知可采用类似解耦——短期历史（如最近 1 秒）直接拼接或注意力融合；长期历史（如过去 10 秒的轨迹）存入记忆银行，通过检索获取相关记忆。

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

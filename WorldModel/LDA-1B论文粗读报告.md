# LDA-1B 模型技术文档

## 目录
1. [项目概述](#1-项目概述)
2. [数据处理流程](#2-数据处理流程)
3. [模型结构设计](#3-模型结构设计)
4. [模型处理流程](#4-模型处理流程)
5. [Loss设计](#5-loss设计)
6. [配置参数说明](#6-配置参数说明)

---

## 1. 项目概述

LDA-1B是一个基于**视觉-语言-动作(Vision-Language-Action, VLA)**的机器人操作模型，采用以下核心架构：
- **视觉语言编码器**: Qwen2.5-VL-3B (预训练的多模态大模型)
- **动作预测头**: Flow Matching + DiT (Diffusion Transformer)

### 核心特点
- 支持多模态输入（图像+语言指令）

- 使用Flow Matching进行连续动作序列生成

- 支持多embodiment（多种机器人类型）

- 基于LeRobot数据集格式

  方案框图

  ![LDA](E:\github\code\LLM2\SOTAFollow\FM基础知识\titok_figures\LDA.png)

---

## 2. 数据处理流程

### 2.1 数据集格式 (LeRobot v2.0/v3.0)

#### 数据目录结构
```
dataset_path/
├── meta/
│   ├── modality.json          # 模态元信息（状态、动作、视频等）
│   ├── episodes.jsonl         # episode信息
│   ├── tasks.jsonl            # 任务描述
│   ├── info.json              # 数据集基本信息
│   └── stats_gr00t.json       # 统计信息（用于归一化）
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # 低维数据（状态、动作）
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        ├── video.base_view/    # 视频数据
        │   ├── episode_000000.mp4
        │   └── episode_000001.mp4
        └── video.ego_view/
```

#### 数据模态定义
| 模态类型 | 示例字段 | 说明 |
|---------|---------|------|
| **视频** | `video.top_head`, `image_0` | RGB图像序列 |
| **状态** | `state.left_eef_position` | 当前机器人状态（位置、旋转、夹爪） |
| **动作** | `action.left_eef_position` | 目标动作序列 |
| **语言** | `annotation.language.action_text` | 自然语言指令 |

### 2.2 数据加载流程

```python
# 核心类: LeRobotSingleDataset (datasets.py)

class LeRobotSingleDataset(Dataset):
    def __init__(self, dataset_path, modality_configs, embodiment_tag, ...):
        # 1. 加载元数据
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        
        # 2. 获取轨迹信息
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        
        # 3. 获取所有采样步骤
        self._all_steps = self._get_all_steps()
        
        # 4. 设置数据变换
        self.set_transforms_metadata(self.metadata)
```

#### 关键参数配置 ([data_config.py](lda/dataloader/gr00t_lerobot/data_config.py))
```python
class BaseDataConfig(ABC):
    # 观察窗口：当前帧和前5帧
    observation_indices = [-5, 0]
    
    # 未来观察索引
    future_observation_indices = [5]
    
    # 历史动作窗口：前5帧到前1帧
    history_action_indices = list(range(-5, 0))
    
    # 动作预测窗口：历史5帧 + 当前帧 + 未来16帧
    action_indices = list(range(-5, 17))  # 共22个时间步
    
    # 图像采样间隔
    img_interval = 3
    
    # 状态字段
    state_keys = [
        "state.left_eef_position",     # 左臂末端位置 (3D)
        "state.left_eef_rotation",     # 左臂末端旋转 (3D/6D)
        "state.left_gripper",          # 左夹爪状态
        "state.right_eef_position",    # 右臂末端位置
        "state.right_eef_rotation",    # 右臂末端旋转
        "state.right_gripper",         # 右夹爪状态
    ]
    
    # 动作字段（与状态对应）
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_rotation",
        "action.left_gripper",
        "action.right_eef_position",
        "action.right_eef_rotation",
        "action.right_gripper",
    ]
```

### 2.3 数据变换Pipeline (Transform Pipeline)

#### 完整变换流程
```
原始数据 → StateActionToTensor → StateActionTransform(Normalize) → ConcatTransform → 输出
           ↓                        ↓
      转为Tensor              q99归一化/旋转转换
```

#### 2.3.1 视频变换 ([video.py](lda/dataloader/gr00t_lerobot/transform/video.py))

```python
class VideoTransform(ModalityTransform):
    """
    视频预处理变换
    - 后端支持: torchvision / albumentations
    - 操作: Resize, Crop, ColorJitter, ToTensor/ToNumpy
    """
    
    # 训练时变换
    train_transform = T.Compose([
        T.Resize(height=224, width=224),      # 统一分辨率
        T.RandomCrop(size=224, scale=0.95),   # 随机裁剪
        T.ColorJitter(                         # 颜色增强
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.08
        ),
    ])
    
    # 评估时变换（无数据增强）
    eval_transform = T.Compose([
        T.Resize(height=224, width=224),
    ])
```

**关键特性**:
- 多视图处理：对所有视角应用相同变换（使用albumentations的ReplayCompose）
- 支持批量处理：`(v b) t c h w -> (v b t) c h w`

#### 2.3.2 状态/动作变换 ([state_action.py](lda/dataloader/gr00t_lerobot/transform/state_action.py))

##### (a) 转换为Tensor
```python
class StateActionToTensor(InvertibleModalityTransform):
    """
    将numpy数组转换为PyTorch Tensor
    - 自动推断dtype
    - 保持维度不变
    """
    def apply(self, data):
        for key in self.apply_to:
            data[key] = torch.tensor(
                data[key], 
                dtype=self.output_dtypes[key]
            )
        return data
```

##### (b) 归一化变换
```python
class Normalizer:
    """
    支持多种归一化模式：
    - q99: 基于分位数的归一化到[-1, 1]
    - mean_std: 标准化 (z-score)
    - min_max: 最小-最大归一化到[-1, 1]
    - binary: 二值化
    """
    
    def forward(self, x):
        if self.mode == "q99":
            # 公式: normalized = 2 * (x - q01) / (q99 - q01) - 1
            # 范围: [-1, 1]
            normalized = (x - q01) / (q99 - q01)
            normalized = 2 * normalized - 1
            normalized = torch.clamp(normalized, -1, 1)
            
        elif self.mode == "mean_std":
            # 公式: normalized = (x - mean) / std
            normalized = (x - mean) / std
            
        elif self.mode == "min_max":
            # 公式: normalized = 2 * (x - min) / (max - min) - 1
            normalized = (x - min) / (max - min)
            normalized = 2 * normalized - 1
```

**q99归一化的优势**:
- 对异常值鲁棒（使用1%和99%分位数而非min/max）
- 输出范围固定在[-1, 1]，适合神经网络训练
- 可逆：支持反归一化恢复原始值

##### (c) 旋转变换
```python
class RotationTransform:
    """
    旋转表示转换
    支持: axis_angle, euler_angles, quaternion, rotation_6d, matrix
    
    典型用法: axis_angle -> rotation_6d
    - axis_angle: 3维 (紧凑但有奇异性)
    - rotation_6d: 6维 (连续、可微、适合学习)
    """
    
    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        # 构建正向和反向变换链
        # axis_angle -> matrix -> rotation_6d
        
    def forward(self, x):
        for func in self.forward_funcs:
            x = func(x)
        return x
```

#### 2.3.3 拼接变换 ([concat.py](lda/dataloader/gr00t_lerobot/transform/concat.py))

```python
class ConcatTransform(ModalityTransform):
    """
    将多个模态沿指定维度拼接
    例如：将左右臂的状态拼接为一个向量
    """
    def apply(self, data):
        # 拼接所有状态字段
        state_concat = torch.cat([data[key] for key in self.state_keys], dim=-1)
        # 拼接所有动作字段  
        action_concat = torch.cat([data[key] for key in self.action_keys], dim=-1)
        return {"state": state_concat, "action": action_concat}
```

### 2.4 数据采样策略

#### 时间窗口采样
```
时间轴:  |---历史动作---|--当前--|----未来动作----|
索引:    [-5,-4,-3,-2,-1] [0] [1,2,...,15,16]
         ↑               ↑    ↑
      observation_indices  action_indices
```

- **观察窗口**: 包含当前帧和历史帧（用于上下文理解）
- **动作窗口**: 包含历史动作 + 未来要预测的动作
- **未来动作窗口大小**: 默认16步 (`future_action_window_size=15` + 当前步)

#### Embodiment标签系统
```python
class EmbodimentTag(Enum):
    """机器人具身类型"""
    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    MOBILE = "mobile"
    HUMANoid = "humanoid"
    # ...
```

支持多embodiment训练，每个embodiment有独立的动作空间。

### 2.5 Batch构建与Collate函数

```python
def collate_fn(batch):
    """
    将多个样本组合成batch
    - 图像: list of PIL Images → 保持list形式（由VLM processor处理）
    - 动作: stack为tensor [B, T, D]
    - 状态: stack为tensor [B, 1, D_state]
    - 语言: 保持list形式
    - embodiment_id: tensor [B]
    """
    return {
        "image": [sample["image"] for sample in batch],
        "lang": [sample["lang"] for sample in batch],
        "action": np.stack([sample["action"] for sample in batch]),
        "state": np.stack([sample["state"] for sample in batch]),
        "embodiment_id": [sample["embodiment_id"] for sample in batch],
    }
```

---

## 3. 模型结构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QwenGR00T Framework                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐     │
│  │                     │    │                                 │     │
│  │  Qwen2.5-VL-3B      │    │     Flow Matching Action Head   │     │
│  │  (Vision-Language   │───▶│                                 │     │
│  │   Encoder)          │    │  ┌─────────────────────────┐   │     │
│  │                     │    │  │   Action Encoder        │   │     │
│  │  Input:             │    │  │   (MLP + Time Embedding)│   │     │
│  │  - Images [B,N,H,W] │    │  └───────────┬─────────────┘   │     │
│  │  - Text Instructions│    │              │                 │     │
│  │                     │    │  ┌───────────▼─────────────┐   │     │
│  │  Output:            │    │  │   Future Tokens Embed   │   │     │
│  │  hidden_states      │    │  │   (Learnable, 32 tokens)│   │     │
│  │  [B, L, H=2048]    │    │  └───────────┬─────────────┘   │     │
│  │                     │    │              │                 │     │
│  └─────────────────────┘    │  ┌───────────▼─────────────┐   │     │
│                             │  │   DiT Transformer       │   │     │
│                             │  │   (Cross-Attention)     │   │     │
│                             │  │   - 8 Layers            │   │     │
│                             │  │   - AdaNorm             │   │     │
│                             │  └───────────┬─────────────┘   │     │
│                             │              │                 │     │
│                             │  ┌───────────▼─────────────┐   │     │
│                             │  │   Action Decoder (MLP)  │   │     │
│                             │  └───────────┬─────────────┘   │     │
│                             │              │                 │     │
│                             │  Output: Predicted Actions  │     │
│                             │  [B, T_future, action_dim]  │     │
│                             └─────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 视觉语言编码器 (Qwen2.5-VL)

**文件**: [QWen2_5.py](lda/model/modules/vlm/QWen2_5.py)

#### 架构细节
```python
class _QWen_VL_Interface(nn.Module):
    def __init__(self, config):
        # 加载预训练的Qwen2.5-VL-3B
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",  # Flash Attention加速
            torch_dtype="auto",  # 自动选择bf16/fp16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
```

#### 关键特性
| 特性 | 说明 |
|------|------|
| **基础模型** | Qwen2.5-VL-3B-Instruct |
| **注意力机制** | Flash Attention 2 (高效内存) |
| **精度** | BFloat16混合精度 |
| **视觉处理** | Patch-based ViT encoder |
| **输出维度** | hidden_size = 2048 |
| **Tokenizer** | 左填充 (padding_side='left') |

#### 输入构建流程
```python
def build_qwenvl_inputs(self, images, instructions):
    """
    构建多模态输入
    
    输入:
        - images: List[List[PIL.Image]]  # [batch_size, num_views]
        - instructions: List[str]        # [batch_size]
    
    处理步骤:
        1. 格式化为chat messages
        2. 应用CoT模板（可选）
        3. Tokenization
        4. 提取视觉特征 (process_vision_info)
        5. 返回BatchFeature包含:
            - input_ids: [B, L_text]
            - attention_mask: [B, L_text]
            - pixel_values: [B, num_patches, C, H, W]
            - image_grid_thw: [B, 3] (时空网格信息)
    """
    
    # Chat格式
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in images[i]],
            {"type": "text", "text": prompt}
        ]
    }]
    
    # 应用processor
    text = processor.apply_chat_template(messages, ...)
    inputs = processor(text=text, images=images, ...)
    return inputs
```

### 3.3 Flow Matching Action Head

**文件**: [GR00T_ActionHeader.py](lda/model/modules/action_model/GR00T_ActionHeader.py)

#### 3.3.1 整体组件

```python
class FlowmatchingActionHead(nn.Module):
    def __init__(self, full_config):
        config = full_config.framework.action_model
        
        # ===== 核心组件 =====
        
        # 1. DiT骨干网络 (Diffusion Transformer)
        self.model = DiT(**diffusion_model_cfg)
        # 配置: DiT-B/L
        # - DiT-B: dim=768, heads=12, head_dim=64
        # - DiT-L: dim=1536, heads=32, head_dim=48
        
        # 2. Action Encoder (将noisy action嵌入到隐空间)
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,        # e.g., 29
            hidden_size=self.input_embedding_dim  # e.g., 768/1536
        )
        # 结构: Linear(action_dim, hidden) -> [+time_embed] -> Linear(2*hidden, hidden) -> Swish -> Linear(hidden, hidden)
        
        # 3. State Encoder (可选，编码当前状态)
        self.state_encoder = MLP(
            input_dim=config.state_dim,           # e.g., 58
            hidden_dim=self.hidden_size,          # e.g., 2560
            output_dim=self.input_embedding_dim   # e.g., 768/1536
        ) if config.state_dim else None
        
        # 4. Action Decoder (从DiT输出解码为动作空间)
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,  # DiT输出维度
            hidden_dim=self.hidden_size,
            output_dim=config.action_dim              # 动作维度
        )
        
        # 5. Future Tokens Embedding (可学习的查询token)
        self.future_tokens = nn.Embedding(
            num_embeddings=config.num_target_vision_tokens,  # 32个token
            embedding_dim=self.input_embedding_dim
        )
        
        # 6. Position Embedding (可选)
        self.position_embedding = nn.Embedding(
            config.max_seq_len,      # 最大序列长度1024
            self.input_embedding_dim
        )
        
        # 7. 时间采样分布 (Flow Matching用)
        self.beta_dist = Beta(
            alpha=config.noise_beta_alpha,  # 1.5
            beta=config.noise_beta_beta     # 1.0
        )
```

#### 3.3.2 Action Encoder详细结构

```python
class ActionEncoder(nn.Module):
    """
    将带噪声的动作轨迹和时间步编码为隐空间表示
    
    输入: 
        - actions: [B, T, action_dim]  (noisy action trajectory)
        - timesteps: [B] (discrete timestep)
    
    输出:
        - features: [B, T, hidden_size]
    """
    def __init__(self, action_dim, hidden_size):
        # 三层MLP + 正弦位置编码
        self.layer1 = nn.Linear(action_dim, hidden_size)      # 投影到隐空间
        self.layer2 = nn.Linear(2*hidden_size, hidden_size)   # 融合action+time
        self.layer3 = nn.Linear(hidden_size, hidden_size)     # 最终投影
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)
    
    def forward(self, actions, timesteps):
        # Step 1: Action embedding
        a_emb = self.layer1(actions)  # [B, T, hidden]
        
        # Step 2: Time embedding (正弦编码)
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, T)  # [B, T]
        tau_emb = self.pos_encoding(timesteps_expanded)  # [B, T, hidden]
        
        # Step 3: Concatenate and process
        x = torch.cat([a_emb, tau_emb], dim=-1)  # [B, T, 2*hidden]
        x = swish(self.layer2(x))                # [B, T, hidden]
        x = self.layer3(x)                       # [B, T, hidden]
        
        return x
```

#### 3.3.3 多Embodiment支持

```python
class CategorySpecificLinear(nn.Module):
    """
    类别特定的线性层，为不同embodiment维护独立权重
    
    用途: 多机器人类型联合训练时，每种机器人有独立的映射
    """
    def __init__(self, num_categories, input_dim, hidden_dim):
        # 为每个类别维护独立的 W 和 b
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(num_categories, hidden_dim))
    
    def forward(self, x, cat_ids):
        # 根据cat_ids选择对应的权重
        selected_W = self.W[cat_ids]  # [B, input_dim, hidden_dim]
        selected_b = self.b[cat_ids]  # [B, hidden_dim]
        
        # 批量矩阵乘法
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class MultiEmbodimentActionEncoder(nn.Module):
    """
    多embodiment版本的Action Encoder
    所有线性层都替换为CategorySpecificLinear
    """
    def __init__(self, action_dim, hidden_size, num_embodiments):
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2*hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
```

### 3.4 DiT (Diffusion Transformer)

**文件**: [cross_attention_dit.py](lda/model/modules/action_model/flow_matching_head/cross_attention_dit.py)

#### 架构细节
```python
class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion Transformer with Cross-Attention
    
    配置示例 (DiT-B):
        - input_embedding_dim: 768
        - num_attention_heads: 12
        - attention_head_dim: 64
        - num_layers: 8
        - dropout: 0.2
        - norm_type: "ada_norm" (Adaptive Layer Norm)
        - cross_attention_dim: 2048 (VLM hidden size)
    """
    
    def __init__(self, 
                 input_embedding_dim=768,
                 num_attention_heads=12,
                 attention_head_dim=64,
                 num_layers=8,
                 cross_attention_dim=2048,
                 output_dim=None,
                 dropout=0.2,
                 norm_type="ada_norm"):
        
        # 1. 时间步编码器
        self.time_proj = Timesteps(num_channels=256)
        self.timestep_embedder = TimestepEmbedding(256, input_embedding_dim)
        
        # 2. Transformer Layers
        self.layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=input_embedding_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,  # 接收VLM特征
                norm_type=norm_type,  # AdaNorm支持时间条件
            ) for _ in range(num_layers)
        ])
        
        # 3. 最终投影
        self.proj_out = nn.Linear(input_embedding_dim, output_dim or input_embedding_dim)
```

#### BasicTransformerBlock结构
```python
class BasicTransformerBlock(nn.Module):
    """
    单个DiT层
    
    结构:
    input → Norm1 → Cross-Attention(with VLM) → Residual → Norm3 → FFN → Residual → output
              ↑                                                      ↑
         AdaNorm(timestep)                                      LayerNorm
    """
    
    def forward(self, hidden_states, encoder_hidden_states, temb):
        # 1. 自适应LayerNorm (条件于时间步)
        if self.norm_type == "ada_norm":
            norm_hidden = self.norm1(hidden_states, temb)
        else:
            norm_hidden = self.norm1(hidden_states)
        
        # 2. Cross-Attention (查询VLM特征)
        attn_output = self.attn1(
            norm_hidden,
            encoder_hidden_states=encoder_hidden_states,  # VLM hidden states
            attention_mask=encoder_attention_mask
        )
        hidden_states = attn_output + hidden_states  # 残差连接
        
        # 3. Feed-Forward Network
        norm_hidden = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden)
        hidden_states = ff_output + hidden_states  # 残差连接
        
        return hidden_states
```

#### AdaLayerNorm (自适应层归一化)
```python
class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep embedding
    
    公式:
        output = LayerNorm(x) * (1 + scale(temb)) + shift(temb)
    
    作用: 让归一化参数随时间步变化，增强模型对噪声水平的感知
    """
    def forward(self, x, temb):
        # 从时间嵌入生成scale和shift
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        
        # 自适应归一化
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x
```

### 3.5 参数量统计 (基于默认配置)

| 模块 | 参数量估计 | 是否可训练 |
|------|-----------|-----------|
| Qwen2.5-VL-3B | ~3B | ❌ 冻结 (默认) |
| DiT (DiT-B, 8层) | ~80M | ✅ |
| Action Encoder | ~2M | ✅ |
| Action Decoder | ~2M | ✅ |
| State Encoder | ~7M | ✅ |
| Future Tokens | ~25K | ✅ |
| Position Embedding | ~786K | ✅ |
| **总可训练参数** | **~91M** | - |

---

## 4. 模型处理流程

### 4.1 训练时前向传播

**文件**: [QwenGR00T.py](lda/model/framework/QwenGR00T.py) - `forward()`方法

```
完整流程图:

Input Batch (examples):
├── images: [B, N_views] PIL Images
├── instructions: [B] str
├── actions: [B, T_full=22, D_action=29]
├── state: [B, 1, D_state=58] (optional)
└── embodiment_ids: [B] int
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Qwen-VL Encoding                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1.1 build_qwenvl_inputs(images, instructions)              │
│      ├─ Format as chat messages                             │
│      ├─ Apply CoT template (optional)                       │
│      └─ Tokenize + Extract vision features                  │
│                                                             │
│  1.2 Qwen2.5-VL Forward                                    │
│      ├─ Visual Encoder: images → patch embeddings           │
│      ├─ Text Embedding: tokens → text embeddings            │
│      ├─ Multimodal Fusion: Transformer layers               │
│      └─ Output: last_hidden_states [B, L, H=2048]           │
│                                                             │
│  Precision: torch.autocast("cuda", dtype=torch.bfloat16)    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Prepare Action Targets                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  actions: [B, T_full=22, D=29]                              │
│      ↓                                                       │
│  actions_target = actions[:, -(future_window+1):, :]        │
│      = actions[:, -16:, :]  # 取最后16帧作为预测目标          │
│      Shape: [B, 16, 29]                                     │
│                                                             │
│  Repeated Diffusion Steps (数据增强)                         │
│  repeated_diffusion_steps = 4                               │
│  → Repeat batch 4 times for robustness                      │
│  actions_target: [4*B, 16, 29]                              │
│  last_hidden: [4*B, L, 2048]                                │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Flow Matching Training                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  3.1 Sample Noise                                           │
│      noise ~ N(0, I)                                        │
│      Shape: [4*B, 16, 29]                                   │
│                                                             │
│  3.2 Sample Time (from Beta Distribution)                   │
│      t ~ Beta(α=1.5, β=1.0)                                │
│      t_continuous ∈ (0, 1)                                  │
│      t_discrete = floor(t * 1000) ∈ {0, ..., 999}          │
│                                                             │
│  3.3 Construct Noisy Trajectory                             │
│      noisy_traj = (1-t) * noise + t * actions_target        │
│      velocity = actions_target - noise  # 目标速度场         │
│                                                             │
│  3.4 Encode Actions                                         │
│      action_features = ActionEncoder(noisy_traj, t_discrete)│
│      Shape: [4*B, 16, 768] (DiT-B)                          │
│      [+ Position Embedding if enabled]                      │
│                                                             │
│  3.5 Encode State (if provided)                             │
│      state_features = StateEncoder(state)                    │
│      Shape: [4*B, 1, 768]                                   │
│                                                             │
│  3.6 Build Sequence for DiT                                 │
│      sa_embs = concat([                                    │
│          state_features,      # [4*B, 1, 768]               │
│          future_tokens,       # [4*B, 32, 768] 可学习       │
│          action_features      # [4*B, 16, 768]               │
│      ], dim=1)                                             │
│      Shape: [4*B, 49, 768]                                  │
│                                                             │
│  3.7 DiT Forward Pass                                       │
│      model_output = DiT(                                    │
│          hidden_states=sa_embs,       # Query               │
│          encoder_hidden_states=vl_embs, # Key/Value from VLM│
│          timestep=t_discretized,                           │
│          encoder_attention_mask=attention_mask              │
│      )                                                     │
│      Shape: [4*B, 49, 768]                                  │
│                                                             │
│  3.8 Decode to Action Space                                 │
│      pred_actions = ActionDecoder(model_output[:, -16:, :]) │
│      Shape: [4*B, 16, 29]                                   │
│                                                             │
│  Precision: torch.autocast("cuda", dtype=torch.float32)     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Compute Loss                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  loss = MSE(pred_actions, velocity)                         │
│      = mean((pred_actions - velocity)²)                     │
│      = mean((pred_actions - (actions_target - noise))²)     │
│                                                             │
│  Return: {"loss": loss}                                     │
└─────────────────────────────────────────────────────────────┘
```

#### 代码实现
```python
def forward(self, examples, **kwargs):
    # ========== Step 1: Qwen-VL Encoding ==========
    batch_images = [ex["image"] for ex in examples]
    instructions = [ex["lang"] for ex in examples]
    
    qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images, 
        instructions=instructions
    )
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, ...)
        last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, 2048]
    
    # ========== Step 2: Prepare Targets ==========
    actions = torch.tensor(np.array(actions))  # [B, 22, 29]
    actions_target = actions[:, -(future_window+1):, :]  # [B, 16, 29]
    
    # 重复扩散步骤（数据增强）
    repeated_steps = 4
    actions_target = actions_target.repeat(repeated_steps, 1, 1)
    last_hidden = last_hidden.repeat(repeated_steps, 1, 1)
    
    # ========== Step 3: Flow Matching ==========
    with torch.autocast("cuda", dtype=torch.float32):
        loss_dict = self.action_model(
            vl_embs=last_hidden,
            actions=actions_target,
            state=state_repeated,
            encoder_attention_mask=attention_mask_repeated,
            embodiment_id=embodiment_ids_repeated
        )
    
    return loss_dict
```

### 4.2 推理时动作生成

**文件**: [QwenGR00T.py](lda/model/framework/QwenGR00T.py) - `predict_action()`方法

```
推理流程图:

Input: 
├── images: [B, N_views] PIL Images
├── instruction: str
├── state: [B, 1, D_state] (optional)
└── embodiment_id: int
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Encode Context (same as training)                   │
├─────────────────────────────────────────────────────────────┤
│  Qwen-VL Forward → last_hidden [B, L, 2048]                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Initialize Action Trajectory                        │
├─────────────────────────────────────────────────────────────┤
│  actions = randn(B, action_horizon=16, action_dim=29)       │
│  (从纯高斯噪声开始)                                          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Iterative Denoising Loop                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for t in range(num_inference_timesteps=4):                 │
│      │                                                      │
│      ├─ t_cont = t / 4  # 连续时间 [0, 0.25, 0.5, 0.75]    │
│      ├─ t_disc = int(t_cont * 1000)  # 离散时间步           │
│      │                                                      │
│      ├─ Encode current actions                              │
│      │   action_features = ActionEncoder(actions, t_disc)   │
│      │                                                      │
│      ├─ DiT Forward                                         │
│      │   model_output = DiT(sa_embs, vl_embs, t_disc)       │
│      │                                                      │
│      ├─ Decode predicted velocity                           │
│      │   pred_velocity = ActionDecoder(model_output)         │
│      │                                                      │
│      └─ Euler Integration Update                            │
│          dt = 1.0 / 4 = 0.25                               │
│          actions = actions + dt * pred_velocity             │
│          (逐步去噪)                                          │
│                                                             │
│  End Loop                                                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Output                                              │
├─────────────────────────────────────────────────────────────┤
│  Return: normalized_actions [B, 16, 29]                      │
│  (后续可反归一化为实际动作空间)                                │
└─────────────────────────────────────────────────────────────┘
```

#### 代码实现
```python
@torch.inference_mode()
def predict_action(self, examples, **kwargs):
    # Step 1: 编码上下文 (与训练相同)
    qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(...)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, ...)
        last_hidden = qwenvl_outputs.hidden_states[-1]
    
    # Step 2: 初始化随机轨迹
    batch_size = last_hidden.shape[0]
    actions = torch.randn(batch_size, self.action_horizon, self.action_dim)
    
    # Step 3: 迭代去噪
    num_steps = self.num_inference_timesteps  # 4
    dt = 1.0 / num_steps  # 0.25
    
    for t in range(num_steps):
        t_cont = t / float(num_steps)
        t_discretized = int(t_cont * self.num_timestep_buckets)
        
        # 编码当前动作
        action_features = self.action_encoder(actions, t_discretized)
        
        # DiT前向传播
        sa_embs = concat([future_tokens, action_features])
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=last_hidden,
            timestep=t_discretized
        )
        
        # 解码预测速度场
        pred_velocity = self.action_decoder(model_output[:, -self.action_horizon:])
        
        # Euler积分更新
        actions = actions + dt * pred_velocity
    
    # Step 4: 返回结果
    return actions.detach().cpu().numpy()
```

### 4.3 反归一化 (推理后处理)

**文件**: [base_framework.py](lda/model/framework/base_framework.py)

```python
@staticmethod
def unnormalize_actions(normalized_actions, action_norm_stats):
    """
    将归一化后的动作恢复到原始空间
    
    输入:
        normalized_actions: [T, D] 范围在[-1, 1]
        action_norm_stats: 包含q01, q99, mask的统计信息
    
    步骤:
        1. Clip到[-1, 1]
        2. 特殊处理第6维 (gripper): 二值化 {0, 1}
        3. 线性反归一化: original = 0.5*(norm+1)*(q99-q01) + q01
        4. 对mask=False的维度保持原值
    """
    mask = action_norm_stats.get("mask", ...)
    action_high = action_norm_stats["q99"]
    action_low = action_norm_stats["q01"]
    
    # Clip
    normalized_actions = np.clip(normalized_actions, -1, 1)
    
    # Gripper二值化 (假设第6维是gripper)
    normalized_actions[:, 6] = np.where(
        normalized_actions[:, 6] < 0.5, 0, 1
    )
    
    # 反归一化
    actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low
    
    # 应用mask (某些维度可能不需要反归一化)
    actions = np.where(mask, actions, normalized_actions)
    
    return actions
```

---

## 5. Loss设计

### 5.1 Flow Matching Loss (核心损失函数)

#### 数学原理

Flow Matching是一种生成建模方法，通过学习向量场（velocity field）来将噪声分布转换为数据分布。

**目标**: 学习一个速度场 $v(x_t, t)$，使得：
$$\frac{dx_t}{dt} = v(x_t, t)$$

其中 $x_t$ 是时刻 $t$ 的状态。

#### 训练目标

对于给定的数据样本 $x_1$ (clean action)，构造插值轨迹：

$$x_t = (1-t) \cdot \epsilon + t \cdot x_1$$

其中：
- $\epsilon \sim \mathcal{N}(0, I)$ 是标准高斯噪声
- $t \sim \text{Beta}(\alpha, \beta)$ 是时间参数
- $x_t$ 是 $t$ 时刻的插值状态

**目标速度场** (ground truth velocity):

$$v_{target} = x_1 - \epsilon = \frac{dx_t}{dt}$$

**Loss函数**:

$$\mathcal{L} = \mathbb{E}_{t, \epsilon, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - \epsilon) \|^2 \right]$$

即 **均方误差(MSE)** between predicted velocity and target velocity。

#### 代码实现

```python
# GR00T_ActionHeader.py - forward()方法

def forward(self, vl_embs, actions, state=None, 
            encoder_attention_mask=None, embodiment_id=None):
    """
    Flow Matching Loss计算
    
    Args:
        vl_embs: [B, L, H] VLM特征
        actions: [B, T, D] 目标动作序列
        state: [B, 1, D_state] 当前状态 (可选)
    """
    device = vl_embs.device
    batch_size = actions.shape[0]
    
    # === 1. 采样噪声 ===
    noise = torch.randn_like(actions)  # [B, T, D]
    
    # === 2. 采样时间 (Beta分布) ===
    t = self.sample_time(batch_size, device, actions.dtype)
    # t shape: [B, 1, 1] 用于广播
    
    # === 3. 构造noisy trajectory ===
    noisy_trajectory = (1 - t) * noise + t * actions
    # 线性插值: 在noise和clean action之间
    
    # === 4. 计算target velocity ===
    velocity = actions - noise
    # 这是我们希望网络学习的目标
    
    # === 5. 离散化时间步 ===
    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    # 范围: [0, 999]
    
    # === 6. 编码 ===
    action_features = self.action_encoder(noisy_trajectory, t_discretized)
    # 加入position embedding (如果启用)
    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1])
        pos_embs = self.position_embedding(pos_ids)
        action_features = action_features + pos_embs
    
    # === 7. 构建完整序列 ===
    future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
    sa_embs = concat([future_tokens, action_features], dim=1)
    # [B, 32+T, hidden_dim]
    
    # === 8. DiT前向传播 ===
    model_output = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embs,
        encoder_attention_mask=encoder_attention_mask,
        timestep=t_discretized
    )
    # [B, 32+T, hidden_dim]
    
    # === 9. 解码 ===
    pred = model_output[:, -actions.shape[1]:]  # 取最后T个位置
    pred_actions = self.action_decoder(pred)
    # [B, T, D]
    
    # === 10. 计算Loss ===
    loss = ((pred_actions - velocity) ** 2).mean()
    # MSE Loss
    
    return {"loss": loss}
```

### 5.2 时间采样策略

```python
def sample_time(self, batch_size, device, dtype):
    """
    从Beta分布采样时间步
    
    分布: Beta(α=1.5, β=1.0)
    
    特点:
    - α > β: 分布偏向较晚的时间步 (接近t=1)
    - 这意味着更多训练样本接近clean data
    - 有助于学习数据分布的精细结构
    
    转换: t_sampled = (s - sample) / s
    其中 s = 0.999 (接近1)
    
    结果范围: t ∈ (0, 0.001) ≈ (0, 1)
    """
    sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
    # Beta(1.5, 1.0) 采样
    return (self.config.noise_s - sample) / self.config.noise_s
    # noise_s = 0.999
```

**Beta分布特性**:

| 参数 | 值 | 影响 |
|------|-----|------|
| alpha (α) | 1.5 | > 1, 分布偏向右侧 |
| beta (β) | 1.0 | = 1, 右偏分布 |
| noise_s | 0.999 | 缩放因子，接近1 |

**效果**: 更多样本集中在 $t \approx 1$ (接近clean data)，有助于学习精确的细节。

### 5.3 重复扩散步骤 (Repeated Diffusion Steps)

```python
# QwenGR00T.py - forward()方法中

repeated_diffusion_steps = 4  # 默认值

# 将整个batch重复4次
actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

# 每次重复会采样不同的噪声和时间步
# 相当于对同一样本做4次不同的flow matching训练
```

**目的**:
- **数据增强**: 同一batch的数据被使用多次，每次有不同的噪声/时间
- **稳定训练**: 增加有效batch size，降低梯度方差
- **效率**: 不需要额外的前向传播计算VLM特征

### 5.4 Loss配置与优化

**配置文件中的相关参数** ([LDA_pretrain.yaml](lda/config/training/LDA_pretrain.yaml)):

```yaml
trainer:
  # 学习率设置 (不同模块可不同)
  learning_rate:
    base: 1e-5              # 基础学习率
    qwen_vl_interface: 1e-5 # VLM编码器 (通常冻结或极低lr)
    action_model: 1e-4      # Action Head (主要训练目标)
  
  # 优化器
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]      # AdamW常用设置
    eps: 1e-8
    weight_decay: 1e-8       # 极小的weight decay
  
  # 学习率调度
  lr_scheduler_type: cosine_with_min_lr  # 余弦退火
  scheduler_specific_kwargs:
    min_lr: 5e-7            # 最小学习率
  
  # 梯度裁剪
  max_grad_norm: 1.0        # 防止梯度爆炸
  
  # 混合精度
  enable_mixed_precision_training: true  # 使用BF16/FP16
  
  # 梯度累积
  gradient_accumulation_steps: 1
  
  # Warmup
  num_warmup_steps: 5000    # 线性warmup
  warmup_ratio: 0.1
```

### 5.5 Loss计算流程总结

```
完整的Loss计算流程:

1. 输入准备
   └─ actions_target: [4B, 16, 29] (重复4次的未来动作)

2. 噪声注入
   ├─ noise ~ N(0,I): [4B, 16, 29]
   └─ t ~ Beta(1.5, 1.0): [4B]

3. 插值构造
   ├─ noisy_traj = (1-t)*noise + t*actions: [4B, 16, 29]
   └─ velocity = actions - noise: [4B, 16, 29] (目标)

4. 特征提取
   ├─ action_features = ActionEncoder(noisy_traj, t): [4B, 16, 768]
   ├─ future_tokens: [4B, 32, 768] (可学习)
   └─ sa_embs = concat(future_tokens, action_features): [4B, 48, 768]

5. 条件生成 (Cross-Attention)
   ├─ Query: sa_embs (action sequence)
   ├─ Key/Value: vl_embs (VLM features) [4B, L, 2048]
   └─ model_output = DiT(sa_embs, vl_embs, t): [4B, 48, 768]

6. 解码
   └─ pred_actions = Decoder(model_output[:, -16:]): [4B, 16, 29]

7. Loss计算
   └─ Loss = MSE(pred_actions, velocity).mean()
       = ||pred_actions - (actions - noise)||² .mean()

8. 反向传播
   └─ Loss.backward() → 更新Action Head参数
      (VLM参数冻结或不更新)
```

### 5.6 Loss的特性分析

#### 优点
1. **简单高效**: 只需MSE loss，无需对抗训练或复杂的匹配目标
2. **稳定训练**: Flow Matching比传统DDPM更稳定的训练动态
3. **灵活采样**: 推理时可调整采样步数平衡质量/速度
4. **多模态友好**: 通过cross-attention自然融合VLM特征

#### 与其他方法的对比

| 方法 | Loss类型 | 采样步数 | 训练稳定性 | 生成质量 |
|------|---------|---------|-----------|---------|
| **DDPM** | 简化ELBO | 1000+ | 中等 | 高 |
| **Flow Matching** | **MSE Velocity** | **4-16** | **高** | **高** ✓ |
| Diffusion Policy | MSE Prediction | 100 | 高 | 高 |
| ACT (Action Chunking) | MSE Direct | 1 | 很高 | 中 |

**LDA-1B选择Flow Matching的原因**:
- ✅ 仅需4步即可高质量采样 (快速推理)
- ✅ 训练稳定 (MSE loss简单鲁棒)
- ✅ 与VLM特征自然结合 (cross-attention)
- ✅ 支持长序列动作预测 (chunk-based)

---

## 6. 配置参数说明

### 6.1 核心超参数

#### Framework配置
```yaml
framework:
  name: "QwenGR00T"                    # 框架名称
  
  qwenvl:
    base_vlm: "./Qwen2.5-VL-3B-Instruct"  # VLM模型路径
    vl_hidden_dim: 2048                 # VLM隐藏维度
  
  action_model:
    action_model_type: "DiT-B"          # DiT变体: DiT-B 或 DiT-L
    hidden_size: 2560                   # Action Head隐藏大小
    action_dim: 29                      # 动作维度 (根据任务)
    state_dim: 58                       # 状态维度 (可选)
    
    # 时间窗口
    future_action_window_size: 15       # 未来动作窗口 (不含当前步)
    past_action_window_size: 0          # 历史动作窗口
    action_horizon: 16                  # 总预测长度 = future + 1
    
    # Flow Matching参数
    repeated_diffusion_steps: 4         # 训练时重复次数
    noise_beta_alpha: 1.5               # Beta分布alpha
    noise_beta_beta: 1.0                # Beta分布beta
    noise_s: 0.999                      # 缩放因子
    num_timestep_buckets: 1000          # 离散时间桶数
    num_inference_timesteps: 4          # 推理去噪步数
    
    # DiT配置
    diffusion_model_cfg:
      cross_attention_dim: 2048         # Cross-attention维度 (=VLM dim)
      num_layers: 8                     # DiT层数
      dropout: 0.2                      # Dropout率
      norm_type: "ada_norm"             # 归一化类型
      
    # 其他
    add_pos_embed: True                 # 是否添加位置编码
    max_seq_len: 1024                   # 最大序列长度
    num_target_vision_tokens: 32        # 可学习query token数
```

#### Dataset配置
```yaml
datasets:
  vla_data:
    dataset_py: "lerobot_datasets"      # 数据集类
    data_root_dir: "/path/to/data"      # 数据根目录
    data_mix: "demo_data"               # 数据混合名称
    
    # 图像设置
    default_image_resolution: [3, 224, 224]  # CHW
    image_size: [224, 224]              # HW
    per_device_batch_size: 16           # Batch size
    
    # 动作设置
    action_type: "delta_ee"             # 动作类型: delta end-effector
    obs: ["image_0"]                    # 观察模态
    
    # CoT (Chain-of-Thought)
    CoT_prompt: "Your task is {instruction}..."  # CoT提示模板
```

#### Trainer配置
```yaml
trainer:
  epochs: 100
  max_train_steps: 100000              # 最大训练步数
  save_interval: 5000                   # 保存间隔
  eval_interval: 100                    # 评估间隔
  
  # 学习率
  learning_rate:
    base: 1e-5
    action_model: 1e-4                  # Action Head使用较高lr
  
  # 优化器
  optimizer:
    name: "AdamW"
    betas: [0.9, 0.95]
    weight_decay: 1e-8
  
  # 调度器
  lr_scheduler_type: "cosine_with_min_lr"
  scheduler_specific_kwargs:
    min_lr: 5e-7
  
  # 训练技巧
  freeze_modules: ''                    # 冻结模块 (空=不冻结)
  enable_gradient_checkpointing: true   # 梯度检查点 (省显存)
  enable_mixed_precision_training: true # 混合精度训练
  gradient_accumulation_steps: 1        # 梯度累积
  max_grad_norm: 1.0                    # 梯度裁剪
```

### 6.2 关键维度关系

```
数据流维度变化:

Input:
  Images: [B, N_views] PIL Images → Processor → [B, num_patches, C, H', W']
  Text: [B] str → Tokenizer → [B, L_text]
  Actions: [B, T_full=22, D_act=29]
  State: [B, 1, D_state=58] (可选)

VLM Output:
  last_hidden: [B, L_vlm, H=2048]

Action Head Internal:
  Action Encoder: [B, T=16, D=29] → [B, T, D_embed=768]  (DiT-B)
  Future Tokens: [B, 32, D_embed=768]
  State Encoder: [B, 1, D_state=58] → [B, 1, D_embed=768]
  
  DiT Input (concat): [B, 49 (or 50), D_embed=768]
  DiT Output: [B, 49 (or 50), D_out=2560]
  
  Action Decoder: [B, 16, D_out=2560] → [B, 16, D_act=29]

Final Output:
  Predicted Actions: [B, 16, 29]
```

### 6.3 训练资源估算

基于默认配置 (DiT-B, batch_size=16, 4 GPUs):

| 项目 | 估计值 |
|------|--------|
| **显存占用** | ~24GB/GPU (BF16 + gradient checkpointing) |
| **训练时间** | ~24-48小时 (100K steps) |
| **吞吐量** | ~ samples/sec (取决于sequence length) |
| **Checkpoint大小** | ~400MB (仅可训练参数) |

---

## 附录A: 关键文件索引

| 文件路径 | 功能描述 |
|---------|---------|
| [train_LDA.py](lda/training/train_LDA.py) | 主训练脚本 |
| [QwenGR00T.py](lda/model/framework/QwenGR00T.py) | 主模型框架 |
| [base_framework.py](lda/model/framework/base_framework.py) | 基础框架类 |
| [GR00T_ActionHeader.py](lda/model/modules/action_model/GR00T_ActionHeader.py) | Flow Matching Action Head |
| [cross_attention_dit.py](lda/model/modules/action_model/flow_matching_head/cross_attention_dit.py) | DiT实现 |
| [QWen2_5.py](lda/model/modules/vlm/QWen2_5.py) | Qwen2.5-VL接口 |
| [datasets.py](lda/dataloader/gr00t_lerobot/datasets.py) | 数据集类 |
| [data_config.py](lda/dataloader/gr00t_lerobot/data_config.py) | 数据配置 |
| [state_action.py](lda/dataloader/gr00t_lerobot/transform/state_action.py) | 状态/动作变换 |
| [video.py](lda/dataloader/gr00t_lerobot/transform/video.py) | 视频变换 |
| [LDA_pretrain.yaml](lda/config/training/LDA_pretrain.yaml) | 训练配置示例 |

---

## 附录B: 参考资料

1. **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
2. **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
3. **GR00T**: NVIDIA GR00T Project (N1.5)
4. **Qwen2.5-VL**: [Qwen2.5-VL Technical Report](https://qwenlm.github.io/blog/qwen2.5-vl/)
5. **LeRobot**: [LeRobot Dataset Format](https://huggingface.co/docs/lerobot/v0.40.0/en/concept_datasets)

## 📋 4个任务的完整定义

```python
# 文件: MMDiT_ActionHeader_rope_embedding.py 第32行
TRAINING_TASKS = ["policy", "forward_dynamics", "inverse_dynamics", "video_gen"]
```

| 任务名称                          | 数学形式 | 物理含义                           | 输入→输出                    |
| --------------------------------- | -------- | ---------------------------------- | ---------------------------- |
| **policy (策略)**                 | o → a    | 知道"现在该做什么"                 | 当前观测 → 动作序列          |
| **forward_dynamics (前向动力学)** | o+a → o' | 知道"做了这个动作会发生什么"       | 当前观测+动作 → 未来观测     |
| **inverse_dynamics (逆动力学)**   | o+o' → a | 知道"要达到那个状态，该做什么动作" | 当前观测+未来观测 → 动作序列 |
| **video_gen (视觉规划)**          | o → o'   | 知道"世界接下来会怎么变化"         | 当前观测 → 未来观测          |

---

## 🔧 机制一：数据层的任务分配

### 核心组件：DistributedTaskBatchSampler

**文件位置**: [task_batch_sampler.py](lda/dataloader/task_batch_sampler.py)

```python
class DistributedTaskBatchSampler(torch.utils.data.Sampler):
    """
    Multi-task batch sampler with:
    - hard constraint: 每个batch至少包含每个任务1个样本
    - soft constraint: 剩余slot按task_weights采样
    """
    
    def __iter__(self) -> Iterator[List[Tuple[int, str]]]:
        for _ in range(len(self)):
            batch: List[Tuple[int, str]] = []
            
            # === Hard Constraint: 保证每个任务至少有1个样本 ===
            for task in self.tasks:  # ["policy", "forward_dynamics", "inverse_dynamics", "video_gen"]
                batch.append((self._next_index(task), task))  # 返回 (index, task_name)
            
            # === Soft Constraint: 剩余位置按权重随机分配 ===
            while len(batch) < self.batch_size:
                task = self._sample_task_by_weight()  # 根据task_weights随机选
                batch.append((self._next_index(task), task))
            
            self.batch_rng.shuffle(batch)  # 打乱顺序
            yield batch
```

### 配置文件中的权重设置

**文件**: [LDA_pretrain.yaml](lda/config/training/LDA_pretrain.yaml#L84)

```yaml
training_task_weights: [1.0, 1.0, 1.0, 1.0]  # [policy, forward_dynamics, inverse_dynamics, video_gen]
```

**含义**：

- 默认等权重（各25%）
- 可以调整比例，如 `[2.0, 1.0, 1.0, 0.5]` 表示策略任务占40%

### 数据流示例

```
Batch Size = 8, 4个任务

Step 1: Hard Constraint (必须)
┌─────────────────────────────────────────────┐
│ (index=100, task="policy")                  │
│ (index=200, task="forward_dynamics")        │
│ (index=300, task="inverse_dynamics")        │
│ (index=400, task="video_gen")               │
└─────────────────────────────────────────────┘

Step 2: Soft Constraint (按权重填充剩余4个slot)
┌─────────────────────────────────────────────┐
│ (index=150, task="policy")       ← 权重大   │
│ (index=250, task="inverse_dynamics")         │
│ (index=350, task="forward_dynamics")         │
│ (index=450, task="video_gen")                │
└─────────────────────────────────────────────┘

Step 3: Shuffle (打乱顺序)
Batch = [
    (250, "inverse_dynamics"),
    (100, "policy"),
    (450, "video_gen"),
    (300, "inverse_dynamics"),  ← 注意：可以有重复！
    (200, "forward_dynamics"),
    (150, "policy"),
    (400, "video_gen"),
    (350, "forward_dynamics")
]

Step 4: Dataset.__getitem__() 接收 (index, task) 元组
返回 dict 中包含 assigned_task 字段
```

---

## 🔧 机制二：模型层的任务区分

### 核心：Task Embedding (4个可学习向量)

**文件位置**: [MMDiT_ActionHeader_rope_embedding.py:582-585](lda/model/modules/action_model/MMDiT_ActionHeader_rope_embedding.py#L582-L585)

```python
class MMDiTActionHeader(nn.Module):
    def __init__(self, config):
        # ... 其他初始化 ...
        
        # 🔑 关键：4个独立的Task Embedding向量
        self.policy_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))      # 策略
        self.fd_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))           # 前向动力学
        self.vg_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))           # 视频生成
        self.id_embedding = nn.Parameter(0.02 * torch.randn(self.inner_dim))           # 逆动力学
        
        # Policy任务专用的learnable tokens（替代真实next_obs）
        self.next_obs_learnable_tokens = nn.Parameter(0.02 * torch.randn(num_chans))
```

### Forward函数中的任务路由逻辑

**文件位置**: [MMDiT_ActionHeader_rope_embedding.py:665-824](lda/model/modules/action_model/MMDiT_ActionHeader_rope_embedding.py#L665-L824)

#### Step 1: 根据 assigned_tasks 分组索引

```python
def forward(self, vl_embs, actions, ..., assigned_tasks):
    # 初始化4个列表
    policy_indices = []           # 存储policy任务的样本索引
    forward_dynamics_indices = [] # 存储FD任务的样本索引
    inverse_dynamics_indices = [] # 存储ID任务的样本索引
    video_gen_indices = []        # 存储VG任务的样本索引
    
    # 遍历batch中每个样本
    for i in range(B):
        task = assigned_tasks[i]  # 从sampler传入的任务标签
        
        if task == "policy":
            policy_indices.append(i)
        elif task == "forward_dynamics":
            forward_dynamics_indices.append(i)
        elif task == "inverse_dynamics":
            inverse_dynamics_indices.append(i)
        elif task == "video_gen":
            video_gen_indices.append(i)
```

**示例输出**：

```python
# 假设B=8，assigned_tasks = ["id", "policy", "vg", "id", "fd", "policy", "vg", "fd"]

policy_indices           = [1, 5]          # 2个策略样本
forward_dynamics_indices = [4, 7]          # 2个前向动力学样本
inverse_dynamics_indices = [0, 3]          # 2个逆动力学样本
video_gen_indices        = [2, 6]          # 2个视频生成样本
```

#### Step 2: 为每组分配对应的Task Embedding

```python
    # 为每组的样本扩展对应的embedding
    policy_embedding = self.policy_embedding.unsqueeze(0).expand(len(policy_indices), -1)           # [2, D]
    fd_embedding = self.fd_embedding.unsqueeze(0).expand(len(forward_dynamics_indices), -1)        # [2, D]
    vg_embedding = self.vg_embedding.unsqueeze(0).expand(len(video_gen_indices), -1)              # [2, D]
    id_embedding = self.id_embedding.unsqueeze(0).expand(len(inverse_dynamics_indices), -1)       # [2, D]
```

#### Step 3: 构建输入差异（关键！）

这是**最核心的部分**——不同任务使用**不同的输入组合**：

```python
    # ════════════════════════════════════════════════
    # 任务分组：预测动作 vs 预测观测
    # ════════════════════════════════════════════════
    pred_action_task_indices = policy_indices + inverse_dynamics_indices     # [0,3,1,5]
    pred_next_obs_task_indices = forward_dynamics_indices + video_gen_indices # [4,7,2,6]
    
    # ──── Policy & Inverse Dynamics: 需要预测动作 ────
    # 提取对应的actions
    policy_action = actions[policy_indices]           # [2, T, action_dim]
    inverse_action = actions[inverse_dynamics_indices] # [2, T, action_dim]
    to_noise_action = torch.cat((policy_action, inverse_action), dim=0)  # [4, T, action_dim]
    
    # 注入噪声（Flow Matching标准操作）
    act_t_sample = self.sample_time(to_noise_action.shape[0], ...)
    action_noise = torch.randn_like(to_noise_action)
    noisy_action = (1 - act_t_sample) * action_noise + act_t_sample * to_noise_action
    
    # 编码noisy actions
    noisy_act_feat = self.action_encoder(noisy_action, act_t_discretized, embodiment_id[pred_action_task_indices])
    
    # ⚠️ Policy vs Inverse Dynamics的关键区别：
    # - Policy: 使用 learnable next_obs tokens（占位符）
    policy_obs_feat = self.next_obs_learnable_tokens.expand(len(policy_indices), num_obs_tokens, -1)
    
    # - Inverse Dynamics: 使用真实的GT next_obs
    inv_obs_feat = next_obs[inverse_dynamics_indices]  # 从future_imgs编码得到
    
    # ──── Forward Dynamics & Video Gen: 需要预测观测 ────
    forward_obs = next_obs[forward_dynamics_indices]    # 真实GT next_obs
    video_gen_obs = next_obs[video_gen_indices]         # 真实GT next_obs
    to_noise_next_obs = torch.cat((forward_obs, video_gen_obs), dim=0)
    
    # 注入噪声到next_obs
    obs_t_sample = self.sample_time(...)
    obs_noise = torch.randn_like(to_noise_next_obs)
    noisy_obs = (1 - obs_t) * obs_noise + obs_t * to_noise_next_obs
    
    # ⚠️ FD vs Video Gen的关键区别：
    # - FD: 使用真实的GT actions（clean，不加噪声）
    t_clean = torch.ones(len(forward_dynamics_indices), ...)
    forward_act_feat = self.action_encoder(actions[forward_dynamics_indices], t_clean, ...)  # Clean!
    
    # - Video Gen: 使用 learnable action tokens（占位符）
    video_gen_act_feat = self.action_learnable_tokens.weight.expand(len(video_gen_indices), -1, -1)
```

#### Step 4: 拼接所有输入送入DiT

```python
    # 按固定顺序拼接（重要！）
    action_features = torch.cat((
        noisy_act_feat,           # [4, T_a, D]  ← Policy + ID的noisy actions
        forward_act_feat,         # [2, T_a, D]  ← FD的clean actions
        video_gen_act_feat        # [2, T_a, D]  ← VG的learnable actions
    ), dim=0)                     # 总计 [8, T_a, D]
    
    noisy_next_obs = torch.cat((
        policy_obs_feat,          # [2, N_obs, D]  ← learnable tokens
        inv_obs_feat,             # [2, N_obs, D]  ← GT next_obs (ID用)
        noisy_obs                 # [4, N_obs, D]  ← Noisy next_obs (FD+VG用)
    ), dim=0)                     # 总计 [8, N_obs, D]
    
    # Task Embedding也按相同顺序拼接
    task_embedding = torch.cat((
        policy_embedding,         # [2, D]
        id_embedding,             # [2, D]
        fd_embedding,             # [2, D]
        vg_embedding              # [2, D]
    ), dim=0)                     # 总计 [8, D]
    
    # 送入DiT Transformer
    image_tokens, action_tokens = self.model(
        image_tokens=obs_tokens,
        action_tokens=action_features,
        text_tokens=vl_embs,
        time_cond=diffusion_t,
        task_embedding=task_embedding,  # 🔑 告诉DiT当前是什么任务
    )
```

#### Step 5: 分别计算Loss

```python
    # 解码action tokens
    pred_actions = self.action_decoder(action_tokens, embodiment_id)
    pred_actions = pred_actions[:, -actions.shape[1]:]  # 取最后T个token
    
    total_loss = 0.0
    
    # ═══ Policy Loss ═══
    policy_pred = pred_actions[:len(policy_indices)]  # [2, T, action_dim]
    policy_loss = MSE(policy_pred, action_velocity[:len(policy_indices)]) * mask
    total_loss += policy_loss
    
    # ═══ Inverse Dynamics Loss ═══
    if not self.only_policy:
        inverse_pred = pred_actions[len(policy_indices):len(pred_action_task_indices)]
        inverse_loss = MSE(inverse_pred, action_velocity[len(policy_indices):]) * mask
        total_loss += inverse_loss  # 通常权重为0.5
    
    # ═══ Forward Dynamics / Video Gen Loss ═══
    if not self.only_policy and not self.only_wo_video_gen:
        pred_next_obs = self.obs_output_projector(image_tokens)  # 从image_tokens解码
        obs_loss = MSE(pred_next_obs, obs_velocity)
        total_loss += obs_loss  # 通常权重为0.1
    
    return {"loss": total_loss, "action_loss": policy_loss, "dynamics_loss": obs_loss}
```

---

## 🎯 四任务完整对比表

| 维度               | Policy             | Forward Dynamics         | Inverse Dynamics     | Video Gen                |
| ------------------ | ------------------ | ------------------------ | -------------------- | ------------------------ |
| **数学目标**       | o → a              | o+a → o'                 | o+o' → a             | o → o'                   |
| **输入Actions**    | ❌ 加噪 (Noisy)     | ✅ 干净 (Clean GT)        | ❌ 加噪 (Noisy)       | ❌ Learnable Tokens       |
| **输入Next Obs**   | ❌ Learnable Tokens | ❌ 加噪 (Noisy)           | ✅ 干净 (GT Next Obs) | ❌ 加噪 (Noisy)           |
| **输出解码器**     | Action Decoder     | **Obs Output Projector** | Action Decoder       | **Obs Output Projector** |
| **Loss目标**       | Action Velocity    | Obs Velocity             | Action Velocity      | Obs Velocity             |
| **Task Embedding** | `policy_embedding` | `fd_embedding`           | `id_embedding`       | `vg_embedding`           |
| **物理意义**       | "做什么？"         | "会怎样？"               | "怎么做？"           | "未来什么样？"           |

---

## 🚀 推理时的任务选择

### 推理方法1：Policy推理（预测动作）

**文件位置**: [MMDiT_ActionHeader_rope_embedding.py:920-1042](lda/model/modules/action_model/MMDiT_ActionHeader_rope_embedding.py#L920-L1042)

```python
@torch.no_grad()
def predict_action(self, vl_embs, state, history_actions, curr_imgs, embodiment_id, attention_mask):
    """
    推理时只做Policy任务！
    """
    B = vl_embs.shape[0]
    
    # 1️⃣ 编码当前观测
    curr_obs = self.encode_curr_obs(curr_imgs)
    
    # 2️⃣ 初始化纯噪声动作
    actions = torch.randn(B, action_horizon, action_dim)
    
    # 3️⃣ 🔑 关键：强制使用Policy Task Embedding
    task_embedding = self.policy_embedding.unsqueeze(0).expand(B, -1)  # 只有policy！
    
    # 4️⃣ Denoising循环
    for step in range(num_inference_steps):
        # 编码当前noisy actions
        action_features = self.action_encoder(actions, timesteps, embodiment_id)
        
        # 使用learnable next_obs tokens（和训练时一致）
        noisy_next_obs = self.next_obs_learnable_tokens.expand(B, num_obs_tokens, -1)
        
        # 送入DiT
        _, action_tokens = self.model(
            image_tokens=obs_tokens,
            action_tokens=action_features,
            text_tokens=vl_embs,
            task_embedding=task_embedding,  # 🔑 Policy embedding
        )
        
        # 解码得到velocity
        pred_velocity = self.action_decoder(action_tokens, embodiment_id)
        
        # Euler积分去噪
        actions = actions + dt * pred_velocity
    
    return actions  # 最终干净的动作序列
```

### 推理方法2：Video Generation推理（预测未来画面）

**文件位置**: [MMDiT_ActionHeader_rope_embedding.py:1044-end](lda/model/modules/action_model/MMDiT_ActionHeader_rope_embedding.py#L1044)

```python
@torch.no_grad()
def video_gen(self, vl_embs, state, history_actions, curr_imgs, embodiment_id, attention_mask):
    """
    推理时做Video Gen任务！
    """
    B = vl_embs.shape[0]
    
    # 1️⃣ 编码当前观测
    curr_obs = self.encode_curr_obs(curr_imgs)
    
    # 2️⃣ 初始化纯噪声的next_obs
    next_obs = torch.randn(B, num_obs_tokens, hidden_size)
    
    # 3️⃣ 🔑 关键：强制使用Video Gen Task Embedding
    task_embedding = self.vg_embedding.unsqueeze(0).expand(B, -1)  # 只有VG！
    
    # 4️⃣ Denoising循环（对next_obs去噪）
    for step in range(num_inference_steps):
        # 使用learnable action tokens（和训练时一致）
        action_features = self.action_learnable_tokens.expand(B, -1, -1)
        
        # 送入DiT
        image_tokens, _ = self.model(
            image_tokens=noisy_next_obs,
            action_tokens=action_features,
            text_tokens=vl_embs,
            task_embedding=task_embedding,  # 🔑 VG embedding
        )
        
        # 从image_tokens解码得到velocity
        pred_velocity = self.obs_output_projector(image_tokens)
        
        # Euler积分去噪
        next_obs = next_obs + dt * pred_velocity
    
    return next_obs  # 最终清晰的未来观测tokens
```

### 推理时的选择逻辑总结

```
用户需求                    调用方法              内部行为
─────────────────────────────────────────────────────────────
"给我动作指令"    →    model.predict_action()    →  task_emb=policy_embedding
                                                    输出: 动作序列

"预测未来画面"    →    model.video_gen()          →  task_emb=vg_embedding  
                                                    输出: 未来观测tokens

"物理仿真"        →    model.forward_dyn()        →  (未实现，但可以用VG近似)
                                                    
"逆向规划"        →    model.inverse_dyn()        →  (未实现，可用Policy近似)
```

---

## 🎨 完整架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        训练时 (Training)                                │
│                                                                         │
│  Batch (B=8)                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Sampler输出: [(idx, "policy"), (idx, "fd"), (idx, "id"), ...]  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ Policy (2样本)    │  │ FD (2样本)        │  │ ID (2样本)        │     │
│  │ Input:           │  │ Input:           │  │ Input:           │     │
│  │ • Noisy Action   │  │ • Clean Action   │  │ • Noisy Action   │     │
│  │ • Learnable Obs  │  │ • Noisy Next_Obs │  │ • GT Next_Obs    │     │
│  │ Emb: policy_emb  │  │ Emb: fd_emb      │  │ Emb: id_emb      │     │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘     │
│           │                     │                     │               │
│  ┌────────┴─────────┐  ┌────────┴─────────┐  ┌────────┴─────────┐     │
│  │ VG (2样本)        │                    │                     │     │
│  │ Input:           │                    │                     │     │
│  │ • Learnable Act  │                    │                     │     │
│  │ • Noisy Next_Obs │                    │                     │     │
│  │ Emb: vg_emb      │                    │                     │     │
│  └────────┬─────────┘                    │                     │     │
│           └──────────────────────────────┼─────────────────────┘     │
│                                          ↓                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              共享 DiT Transformer                               │   │
│  │  输入: [All Actions] + [All Observations] + [Text] + [Task_Emb] │   │
│  └────────────────────────┬───────────────────────────────────────┘   │
│                           ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              分别计算Loss                                       │   │
│  │  Policy_Loss + ID_Loss + FD_Loss + VG_Loss = Total_Loss        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        推理时 (Inference)                               │
│                                                                         │
│  场景A: 预测动作 (Policy)                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Input: Curr_Observation + Text_Prompt                          │   │
│  │  Task_Embedding: policy_embedding (固定!)                       │   │
│  │  Next_Obs: learnable_tokens (占位符)                             │   │
│  │  Loop: Denoise Actions from Noise → Clean                      │   │
│  │  Output: Action_Sequence [B, T, action_dim]                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  场景B: 视觉规划 (Video Gen)                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Input: Curr_Observation + Text_Prompt                          │   │
│  │  Task_Embedding: vg_embedding (固定!)                           │   │
│  │  Action: learnable_tokens (占位符)                               │   │
│  │  Loop: Denoise Next_Obs from Noise → Clear                     │   │
│  │  Output: Future_Observation_Tokens                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 设计哲学深度解读

### 为什么这4个任务构成完整的"具身世界模型"？

```
                    ┌─────────────┐
                    │   观测空间 O  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  Policy (o→a)   │       │ VideoGen (o→o') │
    │  "我该做什么？"  │       │ "世界会如何变？" │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  动作空间 A      │◄──────│  未来观测 O'     │
    └─────────────────┘       └─────────────────┘
             ▲                         │
             │                         │
    ┌────────┴────────┐       ┌────────┴────────┐
    │ InvDyn (o'+o→a) │       │ FwdDyn (o+a→o') │
    │ "怎么做到？"     │       │ "会发生什么？"   │
    └─────────────────┘       └─────────────────┘
```

**完整性证明**：

- **Policy**: Agent的主观决策能力
- **Forward Dynamics**: 客观物理规律建模
- **Inverse Dynamics**: 因果推理能力（从结果反推原因）
- **Video Gen**: 世界模型预测能力（无需知道具体动作）

### 为什么共享同一个DiT？

1. **知识迁移**：学习Forward Dynamics有助于理解物理规律，反过来提升Policy质量
2. **参数效率**：不需要维护4个独立的大模型
3. **统一表示**：所有任务都在同一个语义空间中操作

### Task Embedding的作用

```python
# DiT内部可能的使用方式（推测）:
class MMDiT:
    def forward(self, ..., task_embedding):
        # 将task_embedding注入到attention或FFN层
        # 让模型知道当前在解决什么类型的问题
        for layer in self.layers:
            x = layer(x, task_embedding=task_embedding)  # 条件生成
```

**本质**：Task Embedding类似于"题目类型标签"，告诉模型这道题是选择题还是填空题。

---

## 🔍 调试技巧

### 1. 查看实际的任务分布

```python
# 在训练循环中添加:
for batch in dataloader:
    tasks = batch['assigned_tasks']
    print(f"Task distribution: {Counter(tasks)}")
    # Output: Counter({'policy': 3, 'forward_dynamics': 2, 'inverse_dynamics': 2, 'video_gen': 1})
```

### 2. 验证Task Embedding是否在工作

```python
# 在forward中添加:
print(f"Task embedding norm: {task_embedding.norm(dim=-1)}")
print(f"Policy emb: {policy_embedding[:5].tolist()}")
print(f"FD emb: {fd_embedding[:5].tolist()}")
# 应该看到4组明显不同的向量值
```

### 3. 监控各个任务的Loss变化

```python
# TensorBoard/XView中分别记录:
writer.add_scalar('train/policy_loss', policy_act_loss.item(), step)
writer.add_scalar('train/inverse_loss', inverse_act_loss.item(), step)
writer.add_scalar('train/dynamics_loss', obs_loss.item(), step)
```

---

## 📚 相关代码文件清单

| 文件                                                         | 作用                               |
| ------------------------------------------------------------ | ---------------------------------- |
| [MMDiT_ActionHeader_rope_embedding.py](lda/model/modules/action_model/MMDiT_ActionHeader_rope_embedding.py) | 主模型，包含4任务定义和forward逻辑 |
| [task_batch_sampler.py](lda/dataloader/task_batch_sampler.py) | 数据层任务采样器                   |
| [datasets.py](lda/dataloader/gr00t_lerobot/datasets.py)      | 数据集，接收(index, task)元组      |
| [LDA_pretrain.yaml](lda/config/training/LDA_pretrain.yaml)   | 配置文件，定义任务权重             |
| [\_\_init\_\_.py](lda/dataloader/__init__.py)                | DataLoader构建入口                 |

---

## 🎯 总结

**LDA-1B的四任务机制是一个精巧的设计**：

1. **数据层**：通过`DistributedTaskBatchSampler`保证每个batch都有4种任务
2. **模型层**：通过4个独立的`Task Embedding`向量区分任务
3. **输入层**：每种任务使用**不同的输入组合**（noisy/clean/learnable）
4. **输出层**：通过不同的decoder（action decoder vs obs projector）计算loss
5. **推理层**：提供专门的接口(`predict_action`, `video_gen`)自动选择对应任务

这种设计让一个模型同时学会了：

- **执行能力** (Policy)
- **模拟能力** (Forward Dynamics)
- **规划能力** (Inverse Dynamics)
- **想象能力** (Video Gen)

从而构成了真正的**具身世界模型**！🌟

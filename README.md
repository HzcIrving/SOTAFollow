# 🚀 SOTAFollow - AI/机器人前沿论文追踪

> 分类管理 SOTA（State-of-the-Art）论文，覆盖 VLA、WorldModel、RL、LLM 等领域

---

## 📑 快捷索引

| 🏠 首页 | 📚 [VLA](#-vla) | 🌍 [WorldModel](#-worldmodel) | 🎯 [RL](#-rl) | 🧠 [LLM](#-llm) | 🔧 [FM基础](#-fm基础知识) | 💼 [面筋](#-面筋) |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| [FM基础](#-fm基础知识) | [VLA](#-vla) | [WorldModel](#-worldmodel) | [RL](#-rl) | [LLM](#-llm) | [日报](./RL/日报/) | [DailyAgent](#-dailyagent) |

---

## 🤖 VLA

> Vision-Language-Action Model · 视觉-语言-动作模型

| 论文 | 年份 | 核心贡献 | 文档 |
|:-----|:----:|:---------|:-----|
| **Vega** | 2026 | 统一 Vision-Language-World-Action，InstructScene 100K，NAVSIM EPDMS 89.4 SOTA | [精读](./VLA/Vega_精读报告.md) |
| **Uni-World VLA** | ECCV 2026 | 交错式闭环 VLA，统一生成未来帧+动作 tokens | [精读](./WorldModel/Uni-World_VLA-论文精读-ECCV2026.md) |
| **DVGT-2** | 2026 | Vision-Geometry-Action 端到端，O(1) 帧复杂度，PDMS 90.3 | [精读](./VLA/DVGT-2_精读报告.md) |
| **MINT** ⭐ | 2026 | 频域多尺度动作Tokenizer（SDAT），LIBERO 98.3% SOTA | [精读](./VLA/MINT/MINT-论文精读-arXiv2602.08602.md) |
| **π0.7** ⭐ | 2026 | 多样化上下文条件化，零样本跨本体衬衫折叠，~5B | [精读](./VLA/π0.7_精读报告.md) |
| **VISTA** | 2026 | 世界模型生成视觉子目标，未见场景 14%→69% | [精读](./VLA/VISTA_精读报告.md) |
| **VLA-JEPA** | 2026 | JEPA风格VLA预训练，时序因果注意力，流匹配动作头 | [精读](./VLA/VLA-JEPA_论文精读-arXiv2602.10098.md) |
| Actuate 2025 | 2025 | Sergey Levine & Liyiming Ke：RL Post-training 是关键补充 | [精读](./VLA/Actuate2025_SergeyLevine_精读报告.md) |

---

## 🌍 WorldModel

> World Model · 世界模型 · 自动驾驶

| 论文 | 年份 | 核心贡献 | 文档 |
|:-----|:----:|:---------|:-----|
| **Being-H0.7** ⭐ | 2026 | 先验-后验双分支对齐，MoT 高效实现，6 仿真 SOTA | [精读](./WorldModel/Being-H0.7_精读报告.md) |
| **Latent-WAM** ⭐ | 2026 | SCWE 16-query 压缩 + WorldMirror 几何蒸馏，感知自由 89.3 | [精读](./WorldModel/Latent-WAM_精读报告.md) |
| **Epona** | 2025 | 自回归扩散世界模型，Chain-of-Forward 训练 | [精读](./WorldModel/Epona-精读报告.md) |
| **Fast-WAM** | 2026 | 测试时想象是否必要？端到端规划加速 48× | [精读](./WorldModel/Fast-WAM_精读报告.md) |
| **Uni-World VLA** | ECCV 2026 | 交错式世界建模，冻结幻觉问题，PDMS 89.4 | [精读](./WorldModel/Uni-World_VLA-论文精读-ECCV2026.md) |
| MV-VDP | 2026 | 多视角视频扩散策略，Meta-World 89.1% | [精读](./WorldModel/MV-VDP_精读报告.md) |
| LeWorldModel | 2026 | 首个端到端 JEPA 世界模型，48x 规划加速 | [精读](./WorldModel/LeWorldModel-论文精读报告.md) |
| DreamerAD | 2026 | 解析世界模型，Shortcut Forcing 80× 加速 | [精读](./WorldModel/DreamerAD-论文解读.md) |

### 📝 深度解读

| 主题 | 文档 |
|:-----|:-----|
| 从表征学习角度看 Being-H0.7、Fast-WAM 与 π0.7 | [链接](./WorldModel/从表征学习角度看_Being-H0.7_Fast-WAM与π0.7.md) |
| 世界模型 EP04：Motus（石麻笔记） | [链接](./WorldModel/世界模型EP04_Motus_石麻笔记.md) |
| 硅谷101-世界模型深度解读 | [笔记](./WorldModel/世界模型_视频笔记_硅谷101.md) |

---

## 🎯 RL

> Reinforcement Learning · 强化学习

| 论文 | 年份 | 核心贡献 | 文档 |
|:-----|:----:|:---------|:-----|
| **NPO / AutoNPO** ⭐ | 2026 | 近未来策略优化，Qwen3-VL-8B 57.88→63.15 | [精读](./RL/Near-Future_Policy_Optimization_精读报告.md) |
| **XXPO 系列** | — | PPO/GRPO/GSPO/DAPO/GMPO 算法全景对比 | [精读](./RL/XXPO系列算法精读报告.md) |
| **RAD** | NeurIPS 2025 | 首个3DGS-RL端到端，碰撞率降低3倍 | [精读](./RL/RAD_精读报告.md) |
| **RAD-2** | 2026 | 扩散生成器+RL判别器，碰撞率降低56% | [精读](./RL/RAD-2_精读报告.md) |
| **FlowGRPO** | 2025 | 首个 GRPO + Flow Matching，SD3.5 63%→95% | [精读](./RL/FlowGRPO_精读报告.md) |
| PPO 精读 | — | Clipped Surrogate、GAE、Python 伪代码 | [精读](./RL/PPO_精读报告.md) |

### 📅 日报

| 日期 | 内容 |
|:-----|:-----|
| [RL 日报](./RL/日报/) | 强化学习前沿每日追踪 |

---

## 🧠 LLM

> Large Language Model · 大语言模型

| 论文 | 年份 | 核心贡献 | 文档 |
|:-----|:----:|:---------|:-----|
| **DeepSeek-V4** | 2025 | CSA+HCA 混合稀疏注意力，1M token 上下文 | [精读](./LLM/DeepSeek-V4_论文精读-arXiv2503.24978.md) |

---

## 🔧 FM基础知识

> Foundation Model · 基础模型知识详解

### 📂 核心模块

| 模块 | 主题 | 文档 |
|:----:|:-----|:-----|
| 🧩 **Tokenizer** | TiTok 统一视觉 Tokenizer、VQVAE、1D Tokenization | [详情](#tokenizer) |
| ⚡ **注意力** | FlashAttention、RoPE/3DPE/mRoPE、Kimi Attention Residuals | [详情](#注意力机制) |
| 🎛️ **微调** | LoRA 参数高效微调、ZeRO 优化器 | [详情](#微调优化) |
| 🧠 **世界模型** | 自回归框架、训练 Loss 设计 | [详情](#世界模型) |
| 📍 **位置编码** | RoPE、3DPE、mRoPE（DVGT-2 时序融合） | [详情](#位置编码) |
| 🧩 **记忆机制** | MEM/GMP、MemoryVLA | [详情](#记忆机制) |

### 📄 详细列表

| 主题 | 简介 | 文档 |
|:-----|:-----|:-----|
| **大模型 Roadmap** | Transformer、MoE、量化、RAG、部署全景图 | [详情](./FM基础知识/大模型Roadmap.md) |
| **TiTok** | 统一视觉 Tokenizer，1D 离散化 + VQ-GAN | [精读](./FM基础知识/TiTok-论文精读-arXiv2406.07550.md) |
| **TiTok-1D** | 信息密度自适应分配、背景压缩、主体细节 | [笔记](./FM基础知识/TiTok-1D-Tokenization-学习笔记.md) |
| **VQVAE** | Codebook 机制、视觉表征学习 | [详解](./FM基础知识/VQVAE视觉Tokenizer详解.md) |
| **自回归框架** | 掩码设计、Action Token、VLA 结合 | [详解](./FM基础知识/WorldModel-VLA自回归框架详解.md) |
| **LoRA** | Low-Rank Adaptation、低秩适应机制 | [精读](./FM基础知识/LoRA-论文精读-arXiv2106.09685.md) |
| **FlashAttention** | IO 感知、分块计算 + 重新计算 | [精读](./FM基础知识/FlashAttention-论文精读-arXiv2205.14135.md) |
| **FlashAttention-2** | 并行性改进、循环顺序调换 | [精读](./FM基础知识/FlashAttention2-论文精读-arXiv2307.08691.md) |
| **ZeRO** | 零冗余优化器、数据并行分区 | [精读](./FM基础知识/ZeRO-论文精读-sc20.md) |
| **World Model Loss** | ELBO / KL Balancing / JEPA / LPIPS | [详解](./FM基础知识/WorldModel训练Loss设计详解.md) |
| **DCT** | 频域信号处理基础 | [详解](./FM基础知识/DCT（离散余弦变换）详解.md) |
| **RoPE / 3DPE / mRoPE** | 位置编码技术详解 | [详解](./FM基础知识/RoPE及3DPE技术详解.md) |
| **MEM（记忆机制）** | Gated Memory Policy、MemoryVLA | [精读](./FM基础知识/MEM/GMP_精读报告.md)、[精读](./FM基础知识/MEM/MemoryVLA_精读报告.md) |
| **GCT** | 几何上下文注意力，ETH3D 98.98% F1 | [精读](./FM基础知识/流式三维重建-几何上下文Transformer-精读报告.md) |
| **PETR** | 3D 检测：3D 位置编码融合相机几何 | [精读](./FM基础知识/PETR-论文精读.md) |
| **PETR V2** | Feature-Guided Position Encoding、时序融合 | [精读](./FM基础知识/PETR%20V2-论文精读.md) |

---

## 💼 面筋

> 面试知识点整理 · 子文化搜集与碎碎念

### 📂 分类索引

| 分类 | 简介 | 路径 |
|:----:|:-----|:-----|
| 🎯 **RL** | 强化学习面试相关 | [查看](./面筋/RL/) |
| 🧠 **LLM** | 大语言模型面试相关 | [查看](./面筋/LLM/) |
| 🤖 **VLA** | Vision-Language-Action 面试相关 | [查看](./面筋/VLA/) |
| 🌍 **WM** | WorldModel 面试相关 | [查看](./面筋/WM/) |

### 📄 热门文档

| 主题 | 简介 | 文档 |
|:-----|:-----|:-----|
| MoE_RL 训推不一致 | MoE 做 RL 训练-推理不一致 | [查看](./面筋/RL/MoE_RL_训推不一致.md) |
| 智元 RL 技术凉经 | RL 后训练面试题 | [查看](./面筋/RL/智元RL技术凉经.md) |
| GRPO vs SFT 数据差异 | SFT 与 GRPO 数据需求、难度分布 | [查看](./面筋/RL/GRPO_vs_SFT训练数据差异.md) |
| LLM 面试深度知识点 | 五大模块详解 | [查看](./面筋/LLM/LLM面试深度知识点-五大模块详解.md) |
| LLM 面试入门知识点 | 入门级知识点 | [查看](./面筋/LLM/LLM面试入门知识点.md) |

---

## 🛠️ DailyAgent

> AI 工具使用技巧 · Agent 工作流 · 效率提升策略

| 主题 | 简介 | 文档 |
|:-----|:-----|:-----|
| **Claude Code Context Rot & Rewind** | 1M 上下文会"腐烂"，Rewind 是最佳方案 | [查看](./DailyAgent/ClaudeCode/ClaudeCode_context_rot_rewind.md) |
| **Hermes Agent 新手使用十大技巧** | 主辅模型配置、SOUL.md、记忆机制 | [查看](./DailyAgent/Hermes/Hermes_Agent_新手使用十大技巧.md) |

---

## 📰 最近更新

| 日期 | 类别 | 简报 |
|:----:|:----:|:-----|
| **04-26** 🌟 | ⚙️ CONFIG, 📚 DOCS, 🔬 FM, 🛠️ TOOLS | [FM] MEM文件夹时序融合启示章节大幅扩充：PE编码、时空融合、门控机制完整伪代码 |
| 04-25 | 🌍 WM, 🔀 MERGE | Merge pull request #7 from zhangcollion/lingbot-va |
| 04-24 | 🌳 COMMIT, 🎯 RL, 💬 LLM, 🤖 VLA | [VLA] 新增 VLA-JEPA 论文精读报告 |

### 📜 [查看完整日志 `sotafollow.log`](./sotafollow.log)

_自动按天聚合 · emoji风格 · GitHub快捷导航_

---

## 🕐 历史日志

| 月份 | 总结 |
|:----:|:-----|
| 2026年4月 | [月度总结](./logs/月度总结_2026年04月.md) |

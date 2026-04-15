# 论文目录索引

## 面筋

> 子文化搜集与碎碎念

| 分类 | 简介 | 文档 |
|------|------|------|
| [RL](./面筋/RL/) | 强化学习相关 | （待填充） |
| [LLM](./面筋/LLM/) | 大语言模型相关 | [MoE_RL_训推不一致](./面筋/LLM/MoE_RL_训推不一致.md) |
| [VLA](./面筋/VLA/) | Vision-Language-Action 相关 | （待填充） |
| [WM](./面筋/WM/) | WorldModel 相关 | （待填充） |

---

## VLA

| 论文 | 会议/年份 | 核心贡献 | 文档 |
|------|----------|----------|------|
| Vega | arXiv 2026 | 统一 Vision-Language-World-Action 模型，自然语言指令驾驶，InstructScene 100K 数据集，NAVSIM EPDMS 89.4 SOTA | [Vega_精读报告.md](./VLA/Vega_精读报告.md) |
| Uni-World VLA | ECCV 2026 | 交错式闭环 VLA，统一生成未来帧+动作 tokens | [→ WorldModel](./WorldModel/Uni-World%20VLA-论文精读-ECCV2026.md) |
| DVGT-2 | arXiv 2026 | Vision-Geometry-Action 端到端自动驾驶，O(1) 帧复杂度，NAVSIM PDMS 90.3 | [DVGT-2_精读报告.md](./VLA/DVGT-2_精读报告.md) |
| Actuate 2025 | 视频 2026 | Sergey Levine & Liyiming Ke：第二代 VLA 与机器人 Foundation Model，RL Post-training 是模仿学习的关键补充 | [Actuate2025_SergeyLevine_精读报告.md](./VLA/Actuate2025_SergeyLevine_精读报告.md) |

## WorldModel

| 论文 | 会议/年份 | 核心贡献 | 文档 |
|------|----------|----------|------|
| Uni-World VLA | ECCV 2026 | 交错式世界建模与规划，冻结幻觉问题，NAVSIM PDMS 89.4 | [Uni-World VLA-论文精读-ECCV2026.md](./WorldModel/Uni-World%20VLA-论文精读-ECCV2026.md) |
| MV-VDP | arXiv 2026 | 多视角视频扩散策略，视频基础模型联合预测 RGB 视频 + 热力图，5 demos Meta-World 89.1% | [MV-VDP_精读报告.md](./WorldModel/MV-VDP_精读报告.md) |
| LeWorldModel | arXiv 2026 | 首个端到端 JEPA 世界模型，SIGReg 正则器防崩溃，15M 参数单 GPU 可训，48x 规划加速 | [LeWorldModel-论文精读报告.md](./WorldModel/LeWorldModel-论文精读报告.md) |
| DreamerAD | arXiv 2026 | 基于解析世界模型的自动驾驶车辆控制，Shortcut Forcing 80× 加速，EPDMS 87.7 SOTA | [DreamerAD-论文解读.md](./WorldModel/DreamerAD-论文解读.md) |
| Fast-WAM | arXiv 2026 | 北大&华为：World Action Model 测试时想象是否必要？端到端规划加速 48× | [Fast-WAM_精读报告.md](./WorldModel/Fast-WAM_精读报告.md) |

## RL

（暂无论文）

## FM基础知识

| 主题 | 简介 | 文档 |
|------|------|------|
| TiTok | 统一视觉 Tokenizer，1D离散化 + VQ-GAN，SoTA 图像重建 + 视频理解 | [TiTok-论文精读-arXiv2406.07550.md](./FM基础知识/TiTok-论文精读-arXiv2406.07550.md) |
| VQVAE 视觉 Tokenizer | Codebook 机制、视觉表征学习、World Model 视觉编码器 | [VQVAE视觉Tokenizer详解.md](./FM基础知识/VQVAE视觉Tokenizer详解.md) |
| World Model / VLA 自回归框架 | 掩码设计、Action Token、VLA 与 World Model 结合的自回归范式 | [WorldModel-VLA自回归框架详解.md](./FM基础知识/WorldModel-VLA自回归框架详解.md) |

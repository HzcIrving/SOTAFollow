# 论文目录索引

## 面筋

> 子文化搜集与碎碎念

| 分类 | 简介 | 文档 |
|------|------|------|
| [RL](./面筋/RL/) | 强化学习相关 | （待填充） |
| [LLM](./面筋/LLM/) | 大语言模型相关 | [MoE_RL_训推不一致](./面筋/LLM/MoE_RL_训推不一致.md)、[LLM面试深度知识点-五大模块详解](./面筋/LLM/LLM面试深度知识点-五大模块详解.md) |
| [VLA](./面筋/VLA/) | Vision-Language-Action 相关 | （待填充） |
| [WM](./面筋/WM/) | WorldModel 相关 | （待填充） |

---

## VLA

| 论文 | 会议/年份 | 核心贡献 | 文档 |
|------|----------|----------|------|
| Vega | arXiv 2026 | 统一 Vision-Language-World-Action 模型，自然语言指令驾驶，InstructScene 100K 数据集，NAVSIM EPDMS 89.4 SOTA | [Vega_精读报告.md](./VLA/Vega_精读报告.md) |
| Uni-World VLA | ECCV 2026 | 交错式闭环 VLA，统一生成未来帧+动作 tokens | [→ WorldModel](./WorldModel/Uni-World_VLA-论文精读-ECCV2026.md) |
| DVGT-2 | arXiv 2026 | Vision-Geometry-Action 端到端自动驾驶，O(1) 帧复杂度，NAVSIM PDMS 90.3 | [DVGT-2_精读报告.md](./VLA/DVGT-2_精读报告.md) |
| Actuate 2025 | 视频 2026 | Sergey Levine & Liyiming Ke：第二代 VLA 与机器人 Foundation Model，RL Post-training 是模仿学习的关键补充 | [Actuate2025_SergeyLevine_精读报告.md](./VLA/Actuate2025_SergeyLevine_精读报告.md) |
| **MINT** | arXiv 2026 | 上交大：频域多尺度动作Tokenizer（SDAT），解耦Intent Token与Execution Tokens，One-Shot跨任务迁移超越Fine-tuning 60pp，LIBERO 98.3% SOTA | [MINT-论文精读-arXiv2602.08602.md](./VLA/MINT/MINT-论文精读-arXiv2602.08602.md) |
| **π0.7** | PI 2026 | 多样化上下文条件化，通用模型无需微调匹配专用RL，零样本跨本体衬衫折叠达到人类专家水平，~5B参数（Gemma3 4B + 860M Action Expert） | [π0.7_精读报告.md](./VLA/π0.7_精读报告.md) |

## WorldModel

| 论文 | 会议/年份 | 核心贡献 | 文档 |
|------|----------|----------|------|
| Uni-World VLA | ECCV 2026 | 交错式世界建模与规划，冻结幻觉问题，NAVSIM PDMS 89.4 | [Uni-World VLA-论文精读-ECCV2026.md](./WorldModel/Uni-World_VLA-论文精读-ECCV2026.md) |
| MV-VDP | arXiv 2026 | 多视角视频扩散策略，视频基础模型联合预测 RGB 视频 + 热力图，5 demos Meta-World 89.1% | [MV-VDP_精读报告.md](./WorldModel/MV-VDP_精读报告.md) |
| LeWorldModel | arXiv 2026 | 首个端到端 JEPA 世界模型，SIGReg 正则器防崩溃，15M 参数单 GPU 可训，48x 规划加速 | [LeWorldModel-论文精读报告.md](./WorldModel/LeWorldModel-论文精读报告.md) |
| DreamerAD | arXiv 2026 | 基于解析世界模型的自动驾驶车辆控制，Shortcut Forcing 80× 加速，EPDMS 87.7 SOTA | [DreamerAD-论文解读.md](./WorldModel/DreamerAD-论文解读.md) |
| Fast-WAM | arXiv 2026 | 北大&华为：World Action Model 测试时想象是否必要？端到端规划加速 48× | [Fast-WAM_精读报告.md](./WorldModel/Fast-WAM_精读报告.md) |

## RL

（暂无论文）

## FM基础知识

| 主题 | 简介 | 文档 |
|------|------|------|
| **大模型 Roadmap** | LLM 基础知识全景图：Transformer架构、主流模型、预训练、后训练、量化压缩、MoE、RAG&Agent、部署加速、模型评估、其他结构（SSM/Mamba等） | [大模型Roadmap.md](./FM基础知识/大模型Roadmap.md) |
| TiTok | 统一视觉 Tokenizer，1D离散化 + VQ-GAN，SoTA 图像重建 + 视频理解 | [TiTok-论文精读-arXiv2406.07550.md](./FM基础知识/TiTok-论文精读-arXiv2406.07550.md) |
| VQVAE 视觉 Tokenizer | Codebook 机制、视觉表征学习、World Model 视觉编码器 | [VQVAE视觉Tokenizer详解.md](./FM基础知识/VQVAE视觉Tokenizer详解.md) |
| World Model / VLA 自回归框架 | 掩码设计、Action Token、VLA 与 World Model 结合的自回归范式 | [WorldModel-VLA自回归框架详解.md](./FM基础知识/WorldModel-VLA自回归框架详解.md) |
| LoRA (参数高效微调) | Low-Rank Adaptation 原始论文精读，低秩适应机制、缩放因子设计 | [LoRA-论文精读-arXiv2106.09685.md](./FM基础知识/LoRA-论文精读-arXiv2106.09685.md) |
| FlashAttention | IO 感知的精确注意力算法，分块计算 + 重新计算技术 | [FlashAttention-论文精读-arXiv2205.14135.md](./FM基础知识/FlashAttention-论文精读-arXiv2205.14135.md) |
| FlashAttention-2 | 改进的并行性和工作分配，循环顺序调换，HBM 访问进一步优化 | [FlashAttention2-论文精读-arXiv2307.08691.md](./FM基础知识/FlashAttention2-论文精读-arXiv2307.08691.md) |
| ZeRO 优化器 | 零冗余优化器，数据并行状态分区，万亿参数模型训练 | [ZeRO-论文精读-sc20.md](./FM基础知识/ZeRO-论文精读-sc20.md) |
| World Model 训练 Loss 设计 | ELBO / KL Balancing / JEPA / LPIPS / VQ-VAE Loss 详解，含主流 SOTA 工作Loss汇总 | [WorldModel训练Loss设计详解.md](./FM基础知识/WorldModel训练Loss设计详解.md) |
| **DCT（离散余弦变换）** | 频域信号处理基础：DCT-II定义、频率物理含义、与FFT对比，及其在MINT频域解耦中的作用 | [DCT（离散余弦变换）详解.md](./FM基础知识/DCT（离散余弦变换）详解.md) |
| **Kimi Attention Residuals** | Moonshot AI：跨层选择性注意力残差连接，相同 Loss 下节省 25% 计算资源，Infra 是结构创新上限 | [Kimi_Attention_Residuals_精读报告.md](./FM基础知识/Kimi_Attention_Residuals_精读报告.md) |

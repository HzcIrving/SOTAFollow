# LLM 面试入门知识点

针对2026年大语言模型（LLM）面试的深度知识点，整理涵盖底层架构、训练优化、高效推理及前沿技术的核心指南，重点覆盖社招或高阶算法岗位的“高频重灾区”。

# 一、模型架构深度理解 (Architecture)

不仅仅要懂 Transformer，更要懂它的演进和变体。

## Attention 变体

- **MHA vs. MQA vs. GQA**：重点理解 GQA (Grouped Query Attention) 如何在推理速度（KV Cache 显存占用）和模型效果之间做权衡。
- **MLA (Multi-head Latent Attention)**：DeepSeek 系列的核心，理解它如何通过低秩压缩显著降低推理时的 KV Cache 压力。
- **FlashAttention (1/2/3)**：必考。重点在于它是如何利用 GPU 内存分级（SRAM vs. HBM）来消除 IO 瓶颈，而不是改变 O(n²) 的算法复杂度。

## 位置编码 (Positional Encoding)

- **RoPE (Rotary Positional Embedding)**：深入理解其复数旋转矩阵的形式，以及为什么它在长度外推性上优于绝对位置编码。
- **外推策略**：了解 NTK-aware RoPE、YaRN、LongRoPE 等处理长上下文（如 1M+ token）的技术。

## Normalization & Activation

- **RMSNorm vs. LayerNorm**：为什么大模型多用 RMSNorm（计算量更小，效果接近）。
- **SwiGLU**：理解它相比传统 ReLU 在表达能力上的优势。

# 二、训练与对齐 (Training & Alignment)

## 预训练 (Pre-training)

- **数据质量**：重点讨论如何清洗 PB 级数据、数据配比（Data Mixture）对模型下游能力的影响。
- **Tokenization**：BPE、WordPiece、Unigram 的区别。字节级分词（BBPE）如何解决 OOV 问题。

## 微调 (Fine-tuning / PEFT)

- **LoRA 及其变体**：DoRA（解耦权重大小和方向）、QLoRA（4-bit 量化微调）。
- **P-Tuning v2 vs. Prompt Tuning**：了解其注入位置和参数量的差异。

## 对齐技术 (Alignment)

- **RLHF (PPO)**：重点在于奖励模型（RM）的训练和 PPO 的四个阶段。
- **DPO (Direct Preference Optimization)**：必考。理解它为什么不需要 RM 和复杂的强化学习循环。
- **Reasoning 对齐**：类似 OpenAI o1 的强化学习思维链（CoT）训练方法。

# 三、分布式训练与系统优化 (System & Infrastructure)

## 并行策略 (Parallelism)

- **DP / DDP / FSDP**：重点是 FSDP (Fully Sharded Data Parallel) 带来的显存节省原理。
- **TP vs. PP**：Tensor Parallel (张量并行) 和 Pipeline Parallel (流水线并行) 的切分方式，以及通信开销对比。
- **ZeRO 1/2/3**：理解 Optimizer State、Gradient 和 Parameter 是如何被 Sharding 的。
- **混合精度**：FP16 vs. BF16（为什么 BF16 在训练大模型时更稳定）。

# 四、高效推理与部署 (Inference)

- **KV Cache 管理**：PagedAttention：vLLM 的核心，理解它如何像操作系统内存管理一样解决显存碎片。

- **量化 (Quantization)**：PTQ vs. QAT：了解推理后量化。AWQ / GPTQ：理解权重量化对模型精度的影响。

- **投机采样 (Speculative Decoding)**：使用小模型预测，大模型并行验证，如何打破自回归生成的延迟瓶颈。

  **补充知识**：

  1、SRAM（静态随机存储器）：

  GPU **片上高速缓存**（L1/L2/Shared Memory），**不是普通显存**。速度极快（延迟 < 1ns）、带宽极高（几十 TB/s）、容量极小（MB 级），**只能存小块数据**。价格昂贵；

  **数据流向**：GPU 核心要数据时，先找 **L1** -> 找不到找 **L2** -> 再找不到才去 **HBM 显存**（最慢）

  2、HBM（高带宽存储器）：

  GPU **专用主显存**（3D 堆叠 DRAM），**平时说的 “GPU 显存** 就是它（A100/H100 的显存）。容量大（几十～百 GB）、带宽高（~2–3TB/s）、延迟几十 ns。存模型权重、**KV Cache**、激活等。
  
  华为昇腾使用HBM2e(受制于供应链和封装技术限制),  相比nvidia 落后一代(使用HBM3)
  
  3、内存（RAM / 系统内存）：CPU 用的内存（DDR5），GPU 不能直接访问，需 PCIe 拷贝，**速度慢、带宽低**。
  
  ![SRAM Hierarchy](./fig/sram)
  
  4、GPU组网：
  
  NvLink：**多 GPU 高速互联技术**，用来替代传统 PCIe，让多张 GPU 之间直接高速通信
  
  NviSwitch：**一颗独立的、专门用于 GPU 集群的交换芯片。**作用，让**整个机柜里所有 GPU** 都能互相全互联；如果跨机柜（集群数千GPU）：用 **InfiniBand / 以太网**（Spectrum-X），实现 Gb/s 级通信。

# 五、前沿趋势 (Hot Topics 2026)

- **Agent & Tool Use**：ReAct 框架：观察-思考-行动循环。Multi-Agent：如何设计多智能体协作流（如 AutoGen 模式）。
- **世界模型 (World Models) 与 DiT**：Sora 相关：Diffusion Transformer 的架构，如何将视频切片为 Spacetime Latent Patches。VLA (Vision-Language-Action)：端到端具身智能模型。
- **RAG (检索增强生成)**：GraphRAG：利用知识图谱增强复杂关系的推理。Self-RAG：模型如何学会自主决定何时检索。
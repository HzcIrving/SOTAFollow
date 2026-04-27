# 面试题整理

## 日期：20260427

### LLM（FM基础知识）领域

**论文：《FlashAttention-3: Fast and Memory-Efficient Attention with IO-Awareness》**

1. **[基础概念题]** FlashAttention 系列的核心贡献是将注意力计算从 O(N²) 显存复杂度优化到 O(N)。请解释标准注意力实现为何需要 O(N²) 显存，以及 FlashAttention 如何通过分块计算实现节省显存。
   - **答案/思考提示**：标准实现需要保存完整的注意力矩阵 S = QK^T (N×N) 和后续的 P = softmax(S)。FlashAttention 通过分块计算：① 将 Q/K/V 分成小块；② 在每个块内计算局部注意力；③ 使用在线 softmax 技巧，只保存必要的归一化统计量（m、l）而非完整矩阵；④ 最终通过两次 pass 完成计算。

2. **[深度思考题]** FlashAttention 的 IO-Awareness 优化基于 GPU 内存层次结构。请描述 GPU 的 HBM（High Bandwidth Memory）和 SRAM（Shared Memory）的性能差异，以及 FlashAttention 如何利用这种差异进行优化。
   - **答案/思考提示**：HBM 容量大但带宽低（~1TB/s），SRAM 容量小但带宽高（~20TB/s）。FlashAttention 的核心思想是最小化 HBM 访问——将数据分块加载到 SRAM，在 SRAM 中完成计算后再写回 HBM。通过精心设计的 block size 和 tiling 策略，最大化 SRAM 的复用率。

3. **[实践应用题]** FlashAttention-3 对 FP8 训练的支持需要处理数值精度问题。请解释 FP8 训练中的主要挑战，以及通常如何处理 attention logit 的缩放。
   - **答案/思考提示**：FP8 的动态范围有限（E4M3 vs E5M2），attention logit 通常很大，直接 FP8 计算会溢出。解决方案：① 对 Q/K 进行 per-tensor 或 per-head 缩放；② 对 logit 使用更宽的表示（如 BF16）；③ 分块缩放而非全局缩放；④ 使用特定于层的数值稳定化技术。

4. **[优缺点对比题]** 比较 FlashAttention-1/2/3 三个版本的主要演进。从算法改进、硬件利用、适用场景三个维度分析。
   - **答案/思考提示**：FlashAttention-1 基础分块+在线 softmax；FlashAttention-2 改进了 block size 和 thread block 分配，支持 variable length；FlashAttention-3 利用 Hopper 架构的 WGMMA 指令和 TMA 硬件支持，进一步提升吞吐。FA3 主要针对 H100/H200，FA2 在 A100 等 older 架构上仍是最优选择。

---

**通用 LLM/基础模型思考题**

5. **[基础概念题]** Transformer 中的 Softmax Attention 可以写作 S = softmax(QK^T / √d)V。请解释为什么需要除以 √d，以及这个设计在梯度视角下的含义。
   - **答案/思考提示**：√d 的作用是控制方差——QK^T 的方差随 d 增大，除以 √d 使点积结果的方差保持为 O(1)，避免 softmax 进入饱和区域（梯度消失）。梯度视角：attention score 的梯度与 exp(s)^2 相关，不做缩放会导致大 score 对应的梯度极小。

6. **[深度思考题]** 在 LLM 训练中，loss spike（损失尖峰）和 loss horizon（损失平峰）是常见问题。请分析这两种现象的成因，以及有哪些实用的缓解策略。
   - **答案/思考提示**：loss spike 通常由异常梯度引起（如 large batch 中个别极端样本），缓解方法：梯度裁剪、batch 内 outlier 过滤、学习率 warmup/衰减。loss horizon 指 loss 长时间不下降，可能由 learning rate 过小、gradient starvation、局部最优导致。解决：learning rate 重启、batch 大小调整、模型正则化。

7. **[实践应用题]** 你在训练大模型时如何设置 batch size 和 learning rate？请描述你的 scaling 策略和调参经验。
   - **答案/思考提示**：通常遵循 scaling law——更大的 batch 需要更大的 learning rate（linear scaling rule）。经验公式：lr = base_lr × (batch_size / base_batch_size)。对于超参数搜索，用小模型/小 batch 确定相对超参数关系，再 scale 到大模型。常用策略：cosine annealing、warmup + constant、阶段式衰减。

8. **[深度思考题]** MoE（Mixture of Experts）架构通过稀疏激活实现高效扩展。请解释 MoE 的核心思想，以及为什么它能实现"更少的计算量完成更大的模型容量"。
   - **答案/思考提示**：MoE 将 FFN 层替换为多个 experts（独立的 FFN），每个 token 只激活少数 experts（如 top-2）。模型总参数量随 experts 数量线性增长，但每次前向只计算激活 experts 的参数。路由机制（router）决定哪些 experts 处理哪些 token。挑战：负载均衡（避免少数 experts 被过度使用）、通信开销（分布式部署）、训练稳定性。

9. **[基础概念题]** 请解释 KV Cache 在 LLM 推理中的作用。为什么 KV Cache 对推理效率至关重要？
   - **答案/思考提示**：Autoregressive 生成时，每个新 token 都需要 attend 到之前所有 tokens 的 Key 和 Value。KV Cache 将已计算的 K/V 缓存起来，新 token 只需计算 Q，然后从缓存中获取 K/V。避免重复计算 O(N²) 的 K/V，使每步生成复杂度从 O(N²) 降为 O(N)（Q 计算）+ O(N)（KV 检索）。

10. **[实践应用题]** 在部署 LLM 到生产环境时，你如何做模型量化？请比较 INT8、INT4、FP8 量化方案的适用场景和潜在问题。
    - **答案/思考提示**：INT8：平衡精度和效率，适合延迟不敏感场景；INT4：极致压缩（4x），但可能有明显精度损失，适合 memory-bound 场景；FP8：硬件支持较好，精度损失比 INT 更小。量化策略：QAT（量化感知训练）优于 PTQ（训练后量化）；关注 outlier 特征；不同层可能需要不同的量化精度。

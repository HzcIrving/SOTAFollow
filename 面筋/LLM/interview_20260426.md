## 日期：20260426

### LLM 领域（FM基础知识）

**论文：《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》**

1. **[基础概念题]** FlashAttention 解决了标准注意力计算的什么瓶颈？请从 IO 复杂度角度分析标准注意力的显存问题。
   - 答案/思考提示：标准注意力需要存储 $QK^T$ 中间结果（$O(N^2)$ 显存）和 softmax 输出（$O(N^2)$ 显存）。对于 $N=16K$，单头注意力需要约 2GB 显存。FlashAttention 通过分块计算 + 重新计算技术，将 HBM 读写复杂度从 $O(N^2 d)$ 降到 $O(N d \sqrt{N})$，显存从 $O(N^2)$ 降到 $O(N d)$。关键洞察：计算本身很快，但从 HBM 读写数据很慢——A100 HBM 带宽 ~1.5 TB/s，SRAM 带宽 ~19 TB/s，相差约 10 倍。

2. **[深度思考题]** FlashAttention 的核心思想是"重新计算比存储更快"。为什么在 HBM 带宽受限的场景下，重新计算中间结果反而比从 HBM 读取它们更高效？请从系统层面解释。
   - 答案/思考提示：HBM 带宽是瓶颈（~1 TB/s），而 SRAM 带宽高得多（~19 TB/s）。存储中间结果需要：写回 HBM（一次）+ 后向传播时读回 HBM（一次）= 2 次 HBM 访问。重新计算只需要：在前向时用 SRAM 计算并保留必要统计量（m, l），后向时在 SRAM 中重新计算中间结果，无需读回 HBM。对于 HBM 瓶颈型算子，增加少量计算换取大幅减少内存访问是合算的。

3. **[优缺点对比题]** FlashAttention 与稀疏注意力（如 Longformer）、低秩近似（如 Linformer）相比，核心优势是什么？为什么精度保证很重要？
   - 答案/思考提示：稀疏/低秩方法都是近似算法，有精度损失；FlashAttention 是精确算法，与标准注意力数学等价（mathematically equivalent）。精度保证的重要性：① 可以即插即用，无需重新训练模型；② 没有任何精度损失；③ 更容易被社区采用（这也是 FlashAttention 能成为 Transformer 标准组件的原因）。近似方法虽然也能加速，但需要针对任务调优稀疏模式或近似秩，且可能在某些任务上效果不佳。

4. **[实践应用题]** 在实际部署 FlashAttention 时，分块大小（block size）是关键调优参数。如果你在 A100 GPU（SRAM ~20MB）上运行 FlashAttention，你会如何选择分块大小？需要考虑哪些因素？
   - 答案/思考提示：分块大小需要满足：① 能放入 SRAM 容量限制（A100 SRAM ~20MB）；② 尽量减少 HBM 访问次数。考虑因素：序列长度 N、头维度 d、batch size、SRAM 容量。通常 128 或 256 是不错的默认值。公式约束：$B_r \times d + B_c \times d + \text{output} \leq \text{SRAM size}$。如果 OOM 则减小 block size；如果发现性能下降可能 block size 过大导致频繁的 HBM 读写。

5. **[深度思考题]** FlashAttention 的 online softmax 技巧（分块计算 softmax）是其核心算法之一。请解释为什么标准 softmax 不能直接分块计算，以及 online softmax 如何解决这个问题。
   - 答案/思考提示：Softmax 需要全局归一化（分母是所有指数项的和），不能独立计算每个块的 softmax 后再合并。Online softmax 的关键是维护两个统计量：$m$（当前最大值，用于数值稳定性）和 $\ell$（累计归一化因子）。分块计算时：先用 $\exp(s_{ij} - m_i)$ 归一化局部块，再通过 $m_{new} = \max(m_i, \max(s_{ij}))$ 和 $\ell_{new} = \exp(m_i - m_{new})\ell_i + \sum\exp(s_{ij} - m_{new})$ 递归更新全局统计量，最终正确归一化。

---

**通用 LLM/基础模型知识思考题**

6. **[基础概念题]** VQVAE 作为视觉 Tokenizer 的核心作用是什么？它与 VAE 的关键区别是什么？为什么 World Model 通常选择离散 token 而非连续表征？
   - 答案/思考提示：VQVAE 将高维像素图像压缩到低维离散 token 序列（来自码本），大幅降低计算复杂度（从 ~196K 维到数十个 token）。与 VAE 的关键区别：VAE 输出连续潜在分布，VQVAE 通过最近邻查码本强制离散化。选择离散 token 的原因：① 与语言模型架构兼容（语言模型本身就是离散的 token 序列）；② 码本提供共享的语义概念；③ 离散的表征更易于建模和压缩。

7. **[优缺点对比题]** 自回归语言模型（如 GPT）和 Diffusion Model 在生成方式上有本质不同。这两种范式各适合什么场景？
   - 答案/思考提示：自回归（AR）模型：顺序生成，每步依赖前文，适合文本、代码等序列化生成任务，可控性强但速度受序列长度限制。Diffusion 模型：逐步去噪生成，不依赖自回归结构，适合图像/视频生成，能捕捉丰富的数据分布，但推理速度慢。Flow Matching（如 FlashAttention 论文中提到的）提供了一种介于两者之间的方法，用 ODE 路径替代随机噪声过程，可实现更快速的生成（如 DreamerAD 中的 Shortcut Forcing）。

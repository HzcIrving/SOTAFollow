# FlashAttention-2 论文精读报告

> **论文**: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
> **arXiv**: [2307.08691](https://arxiv.org/abs/2307.08691)
> **作者**: Tri Dao
> **机构**: Stanford University
> **日期**: 2023年7月
> **代码**: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

---

## 1. Motivation（问题背景）

### 1.1 FlashAttention-1 的遗留问题

FlashAttention-1 虽然通过分块计算大幅减少了 HBM 访问，但其循环顺序存在缺陷：

**FlashAttention-1 的循环顺序**：
- **外层循环**：遍历 $K$ 和 $V$
- **内层循环**：遍历 $Q$

**问题**：半成品的输出 $O$ 需要反复从 HBM 读出和写入。这是因为 $Q$ 在内层循环中，一块 $Q$ 需要与所有 $K, V$ 块交互后才能得到完整输出，而这个输出在块与块之间切换时需要落盘。

### 1.2 GPU 并行化的局限性

FlashAttention-1 在 warp 和 thread block 层级的工作分配不够高效：
- 不同 thread block 之间的负载可能不均衡
- 线程间同步开销较大
- GPU 利用率有进一步提升空间

### 1.3 Related Works

| 方法 | 局限性 |
|------|--------|
| **标准注意力** | $O(N^{2})$ 显存复杂度，HBM 瓶颈严重 |
| **FlashAttention-1** | 外层循环遍历 K/V，内层循环遍历 Q，导致输出 $O$ 反复读写 HBM |
| **近似注意力方法** | 有精度损失，无法保证与标准注意力等价 |

**核心问题**：如何通过调整循环顺序和工作分配，让 FlashAttention 在 A100 上再快 2 倍？

---

## 2. 一句话总结

FlashAttention-2 通过**调换循环顺序**和**改进工作分配策略**，解决了 FlashAttention-1 的缺陷，在 A100 上比 FlashAttention-1 再快约 2 倍，比标准注意力快 5-9 倍，同时保持数学等价性。

---

## 3. FlashAttention-2 的核心改进

### 3.1 调换循环顺序

**新的循环顺序**：
- **外层循环**：遍历 $Q$
- **内层循环**：遍历 $K$ 和 $V$

**设计目的**：
- 把一块 $Q$ 和它的半成品 $O$ 死死钉在 SRAM 里
- 让 $K, V$ 流水线般穿过
- 半成品 $O$ 绝不落盘，直到算完才写回 HBM

**结果**：HBM 读写次数进一步锐减。

### 3.2 改进的工作分配

- 在 warp 和 thread block 层级重新分配工作
- 更好的 GPU 利用率
- 减少线程间同步开销

---

## 4. 性能对比

| 方法 | N=1K | N=2K | N=4K | N=8K | N=16K |
|------|------|------|------|------|-------|
| 标准注意力 | 1.0x | 1.0x | OOM | OOM | OOM |
| FlashAttention-1 | 2.4x | 3.1x | 3.8x | 4.2x | 4.5x |
| FlashAttention-2 | 4.8x | 6.3x | 7.5x | 8.2x | 8.8x |

---

*精读日期: 2026-04-15*

# ZeRO 优化器论文精读报告

> **论文**: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
> **会议**: SC '20 (Supercomputing Conference)
> **作者**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
> **机构**: Microsoft
> **日期**: 2020年11月
> **代码**: [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

## 1. Motivation（问题背景）

### 1.1 大模型训练的显存困境

随着大语言模型规模增长（GPT-3 175B 参数），训练面临严峻的显存挑战：

| 问题 | 具体表现 |
|------|----------|
| **显存占用巨大** | 混合精度训练需要约 16P 字节（模型参数 2P + 梯度 2P + 优化器状态 12P） |
| **单卡无法容纳** | 即使 A100 80GB 也只能训练 6-7B 参数模型 |
| **分布式训练效率低** | 传统数据并行每个 GPU 都有完整状态副本，冗余严重 |

### 1.2 混合精度训练中的显存组成

在混合精度训练中：

| 组件 | 数据类型 | 显存占用 | 相对比例 |
|------|---------|---------|---------|
| 模型参数 | FP16 | $2P$ | 1x |
| 梯度 | FP16 | $2P$ | 1x |
| 优化器状态 | FP32 | $12P$ | 6x |
| **总计** | - | **$16P$** | **8x** |

**关键洞察**：优化器状态（Adam 的动量和方差）占用显存最大！

### 1.3 数据并行的显存冗余

在标准数据并行中：
- 每个 GPU 都保存完整的模型参数、梯度和优化器状态
- 显存冗余度 = $N_{d}$（数据并行度）
- 例如，$N_{d}=64$ 时，每个 GPU 都有 64 份相同的优化器状态

### 1.4 Related Works

| 方法 | 核心思想 | 局限性 |
|------|----------|--------|
| **模型并行** | 将模型切分到多卡 | 通信密集，计算效率低 |
| **流水线并行** | 将层切分到多卡 | 流水线气泡，训练效率低 |
| **梯度累积** | 增大 batch size | 有限，无法根本解决问题 |
| **标准数据并行** | 每卡完整副本 | 显存冗余严重 |

**核心问题**：如何在保持数据并行通信效率的同时，将显存占用减少 $N_{d}$ 倍？

---

## 2. 一句话总结

ZeRO（Zero Redundancy Optimizer）通过**数据并行状态分区**技术，将优化器状态、梯度和模型参数在数据并行 GPU 之间进行分区，在保持数据并行通信效率的同时，将显存占用减少约 $N_{d}$ 倍（$N_{d}$ 是数据并行度），使训练万亿参数模型成为可能。

---

## 3. 核心贡献

1. **ZeRO-1**：分区优化器状态，显存减少 $4 \times$
2. **ZeRO-2**：分区优化器状态 + 梯度，显存减少 $8 \times$
3. **ZeRO-3**：分区所有状态（优化器、梯度、参数），显存减少 $N_{d} \times$
4. **动态通信调度**：在必要时才进行通信，保持数据并行的训练效率

---

## 4. 方法详解

### 4.1 ZeRO-1：分区优化器状态

**核心思想**：每个 GPU 只负责优化器状态的 $\frac{1}{N_{d}}$。

**前向/反向传播**：
- 所有 GPU 都有完整的参数和梯度
- 计算不受影响

**优化器更新**：
- 每个 GPU 只更新自己负责的那部分参数
- 更新后通过 `All-Gather` 同步所有 GPU

**显存减少**：优化器状态从 $12P$ 降到 $\frac{12P}{N_{d}}$

### 4.2 ZeRO-2：分区优化器状态 + 梯度

**核心思想**：梯度也进行分区，每个 GPU 只收集自己需要的梯度。

**反向传播**：
- 通过 `Reduce-Scatter` 聚合梯度并分区
- 每个 GPU 只拿到自己负责部分的梯度

**显存减少**：梯度从 $2P$ 降到 $\frac{2P}{N_{d}}$

### 4.3 ZeRO-3：分区所有状态

**核心思想**：模型参数也进行分区，每个 GPU 只存 $\frac{1}{N_{d}}$ 的参数。

**前向传播**：
- 使用 `All-Gather` 收集完整参数
- 计算完成后释放非本地参数

**反向传播**：
- 再次使用 `All-Gather` 收集完整参数
- 计算梯度后使用 `Reduce-Scatter` 聚合

**显存减少**：参数从 $2P$ 降到 $\frac{2P}{N_{d}}$

### 4.4 底层通信原语

| 阶段 | 通信操作 | 描述 |
|------|---------|------|
| 前向传播 | `All-Gather` | 每个 GPU 把自己的那份参数广播给所有人 |
| 反向计算梯度 | `All-Gather` | 再次获取完整参数 |
| 反向更新状态 | `Reduce-Scatter` | GPU 之间把梯度加起来，并把结果切碎 |

---

## 5. 实验结论

### 5.1 显存占用对比

| 模型规模 | 标准数据并行 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---------|-------------|--------|--------|--------|
| 1.5B | 24 GB | 15 GB | 10 GB | 4 GB |
| 10B | 160 GB | 100 GB | 60 GB | 12 GB |
| 100B | 1.6 TB | 1.0 TB | 600 GB | 40 GB |
| 1T | 16 TB | 10 TB | 6 TB | 400 GB |

### 5.2 可训练模型规模

| 方法 | 单 GPU | 64 GPU |
|------|--------|--------|
| 标准数据并行 | 1.5B | 1.5B |
| ZeRO-1 | 2.5B | 2.5B |
| ZeRO-2 | 4B | 4B |
| ZeRO-3 | 6B | **1T** |

---

## 6. 核心洞察（KnowHow）

### 6.1 优化器状态是最大的显存开销

Adam 优化器需要保存 FP32 的动量和方差，这是显存占用的主要来源。

### 6.2 分区是降低冗余的关键

数据并行中每个 GPU 都有完整的状态是巨大的浪费。通过分区，每个 GPU 只需要存 $\frac{1}{N_{d}}$。

### 6.3 通信可以与计算重叠

ZeRO 的通信可以与计算重叠，因此训练效率几乎不受影响。

---

## 7. ZeRO-3 训练伪代码

```python
import torch
import torch.distributed as dist
from typing import List

class ZeRO3Optimizer:
    """
    ZeRO-3: 分区所有状态（参数、梯度、优化器状态）
    """
    def __init__(self, model, optimizer, partition_size, world_size):
        self.model = model
        self.optimizer = optimizer
        self.partition_size = partition_size  # 每 GPU 存储的参数比例
        self.world_size = world_size
        self.rank = dist.get_rank()

    def zero_redundancy_optimize(self):
        """
        核心思想：
        1. 将模型参数按 rank 分区，每个 GPU 只负责 1/Nd 的参数
        2. 前向/反向传播时，通过 All-Gather 收集所需参数
        3. 梯度通过 Reduce-Scatter 聚合并分区
        4. 优化器更新只涉及本地参数分区
        """
        # 初始化分区：将参数分配给不同 GPU
        self._partition_parameters()

        # 训练循环
        for batch in dataloader:
            # ===== 前向传播 =====
            # 1. 收集完整参数（All-Gather）
            params = self._all_gather_parameters()

            # 2. 前向计算
            output = self.model(params, batch)

            # 3. 释放非本地参数（节省显存）
            self._release_nonlocal_parameters()

            # ===== 反向传播 =====
            # 4. 重新收集参数进行反向计算
            params = self._all_gather_parameters()
            output.backward()

            # 5. 梯度分区聚合（Reduce-Scatter）
            self._reduce_scatter_gradients()

            # 6. 释放参数
            self._release_nonlocal_parameters()

            # ===== 优化器更新 =====
            # 7. 只更新本地参数分区
            self.optimizer.step()

            # 8. 同步更新后的参数
            self._broadcast_updated_parameters()

    def _partition_parameters(self):
        """将模型参数分区给不同 GPU"""
        for name, param in self.model.named_parameters():
            # 根据 rank 计算参数分区的起始和结束位置
            start = self.rank * len(param) // self.world_size
            end = (self.rank + 1) * len(param) // self.world_size
            # 每个 GPU 只存储自己负责的参数分区
            param.data = param.data[start:end]

    def _all_gather_parameters(self) -> torch.Tensor:
        """通过 All-Gather 收集完整参数"""
        # 注意：实际实现中只在需要时收集，而非存储完整副本
        tensor_list = [torch.zeros_like(param) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, self.local_param)
        return torch.cat(tensor_list, dim=0)

    def _reduce_scatter_gradients(self):
        """通过 Reduce-Scatter 聚合梯度并分区"""
        # 每个 GPU 只保留自己负责参数的梯度
        dist.reduce_scatter(self.local_grad,
                           op=dist.ReduceOp.SUM,
                           group=self.process_group)

    def _broadcast_updated_parameters(self):
        """将更新后的参数广播给所有 GPU"""
        dist.broadcast(self.local_param, src=self.rank)
```

---

## 8. 通信量分析

| ZeRO Stage | 通信量（每参数字节） | 适用场景 |
|-----------|-------------------|---------|
| ZeRO-1 | $2\phi$ (All-Gather) | 单机多卡，中等规模 |
| ZeRO-2 | $2\phi$ + $2\phi$ (All-Gather + Reduce-Scatter) | 多机多卡 |
| ZeRO-3 | $2\phi N_{d}$ (多次 All-Gather) | 超大规模（万亿参数）|

其中 $\phi$ 是参数字节数（FP16 为 2 字节）。

---

*精读日期: 2026-04-15*

# Image tokenizer 技术分析

---

## 1. Dead Codebook问题

在VQ-VAE 训练过程中，某些 embedding 可能 从未或极少被使用 ；比如

|Embedding 0:  ████████████ 1200次   │
│ Embedding 1:  ██ 50次               │
│ **Embedding 2:  0次 ← DEAD!**           │
│ Embedding 3:  ████████ 800次        │
│ **Embedding 4:  0次 ← DEAD!**   |

Dead Codebook 会影响码本的表达能力，只使用部分Embedding 来表示信息；导致离散化空间变小，重建质量变差

**1、增加EMA**

通过计算器统计每个码本的使用情况

 `self.register_buffer('cluster_size', torch.zeros(self.n_e))  # 初始化为0 `

                     多卡训练同步流程                            ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  GPU 0: curr_cluster_size = [2, 1, 0, ...]                    │
    │  GPU 1: curr_cluster_size = [1, 2, 1, ...]                    │
    │  GPU 2: curr_cluster_size = [0, 3, 2, ...]                    │
    │                    ↓                                           │
    │         all_reduce(SUM) 求和                                   │
    │                    ↓                                           │
    │  所有GPU: curr_cluster_size = [3, 6, 3, ...]  ← 同步结果       │
    │                    ↓                                           │
    │         EMA 更新 cluster_size                                  │
    │                    ↓                                           │
    │  Master (rank 0) 处理 dead codebook                            │
    │                    ↓                                           │
    │         broadcast 广播到所有卡                                  │
    │                    ↓                                           │
    │  所有GPU: cluster_size 和 embed_avg 同步                       │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
**2、Dead Code 重启**

             分布式 Dead Code 重启流程                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Step 1: 只有 rank 0 执行识别和重置                              │
    │  ┌─────────────────────────────────────────┐                   │
    │  │ Rank 0:                                 │                   │
    │  │   - 计算 dead_threshold                 │                   │
    │  │   - 识别 dead_indices                   │                   │
    │  │   - 随机选择样本重置 embedding           │                   │
    │  │   - 更新 embedding.weight.data          │                   │
    │  └─────────────────────────────────────────┘                   │
    │                    ↓                                            │
    │  Step 2: 广播到所有卡                                           │
    │  ┌─────────────────────────────────────────┐                   │
    │  │ dist.broadcast(embedding.weight, src=0) │                   │
    │  │   Rank 0: 发送数据                      │                   │
    │  │   Rank 1,2,3...: 接收数据               │                   │
    │  └─────────────────────────────────────────┘                   │
    │                    ↓                                            │
    │  Step 3: 所有卡同步完成                                         │
    │  ┌─────────────────────────────────────────┐                   │
    │  │ 所有卡的 embedding.weight 完全一致       │                   │
    │  └─────────────────────────────────────────┘                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
只有 rank 0 做 dead code 重置 或相关计算，但每张卡数据分布不均且固定分配会导致其他卡上相关信息利用不足；同步所有节点信息

```
   # 收集所有卡的样本
    all_z = [torch.zeros_like(z_flattened) for _ in range(dist.get_world_size())]
	dist.all_gather(all_z, z_flattened)
	all_z = torch.cat(all_z, dim=0)  # 合并所有卡的样本
   # 从合并后的样本中选择
	sel_indices = torch.randperm(all_z.shape[0], device=z_flattened.device)[:n_resets]
    new_embeds = all_z[sel_indices].detach()
```

 基于预训练的VAE模型，加载encoder  和 decoder后，**冻住encoder**,  只训量化器和decoder

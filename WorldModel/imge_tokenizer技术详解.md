# Image tokenizer 技术分析

**问题描述：**代码原来使用encoder 和 decoder  训练过一个性能不错的VAE模型，然后加上量化部分VectorQuantizer，训练出的模型重建出的图像有马赛克和模糊的问题

**原因分析：**

    ┌─────────────────────────────────────────────────────────────────┐
    │                    根本原因                                      │
    ├─────────────────────────────────────────────────────────────────┤                                                          │
    │  VAE:                                                           │
    │  ────                                                           │
    │  z 是连续的，Encoder 和 Decoder 可以学习任意精度                 │
    │  潜在空间是平滑的，相邻位置特征相似                              │
    │                                                                 │
    │  VQ-VAE:                                                        │
    │  ────────                                                       │
    │  z_q 是离散的，只能从有限的 codebook 中选择                      │
    │  如果 codebook 初始化不好 → 量化误差大                          │
    │  如果 codebook 分布不覆盖 encoder 输出 → 马赛克                 │
    │  如果没有平滑机制 → 空间不连续 → 马赛克边界                     │
    │                                                                 │
    │  本质问题:                                                       │
    │  ══════════                                                     │
    │  Encoder 输出的分布 ≠ Codebook 覆盖的分布                       │
    │  量化误差 = encoder 输出 - 最近邻 embedding                      │
    │  这个误差没有被有效优化                                          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
       量化导致马赛克/模糊的根本原因                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  问题 1: Codebook 初始化范围太小                                 │
    │  ────────────────────────────────                               │
    │  uniform(-1/n_e, 1/n_e) → 范围极小
    │  初始化范围: [-1/4096, 1/4096] = [-0.000244, 0.000244]│
    │  Encoder 输出范围大 → 量化误差巨大   
    │	问题: 所有距离几乎相同！
    │     argmin 没有意义，随机选择 embedding
    │                                                                 │
    │  问题 2: 量化后的空间不连续                                      │
    │  ─────────────────────────                                       │
    │  相邻位置选择不同 embedding → 边界突变                          │
    │  Decoder 无法平滑处理 → 马赛克                                  │
    │                                                                 │
    │  问题 3: z 和 z_q 的维度不匹配                                   │
    │  ─────────────────────────                                       │
    │  quant_conv: z_channels → embed_dim                            │
    │  post_quant_conv: embed_dim → z_channels                       │
    │  量化在 embed_dim 空间进行，但 decoder 需要 z_channels          │
    │  维度转换可能引入额外误差                                        │
    │                                                                 │
    │  问题 4: 缺少量化后平滑                                          │
    │  ─────────────────────────                                       │
    │  z_q 直接送入 decoder → 离散跳跃特征                            │
    │  Decoder 训练时没见过这种特征 → 重建质量差                       │
    │                                                                 │
    │  问题 5: Legacy Loss 计算错误                                    │
    │  ─────────────────────────                                       │
    │  beta 应用到错误的 loss 项                                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘


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

## 解决方案（按优先级排序）
### 方案 1：添加 Post-Quantization Convolution ⭐⭐⭐⭐⭐
这是 最有效的解决方案 ，也是现代 VQ-VAE 的标准做法； 通过post_quantization_conv将量化后的码本特征实现平滑处理

量化前: z_e = [0.1, 0.3, 0.5, ...]  连续值，平滑
量化后: z_q = [0.1, 0.5, 0.9, ...]  离散化，有跳跃
Post-Conv: z_q = [0.15, 0.35, 0.55, ...]  再次平滑

基于预训练的VAE模型，加载encoder  和 decoder后，**冻住encoder**,  只训量化器和decoder

### 方案 2：使用 Pixel Shuffle 减少下采样 ⭐⭐⭐⭐

```
class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        
        self.conv_stack = nn.Sequential(
            # 只做 2 倍下采样（而不是 4 倍）
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 移除第三层或改为 stride=1
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )
      

# 传统方式：转置卷积 (ConvTranspose2d)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2)
# 操作：在输入周围填充零，然后做卷积
# 问题：容易产生棋盘格伪影

# Pixel Shuffle 方式
nn.Conv2d(in_channels, out_channels * r**2, kernel_size=3, padding=1)  # 先增加通道数
nn.PixelShuffle(r)  # 然后重排像素
# 操作：将通道维度转换为空间维度
# 优势：更平滑的上采样，无棋盘格伪影

### decoder 增加 ####
    # 添加：量化后平滑模块
   self.smooth_quant = nn.Sequential(
            nn.Conv2d(cfg.get("z_channels"), cfg.get("z_channels"), kernel_size=3, 					stride=1, padding=1),
            nn.GroupNorm(32, cfg.get("z_channels")),
            nn.SiLU(),
            nn.Conv2d(cfg.get("z_channels"), cfg.get("z_channels"), kernel_size=3, 					stride=1, padding=1),
        )
```


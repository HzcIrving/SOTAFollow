# DVGT-2 论文精读报告

> **论文**: DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale
> **arXiv**: [2604.00813](https://arxiv.org/abs/2604.00813)
> **作者**: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, Jiwen Lu
> **机构**: 清华大学 · 小米汽车 · 澳门大学 · 北京大学
> **日期**: 2026年4月1日
> **代码**: [https://github.com/wzzheng/DVGT](https://github.com/wzzheng/DVGT)

---

## 1. 一句话总结

DVGT-2 提出了 **Vision-Geometry-Action（VGA）** 端到端自动驾驶新范式：用稠密 3D 几何替代语言描述或稀疏感知作为规划的核心表征，通过滑动窗口流式架构实现 **O(1) 每帧复杂度**（仅 266ms 延迟），在 NAVSIM v1 仅用相机达到 **PDMS 90.3（SOTA）**，同时在几何重建任务上超越所有流式和批处理方法。

---

## 2. 为什么这个问题值得研究？

### 2.1 两条老路都走到头了

**第一条路：稀疏感知 + 规划**

UniAD、VAD 等传统端到端方法，用 3D 检测、地图分割等稀疏感知结果来引导规划。但 bounding box 和地图元素会丢弃大量环境细节，voxel 化的 occupancy 存在量化误差。场景结构建模得不完整、不准确，规划的天花板就很低。

**第二条路：Vision-Language-Action（VLA）**

Emma、AutoVLA、DriveVLA 等 VLA 方法借助预训练 VLM 的语义理解能力，用自然语言解读驾驶场景。但语言有两个先天不足：

- **粒度太粗**：语言很难精确描述三维空间关系和几何细节
- **歧义性**：同一张图可以有不同的文字描述，引入不确定性

**那有没有更好的表征？**

DVGT-2 的核心洞察：车辆终归是在一个 **3D 世界**里运作的，理解环境的稠密 3D 几何，才是最直接、最全面的决策依据。

### 2.2 几何重建的老难题：算不起、用不上

**批处理方法（DVGT/VGGT）**：处理 T 帧的复杂度是 **O(T²)**，在线推理时每帧都要重新算全部历史帧，延迟爆炸。

**全历史流式方法（StreamVGGT）**：复杂度降到 O(T)，但内存随序列长度线性增长，开几分钟后缓存就爆了。

DVGT-2 的核心贡献：用滑动窗口流式架构彻底解决这个矛盾——**O(1) 每帧复杂度 + O(1) 常驻内存**，同时联合预测几何和轨迹。

---

## 3. 方法详解

### 3.1 VGA 范式的数学框架

**传统 E2E 范式**：

$$\mathbf{Z} = \mathcal{F}_{\text{perc}}(\mathbf{I}_{t-T:t}), \quad \mathbf{V} = \mathcal{F}_{\text{pred}}(\mathbf{Z}), \quad \mathbf{A}_t = \mathcal{F}_{\text{plan}}(\mathbf{Z}, \mathbf{V})$$

**VLA 范式**：

$$\mathbf{A}_t, \mathbf{L}_t = \mathcal{M}_{\text{VLA}}(\mathbf{I}_{t-T:t})$$

**VGA 范式（DVGT-2）**：

$$\mathbf{A}_t, \mathbf{P}_{t-T:t}, \mathbf{E}_{t-T:t} = \mathcal{M}_{\text{VGA}}(\mathbf{I}_{t-T:t})$$

同时重建稠密 3D 点云 $\mathbf{P}$ 和自车位姿 $\mathbf{E}$，然后直接驱动轨迹规划。点云是连续坐标空间，无量化误差，像素级对齐。

### 3.2 三种几何重建范式的对比

**批处理范式**（DVGT/VGGT）：

$$\mathbf{P}_{t-T:t}, \mathbf{E}_{t-T:t} = \mathcal{G}_{\text{batch}}(\mathbf{I}_{t-T:t})$$

O(T²) 复杂度，每帧重复计算历史帧，无法在线推理。

**全历史流式范式**（StreamVGGT）：

$$\mathbf{P}_t, \mathbf{E}_t, \mathbf{C}_{t-T:t} = \mathcal{G}_{\text{stream}}([\mathbf{I}_t, \mathbf{C}_{t-T:t-1}])$$

O(T) 复杂度，但缓存无限增长，不适合无限长度驾驶。

**DVGT-2 滑动窗口范式**：

$$\mathbf{P}_t, \mathbf{E}_t, \mathbf{C}_{t-W+1:t} = \mathcal{G}_{\text{window}}([\mathbf{I}_t, \mathbf{C}_{t-W:t-1}])$$

固定窗口 $W$，O(1) 每帧复杂度 + O(1) 内存。

FIFO 缓存更新：

$$\mathbf{C}_{t-W+1:t} = \text{FIFO}(\mathbf{C}_{t-W:t-1}, \hat{\mathbf{G}}_t)$$

### 3.3 整体架构

```
多视角图像 I_t (V×H×W×3)
    ↓
[图像编码器 E]  ViT-L / DINOv3 预训练
    ↓
视觉 Token + Pose Token + Trajectory Token
    ↓
[几何 Transformer G]  L=24 层
  - Intra-View Local Attention
  - Cross-View Spatial Attention
  - Temporal Causal Attention（核心创新）
    ↓
[预测头]
  - DPT头 → 稠密 3D 点云 P_t
  - Anchor-based Diffusion头 → 相对位姿 E_t
  - Anchor-based Diffusion头 → 未来轨迹 A_t
```

时序因果注意力采用 **MRoPE-I**（相对时序位置编码），保证缓存特征在任意时间步可复用。

### 3.4 局部点云预测的动机

DVGT-2 预测**当前帧 ego 坐标系下的局部点云**，而非全局点云。这个设计是刻意的：

- **Ray depth（局部几何精度）显著 SOTA**：对自车周围障碍物检测和避让，这个指标比全局点云精度更直接影响安全性
- **无全局累积误差的源头**：全局点云需要从第一帧累积相对位姿，累积误差随时间扩散
- **局部点云可重建全局**：用预测的相对位姿 $E_t$ 迭代变换即可（引入少量累积误差，但 ray depth 不受影响）

### 3.5 Anchor-based Diffusion 预测头

位姿 $E_t$（6DoF）和轨迹 $A_t$（N 步 x-y-yaw）采用基于锚点的扩散头，参考 DiffusionDrive。

**截断扩散策略**：不像标准扩散需要 1000 步去噪，截断扩散只做几步就能得到高质量样本，适合实时控制。两个头各用 20 个预计算锚点（训练数据聚类得到）。

---

## 4. 训练与推理伪代码

### 4.1 训练伪代码

```python
# ===== DVGT-2 训练伪代码 =====
# 两阶段训练：几何重建预训练 → VGA联合训练

# ===== 阶段1：几何重建预训练（关闭流式机制） =====
# 目标：让模型学会从多视角图像中恢复稠密几何
# 数据：2~8 views, 2~24 frames per scene

for iter in range(160000):                        # 160K iterations
    batch = sample_batch(dataloader)

    # 1. 图像编码（ViT-L / DINOv3，冻结主干）
    F_vis = image_encoder(batch.images)            # [B, V*H*W, D]

    # 2. 拼接 pose token 和 trajectory token
    F_combined = concat([F_vis, pose_tokens, traj_tokens])

    # 3. 几何 Transformer 前向（不使用历史缓存，纯帧内+跨视角）
    G_out = geometry_transformer(F_combined)

    # 4. 三路解码
    P_pred = dpt_head(G_out['vis_tokens'])         # → 稠密3D点云
    E_pred = pose_diffusion_head(G_out['pose_tokens'])  # → 相对位姿

    # 5. 几何损失（MoGe-2 深度伪标签监督）
    loss_geo = depth_loss(P_pred, moge2_depth_labels)
    loss_geo.backward()
    optimizer.step()


# ===== 阶段2：VGA联合训练（开启流式 + 轨迹规划） =====
# 目标：联合优化几何重建 + 轨迹规划，引入流式推理机制

for iter in range(80000):                         # 80K iterations
    batch = sample_batch(dataloader)

    # 1. 图像编码
    F_vis = image_encoder(batch.images)

    # 2. 构造输入序列，注入自车状态（速度、加速度、驾驶命令）
    F_combined = concat([F_vis, pose_tokens, traj_tokens])
    F_combined = inject_ego_status(F_combined, ego_velocity, ego_accel, command)

    # 3. 几何 Transformer（含时序因果注意力的滑动窗口流式）
    cache = initialize_cache(window_size=4)         # 初始为空

    all_P, all_E, all_A = [], [], []

    for frame_idx in range(num_frames):
        current_feat = F_combined[frame_idx]
        cached_feat = cache.get()                   # 最近W帧的特征

        # 时序因果注意力：Query=当前帧，KV=历史缓存
        G_current = geometry_transformer_streaming(
            current_feat, cached_feat, window_size=4
        )

        # FIFO 缓存更新
        cache.push(G_current)

        # 三路解码
        P_t = dpt_head(G_current['vis_tokens'])     # 局部点云
        E_t = pose_diffusion_head(G_current['pose_tokens'])  # 相对位姿
        A_t = traj_diffusion_head(G_current['traj_tokens'])  # 未来轨迹

        all_P.append(P_t)
        all_E.append(E_t)
        all_A.append(A_t)

    # 6. 多任务损失
    loss_geo  = depth_loss(all_P, moge2_labels)
    loss_pose = pose_l1_loss(all_E, gt_relative_poses)
    loss_traj = traj_l1_loss(all_A, gt_future_trajs)

    total_loss = loss_geo + loss_pose + loss_traj
    total_loss.backward()
    optimizer.step(lr=1e-4, cosine_annealing)
```

### 4.2 推理伪代码

```python
# ===== DVGT-2 流式推理伪代码 =====
# 核心特点：O(1) 每帧复杂度，O(1) 内存，支持无限长度驾驶

def dvgt2_streaming_inference(multiview_images, window_size=4):
    cache = []                                      # 固定窗口缓存，初始为空
    accumulated_pose = np.eye(4)                    # 累积全局位姿

    encoder = build_vit_encoder(pretrained='dinov3')
    transformer = build_geometry_transformer(layers=24)
    dpt_head = build_dpt_head()
    pose_diffusion_head = build_anchor_diffusion_head(num_anchors=20)
    traj_diffusion_head = build_anchor_diffusion_head(num_anchors=20)

    results = []

    for t, frame_batch in enumerate(multiview_images_stream):
        # 步骤1：图像编码 → 视觉Token
        vis_tokens = encoder(frame_batch)

        # 步骤2：拼接任务Token
        pose_tokens = repeat(learnable_pose_token, num_tokens_per_view)
        traj_tokens = repeat(learnable_traj_token, num_tokens_per_view)
        input_tokens = concat([vis_tokens, pose_tokens, traj_tokens])

        # 步骤3：时序因果注意力（MRoPE-I 相对时序位置编码）
        if len(cache) == 0:
            out_tokens = transformer.intra_cross_attn(input_tokens)
        else:
            out_tokens = transformer.temporal_causal_attn(
                query=input_tokens,
                key_value=cache
            )

        # 步骤4：FIFO 缓存更新（固定大小 W=4）
        current_layer_features = transformer.get_intermediate_features(out_tokens)
        cache = fifo_update(cache, current_layer_features, window_size)

        # 步骤5：三路解码
        P_t = dpt_head(out_tokens['vis_tokens'])         # [V*H*W, 3] 3D点坐标
        E_t_rel = pose_diffusion_head(out_tokens['pose_tokens'])  # [7] 平移+四元数
        A_t = traj_diffusion_head(out_tokens['traj_tokens'])      # [N, 3] x,y,yaw

        # 步骤6：累积全局位姿
        E_t_global = accumulated_pose @ transform_to_matrix(E_t_rel)
        accumulated_pose = E_t_global

        results.append({
            'timestamp': t,
            'local_points': P_t,
            'relative_pose': E_t_rel,
            'global_pose': E_t_global,
            'future_trajectory': A_t,
        })

    return results
```

---

## 5. 实验结果

### 5.1 几何重建性能（OpenScene）

| 方法 | 范式 | Acc↓ | Comp↓ | Abs Rel↓ | δ<1.25↑ | 延迟/帧 |
|------|------|------|-------|---------|---------|---------|
| VGGT* | 全序列 | 1.705 | 1.711 | 0.280 | 0.669 | ~5.31s |
| DVGT | 全序列 | 0.412 | 0.491 | 0.048 | 0.971 | ~1.88s |
| StreamVGGT* | 流式（全历史） | 2.209 | 2.060 | 0.303 | 0.620 | ~1.94s |
| **DVGT-2** | **滑动窗口** | **0.440** | **0.450** | **0.040** | **0.977** | **~0.27s** |

**Abs Rel / δ<1.25（Ray depth）全面 SOTA**；延迟仅 **266ms/帧**，比 DVGT 快 7 倍。

### 5.2 闭环规划（NAVSIM v1）

| 方法 | 输入 | 辅助监督 | PDMS↑ |
|------|------|---------|-------|
| UniAD | Camera | Map+Box+Mot.+Occ | 83.4 |
| Hydra-MDP | Cam+Lidar | Map+Box | 86.5 |
| DiffusionDrive | Cam+Lidar | Map+Box | 88.1 |
| DriveVLA-W0 | Camera | Future States | 90.2 |
| **DVGT-2** | **Camera** | **Dense Geometry** | **88.6** |
| **DVGT-2-NAVSIM** | **Camera** | **Dense Geometry** | **90.3** |

**DVGT-2-NAVSIM 达到 NAVSIM v1 SOTA（PDMS 90.3）**，超越所有 VLA 和端到端方法，仅用相机。

### 5.3 开环规划（nuScenes）

| 方法 | L2@1s↓ | L2@2s↓ | L2@3s↓ | 碰撞率@3s↓ |
|------|--------|--------|--------|-----------|
| UniAD | 0.48 | 0.96 | 1.65 | 0.71 |
| GenAD | 0.36 | 0.83 | 1.55 | 0.43 |
| GaussianAD | 0.40 | 0.64 | 0.88 | 0.81 |
| **DVGT-2** | **0.25** | **0.67** | **1.43** | **0.50** |
| **DVGT-2†** | **0.20** | **0.37** | **0.66** | **0.47** |

†表示使用历史帧平均作为 planning 初始化的通用技巧。

碰撞率显著低于 GenAD（0.50 vs 0.43），说明稠密几何让模型学到了更全面的 3D 结构物理交互。

### 5.4 效率对比

| 指标 | VGGT | DVGT | StreamVGGT | **DVGT-2** |
|------|------|------|-----------|------------|
| 每帧延迟 | 5.31s | 1.88s | 1.94s | **0.27s** |
| 内存增长 | O(T²) | O(T²) | O(T) | **O(1)** |

266ms 每帧 + 常数内存，是目前最接近实际车载部署的几何重建+规划联合模型。

---

## 6. 消融实验

### 6.1 窗口大小的影响

| 窗口大小 W | Acc↓ | Abs Rel↓ |
|-----------|------|---------|
| 2 | 0.613 | 0.042 |
| **4** | **0.480** | **0.042** |
| 6 | 0.474 | 0.042 |
| 8 | 0.501 | 0.042 |

- **Ray depth 与窗口大小无关**（局部几何，只取决于当前帧）
- **W=6 时 Acc 最优**，W=4 次优；W=8 时 Acc 反而下降——更长的累积链引入更多 ego-pose 误差
- **W=4 是精度-效率的最佳平衡点**

### 6.2 轨迹规划多数据集泛化

| 数据集 | L2@1s (m)↓ |
|--------|-----------|
| OpenScene | 0.20 |
| nuScenes | 0.56 |
| Waymo | 0.78 |
| KITTI | 2.12 |
| DDAD | 2.00 |

KITTI/DDAD 误差超 2m，是数据分布偏差——这两个数据集自车速度远高于其他，而 anchor 聚类偏向 OpenScene（占训练数据 75%+）。

---

## 7. 核心洞察

### 7.1 为什么 VGA 比 VLA 更适合自动驾驶？

VLA 用语言作为中间表征，粒度和歧义性是难以克服的根本问题。VGA 用稠密点云：

- **像素级空间精度**：每个像素都有对应 3D 坐标
- **连续空间无量化误差**
- **完整覆盖**：前景物体和背景环境同时建模

### 7.2 局部点云是一个被低估的设计

**Ray depth 才是对规划最有价值的部分**——自车做决策时，最需要知道的是"我周围有多近有什么"，而不是"整个城市的三维地图"。

### 7.3 O(1) 复杂度是工程落地的真正门槛

266ms 每帧 + 常数内存，让 DVGT-2 成为目前最接近实际车载部署的几何重建+规划联合模型。

---

## 8. 总结

DVGT-2 重新回答了"规划到底需要什么样的表征"这个问题。稠密 3D 几何，不是感知的一个额外输出，而是规划的最优输入。

**三大核心贡献：**
1. **VGA 范式**：首次系统验证稠密几何作为规划核心表征的可行性
2. **滑动窗口流式架构**：O(1) 复杂度 + O(1) 内存，266ms/帧，实时可部署
3. **联合几何+轨迹**：不依赖稀疏感知标注、不依赖语言标注，仅用几何监督达到 SOTA 规划性能

---

## 附录：关键符号表

| 符号 | 含义 |
|------|------|
| $\mathbf{I}_t$ | $t$ 时刻的多视角图像输入 |
| $\mathbf{P}_t$ | $t$ 时刻的稠密 3D 点云（当前 ego 坐标系） |
| $\mathbf{E}_t$ | $t$ 时刻的相对自车位姿（6DoF） |
| $\mathbf{A}_t$ | 未来 N 步规划轨迹 |
| $\mathbf{C}_{t-W:t-1}$ | 固定大小 W 的历史特征缓存 |
| $\mathcal{G}_{\text{window}}$ | 滑动窗口流式几何重建 |
| MRoPE-I | 相对时序位置编码，保证缓存特征可跨时间步复用 |
| FIFO | 先进先出缓存更新策略 |

---

*精读日期: 2026-04-07*

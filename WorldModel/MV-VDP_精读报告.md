# MV-VDP 精读报告：Multi-View Video Diffusion Policy

> **论文**：Multi-View Video Diffusion Policy: A 3D Spatio-Temporal-Aware Video Action Model
> **arXiv**：https://arxiv.org/abs/2604.03181（cs.RO）
> **发布**：2026-04-03
> **机构**：中国科学院自动化研究所 · 中国科学院大学 · FiveAges · 清华大学 · 西安交通大学 · 武汉大学 · 南京大学
> **作者**：Peiyan Li, Yixiang Chen, Yuan Xu, Jiabing Yang, Xiangnan Wu, Jun Guo, Nan Sun, Long Qian, Xinghang Li, Xin Xiao, Jing Liu, Nianfeng Liu, Tao Kong, Yan Huang, Liang Wang, Tieniu Tan
> **项目主页**：https://lpy1219.github.io/MV-VDP-Web/

---

## 🔬 一句话总结

**MV-VDP** 将点云与末端执行器姿态通过多视角正交投影转化为 RGB 图 + 热力图，借助视频扩散基础模型（Wan2.2）联合预测未来多视角 RGB 视频与热力图序列，从而解码出连续动作 chunks，在 10 条演示轨迹的极低数据条件下即可完成复杂操作任务。

![MV-VDP Teaser](https://arxiv.org/html/2604.03181v1/x1.png)

---

## 🎯 二、核心贡献

1. **概念创新**：首次利用视频基础模型（Video Foundation Model）构建 3D Video-Action Model（3D VAM），将视频预测与动作微调的表达格式对齐。
2. **方法创新**：多视角正交投影将 3D 点云编码为多视角 RGB 图，将末端执行器姿态编码为多视角热力图；视频扩散模型联合预测未来 RGB 视频 + 热力图序列。
3. **实验验证**：Meta-World 7 任务平均 89.1% 成功率（5 条 demos），真实机器人平台平均 57.1%（10 条 demos），全面超越 Video-prediction-based、3D-based 和 VLA 模型。

---

## 🏗️ 三、方法详解

### 3.1 问题定义

机器人操作需要同时理解环境的 3D 空间结构和时序演化。现有方法存在两个主要局限：

1. **2D 视觉观察 + 图文预训练 backbone**：依赖静态图文对预训练的模型，无法充分建模环境动态
2. **3D 建模不完整**：部分方法虽引入 3D 先验，但缺乏时序建模能力

MV-VDP 的核心思路：用视频基础模型同时预测未来的 RGB 视频和热力图视频，RGB 视频建模环境动态，热力图视频编码动作信息。

### 3.2 整体 Pipeline

```
输入：彩色点云 + 末端执行器姿态
  ↓
① 多视角投影模块
  ├─ 点云 → 多视角正交投影 RGB 图像（3个固定视角）
  └─ 末端执行器姿态 → 多视角 Gaussian 热力图 Hit(u,v)
  ↓
② 多视角视频扩散 Transformer（Wan2.2 + View-Attention）
  ├─ 编码：VAE encoder 将 RGB 图 + 热力图编码为 latent
  ├─ 沿视角维度拼接：(B, V, T×H×W, C)
  ├─ View-Attention：跨视角交互（核心增强）
  └─ DDPM 去噪 → 预测未来多视角 RGB 视频 latent + 热力图 latent
  ↓
③ 动作解码
  ├─ 热力图 back-projection → 3D 末端执行器位置轨迹
  └─ 轻量级 170M Transformer（latent 全局特征 + 热力图峰值局部特征融合）
      → Roll（72 bins，5°分辨率）+ Pitch（72 bins）+ Yaw（72 bins）+ 夹爪状态（二值）
  ↓
输出：连续 action chunk（位置 + 旋转 + 夹爪）
```

![MV-VDP Architecture](https://arxiv.org/html/2604.03181v1/x2.png)

### 3.3 多视角投影（Multi-View Projection）

**输入处理**：裁剪点云至 1m³ 工作空间，减少冗余信息。

**热力图构建**：以末端执行器投影像素为中心，生成 Gaussian 热力图：

$$H_i^t(\mathbf{x}) = \begin{cases} p_i^t(\mathbf{x}), & \text{if } p_i^t(\mathbf{x}) \geq \tau \\ 0, & \text{otherwise} \end{cases}$$

$$p_i^t(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x} - \widehat{\mathbf{x}}_i^t\|^2}{2\sigma^2}\right)$$

其中 $\widehat{\mathbf{x}}_i^t$ 是第 $i$ 个末端执行器在视角 $t$ 上的投影中心，$\sigma$ 控制热力图标准差，$\tau$ 为截断阈值。低于 $\tau$ 的像素置零，热力图经着色后可直接送入预训练 VAE encoder。

### 3.4 多视角视频扩散（Multi-View Video Diffusion）

**基础模型**：Wan2.2（5B 参数视频基础模型，单视角预训练）

**多视角扩展**：每个 DiT block 插入 **View-Attention** 模块，实现跨视角交互。

token reshape：
$$(B, V, T \times H \times W, C) \rightarrow (B, T, V \times H \times W, C)$$

在每个时间步显式建模跨视角交互。

**损失函数**：双分支联合预测

$$L_{\text{diff}} = \lambda L_{\text{vid}} + (1 - \lambda) L_{\text{heat}}$$

实验中 $\lambda = 0.5$（RGB 视频与热力图等权重）。

**训练策略**：LoRA 微调（Full fine-tuning 无收益，但显著增加开销）。

### 3.5 动作解码（Action Decoding）

**位置解码**：热力图峰值 back-projection（已知相机参数）→ 连续 3D 轨迹。

**旋转 & 夹爪解码**：
- 输入：去噪后的 latent（VAE 压缩后的时序特征）+ 热力图峰值位置
- 局部特征（热力图峰值处）+ 全局特征（整个 latent）融合
- Conditioning：首帧 latent 作为 condition，cross-attention 注入

**旋转预测损失**：
$$L_{\text{pred}} = L_{\text{rol}} + L_{\text{pit}} + L_{\text{yaw}} + L_{\text{gri}}$$

Roll/Pitch/Yaw 各 72 bins（5° 分辨率），交叉熵损失；夹爪二分类（开/闭）。

---

## 📋 四、训练与推理伪代码

### 4.1 训练伪代码

```python
# ===== MV-VDP 训练伪代码 =====
# 核心：用视频扩散模型联合预测未来 RGB 视频 + 热力图视频

# 超参数
window_size = 16        # 输入帧数
future_frames = 24      # 预测未来帧数
num_views = 3           # 固定 3 个投影视角
crop_size = 1.0         # 点云裁剪 1m³
heatmap_sigma = ...     # Gaussian 热力图标准差
heatmap_tau = ...       # 热力图截断阈值
lambda_rgb = 0.5        # RGB 损失权重

for iter in range(max_iters):
    batch = sample_batch(dataloader)   # 采样彩色点云 + 末端执行器姿态

    # ===== 步骤1：多视角投影 =====
    for view in range(num_views):
        # 点云正交投影 → RGB 图像
        rgb_view = orthographic_projection(
            point_cloud=crop_pointcloud(batch.points, crop_size),
            view_angle=view_angles[view]
        )
        # 末端执行器姿态 → Gaussian 热力图
        heatmap_view = gaussian_heatmap(
            ee_pose=batch.ee_pose,
            view_angle=view_angles[view],
            sigma=heatmap_sigma,
            tau=heatmap_tau
        )
        # 着色（colorization），使热力图可直接输入 VAE
        heatmap_colored = colorize(heatmap_view)

    # ===== 步骤2：VAE 编码 =====
    input_views = concat([rgb_view, heatmap_colored], dim=channel)  # 沿通道拼接
    z_0 = vae_encoder(input_views)   # (B, V, T*H*W, C) latent

    # ===== 步骤3：DDPM 前向加噪 =====
    t ~ Uniform({1, ..., T})        # 随机采样时间步
    noise ~ N(0, I)
    z_t = sqrt(ᾱ_t) * z_0 + sqrt(1 - ᾱ_t) * noise

    # ===== 步骤4：多视角视频扩散 Transformer =====
    for dit_block in range(num_dit_blocks):
        # View-Attention：跨视角交互
        z_t = view_attention(z_t)   # (B, T, V*H*W, C)
        # 标准 DiT block
        z_t = dit_block(z_t, t)

    # ===== 步骤5：双分支预测 =====
    z_pred = denoiser(z_t)          # 预测 noise / z_0
    rgb_latent_pred = z_pred[..., :C]      # RGB 分支
    heatmap_latent_pred = z_pred[..., C:]  # 热力图分支

    # ===== 步骤6：损失计算 =====
    L_vid = MSE(rgb_latent_pred, rgb_latent_true)
    L_heat = MSE(heatmap_latent_pred, heatmap_latent_true)
    L_diff = lambda_rgb * L_vid + (1 - lambda_rgb) * L_heat

    # ===== 步骤7：动作解码器训练（与扩散模型联合） =====
    # VAE decoder 解码热力图 latent → 视频
    heatmap_video = vae_decoder(heatmap_latent_pred)
    # 峰值检测 + back-projection → 3D 位置轨迹
    positions_3d = back_project_peaks(heatmap_video, camera_params)

    # 全局特征（flatten latent）+ 局部特征（峰值处）融合
    global_feat = flatten(heatmap_latent_pred)  # (B, D)
    local_feat = interpolate_local(heatmap_latent_pred, peak_coords)  # (B, D)
    action_feat = concat([global_feat, local_feat])

    # Cross-attention conditioning with first frame
    cond = first_frame_latent  # (B, D)
    action_feat = cross_attention(q=action_feat, kv=cond)

    # 旋转 + 夹爪预测
    roll_logits = linear(action_feat, 72)     # 72 bins, 5°/bin
    pitch_logits = linear(action_feat, 72)
    yaw_logits = linear(action_feat, 72)
    gripper_logits = linear(action_feat, 2)  # binary

    L_rot = CrossEntropy(roll_logits, gt_roll) + \
            CrossEntropy(pitch_logits, gt_pitch) + \
            CrossEntropy(yaw_logits, gt_yaw)
    L_gri = CrossEntropy(gripper_logits, gt_gripper)
    L_pred = L_rot + L_gri

    # ===== 总损失 =====
    total_loss = L_diff + L_pred
    total_loss.backward()
    optimizer.step()
```

### 4.2 推理伪代码

```python
# ===== MV-VDP 流式推理伪代码 =====
# 核心：1 步去噪即可获得高质量动作预测（热力图分布简单）

def mv_vdp_streaming_inference(colored_point_cloud, ee_pose_init, num_steps=5):
    """
    Args:
        colored_point_cloud: (H, W, 3) 彩色点云
        ee_pose_init: 初始末端执行器姿态
        num_steps: DDPM 去噪步数（论文推荐 N=5，平衡质量与速度）
    Returns:
        action_chunk: (future_frames, 7) 位置(3) + 旋转(3) + 夹爪(1)
    """

    # ===== 步骤1：多视角投影 =====
    for view in range(num_views):
        rgb_view = orthographic_projection(point_cloud, view_angles[view])
        heatmap_view = gaussian_heatmap(ee_pose_init, view_angles[view])
        heatmap_colored = colorize(heatmap_view)
        input_views.append(concat([rgb_view, heatmap_colored]))

    # ===== 步骤2：VAE 编码 → 加噪 =====
    z_t = vae_encoder(input_views)
    z_t = add_noise(z_t)  # 加噪到时间步 T

    # ===== 步骤3：多视角扩散去噪 =====
    for step in range(num_steps):
        t = schedule[step]  # 噪声调度
        for dit_block in range(num_dit_blocks):
            z_t = view_attention(z_t)
            z_t = dit_block(z_t, t)
        z_0 = denoiser(z_t, t)  # 预测原始 latent

    # ===== 步骤4：VAE 解码 → 预测热力图视频 =====
    heatmap_video = vae_decoder(z_0[..., C:])  # (future_frames, V, H, W)

    # ===== 步骤5：位置解码 =====
    positions_3d = []
    for frame_idx in range(future_frames):
        for view in range(num_views):
            # 找热力图峰值
            heatmap_2d = heatmap_video[frame_idx, view]
            peak_uv = argmax(heatmap_2d)
            # Back-projection → 3D 坐标（已知相机内外参）
            pt_3d = back_project(peak_uv, camera_intrinsics[view],
                                 camera_extrinsics[view])
            positions_3d.append(pt_3d)

    position_traj = fuse_multi_view_positions(positions_3d)  # 跨视角融合

    # ===== 步骤6：旋转 & 夹爪解码 =====
    global_feat = flatten(z_0[..., C:])
    peak_coords = get_peak_coords(heatmap_video[0])  # 首帧峰值
    local_feat = interpolate_local(z_0[..., C:], peak_coords)
    action_feat = concat([global_feat, local_feat])
    cond = z_0[0]  # 首帧作为 condition

    roll_bins = softmax(linear(action_feat, 72))
    pitch_bins = softmax(linear(action_feat, 72))
    yaw_bins = softmax(linear(action_feat, 72))
    gripper_state = sigmoid(linear(action_feat, 2))

    # bins → Euler angles（5° × bin_id）
    roll = 5° * argmax(roll_bins)
    pitch = 5° * argmax(pitch_bins)
    yaw = 5° * argmax(yaw_bins)
    gripper = 1 if gripper_state > 0.5 else 0

    # ===== 步骤7：组装 action chunk =====
    action_chunk = concat([position_traj, roll, pitch, yaw, gripper])
    return action_chunk
```

---

## 📊 五、核心实验结果

### 5.1 Meta-World 基准（低数据 regime：5 demos/task）

| 方法 | 平均成功率 |
|------|-----------|
| BC-Scratch | 26.2% |
| BC-R3M | 35.4% |
| Diffusion Policy | 37.7% |
| UniPi | 11.4% |
| AVDC | 58.9% |
| Track2Act | 67.4% |
| DreamZero（相同 Wan2.2 backbone） | 61.1% |
| **MV-VDP（Ours）** | **89.1%** |

> **关键发现**：即使使用相同视频基础模型（Wan2.2），MV-VDP 比 DreamZero 高出 28%，证明多视角热力图表示 + View-Attention 的有效性。

### 5.2 真实机器人实验（10 demos/task）

实验平台：Frank a Research 3 + 3 个 ZED 2i 深度相机。

**基础任务**：
- Put Lion（拾取放置）：10/10（唯一达到 100% 的方法）
- Push-T（推 T 形块）：4/10
- Scoop Tortilla（舀取）：7/10

**跨分布泛化**：
- Put-B（不同背景布）：5/10
- Put-H（高度变化 5.5cm）：6/10
- Push-L（无环境光）：3/10
- Scoop-C（不同物体类别）：5/10

**平均成功率 57.1%**，全面超越 DP3（0%）、π₀.₅（1.4%）、UVA（5.7%）、BridgeVLA（41.4%）。

![MV-VDP Real-World Experiments](https://arxiv.org/html/2604.03181v1/x3.png)

### 5.3 鲁棒性分析

| 超参数 | 变化幅度 | 成功率变化 |
|--------|---------|-----------|
| RGB 损失权重 λ | ±80% | 仅 3.3% |
| 热力图标准差 σ | ±133% | 仅 2.5% |
| 去噪步数 N | 1~50 | **1 步即可保持高性能** |

> **意外发现**：仅 1 步去噪即可达到接近 50 步的性能，因为热力图分布简单、无高频细节。推荐实际使用 N=5（单卡 A100 5Hz 推理）。

### 5.4 消融实验

| 配置 | 平均成功率 |
|------|-----------|
| 完整 MV-VDP | 89.1% |
| - LoRA（Full fine-tuning） | 87.4%（无收益） |
| - View Concat（改用 Channel Concat） | 81.1%（-8%） |
| - Video Prediction（仅预测热力图） | 61.1%（-28%） |
| - 预训练初始化 | 4.6%（灾难性下降） |

> **最重要的消融**：去掉视频预测（仅预测热力图）→ 28% 下降，验证了联合预测 RGB 视频的重要性。

---

## 💡 六、核心洞察（KnowHow）

1. **视频预训练与动作微调的对齐是数据高效的关键**：预训练视频模型学习到的时空动态建模能力可直接迁移到机器人操作，比从零训练或用静态图文预训练的数据效率高得多。

2. **多视角热力图作为动作表示**：将末端执行器姿态用热力图表示，而非直接回归坐标，好处是：
   - 热力图格式与视频预训练的表达格式一致
   - 峰值位置自然对应 3D 坐标，back-projection 简单可靠
   - 扩散模型的去噪过程对热力图尤其高效（1 步去噪即可）

3. **View-Attention 实现跨视角一致性**：在每个 DiT block 插入跨视角注意力，确保三个视角的预测在几何上保持一致。

4. **BridgeVLA 的局限性**：BridgeVLA 虽然也用多视角投影 + 热力图，但其基于静态图文预训练的 VLM，无法预测连续视频/热力图序列，只能预测单帧关键姿态，对 Push-T 这类需要连续动作的任务失败。

5. **可解释性即安全性**：预测未来 RGB 视频允许用户在执行前"预览"动作序列，实验证明可显著降低碰撞事件（6/140 → 0/140）。

6. **为什么 1 步去噪就够？**：热力图没有高频细节，分布模式简单，不像自然图像需要多步去噪恢复细节。

---

## ⚠️ 七、局限性与未来方向

1. **推理速度**：A100 上生成 24 帧 action chunk 约 4.6 秒，不适合高频灵巧操作任务。
2. **未来方向**：
   - 集成 TurboDiffusion 等加速方法（预期 100-200× 加速）
   - 实时 action chunking
   - 适应性选择投影平面（当前固定 3 个视角）
   - 提高热力图分辨率（当前 256×256，每像素≈4mm）

---

## 📚 八、附录：arXiv 论文 Appendix 关键点

### Appendix A：Multi-View Video Diffusion

**A.1 3D VAE Encoder**：沿用 Wan2.2 的 VAE 架构，将 RGB 图和热力图沿通道维度拼接后联合编码。

**A.2 Diffusion Transformer (DiT)**：每个 DiT block 插入 View-Attention 模块，参考 SynCamMaster 的实现。DiT 沿视角维度展开 token，在每个时间步做跨视角交互。

**A.3 3D VAE Decoder**：解码器同时输出 RGB latent 和热力图 latent，分别用于 RGB 视频重建和动作解码。

**A.4 高效推理：Cache 机制**：推理时复用前面帧的 latent特征，仅对新到帧进行编码 + 去噪，显著减少计算量。

### Appendix B：Projection and Back Projection

**B.1 投影过程**：
- **正交投影**：点云沿固定视角方向投影到 2D 平面
- **Z-ordering**：处理遮挡关系，保证近处点优先投影
- **Screen-Space Splatting**：将点属性 splatting 到最近像素

**B.2 反投影过程**：
- 热力图峰值位置 → 2D 像素坐标
- 已知相机内外参，通过射线反向投影求 3D 坐标
- 多视角融合：同一 3D 点在多个视角的投影应汇聚到同一点

### Appendix C：Training & Inference Details

| 参数 | 值 |
|------|-----|
| 点云裁剪范围 | 1.0 m³ |
| 热力图分辨率 | 256×256 |
| RGB/热力图损失权重 λ | 0.5 |
| 热力图标准差 σ | 论文正文 sweep 结果 |
| 预测未来帧数 | 24 |
| 输入帧数 | 16 |
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| 权重衰减 | 1e-5 |

推理速度：A100 单卡，N=5 步去噪，约 5Hz。

### Appendix D：Robustness Analysis for Key Parameters

与正文 5.3 节鲁棒性分析一致。热力图分辨率（256×256，每像素≈4mm）、λ 在大范围变化下成功率稳定。

### Appendix E：Simulation Baselines

- **BC-Scratch / BC-R3M**：行为克隆基线
- **Diffusion Policy**：基于扩散的动作预测
- **UniPi**：基于视频预测的策略，从预测的未来 RGB 帧解码动作
- **AVDC**：从视频中学习稠密对应关系，再解码动作
- **Track2Act**：从预测的 2D 点轨迹解码动作
- **DreamZero**：与 MV-VDP 使用相同 Wan2.2 backbone，但仅预测热力图而非联合预测 RGB

### Appendix F：Real-World Baselines

- **DP3**：3D 扩散策略
- **π₀.₅**：大规模 VLA，用 flow matching 解码动作
- **UVA**：统一视频动作模型，双扩散头预测未来视频和动作
- **BridgeVLA**：与 MV-VDP 最类似的基线，用 VLM + 热力图预测关键姿态

### Appendix G：Video Prediction 可视化

展示了不同去噪步数（N=1, 5, 10, 20, 50）下的 RGB 视频和热力图预测质量。N=1 时 RGB 质量较低，但热力图仍有效；N≥5 时质量显著提升。

---

## 🎯 九、总结

MV-VDP 重新回答了"机器人操作的数据效率从何而来"这个问题。核心在于视频预训练与动作微调的格式对齐。

**三大核心贡献：**
1. **VGA 范式（类比 DVGT-2）**：首次系统验证视频基础模型 + 热力图表示作为动作预测核心表征的可行性
2. **多视角几何-动作联合建模**：点云正交投影 + 热力图 + View-Attention + 联合扩散预测
3. **极低数据条件下的有效性**：5 demos Meta-World 89.1%，10 demos 真实机器人 57.1%，全面超越所有基线

**最重要的洞察**：联合预测 RGB 视频（建模环境动态）比仅预测热力图（仅建模动作）高出 28%，说明让模型同时理解"做什么动作"和"环境如何响应"，是数据高效的关键。

---

## 🏷️ 十、分类与归档

- **类别**：VLA / 具身智能 / 机器人操作
- **归档路径**：`/Users/irving/Documents/SOTAFollow/VLA/`
- **相关方向**：Video Prediction + Manipulation, Diffusion Policy, 3D Visual Servo, World Action Model

---

## 📚 参考文献（关键文献）

- Wan2.2（视频基础模型）：https://www.modelscope.cn/home
- DreamZero（同期工作，相同 backbone）：待补充
- BridgeVLA（最强基线）：arXiv:2506.07961
- Diffusion Policy：IJRR 2025
- π₀.₅（VLA baseline）：arXiv:2504.16054
- AVDC：待补充
- UniPi：待补充
- Track2Act：待补充

---

*精读日期: 2026-04-10*

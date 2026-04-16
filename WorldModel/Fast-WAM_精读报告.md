# Fast-WAM 论文精读报告

> **论文**: Do World Action Models Need Test-time Future Imagination?
> **arXiv**: [2603.16666](https://arxiv.org/abs/2603.16666) [cs.CV]
> **作者**: Tianyuan Yuan, Yanjun Li, Qingyun Wu, Zeyu Gao, Linlin Liu, Qi Wang, Xiaoyan Tian, Qianhong Wang, Hongen Ren, Yao Xu, Hangyu Mao, Song-Chun Zhu, Jinzhuo Wang, Wenhan Yang, Haiyang Sun, Zongzhang Zhang, Dong Che, Mian Zhang, Haichuan Liu, Fangxun Zhong, Wenbo Li, Jie Yang
> **机构**: 北京大学 · 上海人工智能实验室 · 香港大学 · 清华大学 · 华为诺亚方舟实验室 · 多伦多大学 · UCLA · 字节跳动
> **日期**: 2026年3月17日（v1）, 2026年3月23日（v2）
> **代码**: [https://github.com/yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)
> **项目主页**: [https://yuantianyuan01.github.io/FastWAM/](https://yuantianyuan01.github.io/FastWAM/)

---

## 1. 引用信息表

| 信息项 | 内容 |
|--------|------|
| 论文标题 | Do World Action Models Need Test-time Future Imagination? |
| arXiv ID | 2603.16666 [cs.CV] |
| 作者 | Tianyuan Yuan* et al. |
| 机构 | 北京大学、上海AI Lab、港科大、清华、华为诺亚方舟、字节跳动 |
| 提交时间 | 2026-03-17（v1）, 2026-03-23（v2） |
| 代码链接 | [GitHub](https://github.com/yuantianyuan01/FastWAM) |
| 项目主页 | [FastWAM Page](https://yuantianyuan01.github.io/FastWAM/) |

---

## 2. 一句话总结

**Fast-WAM 证明 WAMs 的核心价值在于训练阶段的视频联合建模（video co-training），而非推理阶段显式生成未来观察，移除 video co-training 造成的性能损失远大于移除 test-time 未来想象的损失——从而引发了一场关于 World Model 演进哲学的深刻思考。**

![Fast-WAM Overview](https://arxiv.org/html/2603.16666v2/x1.png)

---

## 3. 研究背景与问题动机

### 3.1 World Action Model 的崛起

构建通用具身智能体需要既能**映射视觉观察到动作**，又能**推理物理世界在交互下如何演化**的策略。这催生了 World Action Models（WAMs）的热潮——WAMs 将未来视觉预测和动作建模统一在同一个框架中，相比标准 Vision-Language-Action（VLA）模型，WAMs 的吸引力在于：建模未来观察有助于捕获物理动力学和任务相关的时序结构。

### 3.2 Imagine-then-Execute 范式的困境

**图1：三种代表性 WAM 范式**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    (A) Joint-modeling WAMs                          │
│   未来视频 tokens 和动作 tokens 一起去噪，共享注意力                  │
│   [ 观察帧 ] ──▶ [ 联合去噪: 未来帧 + 动作 ] ──▶ [ 动作 ]           │
│                                                                     │
│                    (B) Causal WAMs                                   │
│   先生成未来观察，再基于生成的未来表征预测动作                         │
│   [ 观察帧 ] ──▶ [ 生成未来视频 ] ──▶ [ 基于未来预测动作 ]          │
│                                                                     │
│                    (C) Fast-WAM (Ours)                               │
│   保留视频联合训练，推理时跳过显式未来生成，单次前向传播直接预测动作    │
│   [ 观察帧 ] ──▶ [ 单次编码: 获得 world 表征 ] ──▶ [ 直接动作预测 ]  │
└─────────────────────────────────────────────────────────────────────┘
```

大多数现有 WAMs 都遵循 **imagine-then-execute** 范式：
1. 先生成未来观察
2. 再基于想象出的未来条件预测动作

这种设计虽然直观，但存在两个根本性问题：
- **高推理延迟**：迭代式视频去噪带来巨大的测试时开销
- **必要性存疑**：显式未来想象是否真的对动作性能至关重要？

### 3.3 核心问题

WAMs 的有效性可能来自两个截然不同的因素：

| 因素 | 描述 |
|------|------|
| **因素 (1)** | **训练时的视频预测目标**——帮助模型习得更强的物理先验和动作条件表征 |
| **因素 (2)** | **推理时显式生成未来**——为动作预测提供额外的预见能力 |

现有 WAM 系统通常将这两个因素纠缠在一起，难以确定哪个才是真正带来增益的原因。

---

## 4. 方法详解

### 4.1 问题建模

**标准视觉运动策略**建模条件分布：

$$p(a_{1:H} \mid o, l)$$

WAMs 引入未来视觉观察 $v_{1:T}$ 作为中间变量，遵循 imagine-then-execute 分解：

$$p(a_{1:H} \mid o, l) = \int p(v_{1:T} \mid o, l) \cdot p(a_{1:H} \mid o, l, v_{1:T}) \, dv_{1:T}$$

**Fast-WAM 的起点**：将视频预测目标从训练阶段解耦出来，看能否在**不付出推理时未来视频合成成本**的情况下保留其收益。

推理时 Fast-WAM 直接预测动作：

$$p_\theta(a_{1:H} \mid o, l)$$

同时使用**被视频联合训练塑造的 latent world 表征**。形式上，设 $z(o, l)$ 为视频 backbone 基于当前上下文产生的 latent world 表征，Fast-WAM 用它来参数化动作分布：

$$p_\theta(a_{1:H} \mid o, l) = p_\theta(a_{1:H} \mid z(o, l))$$

**与 imagine-then-execute WAMs 的关键区别**：$z(o, l)$ 通过**单次前向编码 pass**获得，而非在推理时显式采样或去噪未来观察 $v_{1:T}$。

### 4.2 模型架构

![Fast-WAM Architecture](https://arxiv.org/html/2603.16666v2/x2.png)

**图2(a)：Fast-WAM 模型架构**

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Fast-WAM: MoT (Mixture-of-Transformer)        │
│                                                                       │
│   输入 Tokens 分为三组：                                                │
│   1. [Clean Latent Tokens] ──观察帧第一帧的干净 latent tokens（共享 anchor）│
│   2. [Noisy Latent Tokens] ──未来视频帧的噪声 latent tokens（仅训练时）   │
│   3. [Action Tokens] ──动作 expert 处理的动作 tokens                    │
│                                                                       │
│   ┌────────────┐      ┌──────────────────────────┐                  │
│   │ Video DiT   │      │ Action Expert DiT         │                  │
│   │ (来自 Wan2.2│      │ (隐藏维度 d_a=1024, 1B参数)│                  │
│   │  -5B)       │      │                          │                  │
│   │ 共享注意力   │◀────▶│  共享注意力               │                  │
│   └────────────┘      └──────────────────────────┘                  │
│         ▲                    ▲                                        │
│         │                    │                                        │
│   [Clean Latent] ────────────┘                                        │
│   (所有 token 通过交叉注意力访问语言嵌入)                              │
└──────────────────────────────────────────────────────────────────────┘
```

**图2(b)：训练和推理掩码**

```
训练时注意力掩码：
┌────────────────┬────────────┬────────────┬────────────┐
│                │ Clean Obs  │ Noisy Vid  │ Action     │
├────────────────┼────────────┼────────────┼────────────┤
│ Clean Obs     │    ✓      │    ✓      │    ✓      │
│ Noisy Video   │    ✓      │    ✓      │    ✗      │
│ Action        │    ✓      │    ✗      │    ✓      │

推理时注意力掩码（移除未来视频分支）：
┌────────────────┬────────────┬────────────┐
│                │ Clean Obs  │ Action     │
├────────────────┼────────────┼────────────┤
│ Clean Obs     │    ✓      │    ✓      │
│ Action        │    ✓      │    ✓      │
└────────────────┴────────────┴────────────┘
```

**关键设计**：
- 构建在预训练视频 DiT（Wan2.2-5B）之上，复用其预训练文本编码器和视频 VAE
- Action Expert DiT 与 Video DiT 共享注意力，组成 MoT 架构
- 结构化注意力掩码控制三组 token 之间的信息流
- **训练时**：未来噪声视频 tokens 双向 attend within video branch，可访问 clean first-frame tokens；动作 tokens 双向 attend within action branch，也可访问 clean first-frame tokens；**关键约束**：动作 tokens 不能 attend 未来视频 tokens，clean first-frame tokens 不能 attend 任何其他 tokens
- **推理时**：完全移除未来视频分支，仅保留 clean first-frame latent tokens，单次前向传播生成动作

### 4.3 训练目标

Fast-WAM 使用**联合 Flow Matching 目标**：

给定目标变量 $y$（动作 chunk 或未来视频 latents），采样高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$ 和时间步 $t \in (0, 1)$，构造插值样本：

$$y_t = (1-t)y + t\epsilon$$

模型训练去预测速度场：

$$\mathcal{L}_{\text{FM}}(y) = \mathbb{E}_{y, \epsilon, t}\left[\left\|f_\theta(y_t, t, o, l) - (\epsilon - y)\right\|_2^2\right]$$

**动作预测损失**：

$$\mathcal{L}_{\text{act}} = \mathcal{L}_{\text{FM}}(a_{1:H})$$

**视频联合训练损失**：

$$\mathcal{L}_{\text{vid}} = \mathcal{L}_{\text{FM}}(z_{1:T})$$

**总损失**：

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{vid}}$$

### 4.4 受控变体设计（对照实验）

为回答核心问题，论文设计了三种受控变体：

| 变体 | 描述 | 对应范式 |
|------|------|---------|
| **Fast-WAM-Joint** | 联合生成范式，未来视频 tokens 和动作 tokens 在共享模型中联合去噪 | 图1(A) |
| **Fast-WAM-IDM** | 先视频后动作范式，先从当前观察和语言上下文生成未来视频，再基于未来表征预测动作 | 图1(B) |
| **Fast-WAM w/o video co-train** | 移除 video co-training 目标，仅保留架构和推理流程不变 | 对照组 |

这三种变体在相同实现框架下进行对比，将 test-time 未来生成与 training-time video co-training 两个因素有效分离。

---

## 5. 实验结果

### 5.1 仿真基准测试

#### 5.1.1 RoboTwin 2.0（双臂操控）

**表1：RoboTwin 结果**

| 方法 | Embodied PT. | Clean | Rand. | Average |
|------|-------------|-------|-------|---------|
| π₀ (with PT.) | ✓ | 65.92 | 58.40 | 62.2 |
| π₀.₅ (with PT.) | ✓ | 82.74 | 76.76 | 79.8 |
| Motus (with PT.) | ✓ | 88.66 | 87.02 | 87.8 |
| LingBot-VA (with PT.) | ✓ | 92.90 | 91.50 | 92.2 |
| **Fast-WAM (Ours)** | ✗ | **91.88** | **91.78** | **91.8** |
| Fast-WAM-Joint | ✗ | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | ✗ | 91.16 | 91.34 | 91.3 |
| Fast-WAM w/o video co-train | ✗ | 82.76 | 84.80 | **83.8** |

**关键洞察**：
- Fast-WAM **无需具身预训练即达 91.8%**，与带具身预训练的 baseline 相当
- Fast-WAM vs Joint/IDM 差异微小（~1%），但 **移除 video co-training 下降 8%**（91.8% → 83.8%）

#### 5.1.2 LIBERO（4个子任务套件）

**表2：LIBERO 结果**

| 方法 | Spatial | Object | Goal | Long | Average |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| **Fast-WAM (Ours)** | **98.2** | **100.0** | **97.0** | **95.2** | **97.6** |
| Fast-WAM-Joint | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| Fast-WAM-IDM | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| Fast-WAM w/o video co-train | 89.2 | 99.2 | 95.4 | 90.0 | **93.5** |

**关键洞察**：
- Fast-WAM 平均 97.6%，与最强具身预训练 WAM 相当
- 移除 video co-training 下降 4.1%，**远大于三种 variant 之间的差异**

### 5.2 真实世界：毛巾折叠任务

![Fast-WAM 毛巾折叠任务](https://arxiv.org/html/2603.16666v2/x3.png)

> **图3**：真实世界毛巾折叠任务 - Galaxea R1 Lite 平台上折叠柔性物体，需要长期规划和精确闭环操控。

![Fast-WAM 真实世界结果](https://arxiv.org/html/2603.16666v2/x4.png)

> **图4**：真实世界毛巾折叠任务结果。左图：成功率 vs 平均完成时间（越左上越好）；右图：推理延迟对比。Fast-WAM 以显著更低的延迟达到强性能，同时移除 video co-training 会导致成功率和完成时间都下降。

**关键发现**（论文原文描述）：
- **π₀.₅**（具身预训练）：最高成功率 + 最短完成时间
- **Fast-WAM 家族**：Fast-WAM-IDM 成功率最高，Fast-WAM 完成时间更短
- **移除 video co-training**：性能显著下降（论文原文："substantially outperform π₀.₅ without pretraining"）
- **推理延迟**：Fast-WAM **190ms**，imagine-then-execute WAMs **>760ms**（4x+ 差距）

### 5.3 受控实验结论

**核心发现**：在仿真和真实世界任务中，模式高度一致：
- Fast-WAM 与 imagine-then-execute 变体性能相当
- **移除 video co-training 造成远大的性能下降**

这表明 **WAM 训练的主要收益可能更多来自训练时视频预测目标对世界表征的塑造，而非推理时显式生成未来观察**。

---

## 6. 核心方法伪代码

### 6.1 训练伪代码

```python
# ===== Fast-WAM 训练伪代码 =====
# 核心：联合视频+动作 Flow Matching，保留 video co-training

def train_fast_wam(dataset, num_iterations):
    # 初始化：Video DiT (Wan2.2-5B) + Action Expert DiT
    video_dit = load_pretrained_video_dit("Wan2.2-5B")
    action_expert = ActionExpertDiT(dim=1024)  # 1B 参数
    optimizer = torch.optim.AdamW(
        [video_dit, action_expert], lr=1e-4, weight_decay=0.01
    )

    for iteration in range(num_iterations):
        # 1. 采样轨迹
        obs_seq, action_seq = dataset.sample_batch()
        # VAE 编码观察帧和未来帧
        z_curr = vae.encode(obs_seq)      # clean tokens
        z_future = vae.encode(next_obs)   # 未来帧 latents
        lang_emb = text_encoder.encode(task_instruction)

        # 2. 构建 token 序列
        # [CLS] + Clean Obs + Noisy Future Video + Action Tokens
        # 使用结构化注意力掩码（防止动作 tokens attend 未来视频）

        # 3. 联合 Flow Matching 损失
        # 动作分支
        loss_act = flow_matching_loss(
            action_expert(z_curr, action_seq, lang_emb),
            action_seq
        )
        # 视频分支（仅在 video co-training 时启用）
        loss_vid = flow_matching_loss(
            video_dit(z_curr, z_future_noisy, lang_emb),
            z_future
        )

        # 4. 总损失（λ 控制视频 co-training 强度）
        loss = loss_act + lambda_vid * loss_vid

        # 5. 反向传播更新（无 EMA）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad()

    return video_dit, action_expert
```

### 6.2 推理伪代码（Fast-WAM 单次前向传播）

```python
# ===== Fast-WAM 推理伪代码 =====
# 核心：移除未来视频分支，单次前向传播直接预测动作

def infer_fast_wam(video_dit, action_expert, current_obs, task_instruction):
    # 1. VAE 编码当前观察
    z_curr = vae.encode(current_obs)  # [B, D] clean latent
    lang_emb = text_encoder.encode(task_instruction)

    # 2. 【关键差异】无需生成未来——单次前向传播获得 world 表征
    world_repr = video_dit.forward_once(z_curr, lang_emb)
    # video_dit 在推理时退化为 world encoder，输出 world 表征

    # 3. 直接基于 world 表征预测动作
    action_chunk = action_expert(world_repr, lang_emb)

    # 4. 动作解码（如需要）
    return action_decoder.decode(action_chunk)
    # 推理延迟：~190ms（单次前向传播，无迭代去噪）
```

### 6.3 Imagine-then-Execute 推理伪代码（对比）

```python
# ===== Joint/IDM WAM 推理伪代码 =====
# 核心：需要迭代式视频去噪，推理成本高

def infer_joint_wam(video_dit, action_expert, current_obs, task_instruction):
    z_curr = vae.encode(current_obs)
    lang_emb = text_encoder.encode(task_instruction)

    # 【关键差异】需要迭代式视频去噪生成未来观察
    z_future = z_curr
    for denoising_step in range(num_steps):  # ~10-50 步
        z_future_noisy = add_noise(z_future, t=denoising_step)
        # 联合去噪：视频 + 动作 tokens 互相影响
        z_future = video_dit.denoise(z_future_noisy, z_curr, lang_emb)

    # 基于生成的未来表征预测动作
    action_chunk = action_expert(z_future, lang_emb)

    return action_decoder.decode(action_chunk)
    # 推理延迟：~800ms+（迭代去噪）
```

---

## 7. 与 LeWorldModel 对比：World Model 演进的哲学思考

### 7.1 两种范式的哲学对立

| 维度 | LeWorldModel | Fast-WAM |
|------|-------------|----------|
| **核心目标** | 学习**紧凑、有物理意义**的 latent 表示 | 证明**训练时**视频联合训练的价值 |
| **训练范式** | 预测下一帧 embedding（JEPA 范式） | 视频 + 动作联合训练（Flow Matching） |
| **推理方式** | Latent 空间 CEM 规划（48x 加速） | 移除 test-time 未来想象，单次前向传播 |
| **表示学习** | SIGReg 正则器防止崩溃 | Video co-training 塑造世界表征 |
| **规划方式** | 显式规划（latent space CEM） | 直接策略（无显式规划） |
| **轻量化** | 15M 参数，<1s 规划 | 6B 总参数，190ms 推理 |

### 7.2 两种哲学的演进关系

```
LeWorldModel 的哲学：
┌─────────────────────────────────────────────────────────────────┐
│  "预测即理解" —— 通过预测未来 embedding 学习世界的底层结构        │
│  ● 表示学习是核心                                              │
│  ● 物理量在 latent 空间被自然提取                               │
│  ● 规划只是一种下游应用                                        │
│  ● SIGReg 确保表示不崩溃                                       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
                  [JEPA 范式收敛]
                  [表示学习登顶]
                          ↓
Fast-WAM 的哲学：
┌─────────────────────────────────────────────────────────────────┐
│  "训练即世界模型，推理只需表征" —— 训练时的视频联合训练才是关键   │
│  ● 视频预测的价值在于塑造表征，而非生成未来                      │
│  ● test-time 显式未来生成可能是"不必要的奢侈"                   │
│  ● 动作预测可以直接从世界表征出发，无需想象未来                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 从 LeWorldModel 到 Fast-WAM：一条演进脉络

**第一阶段：表示学习时代（LeWorldModel）**
- 核心问题：如何稳定地训练端到端 JEPA？
- 解决方案：SIGReg 正则器
- 洞察：**预测下一帧 embedding 的目标自然驱动模型学习环境状态变量**

**第二阶段：解构 WAM 价值（Fast-WAM）**
- 核心问题：WAMs 的增益到底来自训练还是推理？
- 解决方案：受控实验分离两个因素
- 洞察：**video co-training 是主导因素，test-time 未来想象可能是"伪需求"**

**演进规律**：
1. LeWorldModel 证明了**联合嵌入预测架构**可以端到端稳定训练
2. Fast-WAM 则进一步追问：既然表示学习这么重要，**为什么推理时还要费力生成未来？**
3. 本质上，Fast-WAM 将 LeWorldModel 的洞察推向极端：**只保留训练价值的精华，抛弃推理时的"表面形式"**

### 7.4 两种范式的适用场景

| 场景 | 推荐范式 | 原因 |
|------|---------|------|
| **资源受限/实时控制** | Fast-WAM | 190ms 延迟，4x 加速 |
| **需要可解释性** | LeWorldModel | Latent 空间物理量可探测 |
| **长时序规划** | LeWorldModel (CEM) | 显式 latent 空间优化 |
| **数据有限** | Fast-WAM | video co-training 提供更强表示 |
| **需要 zero-shot** | Fast-WAM | 训练收益迁移到 zero-shot |

### 7.5 哲学差异的深层意义

```
┌────────────────────────────────────────────────────────────────────┐
│                     LeWorldModel vs Fast-WAM                        │
│                                                                    │
│   LeWorldModel 问："世界模型应该学习什么表示？"                      │
│   答案：用统计正则器约束表示，强制对齐物理结构                       │
│                                                                    │
│   Fast-WAM 问："世界模型需要在推理时生成未来吗？"                   │
│   答案：不需要——表示已经在训练时被塑造好了                          │
│                                                                    │
│   两者共同指向一个更深的洞察：                                      │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  "世界模型的价值不在于'想象未来'，而在于'理解现在'"        │    │
│   │                                                            │    │
│   │  预测未来只是手段，表征学习才是目的                         │    │
│   │  test-time 想象可能是训练目标的"过度执行"                  │    │
│   └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. KnowHow + 总结评价

### 8.1 论文的重要贡献

1. **首次系统性解耦 WAMs 的训练收益和推理收益**：通过受控实验有力证明了 video co-training 是主导因素
2. **Fast-WAM 架构**：保留训练收益的同时实现 real-time 推理（190ms）
3. **对领域的修正意义**：挑战了"必须有 test-time 未来想象"的直觉，推动社区重新思考 WAM 的设计哲学

### 8.2 局限性与未来方向

- **大规模预训练数据的影响**：论文在相对小规模数据上验证，大规模 scaling 的效果尚不清楚
- **更长时序的任务**：当前实验主要关注短时序操控，长时序规划场景有待验证
- **多模态观察**：主要基于视觉输入，其他传感器模态的泛化待研究
- **Fast-WAM vs LeWorldModel 的融合**：能否将 SIGReg 思想引入 Fast-WAM，进一步提升表示质量？

### 8.3 对自动驾驶 World Model 的启示

对于自动驾驶领域的 World Model 建设：
- **训练优先于推理**：投入更多资源在训练阶段的视频联合建模
- **世界模型的本质是表征学习**：而非显式未来生成
- **实时性要求高**的场景，Fast-WAM 范式更具实用价值
- **长时序决策**仍可参考 LeWorldModel 的 latent 规划思路

---

## 9. 参考链接

| 资源 | 链接 |
|------|------|
| 论文 | [arXiv:2603.16666](https://arxiv.org/abs/2603.16666) |
| arXiv HTML | [HTML Version](https://arxiv.org/html/2603.16666v2) |
| 代码 | [GitHub: FastWAM](https://github.com/yuantianyuan01/FastWAM) |
| 项目主页 | [FastWAM Page](https://yuantianyuan01.github.io/FastWAM/) |

---

## 10. 附录：World Model 演进哲学补充阅读

> 📐 **参考博客**：[从 LeWorldModel 到 Fast-WAM：World Model 演进的两种哲学](https://zhuanlan.zhihu.com/p/2020176998459285938)
>
> 该博客深入分析了 LeWorldModel 和 Fast-WAM 分别代表的两种 World Model 设计哲学：
> - **LeWorldModel** 代表了"**表示学习优先**"的路线，强调通过预测性学习提取物理世界的底层结构
> - **Fast-WAM** 则代表了"**训练即一切**"的路线，主张推理时的未来生成可能是不必要的奢侈
>
> 两者共同揭示了一个趋势：**World Model 的价值正在从"生成未来"向"理解现在"转移**。

---

*精读日期: 2026-04-11*

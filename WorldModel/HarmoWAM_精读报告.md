# HarmoWAM 论文精读报告

> **论文**: HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models  
> **arXiv**: [2605.10942](https://arxiv.org/abs/2605.10942) [cs.RO]  
> **作者**: Qiuxuan Feng, Jiale Yu, Jiaming Liu, Yueru Jia, Zhuangzhe Wu, Hao Chen, Zezhong Qian, Shuo Gu, Peng Jia, Siwei Ma, Shanghang Zhang  
> **机构**: Peking University · Simplexity Robotics · The Chinese University of Hong Kong  
> **日期**: 2026年5月11日（v1）  
> **项目主页**: [https://elbb-yu.github.io/HarmoWAM/](https://elbb-yu.github.io/HarmoWAM/)  
> **代码**: [GitHub（项目页给出，访问待验证）](https://github.com/xuanxuanzzzii/HarmoWAM)

---

## 0. 引用信息表

| 信息项 | 内容 |
|--------|------|
| 论文标题 | HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models |
| arXiv ID | 2605.10942 [cs.RO] |
| 提交时间 | 2026-05-11 |
| 任务类型 | 真实机器人操作，单臂 + 双臂，ID + OOD 泛化 |
| 核心主题 | World Action Model, predictive expert, reactive expert, process-adaptive gating |
| 基座世界模型 | Wan2.2-TI2V-5B，机器人数据上继续预训练约 1.9M trajectories |
| 实验规模 | 6 个真实任务，每任务 100 条示教，每方法每任务 20 个测试 episode |

---

## 1. Motivation（问题背景）

### 1.1 具身操作的两难：泛化到目标附近，还是精确完成交互

机器人真实操作不是单一能力问题，而是两个阶段的组合：

1. **Transit / Approach**：从初始状态移动到目标物附近，要求能处理背景、位置、物体语义变化。
2. **Interaction / Manipulation**：抓取、插入、堆叠、倒液、拉拉链等接触阶段，要求毫米级或姿态级精度。

传统 VLA 模型如 [OpenVLA](https://arxiv.org/abs/2406.09246)、[$\pi_{0.5}$](https://arxiv.org/abs/2504.16054) 依赖大规模视觉语言先验与动作头，但下游操作仍强受示教分布约束。WAM（World Action Model）希望通过预测未来视觉或联合视频-动作建模，把物理动态先验注入策略。

### 1.2 现有 WAM 两大范式各有硬伤

论文将现有 WAM 分成两类：

| 范式 | 代表工作 | 优点 | 短板 |
|------|----------|------|------|
| **Imagine-then-Execute** | [DreamGen](https://arxiv.org/abs/2505.12705), [ViDAR](https://arxiv.org/abs/2507.12898), [WoW](https://arxiv.org/abs/2509.22642), Wan2.2+AnyPos | 世界模型先生成未来轨迹，再由 IDM 转动作，目标搜索/接近能力强 | 未来视频到动作的逆动力学容易失真，接触交互精度不足 |
| **Joint Modeling** | [VPP](https://arxiv.org/abs/2412.14803), [Cosmos-Policy](https://arxiv.org/abs/2601.16163), [Fast-WAM](https://arxiv.org/abs/2603.16666) | 视频表征直接参与动作生成，细粒度动作更稳定 | 行动空间仍被 SFT 轨迹覆盖范围束缚，OOD 目标位置下容易到不了目标 |

这正是 HarmoWAM 的出发点：**世界模型不应该只服务于一种动作路径，而应该在不同阶段调度不同专家**。

### 1.3 论文的关键诊断实验

作者在两个真实任务（Stack Coke Cans、Put Flowers in Vase）上比较两类 WAM，并把评估拆成 Transit 与 Interaction。

| 方法 | Domain | 场景 | Stack Transit | Stack Interaction | Flower Transit | Flower Interaction |
|------|--------|------|---------------|-------------------|----------------|--------------------|
| Imagine-then-Execute | ID | - | 10/10 | 7/10 | 10/10 | 8/10 |
| Imagine-then-Execute | OOD | Background | 10/10 | 6/10 | 10/10 | 6/10 |
| Imagine-then-Execute | OOD | Position | 10/10 | 5/10 | 10/10 | 2/10 |
| Imagine-then-Execute | OOD | Objects | 10/10 | 7/10 | 10/10 | 7/10 |
| Joint Modeling | ID | - | 9/10 | 9/10 | 9/10 | 10/10 |
| Joint Modeling | OOD | Background | 5/10 | 8/10* | 5/10 | 9/10* |
| Joint Modeling | OOD | Position | 3/10 | 10/10* | 0/10 | 10/10* |
| Joint Modeling | OOD | Objects | 0/10 | 10/10* | 6/10 | 10/10* |

`*` 表示 Interaction 由机器人初始化到目标附近后单独测试。这个设计很关键：它说明 Joint Modeling 的问题不在“精细动作”，而在“探索/接近目标”；Imagine-then-Execute 则相反。

---

## 2. 一句话总结

**HarmoWAM 用同一个 world model 同时驱动 predictive expert 和 reactive expert，并通过 Process-Adaptive Gating 在接近阶段走 reactive 泛化路径、在交互阶段走 predictive 精确路径，从而把 WAM 的 OOD transit 能力和高精度 manipulation 能力统一到一个闭环策略里。**

> **图 1：Overview.**  
> We propose HarmoWAM, an end-to-end WAM that jointly achieves generalizable transit and precise manipulation through a world model that provides physical dynamics priors and adaptively coordinates a predictive action expert and a reactive action expert. HarmoWAM achieves SOTA performance in ID settings and exhibits a substantial advantage in OOD scenarios.
>
> ![HarmoWAM Overview](https://arxiv.org/html/2605.10942v1/x1.png)
>
> - **核心冲突**：transit 需要大探索空间，interaction 需要高局部精度。
> - **核心解法**：world model 提供显式未来视频与隐式 latent dynamics；gating 根据当前过程阶段选择专家。
> - **实验信号**：ID 平均 0.89，OOD global avg 0.82，OOD 相对 ID 仅下降 7.9%。

---

## 3. 核心贡献

1. **提出 WAM 范式的阶段性能力拆解**：论文不是直接声称新架构更强，而是先证明 Imagine-then-Execute 擅长泛化 transit、Joint Modeling 擅长精确 interaction。

2. **提出 HarmoWAM 统一框架**：用一个 Wan2.2-TI2V-5B world model 同时产生显式未来视频和隐式时空 latent，分别喂给 reactive expert 与 predictive expert。

3. **提出 Process-Adaptive Gating**：用轻量 MLP 根据当前视觉 token 判断处于 transit 还是 interaction，动态选择 reactive 或 predictive expert，而不是固定平均或手写阶段切换。

4. **真实机器人 OOD 验证扎实**：6 个真实任务，包含 4 个单臂任务和 2 个双臂任务，覆盖 background、position、object semantic 三类训练未见变化。

5. **性能和速度兼顾**：主实验 ID 平均 0.89，OOD global avg 0.82；推理频率达到 48 Hz，action chunk size 为 12。附录中 5-step world generation 在 Put Flowers in Vase 上达到 85% 且保持 4 Hz 视频预测频率。

---

## 4. 方法详述

### 4.1 问题定义

论文把机器人操作建模为条件序列决策问题。在每个时间步 $t$，给定自然语言指令 $\mathbf{l}_{t}$ 与视觉观测 $\mathbf{I}_{t}\in\mathbb{R}^{H\times W\times 3}$，策略预测未来 $H$ 步动作：

$$
\mathbf{a}_{t+1:t+H}\sim\pi_{\theta}(\cdot\mid\mathbf{I}_{t},\mathbf{l}_{t})
$$

单臂控制空间为 7 DoF；双臂任务拼接两只手臂控制向量，形成 14 DoF。

### 4.2 整体 Pipeline

```
输入: 多视角 RGB + proprioception + language instruction
  |
  v
World Model: Wan2.2-TI2V-5B
  |-- 显式输出: 13-frame future video V_{t:t+H}
  |-- 隐式输出: world latent features F^V_t / F^V_s
  |
  +--> Predictive Expert
  |      - 1B Action DiT, 28 Transformer blocks
  |      - 条件: image features + text features + F^V_t
  |      - 优势: 接触阶段、细粒度、时序一致动作生成
  |
  +--> Reactive Expert
  |      - DINOv2-base + Orientation Decoder
  |      - 条件: predicted frame V_s + future latent F^V_s
  |      - 优势: 利用未来视觉演化扩展探索空间
  |
  +--> Process-Adaptive Gating
         - 输入: current visual tokens F^img_t
         - 输出: s_t in [0, 1]
         - s_t > 0.5  -> Predictive Expert
         - s_t <= 0.5 -> Reactive Expert
  |
  v
执行 action chunk，并进入下一闭环观测
```

> **图 2：Framework.**  
> HarmoWAM leverages a world model to bridge spatio-temporal reasoning and motion generation with two complementary action experts.
>
> ![HarmoWAM Framework](https://arxiv.org/html/2605.10942v1/x3.png)
>
> - **World Model**：在 256x320 分辨率下预测 13 帧未来视频，同时输出时空 latent。
> - **Predictive Expert**：把 $\mathcal{F}^{\mathbf{V}}_{t}$ 作为隐式物理动态条件，做扩散式动作去噪。
> - **Reactive Expert**：对每个未来步 $s$，融合 DINOv2 patch feature 与 world latent，直接回归动作。
> - **Gating**：不是融合两个动作，而是按过程阶段选择控制路径。

### 4.3 World Model 到 Predictive Expert

Predictive expert 是一个 1B 参数 Action DiT，包含 28 个 Transformer blocks，并使用 SigLIP image encoder 与 text encoder。它在每个 action diffusion denoising step $k$ 预测动作噪声：

$$
\epsilon_{\theta}=\mathcal{D}_{\theta_{\mathrm{pred}}}({\mathbf{a}}_{t+1:t+H},\tau_{k}\mid\mathcal{F}^{img}_{t},\mathcal{F}^{text},\mathcal{F}^{\mathbf{V}}_{t})
$$

其中 $\mathcal{F}^{\mathbf{V}}_{t}\in\mathbb{R}^{B\times 80\times 3072}$ 是 world model 当前步 latent。直觉上，它不直接相信像素未来，而是把 world model 的时序动态作为动作 DiT 的条件，让动作生成保持局部精确和时间连贯。

### 4.4 World Model 到 Reactive Expert

Reactive expert 更像一个可执行版的 inverse dynamics 路径，但它不只看未来帧，也看 world latent。

对每个未来步 $s\in\{t+1,\dots,t+H\}$：

$$
\mathcal{F}^{\text{patch}}_{s}\in\mathbb{R}^{B\times 1369\times 768}
$$

由 DINOv2 从预测帧 $\mathbf{V}_{s}$ 抽取 patch-level geometry features；同时将 world latent $\mathcal{F}^{\mathbf{V}}_{s}\in\mathbb{R}^{B\times 80\times 3072}$ 平均池化到 768 通道后，与 patch feature 沿 token 维拼接：

$$
\mathcal{F}^{\text{fuse}}_{s}=[\mathcal{F}^{\text{patch}}_{s};{\mathcal{F}}^{\mathbf{V}}_{s}]
$$

Orientation Decoder 输出动作：

$$
\hat{\mathbf{a}}_{s}=\mathcal{D}_{\text{ori}}(\mathcal{F}^{\text{fuse}}_{s})
$$

这个分支承担“泛化接近”的角色：world model 先想象目标相关视觉演化，reactive expert 从未来视觉和 latent 中推断可行动作，因此比纯 SFT action head 更容易跳出训练轨迹覆盖范围。

### 4.5 Process-Adaptive Gating

Gating 网络复用当前观测的视觉 token $\mathcal{F}^{img}_{t}$，输出当前状态属于 interaction phase 的概率 $s_t\in[0,1]$。训练标签来自自动 keyframe pipeline：gripper state change 或任务特定 end-effector height threshold 被视为 interaction cue。

Gating 使用 BCE：

$$
\mathcal{L}_{gate}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_{i}\log(s_{i})+(1-y_{i})\log(1-s_{i})\right]
$$

推理时阈值固定为 0.5：

| 条件 | 路由 | 语义 |
|------|------|------|
| $s_t > 0.5$ | predictive expert | 当前处于精确交互阶段，需要时序一致、高精度动作 |
| $s_t \leq 0.5$ | reactive expert | 当前处于接近/探索阶段，需要利用未来视觉扩展空间 |

论文附录报告 gating classifier 在 held-out demonstrations 上达到 96.95% frame-level accuracy（1,637 test frame pairs）。

### 4.6 两阶段训练目标

**Stage 1: World Model Finetuning**

令 $\mathbf{x}_{1}$ 为 demonstration clean video latent，$\mathbf{x}_{0}\sim\mathcal{N}(0,I)$ 为高斯噪声 latent，条件 $\mathbf{c}$ 包含当前观测和任务指令。采样 $\xi\in[0,1]$：

$$
\mathbf{x}_{\xi}=(1-\xi)\mathbf{x}_{0}+\xi\mathbf{x}_{1}
$$

目标速度：

$$
\mathbf{v}_{\xi}=\frac{d\mathbf{x}_{\xi}}{d\xi}=\mathbf{x}_{1}-\mathbf{x}_{0}
$$

Flow Matching 损失：

$$
\mathcal{L}_{\mathrm{stage1}}=\mathbb{E}_{\mathbf{x}_{0},\mathbf{x}_{1},\xi,\mathbf{c}}\left[w(\xi)\left\|f_{\theta}(\mathbf{x}_{\xi},\xi,\mathbf{c})-\mathbf{v}_{\xi}\right\|_{2}^{2}\right]
$$

**Stage 2: Action Experts Finetuning**

冻结 world model，训练 predictive expert、reactive expert 和 gating。

$$
\mathcal{L}_{\mathrm{pred}}=\mathbb{E}_{\mathbf{a}_{t+1:t+H},\epsilon\sim\mathcal{N}(0,1)}\left[\|\epsilon_{\theta}-\epsilon\|_{2}^{2}\right]
$$

$$
\mathcal{L}_{\mathrm{react}}=\mathbb{E}\left[d(\hat{\mathbf{a}}_{t+1:t+H},\mathbf{a}_{t+1:t+H})\right]
$$

总体损失：

$$
\mathcal{L}_{\mathrm{stage2}}=\mathcal{L}_{\mathrm{pred}}+\lambda_{react}\mathcal{L}_{\mathrm{react}}+\lambda_{\mathrm{gate}}\mathcal{L}_{\mathrm{gate}}
$$

其中 $\lambda_{react}=0.1$，$\lambda_{\mathrm{gate}}=0.05$。附录给出 Smooth L1：

$$
d(x,\hat{x})=\begin{cases}
0.5\cdot\frac{(x-\hat{x})^{2}}{\beta},&\text{if }|x-\hat{x}|<\beta,\\
|x-\hat{x}|-0.5\beta,&\text{otherwise}.
\end{cases}
$$

---

## 5. 训练与推理伪代码

### 5.1 训练伪代码

```python
def train_harmowam(demos, world_model, predictive_expert, reactive_expert, gate):
    # Stage 1: task-specific world model finetuning
    for batch in demos:
        obs, instruction, future_video = batch.obs, batch.lang, batch.future_video
        x1 = video_vae.encode(future_video)
        x0 = normal_like(x1)
        xi = uniform(0.0, 1.0)
        x_xi = (1 - xi) * x0 + xi * x1
        v_xi = x1 - x0
        cond = encode_condition(obs, instruction)

        pred_v = world_model(x_xi, xi, cond)
        loss_stage1 = w(xi) * mse(pred_v, v_xi)
        update(world_model, loss_stage1)

    freeze(world_model)

    # Stage 2: action experts + gate finetuning
    for batch in demos:
        obs, instruction = batch.obs, batch.lang
        action = batch.action_chunk
        gate_label = build_keyframe_label(
            proprio=batch.proprio,
            gripper=batch.gripper,
            task=batch.task_name,
        )

        future_video, world_latents = world_model.predict(obs, instruction)

        # Predictive expert: diffusion denoising action head
        noisy_action, noise, tau = add_action_noise(action)
        eps_pred = predictive_expert(
            noisy_action,
            tau,
            image_features=encode_image(obs),
            text_features=encode_text(instruction),
            world_latent=world_latents.current,
        )
        loss_pred = mse(eps_pred, noise)

        # Reactive expert: future-frame inverse dynamics with latent condition
        patch_features = dinov2(future_video)
        fused = concat_tokens(patch_features, pool_to_768(world_latents.future))
        action_hat = reactive_expert.orientation_decoder(fused)
        loss_react = smooth_l1(action_hat, action)

        # Process-adaptive gating
        score = gate(encode_image(obs))
        loss_gate = binary_cross_entropy(score, gate_label)

        loss = loss_pred + 0.1 * loss_react + 0.05 * loss_gate
        update([predictive_expert, reactive_expert, gate], loss)
```

### 5.2 推理伪代码

```python
def rollout_harmowam(env, instruction, horizon=12, gate_threshold=0.5):
    obs = env.reset()
    done = False

    while not done:
        future_video, world_latents = world_model.predict(obs, instruction)
        score = gate(encode_image(obs))

        if score > gate_threshold:
            # Interaction phase: use predictive expert for precise manipulation
            action_chunk = predictive_expert.sample_actions(
                image_features=encode_image(obs),
                text_features=encode_text(instruction),
                world_latent=world_latents.current,
                horizon=horizon,
            )
        else:
            # Transit phase: use reactive expert for generalizable target reaching
            patch_features = dinov2(future_video)
            fused = concat_tokens(patch_features, pool_to_768(world_latents.future))
            action_chunk = reactive_expert.orientation_decoder(fused)

        for action in action_chunk:
            obs, reward, done, info = env.step(action)
            if done:
                break
```

---

## 6. 实验结论

### 6.1 实验设置

| 维度 | 设置 |
|------|------|
| 机器人 | Dual-arm Franka Research 3；单臂 7 DoF，双臂 14 DoF |
| 观测 | 三路 RGB + proprioception；单臂为 front/top/wrist，双臂为 global + 两个 wrist |
| 任务 | 4 个单臂：Pick Fruit、Stack Cans、Pour Coke、Write "Yes"；2 个双臂：Put Flowers、Put Items to Bag and Zip |
| 数据 | 每任务 100 条 SpaceMouse teleoperation demonstration |
| 评估 | 每方法每任务 20 个独立 episode，随机 tabletop object positions |
| OOD | Unseen Background、Unseen Position、Unseen Objects |
| Baselines | $\pi_{0.5}$、VPP、Wan+AnyPos、QwenVLA-OFT、Cosmos-Policy |

> **图 3：Attention map and execution process.**  
> The upper part presents attention map visualizations from the last-layer features of the reactive and predictive experts, while the lower part illustrates the robot execution process.
>
> ![HarmoWAM Attention and Execution](https://arxiv.org/html/2605.10942v1/x4.png)
>
> - **Reactive expert** 更关注 gripper 与周边任务环境，适合接近和探索。
> - **Predictive expert** 更关注被操作物体，适合接触、对齐、插入等精细操作。

### 6.2 ID 主实验

| 方法 | Pick Fruit | Stack Cans | Pour Coke | Write "Yes" | Put Flowers | Put Items | Avg |
|------|------------|------------|-----------|-------------|-------------|-----------|-----|
| $\pi_{0.5}$ | 0.80 | 0.68 | 0.75 | 0.83 | 0.72 | 0.67 | 0.74 |
| VPP | 0.80 | 0.60 | 0.78 | 0.73 | - | - | 0.73 |
| Wan+AnyPos | 0.88 | 0.60 | 0.78 | 0.72 | 0.53 | 0.52 | 0.67 |
| QwenVLA-OFT | 0.78 | 0.30 | 0.73 | 0.72 | - | - | 0.63 |
| Cosmos-Policy | 0.93 | 0.65 | 0.80 | 0.83 | 0.75 | 0.72 | 0.78 |
| **HarmoWAM** | **0.95** | **0.90** | **0.88** | **0.92** | **0.85** | **0.85** | **0.89** |

结论：HarmoWAM 在所有任务上都是最优或并列最优，平均 0.89；相对最强 VLA $\pi_{0.5}$（0.74）高 15 个百分点，相对最强 WAM Cosmos-Policy（0.78）高 11 个百分点。

### 6.3 OOD 泛化主实验

> **图 4：Generalization experiments.**  
> Red boxes highlight unseen objects, background variations, and manipulated object positions, while blue boxes indicate original training configurations.
>
> ![HarmoWAM Generalization](https://arxiv.org/html/2605.10942v1/x5.png)
>
> - **Background OOD**：加入 5-8 个训练未见干扰物，并改变光照。
> - **Position OOD**：目标物放在训练轨迹未覆盖区域。
> - **Objects OOD**：替换几何与语义不同的目标物，如 carrot -> pepper、Coke can -> Red Bull can。

| 方法 | Background Avg | Position Avg | Objects Avg | Global Avg |
|------|----------------|--------------|-------------|------------|
| $\pi_{0.5}$ | 0.60 | 0.32 | 0.54 | 0.49 |
| VPP | 0.43 | 0.23 | 0.57 | 0.41 |
| Wan+AnyPos | 0.53 | 0.49 | 0.58 | 0.53 |
| QwenVLA-OFT | 0.46 | 0.28 | 0.50 | 0.41 |
| Cosmos-Policy | 0.57 | 0.26 | 0.50 | 0.44 |
| **HarmoWAM** | **0.81** | **0.80** | **0.85** | **0.82** |

结论：HarmoWAM 的 OOD global avg 为 0.82，ID 到 OOD 的降幅只有 7.9%；相比最强 VLA 与 WAM，论文摘要报告平均优势分别为 33% 与 29%。

### 6.4 消融实验

> **图 5：Ablation Study.**  
> We investigate (a) HarmoWAM Structure, (b) Efficacy of Process-Adaptive Gating, and (c) Impact of world model latent features on both action experts. The "-vid" suffix indicates that video latent features are excluded from the action expert's conditioning.
>
> ![HarmoWAM Ablation](https://arxiv.org/html/2605.10942v1/x6.png)
>
> - **去掉 reactive expert**：Position OOD 成功率降到 14%，说明 predictive-only 难以扩展探索空间。
> - **去掉 predictive expert**：Position OOD / object-instance OOD 降到 56% / 60%，说明 reactive-only 精细交互不足。
> - **不用 gating，改平均两个专家输出**：Position OOD 性能下降 46%；keyframe-based averaging 仍下降 31%。
> - **去掉 world latent features**：reactive expert 的 ID/OOD 降到 65%/54%；predictive expert 的 ID 从 95% 降到 62%。

附录还消融了 world model denoising steps：

| Denoising Steps | Success Rate | Inference Frequency |
|-----------------|--------------|---------------------|
| 3 | 80% | 4 Hz |
| 5 | 85% | 4 Hz |
| 10 | 85% | 3.6 Hz |
| 50 | 87% | 3 Hz |

> **图 10：Visual comparison of generated videos under different denoising steps.**
>
> ![Denoising Steps Ablation](https://arxiv.org/html/2605.10942v1/x11.png)
>
> - 3 steps 的未来视频较模糊，关键接触位置不稳。
> - 5 steps 已能保留关键动作阶段与物体位置，且速度不降。
> - 10/50 steps 收益递减，但推理频率下降。

### 6.5 鲁棒性与失败案例

论文的失败分析集中在三类高精度接触：

1. **Stacking final placement**：相机第三视角对前后方向微小偏差不敏感，导致上方物体释放后倾斜或滑落。
2. **Flower insertion**：抓取点轻微变化改变花茎外露长度，最终插入时擦到花瓶边缘。
3. **Zipper pulling**：拉链头太小且材质摩擦不足，gripper 持续拉动时可能打滑。

这说明 HarmoWAM 显著提升了 OOD transit 和大部分 contact-rich manipulation，但仍受限于视觉定位精度、末端执行器摩擦和狭窄容差任务。

---

## 7. KnowHow（核心洞察）

1. **WAM 的价值不止“想象未来”**：world model 的 latent dynamics 可以作为动作专家的隐式物理条件，不一定必须先把未来像素转成动作。

2. **泛化和精度来自不同动作路径**：reactive expert 通过未来视觉扩大搜索空间；predictive expert 通过动作扩散保持接触阶段的时序一致和局部精度。

3. **不要把专家输出直接平均**：消融说明 averaging 会破坏阶段语义，尤其在 Position OOD 下掉得很明显。HarmoWAM 的关键不是 ensemble，而是 routing。

4. **Gating 标签可以从 proprioception 自动生成**：gripper state change 和 end-effector height threshold 足以构造 interaction/transit 标签，避免昂贵人工阶段标注。

5. **世界模型 latent 对两个专家都重要**：去掉 video latent 后，reactive 和 predictive 都明显退化，说明未来帧像素本身不足，latent 表达里有更关键的时空物理先验。

6. **Position OOD 是最能暴露 SFT 轨迹覆盖不足的测试**：Cosmos-Policy 的 OOD position avg 只有 0.26，而 HarmoWAM 仍有 0.80；这正对应论文对 Joint Modeling 的诊断。

7. **真实双臂长程任务是有效压力测试**：Put Items to Bag and Zip 有约 400 steps，包含放入物体、稳定袋子、抓拉链、拉合拉链，多阶段误差传播会快速放大策略弱点。

8. **未来视频步数有性价比拐点**：5 steps 已接近 10 steps 成功率，同时保持 4 Hz；大步数只轻微提升成功率但拖慢推理。

---

## 8. arXiv Appendix 关键点总结

### Appendix A: Real-World Set-up

单臂平台为 Franka Research 3 + 3D printed UMI gripper，使用 front-view D435、top-down D455、wrist-mounted D435，分辨率 640x480。双臂平台为两台 FR3，每臂一个 wrist camera，加一个 global camera；状态维度扩展为 14。

### Appendix B: Self-Collected Data

任务分 4 个单臂和 2 个双臂：

| 任务 | 子阶段 | 平均长度 |
|------|--------|----------|
| Pick Fruit to Plate | banana -> carrot | 约 280 steps |
| Stack Coke Cans | second can beside first -> third can on top | 约 290 steps |
| Pour Coke into Beaker | grasp bottle -> pour | 约 310 steps |
| Write "Yes" | write Y -> e -> s | 约 310 steps |
| Put Flowers in Vase | pick -> handover -> insert | 约 280 steps |
| Put Items to Bag and Zip | place item -> hold bag -> grasp zipper -> pull zipper | 约 400 steps |

### Appendix C: Real-World Experiment Details

Baseline 说明：

| Baseline | 类型 | 说明 |
|----------|------|------|
| $\pi_{0.5}$ | VLA | PaliGemma-based，flow matching 连续动作 |
| QwenVLA-OFT | VLA | Qwen3-VL backbone，$\ell_1$ 并行动作预测 |
| VPP | Joint Modeling WAM | video diffusion 中间特征条件化 diffusion policy head |
| Cosmos-Policy | Joint Modeling WAM | 把 action 编成 latent-space frames，与 video frames 联合去噪 |
| Wan+AnyPos | Imagine-then-Execute WAM | Wan2.2 预测未来视频，AnyPos 做 inverse dynamics |

训练协议：所有模型在 8 张 NVIDIA H20 上训练；world model 从 Wan2.2-TI2V-5B 初始化，基座预训练数据包含 DROID 201,119 trajectories、AgiBot 3,017 trajectories、RoboMIND 1,721,985 trajectories，以及闭源机器人数据。

### Appendix D: Additional Method Details

补充了 IDM 公式：

$$
\mathbf{a}_{t+1:t+H}\sim\pi_{\mathrm{IDM}}(\cdot\mid\mathbf{V}_{t+1:t+H})
$$

并解释了 motivation study 的公平设置：Joint Modeling 的 interaction 通过把机器人初始化到目标附近来单独评估，以区分“到不了目标”和“交互不精确”。

### Appendix E: Additional Visualization

补充 6 个真实任务的完整执行序列，以及 background、position、semantic 三种 OOD 设置下的完整可视化。作者观察到 HarmoWAM 在堆叠对齐、倒液角度控制、花茎插入等细粒度阶段轨迹更平滑。

### Appendix F: Additional Quantitative and Qualitative Results

给出 stage-wise ID/OOD 表格。重点结果：

- ID 单臂平均 0.91，双臂平均 0.85。
- Background OOD：单臂 0.83，双臂 0.76。
- Position OOD：单臂 0.83，双臂 0.70。
- Objects OOD：单臂 0.87，双臂 0.81。
- 5-step world model generation 在成功率与速度上是较优折中。

### Appendix G: Failure Case Analysis

三类失败来自前后位移估计、花茎插入容差、拉链头摩擦/夹持稳定性。这些失败都不是高层语义错误，而是精密接触物理与硬件执行误差。

### Appendix H: Broader Impact

HarmoWAM 可减少任务特定数据采集和环境特定 retraining，但真实部署仍需 safety guardrails、uncertainty-aware control、human supervision 与硬件级约束。

### Appendix I: Limitations

主要限制是 pretrained world model 有固定 future generation horizon，下游任务必须保持同样预测窗口；另外 pixel-level future generation 直观但有额外推理开销。作者提出未来探索 adaptive future-frame generation 与 latent-level predictive representations。

---

## 9. 总结

HarmoWAM 的三大核心贡献是：

1. **把 WAM 的问题从“哪种范式更好”改写成“什么阶段该用哪种能力”**。Imagine-then-Execute 负责泛化接近，Joint Modeling / predictive action generation 负责精确接触。

2. **用 world model 统一显式未来视频与隐式时空 latent**。Reactive expert 用未来视觉扩展探索空间，predictive expert 用 latent dynamics 生成高精度动作，二者不是替代关系，而是互补关系。

3. **用 Process-Adaptive Gating 做闭环阶段调度**。这让 HarmoWAM 在 ID 平均 0.89、OOD global avg 0.82 的同时，把 OOD drop 控制到 7.9%。

最重要的洞察：**真实机器人操作中的泛化与精度并不天然同源；一个强 world model 的正确用法，是让它在不同任务阶段以不同形式进入控制回路。**

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2605.10942](https://arxiv.org/abs/2605.10942) |
| **arXiv HTML** | [HTML](https://arxiv.org/html/2605.10942v1) |
| **DOI** | [10.48550/arXiv.2605.10942](https://doi.org/10.48550/arXiv.2605.10942) |
| **项目主页** | [Project Page](https://elbb-yu.github.io/HarmoWAM/) |
| **代码** | [GitHub（项目页给出，访问待验证）](https://github.com/xuanxuanzzzii/HarmoWAM) |
| **相关：Fast-WAM** | [arXiv:2603.16666](https://arxiv.org/abs/2603.16666) |
| **相关：Cosmos-Policy** | [arXiv:2601.16163](https://arxiv.org/abs/2601.16163) |
| **相关：VPP** | [arXiv:2412.14803](https://arxiv.org/abs/2412.14803) |
| **相关：DreamGen** | [arXiv:2505.12705](https://arxiv.org/abs/2505.12705) |
| **相关：AnyPos** | [arXiv:2507.12768](https://arxiv.org/abs/2507.12768) |

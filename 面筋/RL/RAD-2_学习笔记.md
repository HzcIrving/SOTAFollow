# RAD-2: 强化学习与扩散模型 学习笔记

> 本笔记基于 arXiv:2604.15308 原始论文和 Gemini Chat 对话讲解整理
> 日期: 2026-04-22

---

## 1. 背景与问题定义

### 1.1 为什么需要 RAD-2？

在高级自动驾驶中，运动规划器需要面对复杂、**多模态的未来不确定性**：

```
驾驶场景示例：
                    前车可能的行为：
                    ├── 保持车道 → 跟车行驶
                    ├── 变道到左车道 → 加速超车
                    └── 变道到右车道 → 让道
```

纯模仿学习（IL）训练的扩散模型规划器存在三个致命缺陷：

| 问题 | 表现 | 后果 |
|------|------|------|
| 随机不稳定性 | 扩散模型自带随机性 | 闭环驾驶中出现"抽搐"行为 |
| 缺乏纠错反馈 | 只学"专家怎么开" | 不知道偏离后如何自救 |
| 高维 RL 困境 | 直接对轨迹空间施加 RL | 梯度爆炸，策略崩溃 |

### 1.2 核心洞察

**RAD-2 的核心洞察**：不直接对高维生成器做 RL，而是训练一个"鉴赏家"（判别器）去打分，让生成器迎合鉴赏家的品味。

```
类比理解：
❌ 传统做法：直接教画家每一笔怎么画（对高维模型直接 RL）
✅ RAD-2 做法：训练鉴赏家学会评判，画家迎合鉴赏家的品味
```

---

## 2. 算法框架

### 2.1 Generator-Discriminator 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAD-2 系统架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  交通场景输入 (BEV Feature)                                       │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐     候选轨迹 K 条                          │
│  │  扩散生成器        │ ─────────────────────────────────┐       │
│  │  (Generator)      │                                  │       │
│  └──────────────────┘                                  ▼       │
│                                                          ┌────┐ │
│                                                         │K 条 │ │
│                                                         │轨迹│ │
│  ┌──────────────────┐                                  └──┬──┘ │
│  │  轨迹判别器        │ ◄── RL 训练 ── 闭环仿真器反馈          │  │
│  │  (Discriminator)  │                                    │  │
│  └────────┬─────────┘                                    │  │
│           │                                              │  │
│           ▼                                              │  │
│     Score 打分排序 ──────────────────────────────────────┘  │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  执行最优轨迹      │                                           │
│  └──────────────────┘                                           │
│                                                                  │
│  ═══════════════════════════════════════════════════════════   │
│  生成器预训练(IL) ──► 判别器强化学习(RL) ──► 生成器自进化      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 四阶段训练流程

```
Stage 1: Generator IL 预训练
┌─────────────────────────────────────────┐
│  输入: 人类专家驾驶数据                   │
│  目标: 学习多模态轨迹分布                  │
│  输出: 具备基本驾驶能力的扩散生成器        │
└─────────────────────────────────────────┘
                    │
                    ▼
Stage 2: 闭环 Rollout
┌─────────────────────────────────────────┐
│  Generator 生成 K 条候选轨迹              │
│  联合策略在 BEV-Warp 环境中交互          │
│  收集多样化 rollout 数据                  │
└─────────────────────────────────────────┘
                    │
                    ▼
Stage 3: Discriminator RL (TCR-GRPO)
┌─────────────────────────────────────────┐
│  输入: rollout 数据 + 环境奖励反馈        │
│  目标: 学会评估轨迹的长期驾驶质量        │
│  输出: 精准的轨迹打分模型                │
└─────────────────────────────────────────┘
                    │
                    ▼
Stage 4: Generator 自进化
┌─────────────────────────────────────────┐
│  输入: 判别器筛选的高分轨迹              │
│  目标: 向高奖励流形迁移                  │
│  输出: 进化后的生成器                    │
└─────────────────────────────────────────┘
```

---

## 3. 核心组件详解

### 3.1 Generator (扩散轨迹生成器)

**职责**：生成多样化的候选轨迹候选

```python
class DiffusionGenerator(nn.Module):
    """扩散模型生成器：基于 IL 预训练，负责生成候选轨迹"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=10):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        # 简化的噪声预测网络
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    @torch.no_grad()
    def generate(self, state, num_candidates=16):
        """DDIM 采样生成 K 条候选轨迹"""
        batch_size = state.shape[0]
        # 从纯噪声开始
        trajectories = torch.randn(batch_size, num_candidates, self.action_dim, device=state.device)

        for i in reversed(range(self.num_steps)):
            # 简化的 DDIM 去噪步骤
            noise_pred = self.network(state.unsqueeze(1).expand(-1, num_candidates, -1), trajectories, i)
            trajectories = trajectories - noise_pred * 0.1

        return trajectories  # [batch, num_candidates, action_dim]
```

**特点**：
- 输入：当前状态 + 场景表示
- 输出：K 条多样化的轨迹候选
- 训练：模仿学习初始化 + On-policy Generator Optimization 调优

### 3.2 Discriminator (Transformer 判别器)

**职责**：评估每条轨迹的长期驾驶质量（输出 Q 值/评分）

```python
class TrajectoryDiscriminator(nn.Module):
    """轨迹判别器：评估轨迹的长期价值（Q 值）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量评分
        )

    def forward(self, state, trajectory):
        """
        state: [batch, state_dim]
        trajectory: [batch, action_dim] 或 [batch, num_candidates, action_dim]
        """
        if trajectory.dim() == 3:
            # 批量轨迹打分
            state_expanded = state.unsqueeze(1).expand(-1, trajectory.shape[1], -1)
            x = torch.cat([state_expanded, trajectory], dim=-1)
            return self.network(x).squeeze(-1)  # [batch, num_candidates]
        else:
            x = torch.cat([state, trajectory], dim=-1)
            return self.network(x)  # [batch, 1]
```

**特点**：
- 输出是低维标量评分，与 RL 奖励信号天然对齐
- RL 优化这个低维输出，比直接优化高维生成器稳定得多

### 3.3 BEV-Warp 闭环仿真器

**职责**：在 BEV 特征空间进行高效物理推演

```python
class BEVWarpSimulator:
    """BEV-Warp 闭环仿真器：在特征空间进行物理推演"""
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.current_state = None

    def reset(self, batch_size=1):
        self.current_state = torch.randn(batch_size, self.state_dim)
        return self.current_state

    def step(self, action):
        """
        执行动作，更新状态，计算奖励
        """
        # 简化的物理仿真（实际中使用 BEV 特征空间 warp）
        next_state = self.current_state + action * 0.1 + torch.randn_like(self.current_state) * 0.01

        # 奖励函数
        collision_penalty = -10.0 if torch.rand(1).item() < 0.02 else 0.0
        progress_reward = 1.0
        comfort_reward = -0.1 * torch.norm(action, dim=-1)

        reward = progress_reward + collision_penalty + comfort_reward
        done = collision_penalty < 0

        self.current_state = next_state
        return next_state, reward, done
```

**优势**：
- 高效：无需完整 3D 图像渲染，直接在 BEV 特征空间操作
- 高吞吐：一秒钟可完成数千步评估
- 保真：基于真实 BEV 特征

---

## 4. 核心算法详解

### 4.1 TCR-GRPO (时序一致组相对策略优化)

#### 问题：信用分配难题

```
场景：当前时刻 t 做出了决策，但碰撞发生在 t+5 时刻
问题：如何将碰撞的"惩罚"归因到 t 时刻的决策上？

传统方法：独立比较 → 方差大，不稳定
TCR-GRPO：组内相对比较 → 方差小，更稳定
```

#### 核心思想

来自同一状态 $s_t$ 的 K 个候选轨迹构成一个"组"，组内进行相对优势归一化：

```
同一状态 s_t 生成的 K 条候选轨迹：
traj_1 (score=0.9) ──┐
traj_2 (score=0.7) ──┼── 组内相对优势归一化
traj_3 (score=0.3) ──┤   A_g = (score - μ) / σ
traj_4 (score=0.1) ──┘
```

#### 数学公式

$$
\mathcal{L}_{\text{Discriminator}} = -\mathbb{E}_{(traj, g) \sim D} \left[ \log \sigma(\hat{A}_g) \right]
$$

其中：
- $g$ 表示时序一致的轨迹组
- $\hat{A}_g = \frac{R(traj_g) - \mu_{\text{group}}}{\sigma_{\text{group}}}$ 为组内相对优势
- $\sigma(\cdot)$ 为 sigmoid 函数

#### 代码实现

```python
def tgrpo_loss(discriminator, rollout_data, groups):
    """
    TCR-GRPO 损失计算
    groups: dict {timestep: [(trajectory, reward), ...]}
    """
    total_loss = 0
    count = 0

    for t, group in groups.items():
        # 提取组内所有轨迹和奖励
        trajectories = torch.stack([g[0] for g in group])
        rewards = torch.tensor([g[1] for g in group])

        # 组内相对优势归一化
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        normalized_advantages = (rewards - mean_reward) / std_reward

        # 判别器打分
        scores = discriminator(trajectories)

        # BPR-style 损失
        loss = -torch.mean(torch.log(torch.sigmoid(normalized_advantages * scores)))
        total_loss += loss
        count += 1

    return total_loss / count
```

### 4.2 On-policy Generator Optimization

#### 问题：为什么不能直接对生成器做 RL？

```
直接 RL 的问题：
- 生成器输出: [x1, y1, x2, y2, ..., x10, y10]  (20 维)
- 奖励信号: 标量 (collision=0, safety=1)
- 梯度反向传播: 20 维的每个分量都应该怎么调整？→ 稀疏且不稳定
```

#### 解决思路

不直接对生成器做 RL，而是：
1. 判别器学会打分后，对生成器生成的轨迹进行排序
2. 将低奖励轨迹作为"伪标签"，用监督学习的方式更新生成器

#### 数学公式

$$
\mathcal{L}_{\text{Generator}} = -\mathbb{E}_{traj \sim \mathcal{G}^-} \left[ \log P_{\theta}(traj | s_t) \right]
$$

其中：
- $\mathcal{G}^-$ 为低奖励轨迹集合（reward < 阈值）
- $P_{\theta}(traj | s_t)$ 为生成器给定状态下生成该轨迹的概率

#### 代码实现

```python
def on_policy_generator_update(generator, discriminator, rollout_data, threshold):
    """
    On-policy 生成器优化
    """
    low_reward_trajs = []
    states = []

    for data in rollout_data:
        # 筛选低奖励轨迹
        if data['reward'] < threshold:
            low_reward_trajs.append(data['selected_trajectory'])
            states.append(data['state'])

    if len(low_reward_trajs) > 0:
        low_reward_trajs = torch.stack(low_reward_trajs)
        states = torch.stack(states)

        # 计算生成器对这些低奖励轨迹的 log_prob
        # 简化为 MSE 对比损失
        optimizer.zero_grad()
        generated = generator(states, num_candidates=1).squeeze(1)
        loss = nn.MSELoss()(generated, low_reward_trajs)
        loss.backward()
        optimizer.step()

    return loss
```

### 4.3 完整训练循环

```python
def train_rad2():
    # 超参数
    state_dim = 128
    action_dim = 20  # 10 个未来 (x, y) 坐标
    num_candidates = 16
    gamma = 0.99
    num_epochs = 100
    rollout_steps = 100

    # 初始化
    generator = DiffusionGenerator(state_dim, action_dim).cuda()
    discriminator = TrajectoryDiscriminator(state_dim, action_dim).cuda()
    target_discriminator = TrajectoryDiscriminator(state_dim, action_dim).cuda()
    target_discriminator.load_state_dict(discriminator.state_dict())

    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)
    simulator = BEVWarpSimulator(state_dim)

    for epoch in range(num_epochs):
        # ──────────────────────────────────────
        # Stage 1: Discriminator RL 训练
        # ──────────────────────────────────────
        state = simulator.reset(batch_size=1).cuda()

        for step in range(rollout_steps):
            # 1. Generator 生成 K 条候选轨迹
            candidates = generator.generate(state, num_candidates)

            # 2. Discriminator 批量打分
            scores = discriminator(state, candidates)

            # 3. 选择最优轨迹执行
            best_idx = torch.argmax(scores, dim=1)
            best_trajectory = candidates[0, best_idx[0]]

            # 4. 闭环仿真
            next_state, reward, done = simulator.step(best_trajectory.unsqueeze(0).cuda())

            # 5. 计算 TD Target
            with torch.no_grad():
                next_candidates = generator.generate(next_state.cuda(), num_candidates)
                next_scores = target_discriminator(next_state.cuda(), next_candidates)
                max_next_score = torch.max(next_scores, dim=1)[0]
                td_target = reward + (1 - int(done)) * gamma * max_next_score

            # 6. 更新 Discriminator
            current_score = discriminator(state.cuda(), best_trajectory.unsqueeze(0))
            disc_loss = nn.MSELoss()(current_score.squeeze(), td_target.detach())

            disc_optimizer.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            disc_optimizer.step()

            state = next_state.cuda() if not done else simulator.reset(batch_size=1).cuda()

        # ──────────────────────────────────────
        # Stage 2: On-policy Generator 优化
        # ──────────────────────────────────────
        state = simulator.reset(batch_size=4).cuda()
        for step in range(20):
            candidates = generator.generate(state, num_candidates)
            scores = discriminator(state, candidates)

            # 选择高分轨迹作为"伪标签"
            best_indices = torch.argmax(scores, dim=1)
            # ... (生成器更新逻辑)

        # 定期更新 target network
        if epoch % 10 == 0:
            target_discriminator.load_state_dict(discriminator.state_dict())

        print(f"Epoch {epoch}: Disc Loss={disc_loss.item():.4f}")
```

---

## 5. 实验结果

### 5.1 主实验

| 方法 | 碰撞率 | L2 Distance | Miss Rate |
|------|--------|-------------|-----------|
| 纯 IL 扩散模型 | 8.2% | 1.00 | 12.1% |
| 扩散模型 + RL (直接) | 6.8% | 0.95 | 9.4% |
| **RAD-2** | **3.6%** | **0.82** | **4.9%** |

**关键结论**：碰撞率降低 **56%**（相比纯扩散模型基线）

### 5.2 消融实验

| 组件 | 碰撞率 | 相对变化 |
|------|--------|----------|
| 完整 RAD-2 | 3.6% | - |
| w/o 判别器 RL | 7.1% | +97% |
| w/o 生成器自进化 | 5.2% | +44% |
| w/o TCR-GRPO | 4.8% | +33% |
| 直接 RL 于生成器 | 6.8% | +89% |

**结论**：
- 判别器 RL 是最关键组件，移除后碰撞率翻倍
- 生成器自进化提供 16% 额外降低
- 直接 RL 于生成器效果有限，证明解耦设计的必要性

---

## 6. 面试问题与参考答案

### Q1: RAD-2 和传统 RL+扩散模型方法的本质区别是什么？

**参考答案**：

传统方法尝试直接对高维扩散模型施加 RL 奖励，这面临三个问题：
1. **维度灾难**：生成器输出几十维的连续坐标，稀疏奖励难以指导梯度更新
2. **分布崩溃**：直接 RL 容易破坏扩散模型学到的多模态分布
3. **训练不稳定**：高维输出的梯度方差大，策略容易崩溃

RAD-2 的本质区别是**解耦设计**：
- 不教"画家怎么画"，而是教"鉴赏家怎么评"
- RL 只优化低维判别器输出（标量评分）
- 生成器通过监督学习的方式"迎合鉴赏家的品味"

这类似于GAN的思想，但 discriminator 的训练信号来自真实闭环反馈，而非生成器本身。

---

### Q2: TCR-GRPO 相比传统 PPO/GRPO 有什么改进？为什么？

**参考答案**：

传统 PPO/GRPO 的问题在于**信用分配不准确**：

```
场景：同一状态 s_t 下生成 K 条轨迹
      traj_1: 安全 → reward = +1
      traj_2: 碰撞 → reward = -10

问题：两条轨迹的"优势"差异很大，但这是因为运气还是决策质量？
```

TCR-GRPO 的改进：

1. **组内归一化**：对同一状态产生的轨迹组内进行相对优势计算
   - $A_g = \frac{R - \mu_{group}}{\sigma_{group}}$
   - 消除组间差异，只关注组内相对质量

2. **时序一致性**：来自同一状态的轨迹自然具有可比性
   - 排除了"场景难度不同"的干扰
   - 方差更小，梯度更稳定

3. **效果**：在长程驾驶任务中显著减少信用分配噪声

---

### Q3: 为什么 On-policy Generator Optimization 不用 RL 而用监督学习？

**参考答案**：

核心原因是**生成器的输出空间是高维连续动作空间**：

```
生成器输出: [x1, y1, x2, y2, ..., x10, y10]  # 20 维坐标序列
奖励信号:   scalar (碰撞=0, 安全=1)

问题：这条轨迹的 20 个坐标值，每个应该增大还是减小？
```

监督学习的优势：

1. **信号清晰**：直接告诉生成器"这些坐标应该像高分轨迹那样"
2. **梯度稳定**：MSE loss 的梯度方差小
3. **保持分布**：鼓励生成器生成更像高分轨迹的样本，而非强制改变参数

这实际上是一种**基于判别器筛选的伪标签学习**，生成器被引导向"高奖励流形"迁移。

---

### Q4: BEV-Warp 仿真器相比 3DGS/传统仿真器的优势是什么？

**参考答案**：

| 仿真器 | 渲染方式 | 计算量 | 适用场景 |
|--------|----------|--------|----------|
| 3DGS | 像素级光栅化 | 极高 | 照片级仿真 |
| BEV-Warp | 特征空间仿射变换 | 极低 | 大规模 RL 训练 |
| 传统物理仿真 | 简化动力学模型 | 低 | 快速原型 |

BEV-Warp 的核心优势：

1. **高效**：无需完整图像渲染，直接在 BEV 特征空间做 warp
2. **高吞吐**：一秒钟可完成数千步评估，适合 RL 海量试错
3. **保真**：基于真实 BEV 特征，保留关键结构信息
4. **可扩展**：支持大规模并行训练

---

### Q5: Generator-Discriminator 框架的通用性？

**参考答案**：

这个框架的思想可以迁移到其他高维动作空间的 RL 场景：

1. **机器人操控**：机械臂的关节角度序列（高维连续动作）
   - Generator: 扩散模型生成候选动作序列
   - Discriminator: 评估动作序列的长期成功率和稳定性

2. **视频生成模型优化**：
   - Generator: 视频扩散模型
   - Discriminator: 评估视频质量和动作合理性

3. **LLM 的 RLHF**：
   - 可以看作一种特殊的 Generator-Discriminator
   - Reward Model 即 Discriminator
   - PPO 即 On-policy 更新

**核心思想**：高维生成问题 + 低维评估问题的解耦，是处理复杂生成-评估任务的一般范式。

---

### Q6: RAD-2 的局限性有哪些？

**参考答案**：

1. **BEV 表征的局限性**：
   - BEV 丢失了部分纹理信息
   - 对于需要精细视觉理解的任务可能不足

2. **扩散模型推理延迟**：
   - 生成 K 条候选轨迹需要多次去噪
   - 实时性要求高的场景可能有挑战

3. **判别器容量限制**：
   - 判别器的"打分能力"有上限
   - 如果生成器产生的候选全是很差的，判别器也无能为力

4. **仿真-真实差距 (Sim2Real)**：
   - BEV-Warp 仿真器与真实世界仍存在差距
   - 需要进一步验证大规模部署的泛化性

---

## 7. 核心洞察总结

1. **解耦是高维 RL 的破局之道**：直接对高维扩散模型施加 RL 如同"对画家每笔都下指令"，不现实且易崩溃

2. **判别器的本质是价值函数近似**：判别器输出轨迹的长期 Q 值，与 RL 奖励信号天然对齐

3. **同策略生成器优化打破 IL 上限**：通过判别器筛选"伪标签"，让生成器在闭环反馈中自我进化

4. **BEV-Warp 是高效闭环训练的基础**：特征空间的物理推演使得一秒钟可完成数千步评估

5. **时间一致性是长程决策的关键**：TCR-GRPO 利用组内相对性减少信用分配方差

6. **多模态轨迹生成 + 判别器排序 = 安全与多样性的平衡**

7. **开环指标不可靠，闭环评估才是真标准**

8. **Generator-Discriminator 框架的通用性**：这个思想可迁移到机器人操控、视频生成、LLM 等多个领域

---

## 8. RAD vs RAD-2 对比

| 维度 | RAD | RAD-2 |
|------|-----|-------|
| 核心架构 | 单策略网络 + PPO | Generator-Discriminator |
| 轨迹生成 | 解耦离散动作空间 | 扩散多模态轨迹 |
| RL 算法 | PPO + GAE | TCR-GRPO + On-policy Opt |
| 训练环境 | 3DGS 照片级仿真 | BEV-Warp 特征空间 |
| 优化目标 | 动作维度解耦 | 轨迹流形整体 |
| 碰撞率降低 | 3x vs IL | 56% vs 扩散规划器 |
| 真机部署 | 未提及 | 已验证 |

**技术演进路线**：
```
RAD (NeurIPS 2025)          RAD-2 (arXiv 2026)
    │                            │
    ▼                            ▼
单策略网络                  Generator-Discriminator
PPO + GAE                   TCR-GRPO + On-policy Opt
3DGS 环境                   BEV-Warp 环境
解耦离散动作                扩散多模态轨迹
IL 正则化                   Generator IL + Discriminator RL
```

---

*整理日期: 2026-04-22*
*来源: arXiv:2604.15308 + Gemini Chat RAD-2 讲解对话*

# RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning

**arXiv:** [2502.13144v2](https://arxiv.org/abs/2502.13144) | **NeurIPS 2025**

**Authors:** Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, Xinggang Wang

**Affiliation:** Huazhong University of Science & Technology, Horizon Robotics

---

## 1 一句话总结

RAD 是首个基于 3DGS 的闭环强化学习训练框架，通过在照片级真实的数字孪生环境中进行大规模试错，结合 RL 与 IL 协同优化，终结了端到端自动驾驶中 IL 范式的因果混淆和开环gap问题，碰撞率降低 3 倍。

---

## 2 核心贡献

1. **首个 3DGS-RL 端到端训练框架**：利用 3DGS 技术构建照片级真实数字孪生环境，支持策略大规模探索与试错学习
2. **RL+IL 协同优化范式**：RL 补齐 IL 的因果建模能力和开环gap，IL 补齐 RL 的人类对齐能力
3. **解耦离散动作空间设计**：将动作解耦为横向+纵向，缩短时域 horizon，降低探索复杂度，加速收敛
4. **安全导向的 Reward 设计**：动态碰撞、静态碰撞、位置偏移、航向偏移四类奖励引导策略应对安全关键事件
5. **稀疏奖励缓解方案**：密集辅助目标函数约束完整动作分布，加速 RL 训练收敛

---

## 3 方法详述

### 3.1 问题定义

端到端自动驾驶的目标是学习一个策略 $\pi_\theta(a_t | s_t)$，直接从多视角图像序列映射到车辆控制动作。

现有 IL 范式的问题：
- **因果混淆**：IL 捕获的是相关性而非因果关系，导致 shortcut learning
- **开环gap**：开环训练与闭环部署之间存在误差累积

### 3.2 算法框架

![算法框架图](https://ar5iv.org/html/2502.13144/assets/x2.png)

RAD 采用**三阶段训练范式**：

```
Stage 1: Perception Pre-training
  ├── 输入：多视角图像
  ├── BEV Encoder → BEV Feature
  ├── Map Head → 矢量化的地图元素 (车道线、边界、箭头、交通信号)
  └── Agent Head → 其他交通参与者轨迹
  优化目标：地图和代理的实例级特征

Stage 2: Planning Pre-training (IL Initialization)
  ├── 输入：大规模人类驾驶演示
  ├── 冻结感知模块参数
  └── 仅更新 Image Encoder + Planning Head
  优化目标：模仿专家驾驶行为

Stage 3: Reinforced Post-training (RL + IL)
  ├── N 个并行 Worker 在 3DGS 环境中 rollout
  ├── Rollout 数据存入 buffer
  ├── PPO + GAE 更新策略
  └── IL 作为正则化项
```

### 3.3 动作空间设计

为加速 RL 收敛，设计**解耦离散动作表示**：

动作分解为两个独立分量：
- 横向动作 $a^x$（左/右）
- 纵向动作 $a^y$（加/减）

动作空间构建于 **0.5 秒时域**，假设恒定线速度和角速度：

$$
a^x = v_t \cdot \sin(\delta_t) \cdot \Delta t
$$
$$
a^y = v_t \cdot \cos(\delta_t) \cdot \Delta t
$$

通过解耦 + 短时域 + 简化运动模型，有效降低动作空间维度。

### 3.4 核心数学公式

**规划头输出：**

$$
\pi(a^x | s) = \text{softmax}(\text{MLP}(\phi(E_{\text{plan}}, E_{\text{scene}}) + E_{\text{navi}} + E_{\text{state}}))
$$
$$
\pi(a^y | s) = \text{softmax}(\text{MLP}(\phi(E_{\text{plan}}, E_{\text{scene}}) + E_{\text{navi}} + E_{\text{state}}))
$$

**价值函数：**

$$
V_x(s) = \text{MLP}(\phi(E_{\text{plan}}, E_{\text{scene}}) + E_{\text{navi}} + E_{\text{state}})
$$
$$
V_y(s) = \text{MLP}(\phi(E_{\text{plan}}, E_{\text{scene}}) + E_{\text{navi}} + E_{\text{state}})
$$

**自行车模型（Ego 车辆运动学）：**

$$
x_{t+1}^w = x_t^w + v_t \cos(\psi_t^w) \Delta t
$$
$$
y_{t+1}^w = y_t^w + v_t \sin(\psi_t^w) \Delta t
$$
$$
\psi_{t+1}^w = \psi_t^w + \frac{v_t}{L} \tan(\delta_t) \Delta t
$$

### 3.5 Reward 建模

Reward 由四部分组成：

$$
\mathcal{R} = \{r_{\text{dc}}, r_{\text{sc}}, r_{\text{pd}}, r_{\text{hd}}\}
$$

| 奖励类型 | 触发条件 | 符号 |
|---------|---------|------|
| 动态碰撞 $r_{\text{dc}}$ | 与动态障碍物 bounding box 重叠 | 负 |
| 静态碰撞 $r_{\text{sc}}$ | 与静态障碍物（3DGS 高斯）重叠 | 负 |
| 位置偏移 $r_{\text{pd}}$ | 与专家轨迹欧氏距离超过 $d_{\max}$ | 负 |
| 航向偏移 $r_{\text{hd}}$ | 与专家轨迹航向角差超过 $\psi_{\max}$ | 负 |

![奖励类型示意图](https://ar5iv.org/html/2502.13144/assets/x4.png)

### 3.6 策略优化

**GAE 优势估计：**

$$
\delta_t^x = r_t^x + \gamma V_x(s_{t+1}) - V_x(s_t)
$$
$$
\delta_t^y = r_t^y + \gamma V_y(s_{t+1}) - V_y(s_t)
$$
$$
\hat{A}_t^x = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^x
$$
$$
\hat{A}_t^y = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^y
$$

**Reward 分解：**

$$
r_t^x = r_t^{\text{sc}} + r_t^{\text{pd}} + r_t^{\text{hd}}
$$
$$
r_t^y = r_t^{\text{dc}}
$$

**PPO 目标函数：**

$$
\mathcal{L}_x^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t^x \hat{A}_t^x, \text{clip}(\rho_t^x, 1-\epsilon_x, 1+\epsilon_x) \hat{A}_t^x \right) \right]
$$
$$
\mathcal{L}_y^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t^y \hat{A}_t^y, \text{clip}(\rho_t^y, 1-\epsilon_y, 1+\epsilon_y) \hat{A}_t^y \right) \right]
$$
$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathcal{L}_x^{\text{PPO}}(\theta) + \mathcal{L}_y^{\text{PPO}}(\theta)
$$

---

## 4 训练与推理伪代码

```python
# RAD 三阶段训练伪代码

# ===== Stage 1: Perception Pre-training =====
def perception_pretraining(images, gt_map, gt_agents):
    bev_features = bev_encoder(multi_view_images)
    map_tokens = map_head(bev_features)      # 监督：地图元素
    agent_tokens = agent_head(bev_features)  # 监督：代理运动信息
    return map_tokens, agent_tokens

# ===== Stage 2: Planning Pre-training (IL) =====
def planning_pretraining(images, expert_actions):
    frozen_bev_encoder(), frozen_map_head(), frozen_agent_head()
    image_tokens = image_encoder(images)
    scene_repr = concat(map_tokens, agent_tokens, image_tokens)
    action_dist = planning_head(scene_repr, navi, ego_state)
    loss_il = -log_prob(action_dist, expert_actions)
    optimizer.step(loss_il)

# ===== Stage 3: Reinforced Post-training =====
def reinforced_posttraining(policy, gs_environments, demonstrations):
    rollout_buffer = []
    for worker in range(N):
        env = sample_random(gs_environments)
        state = env.reset()
        for t in range(rollout_steps):
            action = policy.sample(state)
            next_state, reward, done = env.step(action)
            rollout_buffer.append((state, action, reward, next_state, done))
            if done:
                state = env.reset()

    for _ in range(training_steps):
        # PPO update
        batch = sample(rollout_buffer, batch_size)
        advantages = compute_gae(batch)  # Eq.6
        loss_ppo = compute_ppo_loss(policy, batch, advantages)  # Eq.8

        # IL regularization
        demo_batch = sample(demonstrations, batch_size)
        loss_il = -log_prob(policy(demo_batch.state), demo_batch.action)

        # Auxiliary objectives (sparse reward mitigation)
        loss_aux = compute_auxiliary_objectives(policy, batch)  # Eq.9

        loss_total = loss_ppo + alpha * loss_il + beta * loss_aux
        optimizer.step(loss_total)

    return policy
```

---

## 5 实验结论

### 5.1 主实验结果

| Method | CR↓ | DCR↓ | SCR↓ | DR↓ | PDR↓ | HDR↓ | ADD↓ | Long. Jerk↓ | Lat. Jerk↓ |
|--------|-----|------|------|-----|------|------|------|-------------|-------------|
| VAD | 0.335 | 0.273 | 0.062 | 0.314 | 0.255 | 0.059 | 0.304 | 5.284 | 0.550 |
| GenAD | 0.341 | 0.299 | 0.042 | 0.291 | 0.160 | 0.131 | 0.265 | 11.37 | 0.320 |
| VADv2 | 0.270 | 0.240 | 0.030 | 0.243 | 0.139 | 0.104 | 0.273 | 7.782 | 0.171 |
| **RAD** | **0.089** | **0.080** | **0.009** | **0.063** | **0.042** | **0.021** | **0.257** | **4.495** | **0.082** |

**关键结论**：RAD 在多数指标上优于 IL 方法，尤其碰撞率 (CR) 降低约 **3 倍**（0.089 vs 0.335）。

### 5.2 RL:IL Ratio 消融实验

| RL:IL | CR↓ | DCR↓ | SCR↓ | DR↓ | PDR↓ | HDR↓ | ADD↓ | Long. Jerk↓ | Lat. Jerk↓ |
|-------|------|------|------|-----|------|------|------|-------------|-------------|
| 0:1 (Pure IL) | 0.229 | 0.211 | 0.018 | 0.066 | 0.039 | 0.027 | **0.238** | 3.928 | 0.103 |
| 1:0 (Pure RL) | 0.143 | 0.128 | 0.015 | 0.080 | 0.065 | 0.015 | 0.345 | 4.204 | 0.085 |
| 2:1 | 0.137 | 0.125 | 0.012 | 0.059 | 0.050 | 0.009 | 0.274 | 4.538 | 0.092 |
| **4:1** | **0.089** | **0.080** | **0.009** | **0.063** | **0.042** | **0.021** | 0.257 | 4.495 | 0.082 |
| 8:1 | 0.125 | 0.116 | 0.009 | 0.084 | 0.045 | 0.039 | 0.323 | 5.285 | 0.115 |

**结论**：4:1 的 RL:IL 比例达到最佳平衡，CR 最低且 ADD 保持较低水平。

### 5.3 Reward Source 消融实验

| ID | r_dc | r_sc | r_pd | r_hd | CR↓ | ADD↓ |
|----|------|------|------|------|-----|------|
| 1 | ✓ | | | | 0.238 | 0.289 |
| 2 | | ✓ | ✓ | ✓ | 0.238 | 0.265 |
| 3 | ✓ | | ✓ | ✓ | 0.125 | 0.272 |
| 4 | ✓ | ✓ | | ✓ | 0.158 | 0.267 |
| 5 | ✓ | ✓ | ✓ | | 0.114 | 0.278 |
| **6** | **✓** | **✓** | **✓** | **✓** | **0.089** | **0.257** |

**结论**：完整 reward 组合 (ID 6) 达到最低 CR，证明各 reward 分量互补。

### 5.4 定性分析

![闭环对比：IL-only vs RAD](https://ar5iv.org/html/2502.13144/assets/x5.png)

IL-only 在动态环境中频繁碰撞，RAD 能有效避障并处理复杂交通场景。

---

## 6 KnowHow（核心洞察）

1. **3DGS 作为 RL 环境的优势**：3DGS 提供照片级真实渲染 + 高效推理（相比 NeRF），适合大规模闭环数据生成

2. **解耦动作空间的动机**：横向（换道）与纵向（加减速）决策相互独立，解耦后可独立优化，降低探索复杂度

3. **RL+IL 协同的洞见**：
   - RL 解决 IL 的"相关性 vs 因果性"问题
   - IL 解决 RL 的"非人类对齐"问题
   - 两者非竞争而是互补

4. **稀疏奖励缓解**：辅助目标函数通过 penalize 不良行为（碰撞、偏移）的概率，dense 地约束动作分布

5. **终止条件设计**：安全事件（碰撞/偏移）后立即终止 episode，因为后续 3DGS 渲染质量下降，不利于 RL

6. **三阶段解耦训练**：感知任务与规划任务可能冲突，分阶段训练+冻结参数可避免优化目标冲突

7. **GAE 在时序误差传播中的作用**：优势估计通过折扣向前传播，使早期决策也能获得梯度信号

8. **4:1 RL:IL 比例的实践意义**：RL 主导但保留 IL 正则化，在安全性与人类对齐间取得平衡

---

## 7 arXiv Appendix 关键点总结

由于无法直接访问 Appendix，以下列出从正文推断的关键补充内容：

- **A**: 更多闭环可视化对比（10+ 场景）
- **B**: 3DGS 环境重建细节
- **C**: 辅助目标函数的完整数学推导
- **D**: 感知预训练阶段 Map Head 和 Agent Head 的具体监督信号
- **E**: 实施细节：batch size、学习率、GAE 参数 ($\gamma, \lambda$)
- **F**: 更多基线方法对比
- **G**: 局限性讨论（3DGS 非反应性、低光渲染等）

---

## 8 总结

### 三大核心贡献

1. **首个 3DGS-RL 端到端自动驾驶训练框架**：通过照片级真实数字孪生环境实现大规模闭环试错
2. **RL+IL 协同优化新范式**：互补解决 IL 的因果混淆、开环gap问题与 RL 的人类对齐问题
3. **安全导向的 RL 算法设计**：解耦动作空间 + 四类安全奖励 + 密集辅助目标，碰撞率降低 3 倍

### 最重要洞察

**3DGS 不仅是评估工具，更是训练基础设施**。之前的工作（如 StreetGaussians、DrivingGaussion）仅用 3DGS 做闭环评估，RAD 首次将 3DGS 引入 RL 训练流程，释放了"在真实世界中试错"的安全学习可能性。

### 局限性

- 3DGS 环境非反应性（其他交通参与者仅 log-replay）
- 渲染质量可改进（非刚性行人、未观测视角、低光）

---

## 9 RAD vs RAD-2 对比

> 参见 [RAD-2 精读报告](./RAD-2_精读报告.md)

| 维度 | RAD | RAD-2 |
|------|-----|-------|
| **核心架构** | 单策略网络 + PPO | Generator-Discriminator 框架 |
| **轨迹生成** | 解耦离散动作空间 (横向+纵向) | 扩散模型多模态轨迹生成 |
| **RL 算法** | PPO + GAE | TCR-GRPO + On-policy Generator Optimization |
| **训练环境** | 3DGS 照片级数字孪生 | BEV-Warp (BEV 特征空间) |
| **环境交互** | 照片级真实渲染 | 高吞吐 BEV 特征级仿真 |
| **优化目标** | 动作维度解耦优化 | 轨迹流形整体优化 |
| **碰撞率降低** | 3x vs IL 方法 | 56% vs 扩散规划器 |
| **训练阶段** | 三阶段 (感知→规划→RL) | 四阶段 (Generator IL→Rollout→Discriminator→Generator) |
| **真机部署** | 未提及 | 已验证真实道路部署 |

### 关键优化点对比

**RAD 的贡献：**
- 验证了 3DGS 作为 RL 训练环境的可行性
- 证明 RL+IL 协同可以解决 IL 的因果混淆问题
- 解耦动作空间设计降低探索复杂度

**RAD-2 的贡献（相对于 RAD 的改进）：**
- **Generator-Discriminator 架构**：将轨迹生成与质量评估解耦，支持多模态轨迹候选的联合优化
- **扩散模型替代离散动作**：更自然地建模连续轨迹分布，支持多样性和鲁棒性
- **TCR-GRPO 算法**：利用时序一致性减少信用分配方差
- **On-policy Generator Optimization**：结构化纵向优化避免直接在高维空间反向传播
- **BEV-Warp 替代 3DGS**：大幅提升训练吞吐量，降低计算成本
- **真机部署验证**：仿真到真实的迁移能力

### 技术演进路线

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

**References:**
- [arXiv:2502.13144](https://arxiv.org/abs/2502.13144)
- [GitHub: hustvl/RAD](https://github.com/hustvl/RAD)
- [Project Page](https://hgao-cv.github.io/RAD/)
- [RAD-2: arXiv:2604.15308](./RAD-2_精读报告.md)

## 日期：20260428

### RL 领域（通用面试题）

**说明**：本期未处理新的RL论文，生成通用领域面试题

1. **[基础概念题]** PPO（Proximal Policy Optimization）是目前最广泛使用的强化学习算法之一。请详细解释PPO的核心技术：①Clipped Surrogate Objective的设计原理；②为什么要对策略更新幅度进行限制；③PPO与TRPO的核心区别是什么？
   - 答案/思考提示：①Clipped Surrogate Objective：L^CLIP(θ) = min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)，其中r_t = π_θ(a|s)/π_old(a|s)；②限制更新幅度：避免因大幅策略更新导致的性能崩溃，保证训练稳定性；③TRPO使用KL散度约束（二阶优化），PPO使用一阶优化+clipping，更简单高效。

2. **[深度思考题]** 在RLHF（Reinforcement Learning from Human Feedback）中，reward model是如何训练的？为什么需要一个独立的reward model而不是直接使用人类打分？如果reward model和policy model之间出现reward hacking应该如何缓解？
   - 答案/思考提示：①reward model训练：用pairwise comparison数据，学习"人类更偏好A还是B"的二元分类器；②独立reward model：人类打分噪声大、一致性差，用模型学习reward function更稳定；③reward hacking缓解：KL散度约束限制policy偏离SFT模型、加入对抗样本、多种reward组合、averse reward model。

3. **[实践应用题]** 假设你需要训练一个机械臂完成插孔任务（将杆插入孔中），这是一个稀疏奖励场景。请设计一个完整的强化学习训练方案，包括：①如何设计奖励函数（shaped reward vs sparse reward）；②如何处理稀疏奖励带来的探索问题；③如何利用演示数据加速训练（Imitation Learning + RL的结合）。
   - 答案/思考提示：①奖励函数：sparse reward = 1 if 插入成功 else 0；shaped reward可加距离奖励（杆到孔的距离越近奖励越高）、对齐奖励（角度误差越小奖励越高）；②稀疏奖励探索：好奇心驱动（intrinsic motivation）、Hindsight Experience Replay（HER）、课程学习；③演示数据：先用Behavior Cloning预训练，再RL微调（DAPG算法），或用演示数据初始化reward model。

4. **[优缺点对比题]** DPO（Direct Preference Optimization）与PPO+RM（Reward Model）是两种不同的RLHF范式。请从①训练稳定性、②样本效率、③超参数敏感性、④与模型规模的适配性 四个角度对比分析。
   - 答案/思考提示：①训练稳定性：DPO不需要reward model+PPO的迭代，更稳定；PPO需要同时训练RM和policy，可能不稳定；②样本效率：DPO直接用pairwise数据，样本效率高；PPO需要大量采样；③超参：DPO超参少（只需β），PPO有clipping、KL系数等多个超参；④规模化：DPO在大模型上效果已验证，PPO在大模型上训练成本高。

---

**经典算法系列**

5. **[基础概念题]** Q-learning和Policy Gradient是强化学习的两大类算法。请解释：①两者在价值估计和策略更新上的核心区别；②On-policy和Off-policy的概念在这两类算法中如何体现；③为什么Q-learning在连续动作空间难以应用？
   - 答案/思考提示：①Q-learning是value-based，学习最优状态动作价值函数Q*，策略从Q函数导出（ε-greedy）；Policy Gradient是policy-based，直接优化策略参数θ，∇J(θ) = E[∇θ log πθ(a|s) * Q(s,a)]；②Q-learning是off-policy（用ε-greedy策略采样但学习最优Q）；Policy Gradient（如PPO）是on-policy（采样策略和更新策略相同）；③连续动作空间：Q-learning需要argmax_a Q(s,a)，连续空间无解析解，需要额外优化（DQN用replay buffer但仍需离散化或穷举）。

6. **[深度思考题]** 在Actor-Critic架构中，Critic（价值估计）和Actor（策略更新）的作用分别是什么？为什么单独使用Policy Gradient（REINFORCE）会有高方差问题，而加入Critic可以缓解这个问题？Advantage Function在这里起到什么作用？
   - 答案/思考提示：①Critic：估计状态价值V(s)或状态动作价值Q(s,a)，为Actor提供梯度估计的基线；Actor：根据Critic的反馈更新策略参数；②高方差原因：REINFORCE用实际return（蒙特卡洛采样）作为Q估计，单样本方差大；Critic通过函数近似（神经网络）提供更稳定的价值估计，降低方差；③Advantage = Q(s,a) - V(s)：相对优势，衡量某动作相对于平均水平的优势，减少不必要的探索。

7. **[实践应用题]** 你在训练一个强化学习agent时发现训练曲线剧烈震荡、不收敛。请分析可能的原因，并给出至少5种调试/修复策略。
   - 答案/思考提示：①可能原因：学习率过大、奖励尺度不一致、策略更新幅度过大、off-policy算法但replay buffer太小、价值函数估计偏差累积；②修复策略：调小学习率/使用学习率调度；奖励归一化/缩放；引入梯度裁剪；使用PPO的clip机制；增加replay buffer size（off-policy）；减小batch size增加更新频率；价值函数target使用GAE减少偏差。

8. **[优缺点对比题]** Model-Based RL（如Dreamer系列世界模型）和Model-Free RL（如PPO、SAC）在样本效率和泛化能力上有根本差异。请从①样本效率、②泛化能力、③训练复杂度、④对未知环境的适应性 四个角度进行分析。
   - 答案/思考提示：①样本效率：Model-Based利用世界模型进行imagination，样本效率远高于Model-Free（DreamerV3在Atari仅需100K环境步vs Model-Free需要1B步）；②泛化能力：Model-Free在特定任务上可达到更高性能上限，但泛化能力弱；Model-Based的world model可泛化到未见过状态；③训练复杂度：Model-Based需要同时训练世界模型和策略，更复杂；Model-Free相对简单；④未知环境：Model-Based通过世界模型可以快速适应，Model-Free需要重新训练。

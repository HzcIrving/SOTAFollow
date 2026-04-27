# 面试题整理

## 日期：20260427

### RL 领域

**论文：《RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework》**

1. **[基础概念题]** RAD-2 提出的 Generator-Discriminator 框架中，Generator 和 Discriminator 各自的职责是什么？请解释它们如何协作完成轨迹规划任务。
   - **答案/思考提示**：Generator（扩散模型）负责生成多样化的轨迹候选，探索多模态轨迹分布；Discriminator（RL 判别器）负责根据长期驾驶质量对候选轨迹进行重排序，提供密集的信用分配信号。两者解耦设计避免直接在稀疏高维轨迹空间应用 RL。

2. **[深度思考题]** TCR-GRPO（Temporally Consistent Group Relative Policy Optimization）的核心动机是什么？为什么在时序一致的轨迹组内进行相对优势估计可以缓解信用分配问题？
   - **答案/思考提示**：信用分配困难源于独立样本比较方差大。TCR-GRPO 利用时序一致性——来自同一状态的 K 个候选轨迹构成一个组，组内进行相对优势归一化，减少方差。同一场景下的轨迹共享上下文信息，比较更具可靠性。

3. **[实践应用题]** 如果要将 RAD-2 的 Generator-Discriminator 框架应用到无人机控制场景，需要做哪些关键修改？考虑输入模态、动作空间、环境交互方式的变化。
   - **答案/思考提示**：① 输入从 BEV 图像变为 IMU + 视觉/激光雷达；② 动作空间从车辆横纵向变为无人机 6-DOF 控制；③ 仿真环境从 BEV-Warp 变为 AirSim 等无人机仿真器；④ Generator 可能需要适配无人机动力学模型。

4. **[优缺点对比题]** 比较 RAD-2 与 RAD 的核心差异。从架构、训练范式、环境交互三个维度分析 RAD-2 的改进。
   - **答案/思考提示**：架构上 RAD-2 从单策略网络变为 Generator-Discriminator；训练上从 PPO+GAE 变为 TCR-GRPO+On-policy Generator Optimization；环境从 3DGS 照片级渲染变为 BEV-Warp 高吞吐仿真。RAD-2 用扩散模型替代离散动作，更自然建模连续轨迹分布。

5. **[基础概念题]** On-policy Generator Optimization 的核心思想是什么？它如何将闭环反馈转化为结构化优化信号？
   - **答案/思考提示**：识别低奖励轨迹的特征（时间步 t、轨迹形状），沿时间轴调整生成器分布参数。通过 KL 散度约束避免剧烈分布偏移，将"哪条轨迹不好"的反馈转化为"如何调整生成分布"的梯度信号。

---

**通用 RL 领域思考题**

6. **[深度思考题]** 在 Actor-Critic 架构中，Value Function 的 bootstrap 机制可能导致 TD 误差累积，为什么这个问题在长时序任务中尤为严重？有何缓解方法？
   - **答案/思考提示**：多步 bootstrap 时，误差会逐层传播放大。方法：① 减小 bootstrap 步数（用 GAE）；② 使用 Target Network 定期对齐；③ 采用 multi-step return 平衡偏差与方差。

7. **[实践应用题]** 当你在实际项目中遇到 RL 训练不稳定、reward 震荡的情况，你会如何诊断和解决？请列出你的 checklist。
   - **答案/思考提示**：① 检查 reward shaping 是否合理（稀疏/稠密）；② 观察 action 分布熵的变化；③ 检查是否有过拟合到早期经验；④ 尝试 PPO 的 clip range 调整；⑤ 检查环境 stochasticity。

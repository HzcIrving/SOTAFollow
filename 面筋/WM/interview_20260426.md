## 日期：20260426

### WM 领域（World Model）

**论文：《Fast-WAM: Do World Action Models Need Test-time Future Imagination?》**

1. **[基础概念题]** Fast-WAM 提出了一个关键问题：WAMs 的有效性到底来自训练时的视频预测目标还是推理时的显式未来生成？论文通过怎样的实验设计来分离这两个因素？
   - 答案/思考提示：设计了三种受控变体：Fast-WAM-Joint（联合生成范式）、Fast-WAM-IDM（先视频后动作范式）、Fast-WAM w/o video co-train（移除 video co-training 目标）。通过对比发现：移除 video co-training 造成 ~8% 性能下降（91.8% → 83.8%），而三种不同推理方式差异仅 ~1%。证明训练时的视频联合训练才是主导因素。

2. **[深度思考题]** Fast-WAM 指出"test-time 未来想象可能是伪需求"。请从"世界模型的价值"角度阐述这一观点，并对比 LeWorldModel 和 Fast-WAM 两种截然不同的哲学。
   - 答案/思考提示：核心洞察——"世界模型的价值不在于'想象未来'，而在于'理解现在'"。LeWorldModel 的哲学：预测即理解，通过预测未来 embedding 学习世界的底层结构；Fast-WAM 的哲学：训练即世界模型，video co-training 塑造了强表征，推理时只需从中提取动作，无需费力生成未来。两者共同指向：表示学习才是目的，test-time 想象可能只是训练目标的"过度执行"。

3. **[优缺点对比题]** Fast-WAM（190ms 推理延迟）与 imagine-then-execute WAMs（>760ms 推理延迟）相比，在实际部署场景中各有何优劣？
   - 答案/思考提示：Fast-WAM：推理速度快（190ms）、适合实时控制，但牺牲了可解释性（无显式未来预测）和长时序规划能力；Imagine-then-execute WAMs：推理速度慢但提供可解释的未来轨迹想象，适合需要人类介入或长程规划的场景。Fast-WAM 的 paper 证明了在很多任务上，隐式规划（直接动作预测）已经足够好。

4. **[实践应用题]** Fast-WAM 的 MoT（Mixture-of-Transformer）架构中，训练时和推理时的注意力掩码有何不同？这种设计如何实现？
   - 答案/思考提示：训练时三组 token（Clean Obs / Noisy Video / Action）共享注意力，但关键约束：动作 tokens 不能 attend 未来视频 tokens；推理时完全移除未来视频分支，仅保留 clean first-frame latent tokens。实现方式是通过结构化注意力掩码矩阵控制信息流，在推理时将整个视频分支置零。

5. **[深度思考题]** DreamerAD 提出了 Shortcut Forcing 技术将扩散采样从 100 步压缩到 1 步（80× 加速）。与简单的模型蒸馏相比，Shortcut Forcing 的设计巧妙之处在哪里？
   - 答案/思考提示：Shortcut Forcing 不是简单地蒸馏一个 1-step 模型，而是引入 step embedding 让模型感知当前采样步数 $d$。当 $d > d_{min}$ 时，用两个 teacher half-step 各自预测再平均作为 target，让 student 向这个 target 学习。这样模型在任何推理步数下都能工作，且 1-step 推理质量不会崩塌。原始 Epona 1-step 会出现严重模糊和误差累积，而 Shortcut Forcing 避免了这一问题。

---

**通用 World Model 领域思考题**

6. **[基础概念题]** Latent-WAM 在端到端自动驾驶中提出了 SCWE（空间感知压缩世界编码器）和 DLWM（动态潜在世界模型）两大模块。请解释 SCWE 如何通过 16 个 scene query tokens 实现高效压缩，以及几何蒸馏（geometric distillation）的意义。
   - 答案/思考提示：SCWE 使用随机初始化的 16 个 scene query tokens 与图像 patch tokens 拼接后送入 DINO encoder，将数百个 patch tokens 压缩到仅 16 个 tokens，大幅降低计算开销。几何蒸馏使用 WorldMirror（VGGT-based）作为教师，将几何感知能力蒸馏到 vision backbone，增强 3D 空间理解。关键发现：直接拼接几何特征会引入冲突信号（88.0），而端到端蒸馏（89.3）使 backbone 自适应学习与规划对齐的空间表征。

7. **[优缺点对比题]** DreamerAD 的 Shortcut Forcing 与 Latent-WAM 的几何蒸馏，分别从"效率"和"表征质量"角度改进了世界模型。两篇论文的改进思路有何本质不同？你更看好哪个方向？
   - 答案/思考提示：DreamerAD 的 Shortcut Forcing 是推理加速方向的改进，解决"世界模型无法实时闭环 RL"的效率问题；Latent-WAM 的几何蒸馏是表征学习方向的改进，解决"世界模型缺乏 3D 几何感知"的表征问题。两者都在各自的维度上推动了世界模型的实用化，但效率问题更紧迫——没有实时性，表征再好也无法落地。

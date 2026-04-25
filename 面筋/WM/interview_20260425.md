# World Model 世界模型面试题 & 思考题

**Date**: 2026-04-25
**来源**: 基于仓库 WorldModel 论文（LeWorldModel、DreamerAD）

---

## 一、基础概念题

### 1. JEPA (Joint Embedding Predictive Architecture) 的核心思想是什么？它与重建式方法（如像素级 VAE）的本质区别？

**答案提示**：
- JEPA：在隐空间而非像素空间做预测
- 避免像素级重建的计算开销和语义缺失

| 方法 | 表征空间 | 优化目标 | 计算成本 |
|------|---------|---------|---------|
| VAE 重建 | 像素空间 | 重构误差 | 高 |
| JEPA 预测 | 隐空间 | 预测误差 | 低 |

---

### 2. LeWorldModel 提出的 SIGReg 正则器是什么？它解决了什么问题？

**答案提示**：
- SIGReg = Signature Regularizer，基于 Cramér-Wold 定理
- 将 latent embedding 的分布正则化到各向同性高斯分布
- 解决：端到端 JEPA 训练的表示崩溃问题

**数学形式**：

$$\text{SIGReg}(Z) = \frac{1}{M} \sum_{m=1}^{M} T\left( h^{(m)} \right)$$

其中 $h^{(m)} = Z \cdot u^{(m)}$ 是随机投影，$T$ 是正态性检验统计量

---

### 3. DreamerAD 的 Shortcut Forcing 机制是什么？解决了什么问题？

**答案提示**：
- 问题：Pixel-level diffusion world model 每帧推理 ~2秒，无法支持高频 RL 交互
- 解决：将 100 步扩散采样压缩到 1 步，80× 推理加速
- 关键：不简单地蒸馏 1-step 模型，而是引入 step embedding 让模型感知采样步数

---

## 二、深度思考题

### 4. 为什么 DreamerAD 选择在隐空间而非像素空间做 RL 训练？

**答案提示**：
1. **效率**：像素空间每帧 2 秒延迟 vs 隐空间大幅降低
2. **语义对齐**：像素级 loss 关注视觉保真度，隐空间 loss 关注驾驶安全性
3. **泛化**：隐空间特征对模糊/清晰图像的表征更一致

---

### 5. LeWorldModel 为什么只用 2 个 loss 项就能稳定训练？传统 JEPA 需要多少个？为什么？

**答案提示**：
- 传统端到端 JEPA 需要：预测 loss + 对比学习 loss + EMA 更新 + ...
- LeWorldModel：预测 loss + SIGReg 正则（仅 2 项）
- SIGReg 从根本上消除表示崩溃，不需要复杂的防崩溃机制

---

### 6. DreamerAD 的"探索词表"(Vocabulary) 是什么？如何构造的？

**答案提示**：
- 目的：解决 flow matching 确定性采样导致轨迹不平滑的问题
- 构造：从 8192 条轨迹的端点状态过滤（横向≤5m，纵向≤10m，航向≤20°），再均匀采样
- 最终保留 K=256 条代表性轨迹，Gaussian 采样在其邻域内探索

---

## 三、优缺点对比题

### 7. LeWorldModel vs DreamerAD：同样是 World Model for RL，它们的核心区别是什么？

| 方面 | LeWorldModel | DreamerAD |
|------|-------------|-----------|
| 表征空间 | 隐空间（JEPA） | 隐空间（Video DiT） |
| 任务类型 | 2D/3D 控制 | 自动驾驶 |
| RL 算法 | CEM 规划 | GRPO |
| 效率 | 15M 参数，单 GPU 数小时 | 80× 采样加速 |

**思考**：两者都避免像素级世界建模，它们在隐空间建模上的具体实现有何不同？

---

### 8. DreamerAD 的 Shortcut Forcing vs 简单的步数蒸馏：核心区别是什么？

**答案提示**：
- 简单蒸馏：直接学习 1-step 去噪器
- Shortcut Forcing：引入 step embedding，中间步用 teacher half-step 平均
- 结果：Shortcut Forcing 在 1-step 时质量不崩塌，原始 Epona 1-step 会严重模糊

---

## 四、实践应用题

### 9. 如果你在自动驾驶场景中部署 DreamerAD 的 world model，推理延迟要求 10Hz，你会怎么优化？

**思考方向**：
- Shortcut Forcing 已经将 100 步压缩到 1 步
- 进一步优化：量化、剪枝、知识蒸馏
- 关键是保证 1-step 质量不下降

---

### 10. SIGReg 正则器需要 1024 个随机投影方向，计算量是否太大？有何优化空间？

**答案提示**：
- 1024 是默认配置，可以在精度和效率间权衡
- 可能的优化：随机投影复用（跨 batch）、PCA 降维后投影
- 关键洞察：Cramer-Wold 定理保证所有方向都需要覆盖

---

## 五、开放性思考题

### 11. LeWorldModel 提出 SIGReg 来解决表示崩溃，DreamerAD 用什么机制避免崩溃？两者的设计哲学有何不同？

**思考方向**：
- LeWorldModel：预防性正则——强制分布匹配高斯
- DreamerAD：利用 Video DiT 的去噪 latent feature 天然具有语义结构（"挖宝"思路）
- 哪个更优雅？哪个更通用？

---

### 12. World Model + RL 的范式在机器人操作任务中的主要挑战是什么？与自动驾驶场景有何不同？

**提示**：
- 自动驾驶：状态空间相对结构化（道路、车辆）
- 机器人操作：物体交互的多样性、接触力学的复杂性
- 共同挑战：世界模型的泛化能力、reward 的设计

---

## 参考来源

| 论文 | 核心考点 |
|------|---------|
| LeWorldModel (2026) | JEPA, SIGReg, 端到端训练, CEM规划 |
| DreamerAD (2026) | Shortcut Forcing, 隐空间RL, 探索词表, GRPO |

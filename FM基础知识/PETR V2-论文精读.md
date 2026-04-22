# PETR V2 论文精读报告

**论文标题**：PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images

**arXiv 编号**：2206.01256v3

**发表时间**：2022 年 11 月 14 日

**作者**：Yingfei Liu, Junjie Yan, Tiancai Wang*, Xiangyu Zhang, Fan Jia, Shuailin Li, Aqi Gao, Jian Sun

**单位**：MEGVII Technology（旷视科技）

**开源地址**：https://github.com/megvii-research/PETR

**基础工作**：基于 PETR（arXiv:2203.05625）扩展

**核心一句话总结**：PETRv2 在 PETR 基础上，通过 **3D 坐标时序对齐** 实现时序建模、提出 **特征引导位置编码器 FPE** 生成数据驱动的 3D PE，并设计 **稀疏任务专属查询**，统一支持 3D 目标检测、BEV 分割、3D 车道检测三大任务，在全部任务上达到 SOTA，并系统验证了鲁棒性。

------

## 1. Motivation（问题背景）

### 1.1 多视图 3D 感知的方法分类

多相机 3D 感知分为两类：

- **BEV 类**：显式将多视图特征转为 BEV 表示（如 BEVDet、BEVFormer）。
- **DETR 类**：将 3D 目标建模为 object query，端到端检测（如 DETR3D、PETR）。

PETR 核心思路：将 3D 位置信息编码进 2D 特征，形成 **3D 位置感知特征**，让 query 直接在 3D 空间交互。

### 1.2 PETR 存在的问题

| 问题 | 具体表现 |
|------|----------|
| **无时序信息** | 仅单帧推理，速度估计不准、定位不稳定 |
| **3D PE 数据无关** | 仅由固定网格 3D 坐标生成，与图像内容无关 |
| **单任务** | 只支持 3D 检测，无法做分割、车道线 |
| **鲁棒性不足** | 对相机外参噪声、传感器丢帧敏感 |

### 1.3 Related Works

| 工作 | 核心思想 | 局限性 |
|------|----------|--------|
| **BEVDet / BEVDet4D** | 基于 LSS 显式升维到 BEV | Z 轴量化误差 |
| **BEVFormer** | 使用 BEV query + 空间/时序注意力 | 计算量大 |
| **DETR3D** | 3D query 投影到 2D 采样特征 | 反复投影误差累积 |
| **PETR** | 用 3D PE 一次性编码 3D 位置 | 无时序、数据无关 |

### 1.4 核心问题

> 如何在 PETR 基础上实现**时序建模**、让 3D PE **数据驱动**、并用**稀疏查询**统一多任务？

---

## 2. 相关工作

## 2.1 多视图 3D 检测

- BEVDet / BEVDet4D：基于 LSS 显式升维到 BEV。
- BEVFormer：使用 BEV query + 空间 / 时序注意力。
- DETR3D：3D query 投影到 2D 采样特征，需反复投影。
- PETR：用 3D PE 一次性编码 3D 位置。

## 2.2 BEV 分割

VPN、Lift-Splat-Shoot、M²BEV、BEVFormer、BEVSegFormer 等，大多基于稠密 BEV 表示。

## 2.3 3D 车道检测

3D-LaneNet、Gen-LaneNet、PersFormer 等，多基于 IPM 或 BEV。

------

# 4 方法

## 4.1 整体架构

输入：

- 多视图图像序列（t 帧 + t−1 帧）
- 相机内参、外参、自车运动位姿

流程：

1. **Backbone** 提取 2D 图像特征。
2. **3D 坐标生成**：同 PETR，由锥台网格生成 3D 点。
3. **时序坐标对齐**：将 t−1 帧 3D 点变换到 t 帧坐标系。
4. **concat** 拼接两帧特征与 3D 坐标。
5. **FPE 编码器**生成 3D 位置感知特征，输出 **key / value**。
6. **任务专属查询**（Det/Seg/Lane）输入解码器。
7. Transformer 解码器交互更新。
8. **任务头**输出 3D 框、BEV 分割、3D 车道。
9. ![PETR V2 Pipeline](titok_figures/image-20260418205057807.png "width=800")

------

## 4.2 时序建模：3D Coordinates Alignment（核心）

### 4.2.1 坐标系定义

- c(t)：t 时刻相机坐标系
- e(t)：t 时刻自车坐标系
- l(t)：t 时刻激光雷达坐标系（PETRv2 默认 3D 空间）
- g：全局世界坐标系
- Tsrcdst：源坐标系 → 目标坐标系变换矩阵

### 4.2.2 单帧 3D 点计算（公式 1）

Pil(t)(t)=Tci(t)l(t)⋅Ki−1⋅Pm(t)

- Pm(t)：相机锥台网格点（像素 + 深度）
- Ki−1：**4×4 逆投影矩阵**（内参 + 外参）
- Tci(t)l(t)：相机 → 激光坐标系
- 输出：**激光系下 3D 点**

### 4.2.3 时序对齐（公式 2、3）

目标：把 t−1 帧的 3D 点 **对齐到 t 帧激光坐标系**。

Pil(t)(t−1)=Tl(t−1)l(t)⋅Pil(t−1)(t−1)

其中跨帧变换矩阵：

Tl(t−1)l(t)=Te(t)l(t)⋅Tge(t)⋅(Tge(t−1))−1⋅(Te(t−1)l(t−1))−1

**意义**：

- 仅对 **3D 坐标做刚性变换对齐**，不做特征变换。
- 时序信息完全由 **3D PE** 带入模型，极简、无额外延迟。

------

## 4.3 多任务学习：稀疏任务专属查询

PETRv2 不使用稠密 BEV query，而是为每个任务设计 **少量稀疏可学习 query**。

### 4.3.1 检测查询（Det Query）

- 空间：**3D 世界空间**
- 初始化：均匀分布的 3D 可学习锚点 → MLP → query
- 数量：900 / 1500
- 输出：3D 框、类别、速度

### 4.3.2 分割查询（Seg Query）

- 空间：**BEV 空间**
- 初始化：BEV 网格固定锚点 → MLP → query
- 每个查询负责一个 patch（如 25×25）
- 数量：625（对应 200×200 BEV 图）
- 输出：可行驶区域、车道、车辆

### 4.3.3 车道查询（Lane Query）

- 空间：**3D 空间**
- 形式：**锚定线（anchor lane）**
- 构造：沿 Y 轴均匀采样 N 个 3D 点（默认 10，可扩展到 100/300/400）
- 预测：相对于锚线的偏移 (Δx,Δz) + 可见性掩码
- 输出：3D 车道线实例

------

## 4.4 特征引导位置编码器 FPE（核心创新）

### 4.4.1 原 PETR 3D PE

PEi3d(t)=ψ(Pil(t)(t))

- ψ：MLP
- **缺点**：data-independent，只与几何有关，与图像无关。

### 4.4.2 PETRv2 FPE（公式）

PEi3d(t)=ξ(Fi(t))⊙ψ(Pil(t)(t))

- ξ(Fi(t))：2D 特征 → 1×1 Conv → MLP → Sigmoid → **注意力权重**

- ψ(⋅)：3D 坐标 → MLP → 基础 3D PE

- ⊙：逐元素相乘

  ![PETR V2 Architecture](titok_figures/image-20260418205029065.png)

### 4.4.3 FPE 详细流程

1. 2D 图像特征经过 1×1 Conv 降维。
2. 送入小 MLP + Sigmoid → 得到 **权重图**。
3. 3D 坐标送入另一个 MLP → 基础 3D PE。
4. 权重 × 3D PE → **数据驱动 3D PE**。
5. 3D PE + 2D 特征 → **key**。
6. 2D 特征直接作为 **value**。
7. 送入 Transformer 解码器。

------

## 4.5 损失函数

### 4.5.1 3D 检测

- 分类：Focal Loss
- 回归：L1 Loss
- 匹配：匈牙利算法
- 加入 DN-DETR 去噪查询加速收敛

### 4.5.2 BEV 分割

- 逐类别 Focal Loss

### 4.5.3 3D 车道检测

- 类别：Focal Loss
- 可见性：Focal Loss
- 偏移：L1 Loss

------

# 5 实验设置

## 5.1 数据集

- **3D 检测 / BEV 分割**：nuScenes（1000 场景，700/150/150）
- **3D 车道检测**：OpenLane（200K 帧，880K 车道标注）

## 5.2 实现细节

- 骨干：ResNet-50/101、VoVNetV2-99、Swin-T/S-B
- 特征层：P4（1/16 分辨率）
- 深度采样：D=64
- 优化器：AdamW，weight_decay=0.01
- 学习率：2e-4，cosine 退火
- 训练轮数：24 epoch
- 时序帧选择：
  - 训练：随机 [3T, 27T]
  - 推理：固定 15T（T≈0.083s）
- 查询数量：
  - Det：900/1500
  - Seg：625
  - Lane：100

------

# 6 实验结果

## 6.1 3D 目标检测（nuScenes test）

- PETRv2* (VoVNetV2)：**59.1% NDS / 50.8% mAP**
- mAVE：**0.343 m/s**（PETR 为 0.808）
- 显著超过 BEVFormer、BEVDet4D、DETR3D 等

## 6.2 BEV 分割（nuScenes val）

- Driveable：85.6%

- Lane：49.0%

- Vehicle：46.3%



  超过 BEVFormer 等 SOTA

## 6.3 3D 车道检测（OpenLane）

- PETRv2-E（EfficientNet）：51.9% F1
- PETRv2-V（VoVNetV2）：57.8% F1
- 400 点锚线：**61.2% F1**

## 6.4 消融实验（VoVNet-99）

| 方法   | CA   | FPE  | NDS      | mAP      | mAVE      |
| :----- | :--- | :--- | :------- | :------- | :-------- |
| PETR   |      |      | 44.9     | 38.1     | 0.838     |
| +CA    | ✓    |      | 46.1     | 38.4     | 0.605     |
| +FPE   |      | ✓    | 46.4     | 39.3     | 0.736     |
| PETRv2 | ✓    | ✓    | **49.6** | **40.1** | **0.429** |

结论：

- **CA 显著提升时序与速度估计**
- **FPE 提升精度与鲁棒性**
- **CA+FPE 联合增益最大**

------

# 7 鲁棒性分析

PETRv2 系统测试三种真实噪声：

## 7.1 相机外参噪声（旋转噪声）

- Rmax=2/4/6/8 度
- 结论：**FPE 可显著提升抗外参噪声能力**
- PETRv2 下降幅度远小于 PETR

## 7.2 相机丢失（Camera Miss）

- 去掉前 / 后 / 左 / 右等相机
- 后相机缺失影响最大：**-13.19% mAP**
- 前相机：-5.05% mAP
- 其余平均：-2.93% mAP

## 7.3 时间延迟（Time Delay）

- T≈0.083s
- 延迟 1T：mAP 下降 3.19%
- 延迟 3T（≈0.25s）：mAP 下降 9.17%
- 结论：时间延迟影响极大，需时序补偿

------

# 8 结论

1. PETRv2 基于 **3D 坐标对齐** 实现极简高效时序建模。
2. **FPE** 使 3D PE 从数据无关变为数据驱动，提升精度与鲁棒性。
3. **稀疏任务查询** 统一 3D 检测、BEV 分割、3D 车道检测。
4. 在全部任务上达到 SOTA。
5. 提供多维度鲁棒性分析，可作为多相机 3D 感知统一基线。

------

# 9 PETR vs PETRv2 精确对比

| 模块       | PETR         | PETRv2                    |
| :--------- | :----------- | :------------------------ |
| 时序       | 单帧         | 3D 坐标对齐时序融合       |
| 3D PE      | 数据无关     | 数据驱动（FPE 加权）      |
| 输入       | 单帧         | 双帧时序                  |
| 任务       | 仅 3D 检测   | 检测 + BEV 分割 + 3D 车道 |
| 查询       | 仅 Det Query | Det/Seg/Lane 三查询       |
| 速度估计   | 差           | 好（时序加持）            |
| 抗外参噪声 | 弱           | 强（FPE）                 |
| 框架       | 单任务检测器 | 统一多任务感知框架        |

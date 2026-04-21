# PETR 论文精读报告

**论文标题**：PETR: Position Embedding Transformation for Multi-View 3D Object Detection

**arXiv**：2203.05625v3

**作者**：Yingfei Liu, Tiancai Wang, Xiangyu Zhang, Jian Sun

**机构**：MEGVII Technology（旷视科技）

**代码**：https://github.com/megvii-research/PETR

**核心一句话总结**：PETR 通过**3D 位置嵌入变换**，将多视图 2D 图像特征转化为 3D 位置感知特征，让目标查询直接在 3D 空间交互，实现无复杂投影采样的端到端多视图 3D 目标检测，在 nuScenes 数据集达 SOTA（50.4% NDS、44.1% mAP）。

------

## 一、研究背景与动机

### 1. 问题背景

多视图 3D 目标检测是自动驾驶低成本感知方案，基于 DETR 的端到端检测成为主流，但现有方法存在缺陷：

- **DETR3D**：需反复将 3D 参考点投影回 2D 图像采样特征，存在投影误差、采样不全面、流程复杂问题。
- **BEV 类方法**：易引入 Z 轴误差，适配 3D 车道线等任务效果差。

### 2. 核心动机

受隐式神经表示（INR）启发，将**3D 坐标信息编码进 2D 特征**，生成 3D 位置感知特征，让目标查询直接在 3D 空间交互，规避 2D→3D 投影与特征采样。

------

## 二、核心贡献

1. 提出**PETR 框架**：通过 3D 位置嵌入变换，把多视图 2D 特征转为 3D 位置感知特征，实现简洁端到端 3D 检测。
2. 设计**3D 位置编码器**：用 MLP 将 3D 坐标转为位置嵌入，与 2D 特征融合，建立多视图 3D 空间关联。
3. 提出**3D 空间锚点查询生成**：用 3D 空间可学习锚点初始化目标查询，解决 3D 场景收敛难题。
4. 达 SOTA 性能：nuScenes 测试集**50.4% NDS、44.1% mAP**，为首个超 50% NDS 的纯视觉方法。

------

## 三、方法详解

### 1. 整体架构

1. **Backbone**：ResNet/Swin-Transformer 提取多视图 2D 图像特征。
2. **3D 坐标生成器**：将相机锥台空间离散为 3D 网格，经相机参数转换为 3D 世界坐标并归一化。
3. **3D 位置编码器**：3D 坐标→MLP→3D 位置嵌入，与 2D 特征融合为 3D 位置感知特征。
4. **查询生成器 + Transformer 解码器**：3D 锚点→MLP→目标查询，与 3D 特征交互更新。
5. **检测头**：分类（Focal Loss）+3D 框回归（L1 Loss），匈牙利算法匹配标签。

### 2. 3D 坐标生成器

目的：把**相机锥台空间**离散成网格点 → 投影到**3D 世界空间** → **归一化** → 输出最终 3D 坐标

1. 离散相机锥台空间为 **(W_F, H_F, D)** 3D 网格，生成带深度的像素点坐标。

2. 逆投影变换：通过4×4 完整逆投影矩阵 Kᵢ⁻¹ 转为**3D 世界坐标**。

3. 归一化：将 3D 坐标缩至 [0,1]，统一输入尺度。

   ![PETR 3D Position Encoding](titok_figures/image-20260418202437582.png "width=800")

### 3. 3D 位置编码器

1. 3D 坐标经**MLP**生成 3D 位置嵌入（3D PE）。

2. 2D 特征经 1×1 卷积降维，与 3D PE**逐元素相加**融合。

3. 展平后作为 Transformer 解码器的键值对，输出 3D 位置感知特征。

   PETR 3DPE 完整伪代码（Python）

   ```
   import torch
   import torch.nn as nn
   import numpy as np
   
   # ==============================================
   # 【步骤 1】3D Coordinates Generator（坐标生成）
   # 严格对应 PETR 公式 (1)
   # ==============================================
   def generate_3d_world_points(cam_param, H, W, D=64):
       """
       输入：
           cam_param: 相机参数 (内参 K + 外参 RT + 自车位姿)
           H, W: 特征图尺寸
           D: 深度采样数（PETR 默认 64）
       输出：
           points_3d_world: (H, W, D, 3) —— 3D 世界坐标
       """
       # -------------------- 1. 生成像素网格 (u, v) --------------------
       u = torch.linspace(0, W-1, W)
       v = torch.linspace(0, H-1, H)
       uu, vv = torch.meshgrid(u, v, indexing='xy')  # (W, H) → (H, W)
       
       # -------------------- 2. 生成深度 d --------------------
       d = torch.linspace(1.0, 60.0, D)  # PETR 深度范围 1~60m
       
       # -------------------- 3. 构建锥台点 (u*d, v*d, d) --------------------
       uu = uu.unsqueeze(-1).repeat(1,1,D)  # (H, W, D)
       vv = vv.unsqueeze(-1).repeat(1,1,D)
       dd = d.view(1,1,D).repeat(H,W,1)
       
       # 像素齐次坐标：(u*d, v*d, d, 1)
       points_cam = torch.stack([uu*dd, vv*dd, dd, torch.ones_like(dd)], dim=-1)  # (H, W, D, 4)
   
       # -------------------- 4. 逆投影 → 3D 世界坐标 --------------------
       # ✅ 关键：cam_param 是 4×4 投影矩阵 P = K * RT * ego2world
       # ✅ 不是单纯内参 K！
       P_inv = torch.inverse(cam_param)  # 逆投影矩阵
       
       # 矩阵乘法：世界坐标 = P_inv × 锥台点
       points_3d_world = torch.matmul(points_cam, P_inv.T)  # (H, W, D, 4)
       
       # 转欧式坐标（忽略齐次项 w）
       points_3d_world = points_3d_world[..., :3]  # (H, W, D, 3)
   
       return points_3d_world
   
   
   # ==============================================
   # 【步骤 2】3D Position Embedding（3D PE 生成）
   # PETR 核心：归一化 + MLP 编码
   # ==============================================
   class PETR_3D_PE(nn.Module):
       def __init__(self, embed_dims=256):
           super().__init__()
           self.embed_dims = embed_dims
           
           # PETR 使用 2 层 MLP 生成 3D PE
           self.pe_mlp = nn.Sequential(
               nn.Linear(3, embed_dims),
               nn.ReLU(),
               nn.Linear(embed_dims, embed_dims)
           )
   
       def forward(self, points_3d_world):
           """
           输入：3D 世界坐标 (H, W, D, 3)
           输出：3D PE (H, W, D, embed_dims)
           """
           # -------------------- 1. 坐标归一化（PETR 必须步骤） --------------------
           # nuScenes 全局坐标范围：x [-50,50], y [-50,50], z [-5,3]
           x = (points_3d_world[..., 0] + 50.0) / 100.0
           y = (points_3d_world[..., 1] + 50.0) / 100.0
           z = (points_3d_world[..., 2] + 5.0) / 8.0
           points_norm = torch.stack([x, y, z], dim=-1)  # (H, W, D, 3) → [0,1]
   
           # -------------------- 2. MLP 编码 → 3D PE --------------------
           pe_3d = self.pe_mlp(points_norm)  # (H, W, D, C)
   
           return pe_3d
   
   
   # ==============================================
   # 【步骤 3】最终融合：2D 图像特征 + 3D PE
   # ==============================================
   def get_3d_aware_feature(img_feature, pe_3d):
       """
       img_feature: (H, W, C)  2D 图像特征
       pe_3d: (H, W, D, C)     3D 位置嵌入
       返回：3D 位置感知特征 (H, W, D, C)
       """
       # 扩展维度 + 逐元素相加（PETR 原文融合方式）
       img_feature = img_feature.unsqueeze(2)  # (H, W, 1, C)
       feat_3d = img_feature + pe_3d           # (H, W, D, C)
       return feat_3d
   
   
   # ====================== 测试调用 ======================
   if __name__ == "__main__":
       # 模拟参数
       H, W, D = 24, 48, 64
       embed_dims = 256
       cam_param = torch.eye(4)  # 模拟 4×4 投影矩阵
       
       # 1. 生成 3D 世界坐标
       points_3d = generate_3d_world_points(cam_param, H, W, D)
       print("3D world points shape:", points_3d.shape)  # (24,48,64,3)
   
       # 2. 生成 3D PE
       pe_gen = PETR_3D_PE(embed_dims)
       pe_3d = pe_gen(points_3d)
       print("3D PE shape:", pe_3d.shape)  # (24,48,64,256)
   ```

   

### 4. 查询生成与解码器

- **查询生成**：3D 空间均匀分布的可学习锚点→两层 MLP→初始目标查询，保障 3D 场景收敛。
- **解码器**：标准 DETR 解码器，目标查询与 3D 位置感知特征做多头注意力交互，迭代更新。

### 5. 损失函数

$$
L(y, \hat{y}) = \lambda_{cls} \cdot L_{cls}(c, \sigma(\hat{c})) + L_{reg}(b, \sigma(\hat{b}))
$$

- Lcls：分类用 Focal Loss
- Lreg：3D 框回归用 L1 Loss
- λcls=2.0：平衡分类与回归损失
- σ：匈牙利算法最优匹配函数

------

## 四、实验设置

### 1. 数据集与指标

- **数据集**：nuScenes（1000 个场景，6 摄像头 + 激光雷达 + 雷达）
- **核心指标**：NDS（nuScenes Detection Score）、mAP（平均精度）

### 2. 实现细节

- 骨干网络：ResNet-50/101、Swin-Transformer、VoVNetV2
- 优化器：AdamW，权重衰减 0.01，学习率 2e-4，余弦退火
- 训练：8×V100，batch size 8，24 epoch，多尺度训练
- 深度采样：沿深度轴采样 64 点，线性递增离散（LID）

------

## 五、实验结果

### 1. 验证集对比

PETR 在同配置下超越 DETR3D、BEVDet、FCOS3D 等方法，ResNet-101 骨干 NDS 达**44.2%**。

### 2. 测试集 SOTA

- 无外部数据：Swin-B 骨干→**48.3% NDS、44.5% mAP**
- 有外部数据：VoVNetV2 骨干→**50.4% NDS、44.1% mAP**（纯视觉首个破 50% NDS）

### 3. 消融实验

1. **3D PE 关键作用**：仅 3D PE 即可达 30.5% mAP，融合 2D PE + 多视图先验小幅提升。
2. **编码器设计**：MLP 生成 3D PE、1×1 卷积 + 逐元素相加为最优配置。
3. **锚点设计**：3D 可学习锚点 > 3D 固定锚点 > BEV 锚点 > 无锚点，1500 个锚点性能最优。
4. **离散方式**：均匀离散（UD）与线性递增离散（LID）性能接近。

### 4. 收敛与速度

- 收敛：前期慢于 DETR3D，最终性能更高，需更长训练周期。
- 速度：1056×384 输入下单卡 V100 达**10.7 FPS**，快于 BEVDet。

------

## 六、可视化与失败案例

1. **定性结果**：3D 检测框与真值高度吻合，BEV 视图与图像视图匹配准确。
2. **注意力图**：目标查询可在多视图聚焦同一目标，3D PE 建立跨视图空间关联。
3. **失败案例**：小目标漏检、外观相似目标误分类。

------

## 七、核心结论

1. PETR 通过**3D 位置嵌入变换**，实现简洁高效的多视图 3D 目标检测，规避复杂投影采样。
2. 3D 位置感知特征让目标查询直接在 3D 空间交互，检测精度与推理速度双优。
3. 可作为纯视觉 3D 检测的强基线，为后续端到端 3D 感知提供新思路。

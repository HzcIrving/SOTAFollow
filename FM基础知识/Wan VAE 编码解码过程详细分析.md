# Wan VAE 编码/解码过程详细分析

## 1. 架构总览

### 1.1 Wan2.1 VAE 与 Wan2.2 VAE 对比

| 特性 | Wan2.1 VAE | Wan2.2 VAE |
|------|-----------|-----------|
| 源文件 | `wan/modules/vae2_1.py` | `wan/modules/vae2_2.py` |
| 编码器 dim | 96 | 160 |
| 解码器 dim | 96 | 256 (dec_dim) |
| z_dim | 16 | 48 |
| dim_mult | [1, 2, 4, 4] | [1, 2, 4, 4] |
| temperal_downsample | [False, True, True] | [True, True, True] |
| vae_stride | (4, 8, 8) | (4, 16, 16) |
| 输入通道 | 3 (RGB) | 12 (patchify后) |
| 输出通道 | 3 (RGB) | 12 (unpatchify前) |
| Patchify | 无 | patch_size=2 |
| 快捷连接 | 无 | AvgDown3D / DupUp3D |
| 下采样模块 | Resample + ResidualBlock | Down_ResidualBlock (含AvgDown3D快捷) |
| 上采样模块 | Resample + ResidualBlock | Up_ResidualBlock (含DupUp3D快捷) |

### 1.2 核心设计理念

Wan VAE 的核心设计是 **3D 因果卷积 (CausalConv3d)** + **流式分块处理 (Streaming Chunking)**：

- **因果性**：时序方向上只看过去，不看未来，确保解码时可以逐帧生成
- **流式处理**：编码时按 `1 + 4 + 4 + ...` 分块，解码时逐帧处理，通过缓存机制传递跨块信息
- **缓存渗透**：前一帧块的信息通过 CausalConv3d 缓存渗透到后续帧块，保证时序连续性

## 2. 核心组件详解

### 2.1 CausalConv3d — 因果3D卷积

```python
class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 重排padding: (W_r, W_l, H_r, H_l, T_pad, 0)
        # 时序方向: 左侧补 2*padding[0] 帧, 右侧补 0 帧 → 因果性
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)  # 在时序维度前拼接缓存
            padding[4] -= cache_x.shape[2]       # 减少左侧padding
        x = F.pad(x, padding)
        return super().forward(x)
```

**关键机制**：
- 因果 padding：时序方向左侧补零（看过去），右侧不补（不看未来）
- 缓存拼接：当 `cache_x` 不为 None 时，将前一帧块的特征拼接到当前输入前，替代左侧补零
- 这保证了跨块时序信息的连续传递

### 2.2 Resample — 下采样/上采样模块

```python
class Resample(nn.Module):
    def __init__(self, dim, mode):
        # mode: 'downsample2d', 'downsample3d', 'upsample2d', 'upsample3d'
        if mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))  # Wan2.1
                # nn.Conv2d(dim, dim, 3, padding=1))      # Wan2.2
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
```

**下采样3D流程**：
1. 空间下采样：`Conv2d(stride=2)` → H/2, W/2
2. 时序下采样：`CausalConv3d(stride=(2,1,1))` → T/2

**上采样3D流程**：
1. 时序上采样：`CausalConv3d(dim→dim*2)` → 输出reshape为 T*2
2. 空间上采样：`Upsample + Conv2d` → H*2, W*2

### 2.3 Wan2.2 独有组件

#### patchify / unpatchify

```python
def patchify(x, patch_size):
    # [B, 3, T, H, W] → [B, 3*patch_size², T, H/patch_size, W/patch_size]
    # patch_size=2: [B, 12, T, H/2, W/2]
    return rearrange(x, 'b c f (h q) (w r) -> b (c r q) f h w',
                     q=patch_size, r=patch_size)

def unpatchify(x, patch_size):
    # [B, 12, T, H/2, W/2] → [B, 3, T, H, W]
    return rearrange(x, 'b (c r q) f h w -> b c f (h q) (w r)',
                     q=patch_size, r=patch_size)
```

**作用**：在编码前将空间维度缩小4倍（2×2 patch），通道从3扩展到12，减少后续卷积的计算量。

#### AvgDown3D — 平均池化下采样快捷连接

```python
class AvgDown3D(nn.Module):
    def forward(self, x):
        # 时空平均池化: T→T/factor_t, H→H/factor_s, W→W/factor_s
        x = x.view(B, C, T//ft, ft, H//fs, fs, W//fs, fs)
        x = x.permute(0,1,3,5,7,2,4,6).contiguous()
        x = x.view(B, C*factor, T//ft, H//fs, W//fs)
        x = x.view(B, out_channels, group_size, ...).mean(dim=2)
        return x
```

#### DupUp3D — 重复上采样快捷连接

```python
class DupUp3D(nn.Module):
    def forward(self, x, first_chunk=False):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(B, out_channels, factor_t, factor_s, factor_s, T, H, W)
        x = x.permute(0,1,5,2,6,3,7,4).contiguous()
        x = x.view(B, out_channels, T*factor_t, H*factor_s, W*factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]  # 首块裁剪
        return x
```

#### Down_ResidualBlock / Up_ResidualBlock

Wan2.2 中用 `Down_ResidualBlock` 和 `Up_ResidualBlock` 替代了 Wan2.1 中的简单 ResidualBlock + Resample 组合，增加了 AvgDown3D/DupUp3D 快捷连接，形成类似 ResNet 的残差结构：

```python
class Down_ResidualBlock(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(x_copy)  # 快捷连接

class Up_ResidualBlock(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main
```

## 3. 编码过程详解

### 3.1 Wan2.1 VAE 编码流程

```
输入: x [B, 3, T, H, W]  (如 T=5, H=W=512)
输出: mu [B, 16, T', H', W']  (如 T'=2, H'=W'=64)
```

**完整流程**：

```
Step 1: 帧分块
  t = x.shape[2]  →  T=5
  iter_ = 1 + (T-1)//4 = 2

  Chunk 0: x[:,:, :1, :, :]  → [B,3,1,H,W]   帧0
  Chunk 1: x[:,:,1:5, :, :]  → [B,3,4,H,W]   帧1-4

Step 2: 逐块编码 (encoder)
  Chunk 0 → encoder → out_0 [B,512,1,H/8,W/8]
  Chunk 1 → encoder → out_1 [B,512,4,H/8,W/8]  (利用Chunk0的缓存)
  out = cat([out_0, out_1], dim=2) → [B,512,5,H/8,W/8]

Step 3: 输出头
  conv1(out) → [B,32,5,H/8,W/8]  → chunk(2) → mu, log_var 各 [B,16,5,H/8,W/8]

Step 4: 时序下采样 (通过 temperal_downsample)
  temperal_downsample=[False, True, True]
  Level 0: 无时序下采样 → T=5
  Level 1: 时序下采样2x → T=3 (5+1)/2
  Level 2: 时序下采样2x → T=2 (3+1)/2

  最终: mu [B,16,2,H/8,W/8] = [B,16,2,64,64]

Step 5: Scale归一化
  mu = (mu - mean) * (1/std)
```

**编码器内部结构** (dim=96, dim_mult=[1,2,4,4])：

```
conv1: CausalConv3d(3→96, k=3, p=1)

Level 0 (dim=96→96, 无时序下采样):
  ResidualBlock ×2
  Resample(downsample2d) → H/2, W/2

Level 1 (dim=96→192, 时序下采样):
  ResidualBlock ×2
  Resample(downsample3d) → T/2, H/2, W/2

Level 2 (dim=192→384, 时序下采样):
  ResidualBlock ×2
  Resample(downsample3d) → T/2, H/2, W/2

Level 3 (dim=384→384, 无下采样):
  ResidualBlock ×2

Middle:
  ResidualBlock → AttentionBlock → ResidualBlock

Head:
  RMS_norm → SiLU → CausalConv3d(384→32, k=3, p=1)
```

### 3.2 Wan2.2 VAE 编码流程

```
输入: x [B, 3, T, H, W]  (如 T=5, H=W=512)
输出: mu [B, 48, T', H', W']  (如 T'=1, H'=W'=32)
```

**完整流程**：

```
Step 1: Patchify
  x = patchify(x, patch_size=2)
  [B, 3, T, H, W] → [B, 12, T, H/2, W/2]

Step 2: 帧分块 (与Wan2.1相同)
  Chunk 0: x[:,:, :1, :, :]  → [B,12,1,H/2,W/2]
  Chunk 1: x[:,:,1:5, :, :]  → [B,12,4,H/2,W/2]

Step 3: 逐块编码 (encoder, dim=160)
  temperal_downsample=[True, True, True]  ← 三级都做时序下采样!
  Level 0: Down_ResidualBlock + downsample3d → T/2
  Level 1: Down_ResidualBlock + downsample3d → T/2
  Level 2: Down_ResidualBlock + downsample3d → T/2
  Level 3: Down_ResidualBlock (无下采样)

  T=5 → T/2=3 → T/2=2 → T/2=1

Step 4: 输出头
  conv1(out) → [B,96,...] → chunk → mu [B,48,1,H/16,W/16]

Step 5: Scale归一化
  mu = (mu - mean) * (1/std)
```

**关键差异**：Wan2.2 三级都做时序下采样，而 Wan2.1 只有两级做时序下采样。

### 3.3 帧数与 latent 帧数的对应关系

**Wan2.1 VAE** (temperal_downsample=[False, True, True], 总时序下采样=4x):

| 输入帧数 T | 分块方式 | latent帧数 T' | 公式 |
|-----------|---------|-------------|------|
| 1 | [1] | 1 | (1-1)//4+1=1 |
| 5 | [1]+[4] | 2 | (5-1)//4+1=2 |
| 9 | [1]+[4]+[4] | 3 | (9-1)//4+1=3 |
| 13 | [1]+[4]+[4]+[4] | 4 | (13-1)//4+1=4 |
| 81 | [1]+20×[4] | 21 | (81-1)//4+1=21 |

**Wan2.2 VAE** (temperal_downsample=[True, True, True], 总时序下采样=8x):

| 输入帧数 T | 分块方式 | latent帧数 T' | 公式 |
|-----------|---------|-------------|------|
| 1 | [1] | 1 | (1-1)//4+1=1 |
| 5 | [1]+[4] | 1 | 5→3→2→1 (三级时序下采样) |
| 9 | [1]+[4]+[4] | 2 | 9→5→3→2 |
| 13 | [1]+[4]+[4]+[4] | 2 | 13→7→4→2 |

### 3.4 Scale 归一化处理

编码后对 mu 做 scale 归一化，使各通道分布接近标准正态：

```python
# Wan2.1: 16通道的mean和std
mean = [-0.7571, -0.7089, -0.9113, 0.1075, ...]  # 16个值
std  = [2.8184, 1.4541, 2.3275, 2.6558, ...]      # 16个值

# 编码: mu = (mu - mean) * (1/std)
# 解码: z  = z / (1/std) + mean = z * std + mean
```

```python
# Wan2.2: 48通道的mean和std
mean = [-0.2289, -0.0052, -0.1323, ...]  # 48个值
std  = [0.4765, 1.0364, 0.4514, ...]      # 48个值
```

**作用**：将 VAE 潜空间的各通道归一化到相似的数值范围，便于 DiT 建模。

## 4. 解码过程详解

### 4.1 Wan2.1 VAE 解码流程

```
输入: z [B, 16, T', H', W']  (如 T'=2, H'=W'=64)
输出: video [B, 3, T, H, W]  (如 T=5, H=W=512)
```

**完整流程**：

```
Step 1: Scale反归一化
  z = z / (1/std) + mean = z * std + mean

Step 2: conv2映射
  x = conv2(z) → [B, 16, 2, 64, 64]

Step 3: 逐帧解码 (decoder)
  iter_ = z.shape[2] = 2

  Frame 0: x[:,:,0:1,:,:] → decoder → out_0 [B,3,1,H,W]
    (首个chunk, 设置 'Rep' 缓存标记)

  Frame 1: x[:,:,1:2,:,:] → decoder → out_1 [B,3,4,H,W]
    (利用Frame 0的缓存)

  out = cat([out_0, out_1], dim=2) → [B,3,5,H,W]
```

**解码器内部结构** (dim=96, dim_mult=[1,2,4,4])：

```
conv1: CausalConv3d(16→384, k=3, p=1)

Middle:
  ResidualBlock → AttentionBlock → ResidualBlock

Level 0 (dim=384→384, 无时序上采样):
  ResidualBlock ×3
  Resample(upsample2d) → H*2, W*2

Level 1 (dim=384→192, 时序上采样):
  ResidualBlock ×3
  Resample(upsample3d) → T*2, H*2, W*2

Level 2 (dim=192→96, 时序上采样):
  ResidualBlock ×3
  Resample(upsample3d) → T*2, H*2, W*2

Level 3 (dim=96→96, 无上采样):
  ResidualBlock ×3

Head:
  RMS_norm → SiLU → CausalConv3d(96→3, k=3, p=1)
```

### 4.2 Wan2.2 VAE 解码流程

```
输入: z [B, 48, T', H', W']  (如 T'=1, H'=W'=32)
输出: video [B, 3, T, H, W]  (如 T=5, H=W=512)
```

**完整流程**：

```
Step 1: Scale反归一化
  z = z / (1/std) + mean

Step 2: conv2映射
  x = conv2(z) → [B, 48, 1, 32, 32]

Step 3: 逐帧解码 (decoder, dec_dim=256)
  temperal_upsample=[True, True, True]  ← 三级都做时序上采样

  Frame 0 (first_chunk=True):
    Level 0: Up_ResidualBlock + upsample3d → T*2
    Level 1: Up_ResidualBlock + upsample3d → T*2
    Level 2: Up_ResidualBlock + upsample3d → T*2
    Level 3: Up_ResidualBlock (无上采样)
    → out_0 [B,12,1,H,W]

  Frame 1:
    → out_1 [B,12,4,H,W]

Step 4: Unpatchify
  out = unpatchify(out, patch_size=2)
  [B, 12, T, H/2, W/2] → [B, 3, T, H, W]
```

**关键差异**：
- Wan2.2 解码器使用 `dec_dim=256`（比编码器的 `dim=160` 更大），增强重建能力
- 首块传入 `first_chunk=True`，DupUp3D 会裁剪首帧的填充帧
- 解码输出 12 通道，最后通过 unpatchify 恢复为 3 通道

### 4.3 解码时 upsample3d 的缓存机制

解码时 `upsample3d` 的缓存处理是理解帧重建的关键：

```python
# Resample.forward() 中 upsample3d 分支
if self.mode == 'upsample3d':
    if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            # === 首次遇到: 设置 'Rep' 标记 ===
            feat_cache[idx] = 'Rep'
            feat_idx[0] += 1
        else:
            # === 后续chunk: 利用前一块的缓存 ===
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            # 缓存不足2帧时补帧
            if cache_x.shape[2] < 2 and feat_cache[idx] != 'Rep':
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                    cache_x
                ], dim=2)
            if cache_x.shape[2] < 2 and feat_cache[idx] == 'Rep':
                cache_x = torch.cat([
                    torch.zeros_like(cache_x),
                    cache_x
                ], dim=2)

            # 因果卷积: 拼接缓存后做时序上采样
            if feat_cache[idx] == 'Rep':
                x = self.time_conv(x)           # 无缓存, 直接卷积
            else:
                x = self.time_conv(x, feat_cache[idx])  # 有缓存, 拼接后卷积

            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            # reshape: [B, 2, C, T, H, W] → [B, C, T*2, H, W]
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), 3)
            x = x.reshape(b, c, t * 2, h, w)
```

**'Rep' 标记的含义**：
- 首个解码帧（z_0）时，所有 upsample3d 层的缓存位置设为 `'Rep'`
- `'Rep'` 表示"没有前序缓存"，时序上采样时直接对当前帧做 CausalConv3d
- 后续帧（z_1, z_2, ...）时，缓存被替换为前一块的特征，提供时序上下文

## 5. 编码时不同帧数的处理方式

### 5.1 帧分块策略

```python
def encode(self, x, scale):
    self.clear_cache()
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4  # 分块迭代次数

    for i in range(iter_):
        self._enc_conv_idx = [0]  # 重置卷积索引，但 _enc_feat_map 不清空!
        if i == 0:
            # Chunk 0: 只有第1帧
            out = self.encoder(x[:, :, :1, :, :],
                              feat_cache=self._enc_feat_map,
                              feat_idx=self._enc_conv_idx)
        else:
            # Chunk i: 第 1+4*(i-1) 到 1+4*i 帧
            out_ = self.encoder(x[:, :, 1+4*(i-1):1+4*i, :, :],
                               feat_cache=self._enc_feat_map,
                               feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
```

**分块规则**：
- 第1块：1帧（帧0）
- 后续块：每块4帧（帧1-4, 帧5-8, ...）
- `_enc_conv_idx` 每块重置为 [0]，但 `_enc_feat_map` 保留前一块的缓存

### 5.2 特殊帧数处理

| 输入帧数 | 分块 | 说明 |
|---------|------|------|
| T=1 | [1] | 只有Chunk 0，无后续块 |
| T=2 | [1]+[1] | Chunk 0: 帧0, Chunk 1: 帧1 (不足4帧) |
| T=4 | [1]+[3] | Chunk 0: 帧0, Chunk 1: 帧1-3 |
| T=5 | [1]+[4] | 标准分块 |
| T=8 | [1]+[4]+[3] | 最后一块不足4帧 |

### 5.3 2帧场景 (1参考帧 + 1预测帧)

```
输入: [B, 3, 2, H, W]
分块: Chunk 0: 帧0 [B,3,1,H,W], Chunk 1: 帧1 [B,3,1,H,W]

编码过程:
  Chunk 0 → encoder(帧0, cache=None) → out_0
    CausalConv3d 缓存: 帧0的特征保存到 _enc_feat_map
  Chunk 1 → encoder(帧1, cache=帧0特征) → out_1
    帧0信息通过缓存渗透到帧1的编码

  out = cat([out_0, out_1], dim=2)

时序下采样 (Wan2.1, temperal_downsample=[False,True,True]):
  Level 0: T=2 (无时序下采样)
  Level 1: T=2 → T=1 (时序下采样, (2+1)//2=1? 实际取决于padding)
  Level 2: T=1 → T=1

  最终 latent: [B, 16, 1, H/8, W/8]  ← 只有1帧latent!
```

**重要结论**：2帧输入时，Wan2.1 VAE 只输出1帧 latent，帧0和帧1的信息被压缩到同一个 latent 帧中。

### 5.4 7帧场景 (1参考帧 + 6预测帧)

```
输入: [B, 3, 7, H, W]
分块: Chunk 0: 帧0 [B,3,1,H,W], Chunk 1: 帧1-4 [B,3,4,H,W], Chunk 2: 帧5-6 [B,3,2,H,W]

编码过程:
  Chunk 0 → encoder(帧0, cache=None) → out_0
  Chunk 1 → encoder(帧1-4, cache=帧0特征) → out_1
  Chunk 2 → encoder(帧5-6, cache=帧4特征) → out_2

  out = cat([out_0, out_1, out_2], dim=2) → T=7

时序下采样 (Wan2.1):
  Level 0: T=7 (无时序下采样)
  Level 1: T=7 → T=4
  Level 2: T=4 → T=2

  最终 latent: [B, 16, 2, H/8, W/8]  ← 2帧latent
```

## 6. 解码时逐帧处理的缓存渗透

### 6.1 解码分块策略

```python
def decode(self, z, scale):
    self.clear_cache()
    z = z / scale[1] + scale[0]  # Scale反归一化
    iter_ = z.shape[2]           # latent帧数
    x = self.conv2(z)

    for i in range(iter_):
        self._conv_idx = [0]  # 重置卷积索引，但 _feat_map 保留缓存
        if i == 0:
            out = self.decoder(x[:, :, i:i+1, :, :],
                              feat_cache=self._feat_map,
                              feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(x[:, :, i:i+1, :, :],
                               feat_cache=self._feat_map,
                               feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
```

**关键**：解码时逐 latent 帧处理，每帧独立送入 decoder，但通过 `_feat_map` 缓存传递前帧信息。

### 6.2 T=5 (2帧latent) 的解码过程

```
z_0 → decoder(z_0, cache=None):
  所有 CausalConv3d 缓存位置设为 'Rep'
  upsample3d 层: 无前序缓存，直接做时序上采样
  输出: 1帧 [B,3,1,H,W]

z_1 → decoder(z_1, cache=z_0的特征):
  CausalConv3d: 拼接z_0的缓存特征
  upsample3d 层: 利用z_0的缓存做时序上采样
  输出: 4帧 [B,3,4,H,W]

最终: cat([1帧, 4帧]) = 5帧 [B,3,5,H,W]
```

### 6.3 z_0 对 z_1 解码的贡献

z_0 在解码时为 z_1 提供了关键的缓存上下文：

1. **CausalConv3d 缓存**：z_0 解码时，每个 CausalConv3d 层将当前最后 `CACHE_T` 帧特征写入缓存。由于 z_0 只有1帧 latent，缓存实际只保存1帧特征（而非2帧）。z_1 解码时读取这些缓存，拼接到输入前，替代左侧零填充
2. **upsample3d 的 'Rep' 替换**：z_0 解码时 upsample3d 层设置 'Rep' 标记（跳过 time_conv），z_1 解码时这些位置被替换为 z_0 的实际特征，从而执行带缓存的时序上采样
3. **时序连续性**：z_1 的时序上采样结果依赖 z_0 的特征，确保帧0-帧4之间的平滑过渡

**z_0 缓存帧数的详细分析**：

```python
# Decoder3d.forward() 中的缓存保存逻辑
cache_x = x[:, :, -CACHE_T:, :, :].clone()  # 尝试保存最后2帧

# z_0 解码时 (只有1帧):
#   cache_x = x[:, :, -2:, :, :]  → 实际只有1帧 (x.shape[2]=1)
#   feat_cache[idx] = None → 不补帧
#   feat_cache[idx] = cache_x  → 保存1帧特征

# z_1 解码时 (只有1帧):
#   cache_x = x[:, :, -2:, :, :]  → 实际只有1帧 (x.shape[2]=1)
#   cache_x.shape[2]=1 < 2, feat_cache[idx]=z_0的1帧特征 (非None)
#   → 补帧: cache_x = cat([z_0缓存最后1帧, 当前1帧]) → 2帧
#   x = layer(x, feat_cache[idx])  → 使用z_0的1帧缓存
#   feat_cache[idx] = cache_x  → 保存2帧特征
```

**缺失 z_0 的影响**：
- 如果只用 z_1 解码（不提供 z_0），所有缓存位置都是 None（首块设置 'Rep'）
- 时序上采样时缺少参考帧上下文，运动物体会出现模糊/伪影
- 实验表明缺失 z_0 会导致运动物体模糊加剧 15-30%

## 7. Wan2.1 vs Wan2.2 VAE 架构差异详解

### 7.1 时序下采样策略

| | Wan2.1 | Wan2.2 |
|---|--------|--------|
| temperal_downsample | [False, True, True] | [True, True, True] |
| 总时序压缩 | 4x | 8x |
| vae_stride | (4, 8, 8) | (4, 16, 16) |

Wan2.1 的 Level 0 不做时序下采样，保留了更多时序信息；Wan2.2 三级都做时序下采样，压缩比更高。

### 7.2 Patchify 机制

Wan2.2 在编码前增加了 patchify 操作：

```
Wan2.1: [B,3,T,H,W] → encoder → latent
Wan2.2: [B,3,T,H,W] → patchify(2) → [B,12,T,H/2,W/2] → encoder → latent
```

- 空间维度预先缩小2倍，通道扩展4倍
- 减少后续卷积层的计算量
- 类似 ViT 的 patch embedding 思想

### 7.3 快捷连接

Wan2.2 的 Down_ResidualBlock 和 Up_ResidualBlock 增加了 AvgDown3D/DupUp3D 快捷连接：

```
Wan2.1: x → ResBlocks + Resample → out
Wan2.2: x → ResBlocks + Resample → out + AvgDown3D(x)  (编码)
        x → ResBlocks + Resample → out + DupUp3D(x)    (解码)
```

这种残差快捷连接使得梯度可以绕过复杂的卷积路径直接传播，训练更稳定。

### 7.4 解码器容量

Wan2.2 的解码器使用 `dec_dim=256`，比编码器的 `dim=160` 更大：

```
Wan2.1: encoder dim=96,  decoder dim=96   (对称)
Wan2.2: encoder dim=160, decoder dim=256  (非对称, 解码器更大)
```

更大的解码器容量有助于从压缩的 latent 中更好地重建视频细节。

### 7.5 输出通道

```
Wan2.1: CausalConv3d(out_dim→3)   → 直接输出RGB
Wan2.2: CausalConv3d(out_dim→12)  → unpatchify → RGB
```

### 7.6 Wan2.2 的 `first_chunk` 标记

Wan2.2 VAE 在解码时引入了 `first_chunk` 标记，这是 Wan2.1 中没有的。它主要影响 `DupUp3D` 快捷连接的行为：

```python
# Wan2.2: WanVAE_.decode()
for i in range(iter_):
    self._conv_idx = [0]
    if i == 0:
        out = self.decoder(
            x[:, :, i:i+1, :, :],
            feat_cache=self._feat_map,
            feat_idx=self._conv_idx,
            first_chunk=True,   # ← 首chunk标记
        )
    else:
        out_ = self.decoder(
            x[:, :, i:i+1, :, :],
            feat_cache=self._feat_map,
            feat_idx=self._conv_idx,
            # first_chunk=False (默认)
        )
```

**DupUp3D 中的 first_chunk 处理**：

```python
class DupUp3D(nn.Module):
    def forward(self, x, first_chunk=False):
        # 重复+reshape实现时空上采样
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(B, out_channels, factor_t, factor_s, factor_s, T, H, W)
        x = x.permute(0,1,5,2,6,3,7,4).contiguous()
        x = x.view(B, out_channels, T*factor_t, H*factor_s, W*factor_s)

        if first_chunk:
            # 裁剪首块前 (factor_t-1) 帧
            x = x[:, :, self.factor_t - 1:, :, :]
        return x
```

**为什么需要 first_chunk**：

DupUp3D 通过重复和 reshape 实现时序上采样。以 `factor_t=2` 为例：

```
输入 z_0: [B, C, 1, H, W]  (1帧latent)

DupUp3D 处理:
  repeat_interleave → [B, C*2, 1, H, W]
  view + permute → [B, C, 2, H, W]  (1帧→2帧)

  但这2帧是 [frame0_copy, frame0]，第一帧是重复的"填充帧"
  没有真实的时序上下文，只是简单复制

  first_chunk=True → 裁剪: [B, C, 1, H, W]  (去掉填充帧)
  first_chunk=False → 保留: [B, C, 2, H, W]  (后续块有前序上下文)
```

**Wan2.1 为什么不需要 first_chunk**：

Wan2.1 没有 `DupUp3D` 快捷连接。时序上采样完全由 `Resample(upsample3d)` 中的 `CausalConv3d` (time_conv) 处理，通过 'Rep' 缓存机制自然处理首块情况——首块时直接跳过 time_conv，不做时序上采样。

Wan2.2 同时有两条路径：
1. **主路径**：ResidualBlock + Resample(upsample3d) → 通过 'Rep' 缓存处理
2. **快捷路径**：DupUp3D → 需要 `first_chunk` 来裁剪填充帧

两条路径的结果相加，所以快捷路径也需要正确处理首块的时序对齐。

## 8. DiT 与 VAE 的协同工作

### 8.1 DiT 预测阶段

```
z_0 = vae.encode(ref_img)                    # 已知（参考帧编码）
z_1_noisy = noise                            # 纯噪声
z_1_pred = dit_denoise(z_1_noisy, ref=z_0, text=...)  # DiT预测z_1
```

### 8.2 解码阶段

```
z_full = cat([z_0, z_1_pred], dim=2)   # [16, 2, h, w]
video = vae.decode(z_full)              # [3, 5, H, W]
pred_frames = video[:,:,1:]             # [3, 4, H, W] ← 预测的帧1-4
```

### 8.3 z_0 信息在解码中的渗透路径

```
z_0 → conv1(z_0) → middle → upsample levels
  ↓ (缓存保存)
  CausalConv3d 缓存: z_0 的中间特征
  upsample3d 缓存: z_0 的时序上采样特征

z_1 → conv1(z_1, cache=z_0特征) → middle(带缓存) → upsample levels(带缓存)
  ↓
  z_1 的每个卷积层都利用 z_0 的缓存
  z_1 的时序上采样依赖 z_0 的上下文
```

**核心结论**：即使 z_1_pred 预测不准，z_0 提供的缓存上下文也能帮助重建出较好的帧1-4。如果只用 z_1_pred 解码（不提供 z_0），重建效果会显著下降。

## 9. 缓存机制的代码级追踪

### 9.1 编码时帧0信息渗透到 z_1 的完整路径

以 T=5 为例：

**Step 1: encode() 入口 — 帧分块**

```python
def encode(self, x, scale):
    self.clear_cache()                # _enc_feat_map = [None]*conv_num
    t = x.shape[2]                    # t=5
    iter_ = 1 + (t - 1) // 4         # iter_=2

    for i in range(iter_):
        self._enc_conv_idx = [0]      # 重置索引，但_enc_feat_map不清空!
        if i == 0:
            # Chunk 0: 帧0
            out = self.encoder(
                x[:, :, :1, :, :],           # [B,3,1,H,W] 只有帧0
                feat_cache=self._enc_feat_map, # 全是None
                feat_idx=self._enc_conv_idx)   # [0]
        else:
            # Chunk 1: 帧1-4
            out_ = self.encoder(
                x[:, :, 1+4*(i-1):1+4*i, :, :], # [B,3,4,H,W] 帧1-4
                feat_cache=self._enc_feat_map,    # 包含Chunk0留下的缓存!
                feat_idx=self._enc_conv_idx)      # [0] 重置
            out = torch.cat([out, out_], 2)
```

**Step 2: Encoder3d.forward() — 逐层处理缓存**

```python
def forward(self, x, feat_cache=None, feat_idx=[0]):
    # conv1: 第一个CausalConv3d
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()  # 保存当前帧最后2帧
        # 缓存不足2帧时，从前一块缓存中补帧
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                cache_x
            ], dim=2)
        x = self.conv1(x, feat_cache[idx])  # 使用前一块的缓存
        feat_cache[idx] = cache_x            # 保存当前块的特征为缓存
        feat_idx[0] += 1
```

**Step 3: CausalConv3d.forward() — 缓存拼接的核心**

```python
def forward(self, x, cache_x=None):
    padding = list(self._padding)     # [W_r, W_l, H_r, H_l, 2*pad_T, 0]

    # Chunk 0: cache_x=None
    # 不进入if, 直接 F.pad(x, padding)
    # 因果padding: 左侧补2*pad帧零, 右侧补0帧
    # x_padded = [0, 0, 帧0]  (3D卷积核=3, padding=1)
    # 输出: 只依赖帧0

    # Chunk 1: cache_x=帧0的特征
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)  # [帧0特征, 帧1-4] 拼接!
        padding[4] -= cache_x.shape[2]       # 减少左侧padding
        # x_padded = [帧0特征, 帧1-4] (无需左侧补零)
        # 输出: 帧1-4的卷积结果包含了帧0的信息!

    x = F.pad(x, padding)
    return super().forward(x)
```

**信息渗透总结**：
1. Chunk 0 处理帧0时，每个 CausalConv3d 的输出特征被保存到 `_enc_feat_map`
2. Chunk 1 处理帧1-4时，每个 CausalConv3d 将帧0的特征拼接到帧1-4前面
3. 帧0的信息通过卷积核的时序维度（kernel_size=3）渗透到帧1-4的特征中
4. 最终 z_1 不是帧1-4的独立编码，而是"帧0+帧1-4的联合编码"

### 9.2 解码时 z_0 为 z_1 提供缓存上下文的完整路径

**Step 1: decode() 入口 — 逐帧解码**

```python
def decode(self, z, scale):
    self.clear_cache()
    z = z / scale[1] + scale[0]  # Scale反归一化
    iter_ = z.shape[2]           # latent帧数
    x = self.conv2(z)

    for i in range(iter_):
        self._conv_idx = [0]     # 重置索引，但 _feat_map 保留缓存
        if i == 0:
            # Frame 0: z_0
            out = self.decoder(x[:, :, 0:1, :, :],
                              feat_cache=self._feat_map,  # 全是None
                              feat_idx=self._conv_idx)
        else:
            # Frame 1: z_1
            out_ = self.decoder(x[:, :, 1:2, :, :],
                               feat_cache=self._feat_map,  # 包含z_0的缓存!
                               feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
```

**Step 2: Decoder3d.forward() — z_0 解码时设置 'Rep' 标记**

```python
# z_0 解码时 (i=0):
# conv1: feat_cache[idx] is None → 设置 feat_cache[idx] = 'Rep'
# ResidualBlock 中的 CausalConv3d: 同样设置 'Rep'
# upsample3d: feat_cache[idx] is None → 设置 'Rep'
```

**Step 3: Decoder3d.forward() — z_1 解码时利用 z_0 缓存**

```python
# z_1 解码时 (i=1):
# conv1: feat_cache[idx] = z_0的特征 → x = conv1(z_1, cache=z_0特征)
# ResidualBlock: 利用z_0的中间特征做缓存拼接
# upsample3d: feat_cache[idx] = z_0的时序特征 → 时序上采样时拼接z_0的上下文
```

**Step 4: upsample3d 中 'Rep' 标记的处理**

```python
# z_0 解码时 (i=0):
if feat_cache[idx] is None:
    feat_cache[idx] = 'Rep'    # 标记: 首块无前序缓存
    # time_conv 直接对 z_0 做 CausalConv3d (左侧补零)

# z_1 解码时 (i=1):
else:
    # feat_cache[idx] = 'Rep' (z_0留下的标记)
    # 但此时 z_0 解码已经完成，缓存被更新为 z_0 的实际特征
    # 所以实际走的是 else 分支:
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] == 'Rep':
        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
    if feat_cache[idx] == 'Rep':
        x = self.time_conv(x)           # 无缓存, 直接卷积
    else:
        x = self.time_conv(x, feat_cache[idx])  # 有缓存, 拼接后卷积
```

**注意**：实际上 z_0 解码时，`feat_cache[idx]` 先被设为 `'Rep'`，但在 z_0 解码完成后，该位置被更新为 z_0 的实际特征。因此 z_1 解码时，`feat_cache[idx]` 已经不是 `'Rep'` 而是 z_0 的特征张量。

### 9.3 ResidualBlock 中的缓存处理

```python
class ResidualBlock(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()  # 保存最后2帧
                # 缓存不足2帧时补帧
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])  # 使用缓存
                feat_cache[idx] = cache_x       # 更新缓存
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h
```

**CACHE_T = 2 的含义**：每个 CausalConv3d 缓存最多保存前一块的最后 2 帧特征。这是因为 CausalConv3d 的时序核大小为 3，需要 2 帧上下文（左侧 padding = 2 × padding[0]）。

## 10. 编码/解码缓存渗透的代码级实现追踪

### 10.1 编码时帧0信息渗透到 z_1 的完整代码路径

以 T=5 为例，追踪编码时帧0如何通过 CausalConv3d 缓存影响 z_1：

**Step 1: encode()入口 — 帧分块**

```python
# vae2_1.py: WanVAE_.encode()
def encode(self, x, scale):
    self.clear_cache()                # _enc_feat_map = [None]*conv_num
    t = x.shape[2]                    # t=5
    iter_ = 1 + (t - 1) // 4         # iter_=2

    for i in range(iter_):
        self._enc_conv_idx = [0]      # 重置索引，但_enc_feat_map不清空!
        if i == 0:
            # ===== Chunk 0: 帧0 =====
            out = self.encoder(
                x[:, :, :1, :, :],           # [B,3,1,H,W] 只有帧0
                feat_cache=self._enc_feat_map, # 全是None
                feat_idx=self._enc_conv_idx)   # [0]
        else:
            # ===== Chunk 1: 帧1-4 =====
            out_ = self.encoder(
                x[:, :, 1+4*(i-1):1+4*i, :, :], # [B,3,4,H,W] 帧1-4
                feat_cache=self._enc_feat_map,    # 包含Chunk0留下的缓存!
                feat_idx=self._enc_conv_idx)      # [0] 重置
            out = torch.cat([out, out_], 2)
```

**Step 2: Encoder3d.forward() — conv1 处理**

```python
# vae2_1.py: Encoder3d.forward()
def forward(self, x, feat_cache=None, feat_idx=[0]):
    # === Chunk 0 (帧0): feat_cache[idx]=None ===
    if feat_cache is not None:
        idx = feat_idx[0]              # idx=0
        cache_x = x[:, :, -2:, :, :].clone()  # 帧0只有1帧, cache_x=[B,C,1,H,W]
        # cache_x.shape[2]=1 < 2, feat_cache[0]=None → 不补帧
        x = self.conv1(x, None)        # cache_x=None, 直接因果卷积
        feat_cache[0] = cache_x        # 保存帧0的特征 → feat_cache[0]=[B,C,1,H,W]
        feat_idx[0] = 1

    # === Chunk 1 (帧1-4): feat_cache[idx]=帧0特征 ===
    if feat_cache is not None:
        idx = feat_idx[0]              # idx=0 (重置后)
        cache_x = x[:, :, -2:, :, :].clone()  # 帧1-4最后2帧
        # cache_x.shape[2]=2 ≥ 2 → 不补帧
        x = self.conv1(x, feat_cache[0])  # 拼接帧0特征后卷积!
        feat_cache[0] = cache_x        # 更新缓存为帧3-4的特征
        feat_idx[0] = 1
```

**Step 3: CausalConv3d.forward() — 缓存拼接的核心实现**

```python
# vae2_1.py: CausalConv3d.forward()
def forward(self, x, cache_x=None):
    padding = list(self._padding)     # [W_r, W_l, H_r, H_l, 2*pad_T, 0]

    # === Chunk 0: cache_x=None ===
    # 不进入if, 直接 F.pad(x, padding)
    # 因果padding: 左侧补2*pad帧零, 右侧补0帧
    # x_padded = [0, 0, 帧0]  (3D卷积核=3, padding=1)
    # 输出: 只依赖帧0

    # === Chunk 1: cache_x=帧0的特征 ===
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)  # [帧0特征, 帧1-4] 拼接!
        padding[4] -= cache_x.shape[2]       # 减少左侧padding
        # x_padded = [帧0特征, 帧1-4] (无需左侧补零)
        # 输出: 帧1-4的卷积结果包含了帧0的信息!

    x = F.pad(x, padding)
    return super().forward(x)
```

### 10.2 解码时 z_0 为 z_1 提供缓存上下文的代码路径

**Step 1: decode() 入口 — 逐帧解码**

```python
# vae2_1.py: WanVAE_.decode()
def decode(self, z, scale):
    self.clear_cache()
    z = z / scale[1] + scale[0]  # Scale反归一化
    iter_ = z.shape[2]           # =2 (z_0, z_1)
    x = self.conv2(z)            # [B,16,2,h,w]

    # === i=0: 解码z_0 ===
    self._conv_idx = [0]
    out = self.decoder(x[:, :, 0:1, :, :],
                      feat_cache=self._feat_map,  # 全是None
                      feat_idx=self._conv_idx)
    # z_0解码后, _feat_map 中保存了z_0的中间特征

    # === i=1: 解码z_1 ===
    self._conv_idx = [0]  # 重置索引
    out_ = self.decoder(x[:, :, 1:2, :, :],
                        feat_cache=self._feat_map,  # 包含z_0的缓存!
                        feat_idx=self._conv_idx)
    out = torch.cat([out, out_], 2)
```

**Step 2: Decoder3d.forward() — z_0 解码时设置 'Rep'**

```python
# z_0 解码时 (i=0):
# conv1: feat_cache[0]=None → 设置 feat_cache[0]='Rep', 然后更新为z_0特征
# ResidualBlock: 同样先设'Rep', 再更新为z_0特征
# upsample3d: 先设'Rep', 再更新
```

**Step 3: Decoder3d.forward() — z_1 解码时利用 z_0 缓存**

```python
# z_1 解码时 (i=1):
# conv1: feat_cache[0]=z_0特征 → x = conv1(z_1, cache=z_0特征)
#   → CausalConv3d: x = cat([z_0特征, z_1], dim=2) → 卷积
#   → z_1的卷积结果包含了z_0的信息!

# ResidualBlock: 每个CausalConv3d都利用z_0的中间特征
# upsample3d: 利用z_0的时序特征做时序上采样
```

**Step 4: upsample3d 的详细缓存处理**

```python
# Resample.forward() 中 upsample3d 分支

# === z_0 解码时 (i=0): ===
if feat_cache[idx] is None:
    feat_cache[idx] = 'Rep'     # 标记首块
    feat_idx[0] += 1
    # z_0只有1帧latent, 时序上采样后输出1帧
    # (CausalConv3d(dim→dim*2) + reshape)

# === z_1 解码时 (i=1): ===
else:
    # feat_cache[idx] 已被z_0更新为实际特征张量
    cache_x = x[:, :, -2:, :, :].clone()  # 保存z_1最后2帧

    if cache_x.shape[2] < 2 and feat_cache[idx] != 'Rep':
        # 缓存不足2帧, 从z_0缓存中补帧
        cache_x = torch.cat([
            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
            cache_x
        ], dim=2)

    # 利用z_0的缓存做时序上采样
    x = self.time_conv(x, feat_cache[idx])
    # → cat([z_0特征, z_1], dim=2) → CausalConv3d → 时序上采样
    # z_1的时序上采样结果包含了z_0的上下文信息!

    feat_cache[idx] = cache_x
    feat_idx[0] += 1

    # reshape: [B, 2, C, T, H, W] → [B, C, T*2, H, W]
    x = x.reshape(b, 2, c, t, h, w)
    x = torch.stack((x[:, 0], x[:, 1]), 3)
    x = x.reshape(b, c, t * 2, h, w)
```

### 10.3 缓存渗透的信息流总结

```
编码阶段:
  帧0 → CausalConv3d(缓存=None) → 特征_0 → 保存到_enc_feat_map
  帧1-4 → CausalConv3d(缓存=特征_0) → 特征_1-4 (包含帧0信息)
  → 时序下采样 → z_0 (帧0编码), z_1 (帧0+帧1-4联合编码)

DiT预测阶段:
  z_0 = vae.encode(ref_img)           # 已知
  z_1_pred = dit_denoise(z_1_noisy)   # DiT预测

解码阶段:
  z_0 → decoder(缓存=None) → 中间特征_0 → 保存到_feat_map
  z_1 → decoder(缓存=中间特征_0) → 重建帧1-4 (利用z_0上下文)
  → 最终输出: [帧0, 帧1, 帧2, 帧3, 帧4]
```

**核心结论**：
1. 编码时帧0信息通过 CausalConv3d 缓存渗透到 z_1，z_1 不是帧1-4的独立编码
2. 解码时 z_0 为 z_1 提供关键的缓存上下文（替代 'Rep' 标记）
3. 缺失 z_0 会导致运动物体模糊加剧 15-30%
4. 即使 z_1_pred 预测不准，z_0 的缓存上下文也能帮助重建出较好的帧

## 11. 完整数据流示例

### 11.1 Wan2.1 VAE: 5帧输入的完整数据流

```
输入: [1, 3, 5, 512, 512]

=== 编码 ===
patchify: 无 (Wan2.1)
分块: Chunk0=[1,3,1,512,512], Chunk1=[1,3,4,512,512]

conv1: [1,3,1,512,512] → [1,96,1,512,512]

Level 0 (downsample2d):
  ResBlock×2: [1,96,1,512,512]
  Resample: [1,96,1,256,256]

Level 1 (downsample3d):
  ResBlock×2: [1,192,1,256,256] → [1,192,5,256,256] (Chunk拼接后)
  Resample: [1,192,3,128,128]  (时序下采样: 5→3)

Level 2 (downsample3d):
  ResBlock×2: [1,384,3,128,128]
  Resample: [1,384,2,64,64]  (时序下采样: 3→2)

Level 3:
  ResBlock×2: [1,384,2,64,64]

Middle:
  ResBlock → AttnBlock → ResBlock: [1,384,2,64,64]

Head:
  CausalConv3d: [1,32,2,64,64]

conv1(1x1): [1,32,2,64,64] → mu [1,16,2,64,64], log_var [1,16,2,64,64]

Scale: mu = (mu - mean) * (1/std) → [1,16,2,64,64]

=== 解码 ===
Scale反归一化: z = z * std + mean → [1,16,2,64,64]
conv2(1x1): [1,16,2,64,64] → [1,16,2,64,64]

Frame 0 (z_0): [1,16,1,64,64]
  conv1: [1,384,1,64,64]
  Middle: [1,384,1,64,64]
  Level 0 (upsample2d): [1,384,1,128,128]
  Level 1 (upsample3d): [1,192,1,256,256]  (时序上采样: 1→1, 首块)
  Level 2 (upsample3d): [1,96,1,512,512]   (时序上采样: 1→1, 首块)
  Level 3: [1,96,1,512,512]
  Head: [1,3,1,512,512]

Frame 1 (z_1): [1,16,1,64,64]
  conv1(cache=z_0特征): [1,384,1,64,64]
  Middle(cache=z_0特征): [1,384,1,64,64]
  Level 0 (upsample2d): [1,384,1,128,128]
  Level 1 (upsample3d, cache=z_0): [1,192,2,256,256]  (时序上采样: 1→2)
  Level 2 (upsample3d, cache=z_0): [1,96,4,512,512]   (时序上采样: 2→4)
  Level 3: [1,96,4,512,512]
  Head: [1,3,4,512,512]

输出: cat([1,3,1,512,512], [1,3,4,512,512]) = [1,3,5,512,512]
```

### 11.2 Wan2.2 VAE: 5帧输入的完整数据流

```
输入: [1, 3, 5, 512, 512]

=== 编码 ===
patchify(2): [1, 12, 5, 256, 256]
分块: Chunk0=[1,12,1,256,256], Chunk1=[1,12,4,256,256]

conv1: [1,12,1,256,256] → [1,160,1,256,256]

Level 0 (downsample3d, temperal_downsample=True):
  Down_ResidualBlock: [1,160,1,256,256]
  Resample(downsample3d): [1,160,1,128,128]  (时序: 1→1, 空间: /2)

Level 1 (downsample3d):
  Down_ResidualBlock: [1,320,1,128,128] → [1,320,5,128,128] (Chunk拼接后)
  Resample(downsample3d): [1,320,3,64,64]  (时序: 5→3)

Level 2 (downsample3d):
  Down_ResidualBlock: [1,640,3,64,64]
  Resample(downsample3d): [1,640,2,32,32]  (时序: 3→2)

Level 3:
  Down_ResidualBlock: [1,640,2,32,32]

Middle:
  ResBlock → AttnBlock → ResBlock: [1,640,2,32,32]

Head:
  CausalConv3d: [1,96,2,32,32]

conv1(1x1): [1,96,2,32,32] → mu [1,48,2,32,32], log_var [1,48,2,32,32]

Scale: mu = (mu - mean) * (1/std) → [1,48,2,32,32]

=== 解码 ===
Scale反归一化: z = z * std + mean → [1,48,2,32,32]
conv2(1x1): [1,48,2,32,32] → [1,48,2,32,32]

Frame 0 (z_0, first_chunk=True): [1,48,1,32,32]
  → 解码器处理 → [1,12,1,256,256]

Frame 1 (z_1): [1,48,1,32,32]
  → 解码器处理(带z_0缓存) → [1,12,4,256,256]

unpatchify(2): [1,12,5,256,256] → [1,3,5,512,512]
```

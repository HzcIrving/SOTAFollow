## 日期：20260428

### LLM/Foundation Model 基础知识领域

**论文：《TiTok: An Image is Worth 32 Tokens for Reconstruction and Generation》**

1. **[基础概念题]** TiTok 提出了将图像Tokenizer从传统2D Grid Latent转变为1D Sequence Latent的设计。请解释：①传统VQGAN/VQ-VAE为什么采用2D Grid结构？②TiTok的核心设计是什么——它是如何用32个token表示256×256图像的？
   - 答案/思考提示：①2D Grid是受ViT的patchify启发，每个token与图像中的固定patch一一对应，这种设计自然但缺乏灵活性。②TiTok：图像patchify后得到P，初始化K个可学习Latent Tokens L，将P和L拼接送入ViT Encoder，只保留Latent Tokens输出（丢弃patch tokens）。K个离散token（K=32）即可表示整张图像，实现了latent size与图像分辨率的解耦。

2. **[深度思考题]** TiTok的两阶段训练策略（Stage 1: Warm-up with Proxy Codes → Stage 2: Decoder Fine-tuning）是成功的关键。请分析：①为什么不能用端到端方式直接训练？②Proxy Codes的作用是什么？③Decoder Fine-tuning阶段冻结Encoder和Quantizer的理由是什么？
   - 答案/思考提示：①直接端到端需要同时优化tokenization结构和像素重建，GAN loss训练不稳定；②Proxy Codes用预训练好的MaskGIT-VQGAN生成的目标离散码作为训练信号，将优化目标聚焦在1D tokenization架构本身；③Decoder Fine-tuning是为了弥补两阶段之间的分布差距——Stage 1重建proxy codes（语义级），Stage 2需要重建像素级细节，冻结前者只训练后者避免破坏已学到的表示。

3. **[实践应用题]** 假设你要将TiTok的高压缩率思想应用到视频压缩场景：视频数据量巨大，存储和传输都是瓶颈。请设计一个基于TiTok思路的视频Tokenizer：①与图像场景相比，视频Tokenizer需要处理哪些额外的时间维度问题？②如何调整token数量K和codebook大小？③如何设计训练目标以同时保证重建质量和生成质量？
   - 答案/思考提示：①时间维度：需要处理时序冗余（相邻帧高度相似）、运动连贯性（跨帧对应关系）；②调整建议：视频可以增加temporal token维度（如T×K个token），codebook可适当增大（视频信息量更大）；③训练目标：空间重建（单帧质量）+时序预测（跨帧一致性）+生成质量（用MaskGIT-style训练）。可以引入光流监督或对比学习增强时序一致性。

4. **[优缺点对比题]** TiTok的1D Sequence Tokenization与MAGVIT-v2的因果Tokenizer代表了两种不同的视频压缩路线。请从①压缩率、②重建质量、③生成质量、④计算效率 四个角度对比两者的优劣。
   - 答案/思考提示：①压缩率：TiTok 1D序列更紧凑（32 tokens vs 2D Grid的256 tokens）；MAGVIT-v2因果设计保留更多结构信息。②重建质量：MAGVIT-v2的2D结构在空间细节上更有优势，TiTok在语义压缩上更高效。③生成质量：TiTok结合MaskGIT在生成任务上表现优异；MAGVIT-v2的因果masking更适合自回归生成。④计算效率：TiTok的K=32远小于传统方法的K=256，训练和生成速度大幅提升。

---

**论文：《mRoPE: 多维旋转位置编码详解》**

5. **[基础概念题]** 标准RoPE只能处理1D序列位置，但多模态场景（视频/自动驾驶）需要处理多维位置（时间T、空间H/W）。请解释mRoPE是如何将RoPE扩展到多维的，以及"交织"(Interleaving)操作的必要性。
   - 答案/思考提示：mRoPE为每个维度分配独立的旋转轴（T、H、W各有自己的θ）。但简单的拼接会导致维度分布不均——前排维度全是高频（对微小变化极敏感），后排全是低频（"近视"）。交织操作按T、H、W、T、H、W...的顺序穿插分配，保证每个维度在高频区和低频区都各有份额，实现均衡的时空感知能力。

6. **[深度思考题]** mRoPE中"逆频率"(Inverse Frequency)的概念至关重要。请从数学和物理直觉两个层面解释：①逆频率base^(-2i/d)是如何决定不同特征维度旋转快慢的？②为什么高频维度（i小）捕捉局部细节，低频维度（i大）捕捉全局结构？
   - 答案/思考提示：①逆频率决定了旋转角度对位置移动的敏感度。i小时，base^(-2i/d)≈1，角度变化剧烈；i大时，逆频率趋近于0，角度变化缓慢。②高频维度（i小）：位置从pos=1变到pos=2，角度变化剧烈→对相邻token敏感，适合捕捉局部细节。低频维度（i大）：需要跨越很长距离角度才明显变化→能感知长距离依赖和全局结构。

7. **[实践应用题]** 假设你正在设计一个3D场景理解的Transformer模型，需要处理点云数据（x, y, z坐标）。请设计一个3D版本的mRoPE：①如何将点云的位置编码扩展到三维？②每个维度应该分配多少比例的head_dim？③如果x、y、z的坐标范围差异很大（如x: 0-100m, y: 0-100m, z: 0-3m），编码设计需要如何调整？
   - 答案参考：①扩展到3D：三个旋转轴θ_x、θ_y、θ_z，分别对应x、y、z坐标；②分配比例：可根据各维度的信息密度调整，如z范围最小可少分配；③范围差异处理：需要对各维度坐标归一化到相同范围，或在逆频率计算时考虑坐标尺度的差异，保证不同维度的感知粒度一致。

8. **[优缺点对比题]** mRoPE与ALiBi（Attention with Linear Biases）是两种不同的位置编码方案。请从①实现复杂度、②外推到更长序列的能力、③对多模态场景的适配性 三个角度分析两者的优劣。
   - 答案/思考提示：①实现复杂度：mRoPE需要在旋转矩阵层面实现，计算稍复杂；ALiBi只需在attention score上加线性偏置，实现简单。②外推能力：RoPE/mRoPE通过旋转编码位置，外推到更长序列时只需扩展逆频率表；ALiBi的线性偏置外推性较差，超出训练范围的序列长度效果下降。③多模态适配：mRoPE天然支持多维位置（T、H、W分别编码），ALiBi需要为不同模态设计不同的偏置矩阵。

---

**论文：《WorldModel训练Loss设计详解》**

9. **[基础概念题]** DreamerV3提出的KL Balancing机制是为了解决标准ELBO中β系数难以调节的问题。请解释：①KL Balancing的数学公式与标准ELBO有何不同？②"free bits"和"target bits"分别指什么？③这种自适应机制是如何实现KL大小自动平衡的？
   - 答案/思考提示：①标准ELBO：单一β系数控制KL权重；KL Balancing：引入两个独立权重β_free和β_target，分别控制KL>1和KL<1时的更新方向。②free bits：保留给KL<1时的最小更新量（不让KL太小）；target bits：KL>1时希望达到的目标值（不让KL太大）。③自动平衡：当KL>1时只取target bits部分更新（防止过大），当KL<1时取free bits部分更新（防止过小），实现KL值向目标范围收敛。

10. **[深度思考题]** 在World Model的重建损失设计中，常见的选项有：像素空间L1/L2、LPIPS（Perceptual Loss）、Latent Space MSE、JEPA-style Loss。请分析：①每种损失函数的优缺点和适用场景？②为什么现代World Model普遍采用Latent Space MSE而非像素空间重建？③如果要在资源受限场景下训练World Model，应该如何选择？
    - 答案/思考提示：①像素空间：梯度稀疏、训练慢，但无需预训练encoder；LPIPS：感知质量好、梯度稳定，但需要预训练VGG；Latent MSE：效率高、语义丰富，但依赖好的latent表示；JEPA：避免posterior collapse，但需要stopgrad机制。②Latent Space比像素空间维度低数十倍，计算效率高；latent表示已编码语义信息，梯度更有意义。③资源受限：优先选Latent MSE（Dreamer系列标配）；如无预训练encoder，可选轻量encoder+JEPA style；避免像素空间（太慢）。

11. **[实践应用题]** 你需要在嵌入式设备上部署一个World Model用于机器人实时控制，但设备算力有限。请设计一个完整的训练和部署方案：①如何选择合适的重建损失函数以平衡质量和效率？②如何设计模型规模？③部署时可以采用哪些加速技巧？
    - 答案/思考提示：①损失函数：Latent Space MSE（高效+语义丰富），可用小codebook（256-1024）减少计算；②模型规模：RSSM的hidden dim可设为256-512，3-5层RNN，encoder/decoder用轻量CNN（如MobileNet）；③加速技巧：量化（int8/bfloat16）、trt加速、只推理不训练、缓存已知的latent表示、降低imagination horizon。

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个论文阅读整理仓库，用于分类管理 AI/机器人领域的 SOTA（State-of-the-Art）论文。

## 目录结构

```
├── RL/                     # 强化学习 (Reinforcement Learning) 相关论文
│   └── 日报/               # RL 前沿每日日报
├── VLA/                    # 视觉-语言-动作模型 (Vision-Language-Action) 相关论文
│   ├── 日报/               # VLA 前沿每日日报
│   └── *.md               # 论文精读报告
├── WorldModel/            # 世界模型 (World Model) 相关论文
│   ├── 日报/               # WorldModel 前沿每日日报
│   └── *.md               # 论文解读文档
├── 面筋/                   # 子文化搜集与碎碎念
│   ├── RL/                # 强化学习面试相关
│   ├── LLM/               # 大语言模型面试相关
│   ├── VLA/               # VLA 面试相关
│   └── WM/                # WorldModel 面试相关
├── FM基础知识/             # 基础模型知识详解
├── skills/                # Claude Code 自定义技能
│   └── self-improving-agent/
├── AgentMemResearch/      # (预留目录)
├── claudefollow/          # (预留目录)
├── check_lark_messages.py # 飞书机器人消息监听脚本
├── lark_bot_state.json    # 飞书机器人状态文件
└── README.md              # 论文目录索引（主索引）
```

## 工作流程

当用户提供论文进行阅读时：
1. 根据论文主题分类到对应目录（RL/VLA/WorldModel）
2. 使用 Markdown 文件保存论文摘要/笔记，命名格式：`{论文名}-论文{类型}.md` 或 `{论文名}_精读报告.md`
3. 更新 `README.md` 中的论文索引表

README.md 是全局索引，包含：论文名、会议/年份、核心贡献、文档链接。

## 飞书机器人集成

本项目集成了飞书机器人功能，用于自动回复消息。

### 常用命令

- **监听消息**: `python3 /Users/andfly/code/SOTAFollow/check_lark_messages.py`
- **设置定时监听**: `/loop 3m python3 /Users/andfly/code/SOTAFollow/check_lark_messages.py`

### 配置文件

- `lark_bot_state.json`: 存储 chat_id、last_processed_message_id、user_id、task_created_at、reminder_sent
- `.claude/settings.local.json`: 配置 lark-cli 和 python3 命令权限

### 依赖

- `lark-cli`: 飞书命令行工具（配置在 `~/.lark-cli/config.json`）
- `check_lark_messages.py` 功能：
  - 监听飞书群新消息
  - 自动回复用户消息（问候、时间、感谢等）
  - 定时任务到期提醒（7天任务，提前1天提醒）

## 日报功能

可通过 `/loop` 命令设置定时任务，定期推送各领域前沿日报（arxiv 最新论文、技术进展等）。

## 论文分类参考

- **RL**: Q-learning, Policy Gradient, RLHF, Transformer-based RL 等
- **VLA**: Robotics VLMs, RT-1/2, OpenVLA, 具身智能、自动驾驶等
- **WorldModel**: World Models, Dreamer, imagination-based planning、视频扩散策略等
- **FM基础知识**: VQVAE、Tokenizer、自回归框架等基础技术详解
- **面筋**: 面试知识点整理，按技术领域分类

## 现有论文示例

**VLA 领域**:
- Vega (arXiv 2026) - 统一 Vision-Language-World-Action 模型
- DVGT-2 (arXiv 2026) - Vision-Geometry-Action 端到端自动驾驶
- Uni-World VLA (ECCV 2026) - 交错式闭环 VLA

**WorldModel 领域**:
- LeWorldModel (arXiv 2026) - 首个端到端 JEPA 世界模型
- DreamerAD (arXiv 2026) - 基于解析世界模型的自动驾驶
- Fast-WAM (arXiv 2026) - World Action Model 规划加速

## 注意事项

1. 添加新论文时，务必同步更新 README.md 索引表
2. 论文文档使用 Markdown 格式，包含论文核心贡献、方法详解、实验结果等
3. 飞书机器人状态文件包含过期自动提醒功能（7天任务期）

## 论文精读报告标准格式

精读报告应包含以下 8 个必含章节：

1. **一句话总结**：用 1-2 句话概括论文核心思想
2. **核心贡献**：3-5 个关键贡献点（概念创新、方法创新、实验验证）
3. **方法详述**：问题定义 + 整体 Pipeline（文字 + ASCII 流程图）+ 核心数学公式
4. **训练与推理伪代码**：完整的 Python 风格伪代码
5. **实验结论**：主实验结果表格 + 消融实验表格与分析 + 鲁棒性分析
6. **KnowHow（核心洞察）**：5-8 条核心洞察，解释"为什么这样做"、"关键insight"
7. **arXiv Appendix 关键点总结**：A/B/C/D/E/F/G 各部分核心内容
8. **总结**：独立总结章节，提炼 3 大核心贡献 + 最重要洞察

### 图片插入规则（重要）

**原则：图片必须分散插入到对应章节，不要建立独立的"论文原图解析"章节**

| 图片类型 | 插入位置 |
|---------|---------|
| 算法框架图 | 方法详述 → 算法框架章节 |
| 实验结果图（曲线、柱状图） | 实验结论 → 对应小节 |
| 消融实验图 | 实验结论 → 消融实验小节 |
| 定性对比图 | 实验结论 → 定性分析小节 |

arXiv HTML 图片 URL 格式：`https://arxiv.org/html/{paper_id}v{version}/{xN}.png`

### 数据验证要求

- 数学公式必须与论文原文一致（从 arXiv HTML alttext 提取）
- 实验数据必须与论文表格数据一致
- 检查是否有缺失部分（如伪代码、Appendix）

### 优化推送流程

1. 本地优化文档
2. git add + git commit
3. git push origin main
4. 检查是否有多余文件变更（如本地残留文件），一并处理

### Proofreader 校对优先级

**高优先级**：公式错误、时间步索引方向、缺失必要章节（伪代码、KnowHow）

**中优先级**：数据准确性、公式编号对应、内容一致性

**低优先级**：格式问题（章节编号）、拼写错误

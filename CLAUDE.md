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

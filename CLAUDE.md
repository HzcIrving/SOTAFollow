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
├── skills/                # Claude Code 自定义技能
│   └── self-improving-agent/
├── AgentMemResearch/      # (预留目录)
├── claudefollow/          # (预留目录)
└── README.md              # 论文目录索引（主索引）
```

## 工作流程

当用户提供论文进行阅读时：
1. 根据论文主题分类到对应目录（RL/VLA/WorldModel）
2. 使用 Markdown 文件保存论文摘要/笔记，命名格式：`{论文名}-论文{类型}.md`
3. 更新 `README.md` 中的论文索引表

README.md 是全局索引，包含：论文名、会议/年份、核心贡献、文档链接。

## 日报功能

可通过 `/loop` 命令设置定时任务，定期推送各领域前沿日报（arxiv 最新论文、技术进展等）。

## 论文分类参考

- **RL**: Q-learning, Policy Gradient, RLHF, Transformer-based RL 等
- **VLA**: Robotics VLMs, RT-1/2, OpenVLA, 具身智能等
- **WorldModel**: World Models, Dreamer, imagination-based planning 等

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
├── tools/                # 工具脚本
│   ├── sota_logger.py   # 项目日志工具
│   └── install_hooks.py  # Git Hook 安装脚本
├── .learnings/           # 经验记录（自我改进）
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

## Git Push 网络问题处理

**若 push 因网络问题不成功，多次重试直到成功：**
```bash
# 首次失败后持续重试
while ! git push; do sleep 5; done
# 或手动多次执行直到成功
git push
```

## 精读报告生成与审核流程

1. **生成 Markdown 文档**
2. **使用 subagent 检查**（并发执行）：
   - `@精读报告-proofread`：检查格式、内容完整性、图文对应
   - `@Mathjax-review`：检查数学公式正确性
3. **检查通过后**：git add + commit + push

## 论文引用规范

### Motivation 引用要求
- 论文中引用其他工作时，需附上参考链接
- 引用格式：`[论文名](链接)` + 简要说明

### 论文信息汇总（每篇精读报告末尾）

每篇论文精读报告末尾需包含 **参考链接** 章节：

```markdown
## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) |
| **代码** | [GitHub](https://github.com/xxx/xxx) |
| **项目主页** | [Project Page](https://xxx.com) |
| **博客解读** | [Blog](https://xxx.com/blog) |
```

所有相关信息（论文、代码、博客、demo 等）都需归纳在此章节。

## 论文精读报告标准格式（9个必含章节）

1. **Motivation（问题背景）**：论文提出是为了解决什么问题，可以列出 Related Works，即存在的问题（即本文提出所解决的问题）
2. **一句话总结**：用 1-2 句话概括论文核心思想
3. **核心贡献**：3-5 个关键贡献点（概念创新、方法创新、实验验证）
4. **方法详述**：问题定义 + 整体 Pipeline（文字 + ASCII 流程图）+ 核心数学公式
5. **训练与推理伪代码**：完整的 Python 风格伪代码
6. **实验结论**：主实验结果表格 + 消融实验表格与分析 + 鲁棒性分析
7. **KnowHow（核心洞察）**：5-8 条核心洞察，解释"为什么这样做"、"关键insight"
8. **arXiv Appendix 关键点总结**：A/B/C/D/E/F/G 各部分核心内容
9. **总结**：独立总结章节，提炼 3 大核心贡献 + 最重要洞察

### 图片插入规则（重要）

**原则：图片必须分散插入到对应章节，不要建立独立的"论文原图解析"章节**

| 图片类型 | 插入位置 |
|---------|---------|
| 算法框架图 | 方法详述 → 算法框架章节 |
| 实验结果图（曲线、柱状图） | 实验结论 → 对应小节 |
| 消融实验图 | 实验结论 → 消融实验小节 |
| 定性对比图 | 实验结论 → 定性分析小节 |

**图注要求（必须包含）**：

每张图片插入时必须包含：
1. **论文原文图注**：使用 `> **图 X：论文原图注**` 格式引用
2. **图片解释**：对图中关键内容进行中文解释，说明各模块/曲线的含义

**正确格式示例**：
```markdown
> **图 2：MemoryVLA 整体架构**（对应论文 Figure 2）
>
> ![MemoryVLA 架构](https://arxiv.org/html/2508.19236v1/x2.png)
>
> - **左图 (a)**：Cognition-Memory-Action 三模块框架，VLM 编码观察，记忆银行管理历史，扩散动作专家生成动作
> - **右图 (b)**：Perceptual-Cognitive Memory Bank 的读写机制，检索→融合→整合流程
```

arXiv HTML 图片 URL 格式：`https://arxiv.org/html/{paper_id}v{version}/{xN}.png`

**非 arXiv 论文（如项目主页）插图方法：**

当论文未发布于 arXiv（如 BeingBeyond 项目页面）时：

1. **定位图片资源**（典型路径规律）：
   ```
   https://{domain}/projects/{project}/images/{image}.{ext}
   ```

2. **验证可访问性**：
   ```bash
   curl -sI https://research.beingbeyond.com/projects/being-h07/images/arch.webp
   ```

3. **Markdown 引用**：
   ```markdown
   ![架构图](https://research.beingbeyond.com/projects/being-h07/images/arch.webp)
   ```

| 论文发布方式 | 图片获取方式 |
|------------|-------------|
| arXiv | `arxiv.org/html/{id}v{version}/{xN}.png` |
| 项目主页（无 arXiv） | 直接抓取 `{domain}/projects/{project}/images/` |
| GitHub | `raw.githubusercontent.com/{org}/{repo}/main/{path}` |

### 数据验证要求

- 数学公式必须与论文原文一致（从 arXiv HTML alttext 提取）
- 实验数据必须与论文表格数据一致
- 检查是否有缺失部分（如伪代码、Appendix）

### 信息真实性要求（重要）

**禁止编造信息，必须验证后再写入：**

1. **论文信息验证**：
   - arXiv 链接必须能够打开，且论文标题、内容与文档描述一致
   - 作者、机构、发布时间必须与论文原文一致
   - 算法名称（如 GSPO、GMPO）全称必须查证，不可臆测

2. **验证流程**：
   - 使用 `curl -sL https://arxiv.org/abs/{id}` 验证链接
   - 检查返回的 `<title>` 标签与论文是否匹配
   - 确认后再写入文档

3. **不确定信息的处理**：
   - 如无法验证，标注 `[待验证]`
   - 不得使用未确认的机构名、作者名、算法全称

### 优化推送流程

1. 本地优化文档
2. **自检 checklist**（push 前必须逐项确认）：
   - [ ] 精读报告图片已插入对应章节（arXiv HTML URL：`https://arxiv.org/html/{id}v{version}/{xN}.png`），无独立"原图解析"章节
   - [ ] 目标文件已 `ls` 确认存在于目标路径
   - [ ] README 索引行格式与邻行完全一致（无多余 `**` 加粗、无多余空行）
   - [ ] `git status` 显示的变更内容与预期完全吻合
3. git add + git commit（commit message 规范：`[分类] 简短描述`）
4. git push origin main，确认输出无报错

### Proofreader 校对优先级

**高优先级**：公式错误、时间步索引方向、缺失必要章节（伪代码、KnowHow）

**中优先级**：数据准确性、公式编号对应、内容一致性

**低优先级**：格式问题（章节编号）、拼写错误

## 行为准则（遵循 Andrej Karpathy 风格）

### 1. 思考后再编码

**不要假设。不要隐藏困惑。提出权衡方案。**

在实施前：
- 明确陈述假设。如果不确定，主动提问
- 如果存在多种解释，列出备选方案而非静默选择
- 如果存在更简单的方案，直接指出
- 如果有疑问，停下来，明确指出困惑点

### 2. 简洁优先

**最小化解决问题的代码。拒绝投机。**

- 不添加超出需求的功能
- 不为单次使用创建抽象
- 不添加未请求的"灵活性"或"可配置性"
- 不为不可能发生的场景添加错误处理
- 如果能 50 行解决就不要写 200 行

自问："资深工程师会说这太复杂了吗？"如果是，简化为准。

### 3. 精准改动

**只触碰必须改动的。清理自己造成的混乱。**

编辑现有代码时：
- 不"改进"相邻代码、注释或格式
- 不重构没坏的部分
- 遵循已有风格，即使你会用不同方式写
- 发现无关的死代码时：提及但不删除

检验标准：每一行改动都应直接追溯到用户请求。

### 4. 目标驱动执行

**定义成功标准。循环验证直到完成。**

将任务转化为可验证的目标：
- "添加验证" → "为无效输入写测试，然后让它们通过"
- "修复 bug" → "写能复现问题的测试，然后让它通过"
- "重构 X" → "确保测试在之前和之后都通过"

多步骤任务需简述计划：
```
1. [步骤] → 验证：[检查方式]
2. [步骤] → 验证：[检查方式]
3. [步骤] → 验证：[检查方式]
```

强成功标准让你能独立循环。弱标准（"让它工作"）需要不断确认。

## 自我改进学习（Self-Improvement）

项目使用 `.learnings/` 目录记录经验教训，用于持续改进。

### 学习文件

| 文件 | 用途 | 触发场景 |
|------|------|----------|
| `.learnings/LEARNINGS.md` | 经验、纠正、最佳实践 | 用户纠正、发现更好方法、知识差距 |
| `.learnings/ERRORS.md` | 命令/操作失败记录 | 命令返回非零退出码、异常、超时 |
| `.learnings/FEATURE_REQUESTS.md` | 用户请求的功能 | "能不能加一个..."、"希望可以..." |

### 日志格式

**Learning 条目**：
```markdown
## [LRN-YYYYMMDD-XXX] category

**Priority**: low | medium | high | critical
**Status**: pending | resolved | promoted

### Summary
一句话描述学到了什么

### Details
完整上下文：发生了什么、哪里错了、正确答案是什么

### Suggested Action
具体的修复或改进建议

---
```

**Error 条目**：
```markdown
## [ERR-YYYYMMDD-XXX] skill_or_command_name

**Priority**: high
**Status**: pending

### Error
```
实际错误信息
```

### Context
命令/操作尝试了什么、使用的输入或参数

### Suggested Fix
如果能识别，给出可能的修复方案

---
```

### 提升到 CLAUDE.md

当学习具有广泛适用性时，提升到项目永久记忆：

| 学习类型 | 提升目标 |
|----------|----------|
| 行为模式 | `SOUL.md` |
| 工作流改进 | `AGENTS.md` |
| 工具技巧 | `TOOLS.md` |
| 项目规范/约定 | `CLAUDE.md` |

**提升标准**：
- `Recurrence-Count >= 3`（跨至少 2 个不同任务）
- 30 天内出现
- 不是一次性问题

### 检测触发词

**纠正**（→ category: correction）：
- "不对..."、"错了..."、"实际上应该是..."
- "你搞错了..."
- "这个信息过时了..."

**功能请求**（→ feature request）：
- "能不能也..."
- "要是能...就好了"
- "有没有办法..."

**知识差距**（→ category: knowledge_gap）：
- 用户提供了你不知道的信息
- 引用的文档已过时
- API 行为与理解不符

### 优先级指南

| 优先级 | 使用场景 |
|--------|----------|
| `critical` | 阻塞核心功能、数据丢失风险、安全问题 |
| `high` | 重大影响、影响常见工作流、 recurring 问题 |
| `medium` | 中等影响、有 workaround |
| `low` | 小麻烦、边缘情况、可选改进 |

### 最佳实践

1. **立即记录** - 上下文在问题发生后最清晰
2. **具体描述** - 未来的自己需要能快速理解
3. **包含复现步骤** - 特别是错误
4. **关联文件** - 让修复更容易
5. **给出具体建议** - 不要只写"调查一下"
6. **定期审查** - 陈旧的学习会失去价值

## SotaLogger 日志工具

项目使用 `tools/sota_logger.py` 记录所有更新日志。

### 安装 Git Hook

```bash
python3 tools/install_hooks.py
```

安装后，每次 `git commit` 会自动记录到 `sotafollow.log`。

### 命令行用法

```bash
# 记录 commit
python3 tools/sota_logger.py commit "更新 VISTA 精读报告"

# 记录精读报告
python3 tools/sota_logger.py report "VISTA_精读报告.md" created "arXiv 2602.10983"
python3 tools/sota_logger.py report "VISTA_精读报告.md" updated
python3 tools/sota_logger.py report "VISTA_精读报告.md" optimized

# 记录 PR merge
python3 tools/sota_logger.py pr 42 "VISTA: 世界模型生成视觉子目标" "核心：层次化VLA架构"

# 记录通用操作
python3 tools/sota_logger.py operation "新增论文" "VISTA (arXiv 2602.10983)"

# 查看日志
python3 tools/sota_logger.py view 50
tail -f sotafollow.log
```

### Python API

```python
from tools.sota_logger import SotaLogger

logger = SotaLogger()
logger.log_commit("更新 π0.7 精读报告", ["VLA/π0.7_精读报告.md"])
logger.log_reading_report("VISTA_精读报告.md", "created", "arXiv 2602.10983")
logger.log_operation("代码审查", "FlowGRPO 格式修复完成")
```

### 日志格式示例

```
📅 2026-04-22 ⏰ 14:32:15 🌳 [COMMIT]
   📝 更新 π0.7 精读报告
   🤖 VLA: VLA/π0.7_精读报告.md
   └── 👤 Irving

📅 2026-04-22 ⏰ 14:35:00 ✨ [READING REPORT] CREATED
   📄 VISTA_精读报告.md
   🏛️ arXiv 2602.10983
   └── 🤖 Auto-logged by SotaLogger
```

### 日志文件

- 位置：`./sotafollow.log`（项目根目录）
- 轮转：文件达到 10MB 时自动轮转，保留最近 5 个文件


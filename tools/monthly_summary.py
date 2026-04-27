#!/usr/bin/env python3
"""
月度总结生成器
==============
生成月度总结报告，包含本月重点关注内容、重要经验总结
"""

import subprocess
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from sota_logger import SotaLogger


def get_commits_in_range(start_date: str, end_date: str) -> list:
    """获取指定日期范围内的提交"""

    try:
        result = subprocess.run(
            ["git", "log", "--all", "--since", start_date, "--until", end_date,
             "--pretty=format:%H|%s|%an|%ad|%ai", "--date=iso"],
            capture_output=True,
            text=True,
            timeout=30
        )

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                commits.append({
                    "hash": parts[0],
                    "subject": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                    "iso_date": parts[4]
                })
        return commits
    except Exception as e:
        print(f"❌ 获取提交历史失败: {e}")
        return []


def parse_category(msg: str) -> tuple:
    """解析提交消息的类别"""
    if "[VLA]" in msg:
        return ("🤖", "VLA", "视觉-语言-动作模型")
    elif "[RL]" in msg:
        return ("🎯", "RL", "强化学习")
    elif "[WM]" in msg or "[WorldModel]" in msg:
        return ("🌍", "WorldModel", "世界模型")
    elif "[FM]" in msg:
        return ("🔬", "FM", "基础模型")
    elif "[优化]" in msg:
        return ("⚡", "OPTIMIZE", "优化")
    elif "[修复]" in msg:
        return ("🐛", "FIX", "修复")
    elif "[Docs]" in msg or "文档" in msg:
        return ("📚", "DOCS", "文档")
    elif "[Tools]" in msg:
        return ("🛠️", "TOOLS", "工具")
    elif "[Config]" in msg:
        return ("⚙️", "CONFIG", "配置")
    elif "Merge" in msg:
        return ("🔀", "MERGE", "合并")
    elif "[整理]" in msg or "[新增]" in msg:
        return ("📦", "ADD", "新增")
    elif "[LLM]" in msg:
        return ("💬", "LLM", "大语言模型")
    elif "[面筋]" in msg:
        return ("🎓", "INTERVIEW", "面试")
    else:
        return ("🌳", "COMMIT", "提交")


def generate_monthly_summary(year: int, month: int) -> str:
    """生成月度总结"""

    # 计算日期范围
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"

    commits = get_commits_in_range(start_date, end_date)

    if not commits:
        return f"📭 {year}年{month}月 无提交记录"

    # 统计各类别
    categories = defaultdict(lambda: {"count": 0, "items": []})
    authors = defaultdict(int)
    pr_merges = 0

    for commit in commits:
        msg = commit["subject"]
        emoji, cat, cat_name = parse_category(msg)

        categories[cat]["count"] += 1
        if len(categories[cat]["items"]) < 5:  # 最多5条
            categories[cat]["items"].append(msg)

        authors[commit["author"]] += 1
        if "Merge" in msg:
            pr_merges += 1

    # 月份名称
    month_names = ["", "一月", "二月", "三月", "四月", "五月", "六月",
                   "七月", "八月", "九月", "十月", "十一月", "十二月"]

    # 生成报告
    report = f"""# {year}年{month_names[month]}月度总结

> 🤖 SOTAFollow 项目月度报告
> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 📊 本月概览

| 指标 | 数值 |
|------|------|
| 📝 总提交数 | {len(commits)} |
| 🔀 PR 合并 | {pr_merges} |
| 👥 贡献者 | {len(authors)} |

### 贡献者排行榜

"""

    # 按提交数排序
    sorted_authors = sorted(authors.items(), key=lambda x: -x[1])
    for author, count in sorted_authors[:5]:
        report += f"- **{author}**: {count} 次提交\n"

    report += f"""
## 📂 各类别统计

"""

    # 按数量排序
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1]["count"])
    for cat, data in sorted_cats:
        # 从原始消息解析类别
        if data["items"]:
            _, _, cat_display = parse_category(data["items"][0])
        else:
            cat_display = "其他"
        report += f"""### {cat} {cat_display}（{data['count']} 次提交）

"""
        for item in data["items"]:
            report += f"- {item}\n"
        report += "\n"

    report += f"""## 🎯 本月重点

"""

    # 本月最重要的论文
    important_keywords = ["精读报告", "新增", "SOTA", "突破", "首个"]
    important_commits = [c for c in commits if any(kw in c["subject"] for kw in important_keywords)]

    if important_commits:
        report += "**重要论文与突破：**\n\n"
        for commit in important_commits[:5]:
            report += f"- 📄 {commit['subject']}\n"
        report += "\n"

    # 本月经验总结
    report += f"""## 💡 经验总结

### 本月完成的主要工作

"""

    work_summary = {
        "VLA": "视觉-语言-动作模型研究持续推进",
        "RL": "强化学习算法深入探索",
        "WorldModel": "世界模型与规划研究",
        "FM": "基础模型知识体系完善",
        "OPTIMIZE": "代码与文档优化",
        "FIX": "问题修复与改进",
    }

    for cat, summary in work_summary.items():
        if cat in categories:
            report += f"- **{summary}**（{categories[cat]['count']} 项）\n"

    report += f"""
### 技术积累

- 精读报告格式标准化流程完善
- MathJax 公式渲染兼容性优化
- Proofreader 校对机制建立

## 📅 下月计划

- [ ] 持续关注 VLA 前沿进展
- [ ] 完善精读报告审核流程
- [ ] 优化项目工具链

---

*🤖 由 SotaLogger 自动生成*
"""

    return report


def save_monthly_summary(year: int, month: int, output_dir: str = "logs"):
    """保存月度总结"""

    report = generate_monthly_summary(year, month)

    output_path = Path(output_dir) / f"月度总结_{year}年{month:02d}月.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
    else:
        now = datetime.now()
        year = now.year
        month = now.month

    output_path = save_monthly_summary(year, month)
    print(f"✅ 月度总结已生成: {output_path}")

    # 打印到终端
    print("\n" + "=" * 60)
    print(generate_monthly_summary(year, month))

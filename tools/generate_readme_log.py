#!/usr/bin/env python3
"""
生成 README 最近的日志摘要
===========================
提取最近 N 天的日志生成简洁的 README 表格块
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def get_commits_in_days(days: int = 3) -> list:
    """获取最近 N 天的 commits"""
    try:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        result = subprocess.run(
            ["git", "log", f"--since={since}", "--all",
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
                date_str = parts[4].split(" +")[0]
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                commits.append({
                    "hash": parts[0],
                    "subject": parts[1],
                    "author": parts[2],
                    "date": dt
                })
        return commits
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_category(msg: str) -> tuple:
    """获取 commit 类别"""
    if "[VLA]" in msg:
        return ("🤖", "VLA")
    elif "[RL]" in msg:
        return ("🎯", "RL")
    elif "[WM]" in msg or "[WorldModel]" in msg:
        return ("🌍", "WM")
    elif "[FM]" in msg:
        return ("🔬", "FM")
    elif "[优化]" in msg:
        return ("⚡", "OPT")
    elif "[修复]" in msg or "[FIX]" in msg:
        return ("🐛", "FIX")
    elif "[Docs]" in msg or "文档" in msg:
        return ("📚", "DOCS")
    elif "[Tools]" in msg:
        return ("🛠️", "TOOLS")
    elif "[Config]" in msg:
        return ("⚙️", "CONFIG")
    elif "[整理]" in msg or "[新增]" in msg:
        return ("📦", "ADD")
    elif "[LLM]" in msg:
        return ("💬", "LLM")
    elif "[面筋]" in msg:
        return ("🎓", "INTERVIEW")
    elif "Merge" in msg:
        return ("🔀", "MERGE")
    else:
        return ("🌳", "COMMIT")


def generate_readme_block(days: int = 3) -> str:
    """生成 README 用的日志块"""
    commits = get_commits_in_days(days)

    if not commits:
        return "> 📭 暂无更新记录"

    # 按天分组
    daily_data = defaultdict(lambda: {"commits": [], "categories": set(), "summary": []})

    for commit in commits:
        date_key = commit["date"].strftime("%m-%d")
        msg = commit["subject"]

        daily_data[date_key]["commits"].append(msg)
        emoji, cat = get_category(msg)
        daily_data[date_key]["categories"].add(f"{emoji} {cat}")

        # 保存前3条关键信息
        if len(daily_data[date_key]["summary"]) < 3:
            # 简化消息
            short_msg = msg[:50] + "..." if len(msg) > 50 else msg
            daily_data[date_key]["summary"].append(short_msg)

    # 生成 Markdown
    lines = [
        "## 📖 最近更新",
        "",
        "| 日期 | 类别 | 简报 |",
        "|------|------|------|"
    ]

    sorted_dates = sorted(daily_data.keys(), reverse=True)[:days]

    for date_key in sorted_dates:
        data = daily_data[date_key]
        cats = ", ".join(sorted(data["categories"])[:4])  # 最多4个类别

        # 判断是否是今天
        today = datetime.now().strftime("%m-%d")
        date_display = f"**{date_key}** 🌟" if date_key == today else date_key

        summary = " | ".join(data["summary"][:2])

        lines.append(f"| {date_display} | {cats} | {summary} |")

    lines.append("")
    lines.append(f"_共 {len(commits)} 条更新，查看 [完整日志](./sotafollow.log)_")

    return "\n".join(lines)


if __name__ == "__main__":
    days = 3
    if len(sys.argv) >= 2:
        days = int(sys.argv[1])

    output_path = None
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    content = generate_readme_block(days)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ README 块已生成: {output_path}")
    else:
        print(content)

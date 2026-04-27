#!/usr/bin/env python3
"""
SotaLogger 历史导入脚本
=======================
将 Git 历史记录导入到 sotafollow.log
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from sota_logger import SotaLogger


def get_git_history(count: int = 100):
    """获取 Git 历史"""

    try:
        result = subprocess.run(
            ["git", "log", f"--all", f"-{count}", "--pretty=format:%H|%s|%an|%ad|%ai"],
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
                commit_hash, subject, author, date_str, iso_date = parts[:5]
                commits.append({
                    "hash": commit_hash,
                    "subject": subject,
                    "author": author,
                    "date": date_str,
                    "iso_date": iso_date
                })
        return commits
    except Exception as e:
        print(f"❌ 获取 Git 历史失败: {e}")
        return []


def import_history(limit: int = 100, dry_run: bool = False):
    """导入历史记录"""

    commits = get_git_history(limit)

    if not commits:
        print("📭 未获取到 Git 历史")
        return

    print(f"📥 准备导入 {len(commits)} 条提交记录...")

    logger = SotaLogger()

    for i, commit in enumerate(commits):
        # 解析日期
        try:
            dt = datetime.fromisoformat(commit["iso_date"].replace(" +0000", "+00:00"))
            date_str = f"📅 {dt.strftime('%Y-%m-%d')} ⏰ {dt.strftime('%H:%M:%S')}"
        except:
            date_str = f"📅 {commit['date']}"

        # 解析类型
        msg = commit["subject"]
        if "[VLA]" in msg:
            emoji, label = "🤖", "VLA"
        elif "[RL]" in msg:
            emoji, label = "🎯", "RL"
        elif "[WM]" in msg or "[WorldModel]" in msg:
            emoji, label = "🌍", "WM"
        elif "[FM]" in msg:
            emoji, label = "🔬", "FM"
        elif "[优化]" in msg:
            emoji, label = "⚡", "OPTIMIZE"
        elif "[修复]" in msg:
            emoji, label = "🐛", "FIX"
        elif "[Docs]" in msg or "文档" in msg:
            emoji, label = "📚", "DOCS"
        elif "[Tools]" in msg:
            emoji, label = "🛠️", "TOOLS"
        elif "[Config]" in msg:
            emoji, label = "⚙️", "CONFIG"
        elif "Merge" in msg:
            emoji, label = "🔀", "MERGE"
        elif "[整理]" in msg or "[新增]" in msg:
            emoji, label = "📦", "ADD"
        else:
            emoji, label = "🌳", "COMMIT"

        entry = f"""
{date_str} {emoji} [{label}]
   📝 {msg}
   🔗 {commit['hash'][:8]}
   👤 {commit['author']}"""

        if not dry_run:
            with open(logger.log_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

        print(f"  {'[DRY-RUN] ' if dry_run else ''}{emoji} {msg[:50]}...")

    print(f"\n✅ {'Dry-run 完成（无实际写入）' if dry_run else f'成功导入 {len(commits)} 条记录'}")

    if dry_run:
        print("\n💡 实际导入请运行: python3 tools/import_git_history.py")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    limit = 100
    for arg in sys.argv:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    import_history(limit, dry_run)

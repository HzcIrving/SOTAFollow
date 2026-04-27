#!/usr/bin/env python3
"""
SotaLogger: SOTAFollow 项目日志工具
=======================================
按天聚合记录项目更新日志，支持 commit、PR merge、论文精读报告更新等操作
日志风格：emoji + 开源学术风格
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

# 日志轮转配置
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
MAX_ROTATED_FILES = 5


class DailyEntry:
    """单日日志条目"""

    def __init__(self, date_str: str):
        self.date_str = date_str  # "04-26"
        self.full_date = ""  # "2026-04-26"
        self.commits = []
        self.pr_merges = []
        self.operations = []
        self.reading_reports = []
        self.authors = set()
        self.files = set()
        self.is_today = False

    def add_commit(self, msg: str, author: str, files: List[str]):
        self.commits.append(msg)
        self.authors.add(author)
        self.files.update(files[:5])  # 最多记录5个文件

    def add_pr_merge(self, pr_num: int, title: str):
        self.pr_merges.append((pr_num, title))

    def add_operation(self, operation: str, detail: str):
        self.operations.append((operation, detail))

    def add_reading_report(self, report_name: str, action: str, paper_info: str):
        self.reading_reports.append((report_name, action, paper_info))

    def get_categories(self) -> Dict[str, List[str]]:
        """按类别聚合 commits"""
        categories = defaultdict(list)

        for msg in self.commits:
            if "[VLA]" in msg:
                categories["🤖 VLA"].append(msg)
            elif "[RL]" in msg:
                categories["🎯 RL"].append(msg)
            elif "[WM]" in msg or "[WorldModel]" in msg:
                categories["🌍 WM"].append(msg)
            elif "[FM]" in msg:
                categories["🔬 FM"].append(msg)
            elif "[优化]" in msg:
                categories["⚡ OPT"].append(msg)
            elif "[修复]" in msg or "[FIX]" in msg:
                categories["🐛 FIX"].append(msg)
            elif "[Docs]" in msg or "文档" in msg:
                categories["📚 DOCS"].append(msg)
            elif "[Tools]" in msg:
                categories["🛠️ TOOLS"].append(msg)
            elif "[Config]" in msg:
                categories["⚙️ CONFIG"].append(msg)
            elif "[整理]" in msg or "[新增]" in msg:
                categories["📦 ADD"].append(msg)
            elif "[LLM]" in msg:
                categories["💬 LLM"].append(msg)
            elif "[面筋]" in msg:
                categories["🎓 INTERVIEW"].append(msg)
            elif "Merge" in msg:
                pass  # PR merge 单独显示
            else:
                categories["🌳 COMMIT"].append(msg)

        return dict(categories)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        today_mark = " 🌟" if self.is_today else ""

        lines = [
            f"## 📅 {self.full_date} ({self.date_str}){today_mark}",
            ""
        ]

        # 按类别显示 commits
        categories = self.get_categories()
        for cat, msgs in categories.items():
            lines.append(f"**{cat}**")
            for msg in msgs[:5]:  # 最多5条
                lines.append(f"- {msg}")
            if len(msgs) > 5:
                lines.append(f"  _...还有 {len(msgs) - 5} 条_")
            lines.append("")

        # PR merges
        if self.pr_merges:
            lines.append("**🔀 PR Merges**")
            for pr_num, title in self.pr_merges:
                lines.append(f"- #{pr_num}: {title}")
            lines.append("")

        # Operations
        if self.operations:
            lines.append("**⚙️ Operations**")
            for op, detail in self.operations:
                lines.append(f"- {op}: {detail}")
            lines.append("")

        # Reading reports
        if self.reading_reports:
            lines.append("**📄 Reading Reports**")
            for name, action, info in self.reading_reports:
                lines.append(f"- {name} ({action})")
            lines.append("")

        # 统计
        author_list = ", ".join(sorted(self.authors)) if self.authors else "Unknown"
        lines.append(f"👥 **{len(self.authors)} 人** | 📝 **{len(self.commits)} commits** | 📂 **{len(self.files)} files**")

        return "\n".join(lines)


class SotaLogger:
    """SOTAFollow 项目日志记录器 - 按天聚合版本"""

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            self.log_path = Path(__file__).parent.parent / "sotafollow.log"
        else:
            self.log_path = Path(log_path)

        self.index_path = self.log_path.with_suffix(".index.md")
        self._ensure_log_directory()

        # 内存缓冲区：今天的 commits
        self._today_entries: Dict[str, DailyEntry] = {}
        self._load_today_entries()

    def _ensure_log_directory(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_date_key(self, dt: datetime = None) -> str:
        """获取日期键"""
        if dt is None:
            dt = datetime.now()
        return dt.strftime("%m-%d")

    def _get_full_date(self, dt: datetime = None) -> str:
        if dt is None:
            dt = datetime.now()
        return dt.strftime("%Y-%m-%d")

    def _load_today_entries(self):
        """从现有日志加载今天的条目（用于重启后继续累积）"""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                content = f.read()

            today = self._get_date_key()
            today_full = self._get_full_date()

            # 简单解析：找到今天的区块
            lines = content.split("\n")
            in_today = False
            current_date = ""

            for line in lines:
                if line.startswith("## 📅 ") and today in line:
                    in_today = True
                    current_date = today
                    if current_date not in self._today_entries:
                        self._today_entries[current_date] = DailyEntry(current_date)
                        self._today_entries[current_date].full_date = today_full
                        self._today_entries[current_date].is_today = True
                elif line.startswith("## 📅 ") and in_today:
                    break  # 遇到下一天就停止

        except Exception:
            pass

    def _check_rotation(self):
        if not self.log_path.exists():
            return

        file_size = self.log_path.stat().st_size
        if file_size >= MAX_LOG_SIZE:
            self._rotate_logs()

    def _rotate_logs(self):
        oldest = self.log_path.parent / f"{self.log_path.name}.{MAX_ROTATED_FILES}"
        if oldest.exists():
            oldest.unlink()

        for i in range(MAX_ROTATED_FILES - 1, 0, -1):
            current = self.log_path.parent / f"{self.log_path.name}.{i}"
            if current.exists():
                next_file = self.log_path.parent / f"{self.log_path.name}.{i + 1}"
                current.rename(next_file)

        rotated = self.log_path.parent / f"{self.log_path.name}.1"
        if self.log_path.exists():
            self.log_path.rename(rotated)

    def _get_git_user(self) -> str:
        try:
            result = subprocess.run(
                ["git", "config", "user.name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except Exception:
            return "Unknown"

    def _get_git_history(self, count: int = 50) -> List[Dict]:
        """获取最近的 git commits"""
        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--pretty=format:%H|%s|%an|%ai"],
                capture_output=True,
                text=True,
                timeout=30
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    # 日期格式: "2026-04-26 23:43:54 +0800"
                    date_str = parts[3].split(" +")[0]  # 去掉时区
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    commits.append({
                        "hash": parts[0],
                        "subject": parts[1],
                        "author": parts[2],
                        "date": dt
                    })
            return commits
        except Exception:
            return []

    def generate_full_log(self, days: int = 30) -> str:
        """生成完整日志（按天聚合）"""
        commits = self._get_git_history(days * 10)  # 假设每天最多10条

        # 按天分组
        daily_data: Dict[str, DailyEntry] = {}

        for commit in commits:
            date_key = commit["date"].strftime("%m-%d")
            full_date = commit["date"].strftime("%Y-%m-%d")

            if date_key not in daily_data:
                daily_data[date_key] = DailyEntry(date_key)
                daily_data[date_key].full_date = full_date
                if date_key == self._get_date_key():
                    daily_data[date_key].is_today = True

            daily_data[date_key].add_commit(
                commit["subject"],
                commit["author"],
                []
            )

        # 生成 Markdown
        output = [
            "# 📖 SOTAFollow 更新日志",
            "",
            "> 🤖 自动生成 | [查看完整日志](./sotafollow.log)",
            "",
        ]

        # 快捷索引
        dates = sorted(daily_data.keys(), reverse=True)[:7]
        index_items = " | ".join([f"[📅 {d}](#📅-{d})" for d in dates])
        output.append(f"**快捷导航**: {index_items}")
        output.append("")
        output.append("---")
        output.append("")

        # 每日详情
        for date_key in sorted(daily_data.keys(), reverse=True):
            entry = daily_data[date_key]
            output.append(entry.to_markdown())
            output.append("")
            output.append("---")
            output.append("")

        return "\n".join(output)

    def save_log(self, days: int = 30):
        """保存完整日志"""
        self._check_rotation()

        content = self.generate_full_log(days)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(content)

    def view_log(self, days: int = 3):
        """查看最近 n 天日志"""
        content = self.generate_full_log(days)
        print(content)

    def log_commit(self, commit_msg: str, files: List[str] = None):
        """记录 commit - 重新加载完整日志并聚合"""
        files = files or []
        git_user = self._get_git_user()

        # 获取今天日期
        today = datetime.now()
        date_key = self._get_date_key(today)
        full_date = self._get_full_date(today)

        # 添加到今日缓冲区
        if date_key not in self._today_entries:
            self._today_entries[date_key] = DailyEntry(date_key)
            self._today_entries[date_key].full_date = full_date
            self._today_entries[date_key].is_today = True

        self._today_entries[date_key].add_commit(commit_msg, git_user, files)

        # 重新生成完整日志
        self.save_log()

    def log_pr_merge(self, pr_num: int, title: str, summary: str = ""):
        """记录 PR merge"""
        date_key = self._get_date_key()

        if date_key not in self._today_entries:
            self._today_entries[date_key] = DailyEntry(date_key)
            self._today_entries[date_key].full_date = self._get_full_date()
            self._today_entries[date_key].is_today = True

        self._today_entries[date_key].add_pr_merge(pr_num, title)
        self.save_log()

    def log_operation(self, operation: str, detail: str, category: str = "OPERATION"):
        """记录通用操作"""
        date_key = self._get_date_key()

        if date_key not in self._today_entries:
            self._today_entries[date_key] = DailyEntry(date_key)
            self._today_entries[date_key].full_date = self._get_full_date()
            self._today_entries[date_key].is_today = True

        self._today_entries[date_key].add_operation(operation, detail)
        self.save_log()

    def log_reading_report(self, report_name: str, action: str, paper_info: str = ""):
        """记录精读报告更新"""
        date_key = self._get_date_key()

        if date_key not in self._today_entries:
            self._today_entries[date_key] = DailyEntry(date_key)
            self._today_entries[date_key].full_date = self._get_full_date()
            self._today_entries[date_key].is_today = True

        self._today_entries[date_key].add_reading_report(report_name, action, paper_info)
        self.save_log()


def get_logger() -> SotaLogger:
    return SotaLogger()


# 命令行接口
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              SotaLogger 使用指南 (按天聚合版)               ║
╠══════════════════════════════════════════════════════════════╣
║  用法: python3 sota_logger.py <command> [args]             ║
╠══════════════════════════════════════════════════════════════╣
║  命令:                                                        ║
║    view [days]         查看最近 N 天日志 (默认 3 天)         ║
║    save                保存/更新完整日志                     ║
║    commit <msg>        记录 commit 并更新日志               ║
║    pr <num> <title>    记录 PR merge                        ║
║    report <name> <act> 记录精读报告 (created/updated...)   ║
║    op <op> <detail>    记录操作                              ║
╠══════════════════════════════════════════════════════════════╣
║  示例:                                                        ║
║    python3 sota_logger.py view 7                           ║
║    python3 sota_logger.py save                              ║
║    python3 sota_logger.py commit "更新 VISTA 精读报告"       ║
╚══════════════════════════════════════════════════════════════╝
        """)
        sys.exit(0)

    logger = SotaLogger()
    command = sys.argv[1]

    if command == "view" and len(sys.argv) >= 3:
        days = int(sys.argv[2])
        logger.view_log(days)
    elif command == "view":
        logger.view_log(3)

    elif command == "save":
        logger.save_log()
        print("✅ 日志已保存!")

    elif command == "commit" and len(sys.argv) >= 3:
        logger.log_commit(sys.argv[2])
        print("✅ Commit logged!")

    elif command == "pr" and len(sys.argv) >= 4:
        pr_num = int(sys.argv[2])
        title = sys.argv[3]
        summary = sys.argv[4] if len(sys.argv) > 4 else ""
        logger.log_pr_merge(pr_num, title, summary)
        print(f"✅ PR #{pr_num} logged!")

    elif command == "report" and len(sys.argv) >= 4:
        name = sys.argv[2]
        action = sys.argv[3]
        info = sys.argv[4] if len(sys.argv) > 4 else ""
        logger.log_reading_report(name, action, info)
        print(f"✅ Report {action} logged!")

    elif command == "op" and len(sys.argv) >= 4:
        op = sys.argv[2]
        detail = sys.argv[3]
        logger.log_operation(op, detail)
        print("✅ Operation logged!")

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

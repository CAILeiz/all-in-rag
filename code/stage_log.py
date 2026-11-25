"""分阶段日志：把 RAG 流水线每一步打在终端上，方便对着输出学流程。

设计目标只有三条：
  1. 每行都以 `[当前步/总步数]` 开头，一眼看出走到哪了
  2. 每一步都打印「关键入参」和「产出结果」，而不只是一句「开始了」
  3. 自动计时，末尾给总耗时

用法：

    from stage_log import Stage

    stage = Stage(total=6)

    with stage.step("文档开始分片", func="RecursiveCharacterTextSplitter(chunk_size=4000)"):
        chunks = split(docs)
        stage.note(f"输出 {len(chunks)} 个 chunk")

    stage.summary()          # 打印总耗时表

也可以用 `stage.detail("...")` 打缩进的补充信息，`stage.info("...")` 打不占步数的提示行。

环境变量：
    RAG_LOG=quiet   关闭全部阶段日志（脚本原有输出不受影响）
"""
import os
import sys
import time
import unicodedata

# ------------------------------------------------------------------ 样式

_ENABLED = os.getenv("RAG_LOG", "").lower() not in ("quiet", "0", "false", "off")
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

_C = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
}


def _c(text, *styles):
    if not _USE_COLOR or not styles:
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


def _dwidth(text):
    """字符串在终端里的显示宽度：中文/全角算 2，其余算 1。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text, width):
    """按显示宽度左对齐（补空格），用于中文混排的列对齐。"""
    return text + " " * max(0, width - _dwidth(text))


def _rpad(text, width):
    """按显示宽度右对齐。"""
    return " " * max(0, width - _dwidth(text)) + text


def _ms(seconds):
    """耗时人性化：小于 1 秒用 ms，否则用 s。"""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(seconds, 60)
    return f"{int(m)} min {s:.0f} s"


# ------------------------------------------------------------------ 步骤

class _Step:
    """一次 `with stage.step(...)` 的上下文。"""

    def __init__(self, stage, title, meta):
        self.stage = stage
        self.title = title
        self.meta = meta
        self.notes = []
        self._t0 = 0.0
        self._printed_meta = False

    # -- 便捷方法 --------------------------------------------------
    def detail(self, msg):
        """缩进的补充信息，跟在当前步骤下面。传空串就是空行。"""
        if msg:
            self.stage._emit(f"       {_c(msg, 'dim')}")
        else:
            self.stage._emit("")
        return self

    def note(self, msg):
        """登记一条结果，会拼到「完成」那一行后面。"""
        self.notes.append(msg)
        return self

    # -- 上下文协议 ------------------------------------------------
    def __enter__(self):
        self.stage._step_index += 1
        self.stage._t_step = time.perf_counter()
        self._t0 = self.stage._t_step
        self.stage._current = self          # 让 stage.note() 能转发到当前步骤

        prefix = self.stage._prefix()
        self.stage._emit(f"{prefix} {_c('▶ ' + self.title, 'bold', 'cyan')}")
        # 按最长的键动态对齐，避免 "temperature" 这种长键把值挤到一起
        kw = max((_dwidth(k) for k in self.meta), default=0) + 2
        for k, v in self.meta.items():
            self.stage._emit(f"       {_c(_pad(k, kw), 'dim')}{v}")
        self._printed_meta = True
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._t0
        self.stage.elapsed_total += elapsed
        self.stage.timings.append((self.title, elapsed))
        self.stage._current = None

        prefix = self.stage._prefix()
        if exc_type is not None:
            self.stage._emit(f"{prefix} {_c('✗ 失败', 'red', 'bold')} {self.title}"
                             f"  {_c(type(exc).__name__ + ': ' + str(exc)[:90], 'red')}")
            return False  # 不吞异常

        suffix = f"  {_c('| ' + '，'.join(self.notes), 'dim')}" if self.notes else ""
        self.stage._emit(f"{prefix} {_c('✓ 完成', 'green', 'bold')}"
                         f"  {_c(_ms(elapsed), 'green')}{suffix}")
        return False


# ------------------------------------------------------------------ Stage

class Stage:
    """一串步骤的容器，负责编号、计时和汇总。"""

    def __init__(self, total, title=None):
        self.total = total
        self.title = title
        self._step_index = 0          # 已经进入过的步骤数
        self._t_step = 0.0
        self._t_start = time.perf_counter()
        self.elapsed_total = 0.0
        self.timings = []
        self._current = None          # 当前正在执行的 _Step

    # -- 输出 ------------------------------------------------------
    @staticmethod
    def _emit(line):
        if _ENABLED:
            print(line, flush=True)

    def _prefix(self):
        n = min(self._step_index, self.total)
        tag = f"[{n}/{self.total}]"
        return _c(tag, "blue", "bold")

    # -- 公开 API --------------------------------------------------
    def header(self, subtitle="", title=None):
        """打一条分隔线，标记整轮开始。

        title 默认取 Stage(title=...)，都没有就用通用的「RAG 流水线」。
        """
        if not _ENABLED:
            return self
        bar = "═" * 72
        self._emit("")
        self._emit(_c(bar, "cyan"))
        name = title or self.title or "RAG 流水线"
        head = f"  {name}  ·  共 {self.total} 步"
        self._emit(_c(head, "bold", "cyan"))
        if subtitle:
            self._emit(_c(f"  {subtitle}", "dim"))
        self._emit(_c(bar, "cyan"))
        return self

    def step(self, title, **meta):
        """开始一个步骤，返回上下文管理器。

        meta 会以「键 : 值」的形式打印在步骤标题下面，
        用来展示这一步的关键入参（比如用了什么函数、什么模型）。
        """
        return _Step(self, title, meta)

    def note(self, msg):
        """在 with 块内登记结果，会拼到「✓ 完成」那一行的后面。"""
        if self._current is not None:
            self._current.note(msg)
        return self

    def detail(self, msg):
        if msg:
            self._emit(f"       {_c(msg, 'dim')}")
        else:
            self._emit("")
        return self

    def info(self, msg):
        """不占步数的提示行。"""
        self._emit(f"         {_c('· ' + msg, 'yellow')}")
        return self

    def warn(self, msg):
        self._emit(f"         {_c('! ' + msg, 'yellow', 'bold')}")
        return self

    def summary(self):
        """结尾汇总：每步耗时 + 总耗时。"""
        if not _ENABLED:
            return
        wall = time.perf_counter() - self._t_start
        self._emit("")
        self._emit(_c("─" * 72, "dim"))
        self._emit(_c(f"  耗时汇总（流水线共 {len(self.timings)} 步）", "bold"))
        for i, (title, sec) in enumerate(self.timings, 1):
            bar_len = max(1, int(sec / max(self.elapsed_total, 1e-9) * 30))
            bar = "█" * bar_len
            self._emit(f"    {i}. {_pad(title, 26)}{_rpad(_ms(sec), 9)}  {_c(bar, 'cyan')}")
        self._emit(_c("─" * 72, "dim"))
        self._emit(f"    步骤合计 {_ms(self.elapsed_total)}    "
                   f"整轮墙钟 {_ms(wall)}")
        self._emit(_c("─" * 72, "dim"))

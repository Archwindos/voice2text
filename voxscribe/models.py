"""任务在模块之间传递的数据结构。

这些 dataclass 是各模块（asr / refine / export / gui）共同遵守的接口契约，
修改字段前请先同步更新调用方与 ``tests/``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 整理风格：键为程序内部标识，值为界面展示文案
# ---------------------------------------------------------------------------
STYLES: Dict[str, str] = {
    "oral": "口述成文",      # 口语转书面语：去口头禅、补标点、分段落
    "proofread": "口述校对",  # 快速语音输入：纠错别字/同音字 + 去口头禅 + 补标点
    "meeting": "会议纪要",    # 结构化：议题、结论、待办
    "verbatim": "逐字稿润色",  # 保留原话，只做最小限度的清洗
    "summary": "要点摘要",    # 压缩为要点列表
}

EXPORT_FORMATS = ("txt", "md", "docx", "srt")


def format_timestamp(seconds: float) -> str:
    """把秒数格式化为 ``HH:MM:SS``（用于预览与文本时间戳）。"""
    seconds = max(0.0, float(seconds or 0.0))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """SRT 字幕时间戳：``HH:MM:SS,mmm``。"""
    seconds = max(0.0, float(seconds or 0.0))
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


@dataclass
class Segment:
    """一条带时间戳的转写片段。"""

    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"start": float(self.start), "end": float(self.end), "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
        )


@dataclass
class Transcript:
    """一段音频的完整转写结果。"""

    segments: List[Segment] = field(default_factory=list)
    language: str = "zh"
    duration: float = 0.0

    @property
    def text(self) -> str:
        """按行拼接的纯文本（已去掉首尾空白）。"""
        lines = [seg.text.strip() for seg in self.segments if seg.text and seg.text.strip()]
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return not self.text

    def with_timestamps(self) -> str:
        """每行前加 ``[HH:MM:SS]`` 的带时间戳文本。"""
        lines = []
        for seg in self.segments:
            text = (seg.text or "").strip()
            if text:
                lines.append(f"[{format_timestamp(seg.start)}] {text}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "duration": float(self.duration),
            "segments": [seg.to_dict() for seg in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        raw_segments = data.get("segments") or []
        return cls(
            segments=[Segment.from_dict(item) for item in raw_segments],
            language=str(data.get("language") or "zh"),
            duration=float(data.get("duration") or 0.0),
        )


@dataclass
class RefineOptions:
    """交给 DeepSeek 的整理参数。"""

    enabled: bool = True
    style: str = "oral"
    language: str = "zh"
    glossary: str = ""
    extra_instruction: str = ""
    temperature: float = 0.3
    max_chunk_chars: int = 2800
    chunk_overlap_chars: int = 120


@dataclass
class RefineUsage:
    """DeepSeek 返回的 token 用量，用于界面展示本次消耗。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "RefineUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class RefineResult:
    """DeepSeek 整理后的结果。"""

    text: str
    chunks: int = 0
    usage: RefineUsage = field(default_factory=RefineUsage)
    elapsed: float = 0.0


@dataclass
class JobOptions:
    """单个音频文件的一次处理选项（GUI 侧逐文件传入）。"""

    refine: bool = True
    style: str = "oral"
    language: Optional[str] = "zh"  # "zh" / "en" / None(自动)
    export_formats: List[str] = field(default_factory=lambda: ["txt", "md"])
    output_dir: Optional[str] = None
    title: Optional[str] = None


@dataclass
class JobResult:
    """一次完整任务（转写 + 整理 + 导出）的结果。"""

    audio_path: str
    transcript: Transcript
    refined: Optional[RefineResult] = None
    output_files: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def final_text(self) -> str:
        """整理稿优先；未整理时退回原始转写。"""
        if self.refined and self.refined.text.strip():
            return self.refined.text
        return self.transcript.text


@dataclass
class JobEvent:
    """流水线向界面发出的进度事件（由界面线程轮询队列消费）。"""

    stage: str  # prepare / asr / refine / export / done / error / cancelled
    progress: float = -1.0  # 0.0~1.0；-1 表示不确定进度
    message: str = ""
    audio_path: str = ""
    result: Optional[JobResult] = None
    error: Optional[str] = None
    error_hint: str = ""


__all__ = [
    "STYLES",
    "EXPORT_FORMATS",
    "format_timestamp",
    "format_srt_timestamp",
    "Segment",
    "Transcript",
    "RefineOptions",
    "RefineUsage",
    "RefineResult",
    "JobOptions",
    "JobResult",
    "JobEvent",
]

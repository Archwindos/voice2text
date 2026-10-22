"""VoxScribe 语声成文 —— 把语音内容整理成书面文字。

工作流：
    麦克风 / 音频文件
        -> 本地 faster-whisper（优先 NVIDIA GPU / CUDA float16）语音转写
        -> DeepSeek API（deepseek-chat，非思考模式）口语整理与润色
        -> 直接复制 / 导出 txt、md、docx、srt

设计目标：快、轻量、界面克制、隐私友好；托盘常驻，Alt+~ 一键口述。
"""

__version__ = "1.3.3"
APP_NAME = "VoxScribe"
APP_DISPLAY_NAME = "语声成文"

# 部分网络环境下 Hugging Face 的 Xet 下载通道不可用，改用普通 HTTP 更稳。
# 必须在 huggingface_hub 被（faster-whisper 懒加载）import 之前设置。
import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

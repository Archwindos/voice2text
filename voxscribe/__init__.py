"""VoxScribe 语声成文 —— 把语音内容整理成书面文字。

工作流：
    录音 / 音频文件
        -> 本地 faster-whisper（faster-whisper, CPU int8）语音转写
        -> DeepSeek API（deepseek-chat）口语整理与润色
        -> 导出 txt / md / docx / srt

设计目标：轻量、离线可用、CPU 占用可控、模型空闲自动卸载。
"""

__version__ = "1.0.0"
APP_NAME = "VoxScribe"
APP_DISPLAY_NAME = "语声成文"

"""词语对照表自动学习核心模块（纯逻辑：无 GUI、无网络、无第三方依赖）。

模块职责
========
1. :func:`extract_candidates`：比较「直出模式最终稿基线」与「用户修改后的文本」，
   抽取「错误词 → 正确词」纠正候选；
2. :func:`parse_manual_glossary`：把设置页的手动术语表文本解析成结构化条目；
3. :func:`merge_candidates`：把候选并入条目表，处理去重、重复累加与冲突；
4. :func:`load_store` / :func:`save_store`：读写 ``data/glossary.json``（原子写 + ``.bak``）；
5. :func:`build_prompt_glossary`：生成发给 DeepSeek 的术语表文本；
6. :func:`apply_replacements`：在本地应用替换/删除规则（链式映射做传递闭包，检测到环跳过）。

比较规则（与 docs/glossary-auto-learning-design.md 一致）
=======================================================
* NFKC 归一化后做字符级 :class:`difflib.SequenceMatcher` 对齐；
* 只处理 ``replace`` / ``delete``；插入、纯标点、纯空白变化不学习；
* 先裁掉公共前后缀，再收缩到不含标点/空白的词边界；
* 过滤：数字/URL/邮箱/手机号、繁简差异、大小写差异、空白差异、停用词、单字、超长；
* 整段重写（变更字符 > 40 或 > 基线 30%）直接放弃，避免误学；
* 中文单字差异（如「阴→音」）借助「常见双字词表」向左/右扩张成双字词
  （如「阴轨→音轨」）；停用单字（的/了/在/得……）仍然拒绝；
* 英文按空白词取整：字符级片段会扩张成完整单词（audio→video），
  删除只接受整词删除（删掉单词中间几个字母不学习）；
* 候选数量默认上限 5，多个字符级片段扩张成同一对词语时会去重。

日志（用户要求，作为产品试验数据）
=================================
logger 名固定为 ``voxscribe.glossary``；每次抽取/合并/应用都会输出 **一行 JSON**
（``ensure_ascii=False``），方便离线分析。字段包含 ``event`` / ``timestamp`` /
``baseline_chars`` / ``edited_chars`` / ``changed_chars`` / ``change_ratio`` /
``opcode_count`` / ``candidates`` / ``filtered`` / ``accepted`` / ``pending`` /
``conflicts`` / ``elapsed_ms``。DEBUG 级会额外记录每一条被过滤片段的原因。

隐私：**不记录完整原文/修改稿**，只记录长度、计数、候选词本身，以及候选词前后
各最多 8 字的上下文；被过滤片段的 snippet 最多 20 字。
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config

#: 日志 logger 名固定（产品数据分析时按此名过滤）。
LOGGER_NAME = "voxscribe.glossary"
log = logging.getLogger(LOGGER_NAME)

# ---------------------------------------------------------------------------
# 对外常量
# ---------------------------------------------------------------------------
STORE_VERSION = 1
STORE_FILENAME = "glossary.json"

DEFAULT_MAX_LEN = 12
DEFAULT_MAX_CHANGED_CHARS = 40
DEFAULT_MAX_CHANGE_RATIO = 0.3
DEFAULT_MAX_CANDIDATES = 5
DEFAULT_AUTO_APPLY_THRESHOLD = 0.85
DEFAULT_MAX_ENTRIES = 200

#: 候选中候选词前后各保留的上下文字符数。
CONTEXT_CHARS = 8
#: 被过滤片段写进日志时的最大字符数。
SNIPPET_CHARS = 20

#: 置信度规则（见设计文档 3.4）。
CONFIDENCE_REPLACE = 0.75
CONFIDENCE_DELETE = 0.5
#: 同一错误重复出现时每次累加的置信度（0.75 -> 0.85 恰好达到自动应用阈值）。
CONFIDENCE_BUMP = 0.1

EVENT_EXTRACT = "extract"
EVENT_MERGE = "merge"
EVENT_APPLY = "apply"
EVENT_CONFLICT = "conflict"
EVENT_UNDO = "undo"
#: 单条被过滤片段的 DEBUG 事件类型（主事件类型之外的可观测性补充）。
EVENT_FILTER = "filter"

#: 过滤原因（测试与界面文案共用，尽量简短且含关键词）。
REASON_EMPTY = "空片段"
REASON_INSERT = "插入不是纠错"
REASON_PUNCT = "标点/空白"
REASON_SENSITIVE = "数字/敏感信息"
REASON_URL = "URL/邮箱"
REASON_SCRIPT = "繁简差异"
REASON_CASE = "大小写差异"
REASON_SPACE = "空白差异"
REASON_STOPWORD = "停用词"
REASON_SINGLE = "单字"
REASON_LONG = "超长"
REASON_BIG_CHANGE = "大改"
REASON_LIMIT = "候选超过上限"
REASON_DELETE_JOIN = "删除破坏句子结构"

# ---------------------------------------------------------------------------
# 停用词 / 敏感信息 / 繁简对照等静态规则
# ---------------------------------------------------------------------------
#: 常见停用词、语气词、代词、连接词：old/new 任一侧命中都不学习。
_STOPWORDS = frozenset(
    """
    的 了 着 过 吗 吧 呢 啊 嗯 哦 呀 嘛 么 是 在 有 和 与 及 或 就 都 也 还 又
    很 太 这 那 这个 那个 这些 那些 这里 那里 哪里 然后 就是 但是 可是 不过
    因为 所以 如果 虽然 而且 并且 或者 还是 一个 一下 一些 什么 怎么 怎样
    如何 这样 那样 其实 已经 可以 可能 应该 我们 你们 他们 她们 它们 自己
    大家 这么 那么 以及 等等 的话 一样 一点 有些 有的 没有 不是 只是 于是
    因此 然而 否则 不然 总之 首先 其次 最后 时候 现在
    """.split()
)
#: 英文停用词（按空白分词，统一小写比较）。
_STOPWORDS_EN = frozenset(
    """
    the a an and or but if then than that this these those is are was were be
    been being am do does did have has had to of in on at by for with from as
    it its i you he she we they them his her our their not no yes so such very
    just can could should would will
    """.split()
)
#: 单字停用字：这些单字差异默认不学习（「的→得」这类）。
_SINGLE_STOP_CHARS = frozenset(
    "的了着過过吗嗎吧呢啊嗯哦呀嘛么麼是在有和与與及或就都也还還又很太这這那哪"
    "个個們们地得之而且并並则則于於把被让讓给給对對从從向到以所没沒不"
)

#: 疑似敏感信息：URL / 邮箱 / 手机号 / 长数字串。
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s-]{5,}\d)")
_DIGIT_RE = re.compile(r"\d")

#: 句末标点（判断删除后是否出现「两个标点贴在一起」）。
_SENTENCE_END = frozenset("。！？；!?;")

#: 常见繁体 → 简体字形对照（只用于识别「繁简差异」，不用于实际转换）。
#: 词表不求完整，覆盖常用字即可；未覆盖的繁简差异会按普通错词候选处理。
_TRAD_TO_SIMP: Dict[str, str] = {
    pair[0]: pair[1]
    for pair in """
    這这 個个 們们 來来 時时 說说 會会 對对 開开 關关 還还 沒没 經经 過过 發发
    現现 實实 點点 問问 題题 學学 習习 樣样 後后 從从 裡里 裏里 麼么 為为 與与
    無无 聽听 寫写 聲声 數数 據据 網网 絡络 電电 腦脑 資资 訊讯 處处 應应 該该
    總总 結结 種种 詳详 細细 單单 簡简 繁繁 體体 臺台 灣湾 慶庆 週周 歲岁 漢汉
    語语 詞词 彙汇 譯译 書书 讀读 買买 賣卖 錢钱 價价 貴贵 費费 銀银 鐵铁 門门
    馬马 鳥鸟 魚鱼 龍龙 鳳凤 龜龟 貓猫 豬猪 雞鸡 鴨鸭 鵝鹅 蟲虫 貝贝 車车 軍军
    農农 運运 邊边 進进 遠远 連连 達达 選选 遞递 適适 邏逻 輯辑 錯错 誤误 標标
    準准 確确 認认 檢检 測测 試试 驗验 證证 檔档 夾夹 帶带 頁页 碼码 編编 號号
    權权 賬账 帳帐 戶户 郵邮 鏈链 擊击 鍵键 盤盘 顯显 隱隐 刪删 複复 製制 貼贴
    動动 擇择 設设 啟启 暫暂 繼继 續续 錄录 轉转 歸归 畢毕 業业 獎奖 歷历 曆历
    圍围 團团 圓圆 園园 國国 圖图 層层 屬属 島岛 嶺岭 峽峡 巖岩 幣币 幫帮 廠厂
    廣广 廳厅 張张 強强 彈弹 當当 況况 滅灭 滿满 漁渔 漲涨 潔洁 濕湿 灑洒 烏乌
    煙烟 熱热 燈灯 營营 爾尔 牆墙 狀状 獨独 獲获 獻献 環环 產产 畫画 異异 疊叠
    盡尽 鹽盐 監监 眾众 礙碍 礎础 稅税 穩稳 窮穷 競竞 筆笔 節节 範范 築筑 簽签
    籃篮 籌筹 糧粮 紅红 級级 紀纪 約约 純纯 紙纸 終终 組组 絕绝 統统 絲丝 綠绿
    維维 綱纲 緊紧 線线 練练 織织 罰罚 義义 聳耸 聯联 職职 脈脉 腳脚 臉脸 臨临
    舉举 舊旧 艙舱 藝艺 華华 萬万 葉叶 蓋盖 蔣蒋 蕭萧 薩萨 蘇苏 蘋苹 虛虚 補补
    裝装 褲裤 見见 規规 視视 親亲 覺觉 觀观 計计 訂订 討讨 訓训 記记 訪访 許许
    訴诉 診诊 註注 証证 評评 詩诗 話话 誠诚 誰谁 課课 調调 談谈 請请 論论 諮咨
    諾诺 謀谋 講讲 謝谢 識识 譜谱 議议 護护 變变 讓让 讚赞 豐丰 負负 財财 貢贡
    貨货 販贩 貪贪 責责 貸贷 賓宾 賤贱 賦赋 質质 賭赌 賴赖 購购 賽赛 贈赠 贊赞
    贏赢 趨趋 跡迹 躍跃 軌轨 軟软 較较 載载 輔辅 輕轻 輛辆 輝辉 輪轮 輸输 辦办
    辭辞 迴回 遊游 違违 遜逊 遲迟 遺遗 鄉乡 鄰邻 醫医 釋释 針针 釣钓 鈔钞 鈴铃
    鉤钩 銅铜 銘铭 銷销 鋒锋 鋪铺 鋼钢 錶表 鎂镁 鐘钟 鑄铸 鑑鉴 長长 閃闪 閉闭
    閒闲 間间 閔闵 闡阐 陽阳 陰阴 陳陈 陸陆 隊队 隨随 險险 雖虽 雙双 雜杂 離离
    難难 雲云 霧雾 靈灵 靜静 頂顶 項项 順顺 須须 預预 領领 頭头 頻频 額额 顏颜
    願愿 類类 顧顾 風风 飛飞 飯饭 飲饮 飾饰 養养 館馆 駕驾 騎骑 驅驱 騰腾 驚惊
    髮发 鬧闹 鬥斗 鮮鲜 鳴鸣 鷹鹰 麗丽 麥麦 黃黄 黨党 於于 兇凶
    """.split()
    if len(pair) == 2
}

#: 常见双字词表：仅用于把中文单字差异扩张成双字词时的「哪一侧更像词」判断。
#: 不是分词器，也不追求完整；两侧都判断不出时优先向右扩张。
_COMMON_WORDS = frozenset(
    """
    我们 你们 他们 什么 怎么 这样 那样 时候 现在 已经 可以 因为 所以 但是 如果
    然后 或者 而且 虽然 不过 因此 于是 其实 可能 一定 非常 比较 特别 开始 结束
    完成 打开 关闭 保存 删除 设置 文件 目录 数据 网络 连接 登录 登入 账号 帐号
    密码 用户 项目 问题 时间 内容 结果 方法 方式 情况 状态 进度 错误 正确 检查
    确认 处理 分析 设计 实现 测试 发布 版本 系统 软件 硬件 电脑 手机 音频 音轨
    视频 图片 会议 记录 讨论 决定 计划 目标 任务 工作 学习 生活 公司 产品 服务
    客户 市场 价格 成本 收入 支出 费用 金额 银行 支付 订单 地址 电话 邮件 城市
    国家 地区 世界 今天 明天 昨天 上午 下午 晚上 今年 去年 明年 这个 那个 哪个
    这些 那些 这里 那里 前面 后面 上面 下面 里面 外面 中间 旁边 东西 事情 答案
    原因 办法 意见 建议 要求 规定 标准 规则 格式 标题 正文 文档 表格 报告 总结
    方案 流程 步骤 操作 功能 页面 窗口 按钮 菜单 消息 提示 异常 日志 配置 参数
    选项 默认 手动 自动 本地 在线 下载 上传 安装 启动 运行 停止 继续 返回 退出
    进入 添加 修改 编辑 复制 粘贴 移动 选择 输入 输出 发送 接收 显示 隐藏 刷新
    更新 升级 备份 恢复 清理 缓存 存储 内存 磁盘 硬盘 空间 速度 性能 质量 安全
    隐私 权限 账户 设备 驱动 模型 语言 中文 英文 简体 繁体 图像 声音 录音 播放
    暂停 录制 转写 整理 翻译 校对 词汇 术语 词语 短语 句子 段落 章节 文章 书籍
    新闻 信息 知识 经验 能力 技术 科学 数学 物理 化学 历史 地理 音乐 电影 游戏
    体育 健康 医院 医生 药物 疾病 食物 餐厅 酒店 旅游 交通 汽车 火车 飞机 机场
    车站 商店 超市 购买 销售 优惠 活动 节日 生日 礼物 家庭 朋友 同事 老板 员工
    学生 老师 学校 课程 考试 成绩 毕业 大学 专业 研究 实验 论文 期刊 展览 比赛
    冠军 运动 训练 健身 天气 温度 下雨 晴天 季节 春天 夏天 秋天 冬天 颜色 红色
    蓝色 绿色 白色 黑色 数字 字母 符号 标点 空格 换行 排版 字体 大小 位置 方向
    距离 高度 宽度 长度 数量 重量 体积 比例 概率 统计 平均 最大 最小 总共 部分
    全部 其他 另外 例如 比如 包括 除了 关于 根据 通过 由于 为了 甚至 尤其 一般
    通常 经常 偶尔 从来 曾经 马上 立刻 刚刚 正在 一直 仍然 依然 终于 果然 居然
    竟然 当然 确实 的确 显然 明显 清楚 明白 理解 认为 觉得 希望 打算 准备 同意
    反对 支持 帮助 感谢 抱歉 对不起 没关系 再见 你好 好的 是的 不是 没有 不要
    不能 不会 不用 不对 不行 可爱 漂亮 美丽 难看 好看 有趣 无聊 高兴 开心 快乐
    难过 生气 担心 害怕 紧张 放松 舒服 辛苦 厉害 重要 简单 复杂 容易 困难 危险
    干净 拥挤 安静 热闹 新鲜 便宜 昂贵 快速 缓慢 准时 迟到
    """.split()
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
def _clamp_confidence(value: Any) -> float:
    """把置信度夹到 0~1；非法值按 0 处理。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


@dataclass
class GlossaryEntry:
    """词语表里的一条规则。

    ``kind == "replace"`` 时把 ``wrong`` 替换为 ``correct``；
    ``kind == "delete"`` 时从文本中删除 ``wrong``（``correct`` 为空）。
    """

    wrong: str
    correct: str = ""
    kind: str = "replace"  # "replace" / "delete"
    source: str = "manual"  # "manual" / "learned"
    enabled: bool = True
    confidence: float = 1.0
    count: int = 1
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.wrong = (self.wrong or "").strip()
        self.correct = (self.correct or "").strip()
        if self.kind not in ("replace", "delete"):
            self.kind = "replace"
        if self.source not in ("manual", "learned"):
            self.source = "learned"
        self.enabled = bool(self.enabled)
        self.confidence = _clamp_confidence(self.confidence)
        try:
            self.count = max(1, int(self.count))
        except (TypeError, ValueError):
            self.count = 1
        self.created_at = str(self.created_at or "")
        self.updated_at = str(self.updated_at or "")

    def to_dict(self) -> Dict[str, Any]:
        """转成可直接写入 JSON 的字典（字段名即存储格式）。"""
        return {
            "wrong": self.wrong,
            "correct": self.correct,
            "kind": self.kind,
            "source": self.source,
            "enabled": self.enabled,
            "confidence": round(float(self.confidence), 4),
            "count": int(self.count),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlossaryEntry":
        """从 JSON 字典还原；字段缺失/类型异常时尽量降级而不是崩溃。"""
        if not isinstance(data, dict):
            raise ValueError("词条必须是 JSON 对象")
        return cls(
            wrong=str(data.get("wrong") or ""),
            correct=str(data.get("correct") or ""),
            kind=str(data.get("kind") or "replace"),
            source=str(data.get("source") or "learned"),
            enabled=bool(data.get("enabled", True)),
            confidence=float(data.get("confidence", 1.0) or 0.0),
            count=int(data.get("count", 1) or 1),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class LearnCandidate:
    """一次抽取得到的「待确认」候选（尚未写入词语表）。"""

    wrong: str
    correct: str = ""
    kind: str = "replace"
    confidence: float = 0.0
    context: str = ""

    def __post_init__(self) -> None:
        self.wrong = (self.wrong or "").strip()
        self.correct = (self.correct or "").strip()
        if self.kind not in ("replace", "delete"):
            self.kind = "replace"
        self.confidence = _clamp_confidence(self.confidence)
        self.context = (self.context or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wrong": self.wrong,
            "correct": self.correct,
            "kind": self.kind,
            "confidence": round(float(self.confidence), 4),
            "context": self.context,
        }


@dataclass
class ExtractionResult:
    """一次抽取的完整结果 + 统计信息（供日志与界面展示）。"""

    candidates: List[LearnCandidate] = field(default_factory=list)
    baseline_chars: int = 0
    edited_chars: int = 0
    changed_chars: int = 0
    change_ratio: float = 0.0
    opcode_count: int = 0
    #: 被过滤的变更片段：[{"reason": 原因, "snippet": 片段}]，不落原文。
    filtered: List[Dict[str, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """本地时区的 ISO 时间戳（秒级），如 ``2026-10-23T01:46:07+08:00``。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize(text: str) -> str:
    """NFKC 归一化 + 去首尾空白；全角/半角标点在此统一。"""
    return unicodedata.normalize("NFKC", text or "").strip()


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """把片段压成一行并截断（日志/界面展示用，绝不记录整段原文）。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _make_context(text: str, start: int, end: int) -> str:
    """候选词左边/右边各取最多 :data:`CONTEXT_CHARS` 字，用「…」连接。"""
    left = text[max(0, start - CONTEXT_CHARS) : start]
    right = text[end : end + CONTEXT_CHARS]
    return f"{left}…{right}"


def _is_cjk(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_punct_or_space(char: str) -> bool:
    """标点、分隔符、空白（用于词边界收缩与「纯标点」过滤）。"""
    if not char:
        return True
    if char.isspace():
        return True
    return unicodedata.category(char)[0] in ("P", "Z")


def _contains_punct_or_space(text: str) -> bool:
    return any(_is_punct_or_space(char) for char in text)


def _sensitive_reason(text: str) -> Optional[str]:
    """疑似敏感信息（URL/邮箱/手机号/数字）返回原因，否则 ``None``。"""
    if _URL_RE.search(text):
        return REASON_URL
    if _EMAIL_RE.search(text):
        return REASON_URL
    if _PHONE_RE.search(text) or _DIGIT_RE.search(text):
        return REASON_SENSITIVE
    return None


def _is_stopword(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    if value in _STOPWORDS:
        return True
    return value.lower() in _STOPWORDS_EN


def _simplified(text: str) -> str:
    """按内置常用字表做繁→简映射（仅用于繁简差异判断）。"""
    return "".join(_TRAD_TO_SIMP.get(char, char) for char in text)


def _same_script_variant(old: str, new: str) -> bool:
    """old / new 是否只是繁简字形差异（如 這個 → 这个）。"""
    return old != new and _simplified(old) == _simplified(new)


def _same_case_variant(old: str, new: str) -> bool:
    return old != new and old.lower() == new.lower()


def _same_space_variant(old: str, new: str) -> bool:
    return old != new and re.sub(r"\s+", "", old) == re.sub(r"\s+", "", new)


def _shrink_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """把片段向两侧收缩到不含标点/空白的词边界，返回新的下标区间。"""
    while start < end and _is_punct_or_space(text[start]):
        start += 1
    while end > start and _is_punct_or_space(text[end - 1]):
        end -= 1
    return start, end


_ASCII_WORD_CHAR = re.compile(r"[A-Za-z0-9]")


def _is_ascii_word_text(value: str) -> bool:
    """value 是否全部由 ASCII 字母/数字组成（英文单词片段）。"""
    return bool(value) and all(_ASCII_WORD_CHAR.match(char) for char in value)


def _expand_word_token(text: str, start: int, end: int) -> Tuple[int, int]:
    """把 ASCII 片段向外扩张成完整的「空白分隔词」（英文按词取整）。"""
    while start > 0 and _ASCII_WORD_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and _ASCII_WORD_CHAR.match(text[end]):
        end += 1
    return start, end


def _trim_common_edges(old: str, new: str) -> Tuple[str, str, int, int]:
    """裁掉 old/new 的公共前后缀，返回 ``(裁剪后 old, 裁剪后 new, 前缀长, 后缀长)``。"""
    prefix = 0
    limit = min(len(old), len(new))
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]
    ):
        suffix += 1
    return old[prefix : len(old) - suffix], new[prefix : len(new) - suffix], prefix, suffix


def _log_event(level: int, event: str, **fields: Any) -> None:
    """输出一行 JSON 日志（ensure_ascii=False），字段缺失用 0 补齐。"""
    candidates = fields.pop("candidates", None) or []
    filtered = fields.pop("filtered", None) or []
    payload: Dict[str, Any] = {
        "event": event,
        "timestamp": _now_iso(),
        "baseline_chars": int(fields.pop("baseline_chars", 0) or 0),
        "edited_chars": int(fields.pop("edited_chars", 0) or 0),
        "changed_chars": int(fields.pop("changed_chars", 0) or 0),
        "change_ratio": round(float(fields.pop("change_ratio", 0.0) or 0.0), 4),
        "opcode_count": int(fields.pop("opcode_count", 0) or 0),
        "candidates": [
            item.to_dict() if isinstance(item, LearnCandidate) else dict(item)
            for item in candidates
        ],
        "filtered": [dict(item) for item in filtered],
        "accepted": int(fields.pop("accepted", 0) or 0),
        "pending": int(fields.pop("pending", 0) or 0),
        "conflicts": int(fields.pop("conflicts", 0) or 0),
        "elapsed_ms": round(float(fields.pop("elapsed_ms", 0.0) or 0.0), 3),
    }
    payload.update(fields)
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # 理论上不会发生；兜底保证永远是合法单行 JSON
        line = json.dumps(
            {"event": event, "timestamp": payload["timestamp"], "log_error": True},
            ensure_ascii=False,
        )
    log.log(level, line)


def _log_filtered(reason: str, snippet: str, kind: str = "") -> None:
    """DEBUG 级记录单条被过滤片段（原因 + 片段，不含整段原文）。"""
    _log_event(
        logging.DEBUG,
        EVENT_FILTER,
        reason=reason,
        snippet=_snippet(snippet),
        kind=kind,
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


# ---------------------------------------------------------------------------
# 抽取候选
# ---------------------------------------------------------------------------
def _expand_single_char(
    base: str,
    edit: str,
    old: str,
    new: str,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
) -> Optional[Tuple[str, str, int, int, int, int]]:
    """把中文单字差异扩张成双字词，返回扩张后的 (old, new, i1, i2, j1, j2)。

    例：``我打开阴轨`` → ``我打开音轨`` 的差异是单字「阴→音」，
    扩张后得到「阴轨→音轨」；哪一侧更像常见词由 :data:`_COMMON_WORDS` 判断，
    两侧都无法判断时优先向右扩张；停用单字直接返回 ``None``（拒绝学习）。
    """
    if not old or not new or len(old) != 1 or len(new) != 1:
        return None
    if old in _SINGLE_STOP_CHARS or new in _SINGLE_STOP_CHARS:
        return None

    best: Optional[Tuple[int, str, str, int, int, int, int]] = None
    # 右侧：base[i2] / edit[j2] 必须是同一汉字
    if i2 < len(base) and j2 < len(edit) and base[i2] == edit[j2] and _is_cjk(base[i2]):
        right_old = old + base[i2]
        right_new = new + edit[j2]
        score = int(right_old in _COMMON_WORDS) + int(right_new in _COMMON_WORDS)
        best = (score, right_old, right_new, i1, i2 + 1, j1, j2 + 1)
    # 左侧：base[i1-1] / edit[j1-1] 必须是同一汉字
    if i1 > 0 and j1 > 0 and base[i1 - 1] == edit[j1 - 1] and _is_cjk(base[i1 - 1]):
        left_old = base[i1 - 1] + old
        left_new = edit[j1 - 1] + new
        score = int(left_old in _COMMON_WORDS) + int(left_new in _COMMON_WORDS)
        if best is None or score > best[0]:
            best = (score, left_old, left_new, i1 - 1, i2, j1 - 1, j2)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5], best[6]


def _replacement_reason(old: str, new: str, max_len: int) -> Optional[str]:
    """replace 候选的过滤原因；``None`` 表示通过。"""
    if not old or not new:
        return REASON_EMPTY
    if old == new:
        return REASON_EMPTY
    if len(old) <= 1 or len(new) <= 1:
        return REASON_SINGLE
    sensitive = _sensitive_reason(old) or _sensitive_reason(new)
    if sensitive:
        return sensitive
    if _contains_punct_or_space(old) or _contains_punct_or_space(new):
        return REASON_PUNCT
    if _same_script_variant(old, new):
        return REASON_SCRIPT
    if _same_case_variant(old, new):
        return REASON_CASE
    if _same_space_variant(old, new):
        return REASON_SPACE
    if _is_stopword(old) or _is_stopword(new):
        return REASON_STOPWORD
    if len(old) > max_len or len(new) > max_len:
        return REASON_LONG
    return None


def _delete_reason(old: str, max_len: int, edit: str, j1: int) -> Optional[str]:
    """delete 候选的过滤原因；``None`` 表示通过。"""
    if not old:
        return REASON_EMPTY
    sensitive = _sensitive_reason(old)
    if sensitive:
        return sensitive
    if _contains_punct_or_space(old):
        return REASON_PUNCT
    if _is_stopword(old):
        return REASON_STOPWORD
    if len(old) < 2:
        return REASON_SINGLE
    if len(old) > max_len:
        return REASON_LONG
    # 删除后左右两侧若都变成标点/空白，句子结构会被破坏
    prev = edit[j1 - 1] if j1 > 0 else ""
    nxt = edit[j1] if j1 < len(edit) else ""
    if prev and nxt and _is_punct_or_space(prev) and _is_punct_or_space(nxt):
        return REASON_DELETE_JOIN
    if prev and nxt and prev in _SENTENCE_END and nxt in _SENTENCE_END:
        return REASON_DELETE_JOIN
    return None


def _finish_extraction(
    result: ExtractionResult,
    started: float,
    *,
    big_change: bool = False,
) -> ExtractionResult:
    """补齐耗时并按统一格式输出 extract 事件。"""
    result.elapsed_ms = _elapsed_ms(started)
    _log_event(
        logging.INFO,
        EVENT_EXTRACT,
        baseline_chars=result.baseline_chars,
        edited_chars=result.edited_chars,
        changed_chars=result.changed_chars,
        change_ratio=result.change_ratio,
        opcode_count=result.opcode_count,
        candidates=list(result.candidates),
        filtered=list(result.filtered),
        accepted=0,
        pending=0,
        conflicts=0,
        candidate_count=len(result.candidates),
        filtered_count=len(result.filtered),
        big_change=bool(big_change),
        elapsed_ms=result.elapsed_ms,
    )
    return result


def extract_candidates(
    before: str,
    after: str,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    max_changed_chars: int = DEFAULT_MAX_CHANGED_CHARS,
    max_change_ratio: float = DEFAULT_MAX_CHANGE_RATIO,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ExtractionResult:
    """比较基线文本与用户修改后的文本，抽取「错误词 → 正确词」候选。

    参数
    ----
    before:
        直出模式最终稿基线（用户开始编辑前看到的文本）。
    after:
        用户编辑后的文本。
    max_len:
        候选词最长字符数，超长片段视为改写而非词语纠正。
    max_changed_chars / max_change_ratio:
        整段改写阈值；变更字符数 > ``max_changed_chars`` **或**
        变更比例 > ``max_change_ratio`` 时直接放弃学习。
    max_candidates:
        单次最多产出的候选数，超出部分进入 ``filtered``（原因「候选超过上限」）。

    返回
    ----
    :class:`ExtractionResult`：候选、过滤明细与统计信息；不会抛异常。
    """
    started = time.perf_counter()
    base = _normalize(before)
    edit = _normalize(after)
    result = ExtractionResult(baseline_chars=len(base), edited_chars=len(edit))

    # ---- 空文本 / 无变化：不学习，但仍然记一条汇总日志 ----
    if not base or not edit or base == edit:
        reason = ""
        if not base and not edit:
            reason = "基线为空"
        elif not base:
            reason = "基线为空"
        elif not edit:
            reason = "编辑后为空"
        if reason:
            result.filtered.append({"reason": reason, "snippet": ""})
            _log_filtered(reason, "", "extract")
        return _finish_extraction(result, started)

    # ---- 字符级对齐 ----
    try:
        matcher = difflib.SequenceMatcher(None, base, edit, autojunk=False)
        opcodes = matcher.get_opcodes()
    except Exception:  # pragma: no cover - difflib 理论上不会抛，这里防御未知文本
        result.filtered.append({"reason": "对齐失败", "snippet": ""})
        return _finish_extraction(result, started)

    change_ops = [op for op in opcodes if op[0] != "equal"]
    changed_chars = 0
    for tag, i1, i2, j1, j2 in change_ops:
        if tag == "replace":
            changed_chars += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            changed_chars += i2 - i1
        else:  # insert
            changed_chars += j2 - j1
    result.changed_chars = changed_chars
    result.opcode_count = len(change_ops)
    result.change_ratio = (changed_chars / len(base)) if base else 0.0

    # ---- 整段重写快速拒绝（宁缺毋滥） ----
    if changed_chars > max_changed_chars or result.change_ratio > max_change_ratio:
        reason = (
            f"{REASON_BIG_CHANGE}拒绝：变更 {changed_chars} 字，"
            f"比例 {result.change_ratio:.0%}，超过阈值"
        )
        result.filtered.append({"reason": reason, "snippet": ""})
        _log_filtered(reason, "", "extract")
        return _finish_extraction(result, started, big_change=True)

    # ---- 逐段处理 ----
    seen_candidates: set = set()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        # 插入不是「错词纠正」，不学习，仅记录
        if tag == "insert":
            reason = REASON_INSERT
            inserted = edit[j1:j2]
            result.filtered.append({"reason": reason, "snippet": _snippet(inserted)})
            _log_filtered(reason, inserted, "insert")
            continue

        raw_old = base[i1:i2]
        raw_new = edit[j1:j2] if tag == "replace" else ""

        # 1) 裁掉公共前后缀（SequenceMatcher 偶尔会把公共字包进变更片段）
        if tag == "replace" and raw_old and raw_new:
            old, new, prefix, suffix = _trim_common_edges(raw_old, raw_new)
            if prefix or suffix:
                i1 += prefix
                i2 -= suffix
                j1 += prefix
                j2 -= suffix
                raw_old = base[i1:i2]
                raw_new = edit[j1:j2]
        else:
            old, new = raw_old, raw_new

        # 2) 收缩到不含标点/空白的词边界（old/new 两侧分别收缩）
        if tag == "replace":
            old_start, old_end = _shrink_span(base, i1, i2)
            new_start, new_end = _shrink_span(edit, j1, j2)
            i1, i2, j1, j2 = old_start, old_end, new_start, new_end
            old = base[i1:i2]
            new = edit[j1:j2]
        else:
            old_start, old_end = _shrink_span(base, i1, i2)
            i1, i2 = old_start, old_end
            old = base[i1:i2]
            new = ""

        # 纯标点/空白变化：收缩后片段为空
        if not old and not new:
            reason = REASON_PUNCT
            result.filtered.append({"reason": reason, "snippet": _snippet(raw_old)})
            _log_filtered(reason, raw_old, tag)
            continue

        # 2.5) 英文按空白分词取整：把 recieve→receive 的字符级片段扩张成完整单词
        if tag == "replace" and _is_ascii_word_text(old) and _is_ascii_word_text(new):
            i1, i2 = _expand_word_token(base, i1, i2)
            j1, j2 = _expand_word_token(edit, j1, j2)
            old = base[i1:i2]
            new = edit[j1:j2]
        elif tag == "delete" and _is_ascii_word_text(old):
            # 只接受「整词删除」；删词里的几个字母不是词语纠正
            expanded_start, expanded_end = _expand_word_token(base, i1, i2)
            if (expanded_start, expanded_end) != (i1, i2):
                reason = "英文删除片段不完整"
                result.filtered.append(
                    {"reason": reason, "snippet": _snippet(raw_old)}
                )
                _log_filtered(reason, raw_old, tag)
                continue
            i1, i2 = expanded_start, expanded_end
            old = base[i1:i2]

        # 3) 中文单字差异尝试扩张成双字词（阴→音 => 阴轨→音轨）
        if tag == "replace" and len(old) == 1 and len(new) == 1:
            expanded = _expand_single_char(base, edit, old, new, i1, i2, j1, j2)
            if expanded is None:
                reason = REASON_SINGLE
                result.filtered.append(
                    {"reason": reason, "snippet": _snippet(f"{old}→{new}")}
                )
                _log_filtered(reason, f"{old}→{new}", tag)
                continue
            old, new, i1, i2, j1, j2 = expanded

        # 4) 规则过滤
        if tag == "replace":
            reason = _replacement_reason(old, new, max_len)
        else:
            reason = _delete_reason(old, max_len, edit, j1)
        if reason:
            snippet = f"{old}→{new}" if tag == "replace" else old
            result.filtered.append({"reason": reason, "snippet": _snippet(snippet)})
            _log_filtered(reason, snippet, tag)
            continue

        # 5) 候选数量上限
        if len(result.candidates) >= max_candidates:
            reason = f"{REASON_LIMIT}（最多 {max_candidates} 条）"
            result.filtered.append({"reason": reason, "snippet": _snippet(old)})
            _log_filtered(reason, old, tag)
            continue

        # 6) 去重（多个字符级片段可能扩张成同一对词语，如 audio→video）
        kind = "replace" if tag == "replace" else "delete"
        key = (old, new if tag == "replace" else "", kind)
        if key in seen_candidates:
            _log_filtered("重复候选", old, tag)
            continue
        seen_candidates.add(key)

        # 7) 生成候选
        confidence = (
            CONFIDENCE_REPLACE if tag == "replace" else CONFIDENCE_DELETE
        )
        result.candidates.append(
            LearnCandidate(
                wrong=old,
                correct=new if tag == "replace" else "",
                kind=kind,
                confidence=confidence,
                context=_make_context(base, i1, i2),
            )
        )

    return _finish_extraction(result, started)


# ---------------------------------------------------------------------------
# 手动术语表解析
# ---------------------------------------------------------------------------
def parse_manual_glossary(text: str) -> List[GlossaryEntry]:
    """把设置页的手动术语表文本解析成结构化条目。

    格式（每行一条）：

    * ``正确写法：常见误写``（冒号可用中文/英文）；
    * 右侧可用「、，,;；|」分隔多个误写，各生成一条；
    * 没有冒号的行视为「只有正确写法」的词条（``wrong`` 为空，仅参与 prompt）；
    * ``#`` / ``//`` 开头的行是注释；空行忽略。

    返回的条目 ``source="manual"``、``enabled=True``、``confidence=1.0``、``count=1``。
    """
    entries: List[GlossaryEntry] = []
    seen: set = set()
    now = _now_iso()

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if "：" in line:
            correct_part, wrong_part = line.split("：", 1)
        elif ":" in line:
            correct_part, wrong_part = line.split(":", 1)
        else:
            correct_part, wrong_part = line, ""
        correct = correct_part.strip()
        wrong_part = wrong_part.strip()
        if not correct and not wrong_part:
            continue

        wrongs = [item.strip() for item in re.split(r"[、，,;；|]", wrong_part)]
        wrongs = [item for item in wrongs if item]
        if not wrongs:
            # 只有正确写法：仅用于 prompt 术语表
            key = ("", correct)
            if not correct or key in seen:
                continue
            seen.add(key)
            entries.append(
                GlossaryEntry(
                    wrong="",
                    correct=correct,
                    kind="replace",
                    source="manual",
                    enabled=True,
                    confidence=1.0,
                    count=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            continue

        if not correct:
            _log_filtered("手动术语表缺少正确写法", line, "manual")
            continue
        for wrong in wrongs:
            if wrong == correct:
                continue
            key = (wrong, correct)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                GlossaryEntry(
                    wrong=wrong,
                    correct=correct,
                    kind="replace",
                    source="manual",
                    enabled=True,
                    confidence=1.0,
                    count=1,
                    created_at=now,
                    updated_at=now,
                )
            )
    return entries


# ---------------------------------------------------------------------------
# 合并候选
# ---------------------------------------------------------------------------
def _new_learned_entry(
    candidate: LearnCandidate, now: str, *, enabled: bool = False
) -> GlossaryEntry:
    return GlossaryEntry(
        wrong=candidate.wrong,
        correct=candidate.correct if candidate.kind == "replace" else "",
        kind=candidate.kind,
        source="learned",
        enabled=enabled,
        confidence=candidate.confidence,
        count=1,
        created_at=now,
        updated_at=now,
    )


def _conflict_event(
    *,
    wrong: str,
    kept_correct: str,
    rejected_correct: str,
    reason: str,
    candidate: Optional[LearnCandidate] = None,
) -> None:
    _log_event(
        logging.WARNING,
        EVENT_CONFLICT,
        candidates=[candidate] if candidate else [],
        conflicts=1,
        wrong=wrong,
        kept_correct=kept_correct,
        rejected_correct=rejected_correct,
        reason=reason,
    )


def merge_candidates(
    entries: Sequence[GlossaryEntry],
    candidates: Iterable[LearnCandidate],
    *,
    manual_entries: Optional[Iterable[GlossaryEntry]] = None,
    auto_apply_threshold: float = DEFAULT_AUTO_APPLY_THRESHOLD,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> Tuple[List[GlossaryEntry], List[GlossaryEntry], List[GlossaryEntry]]:
    """把新候选合并进词语表，返回 ``(entries, pending, conflicts)``。

    返回语义
    --------
    * ``entries``：合并后的完整条目表（手动 + 学习；含待确认条目，``enabled=False``），
      可直接交给 :func:`save_store`；
    * ``pending``：``entries`` 中仍待用户确认的学习条目（``enabled=False``），
      供界面展示；
    * ``conflicts``：本次冲突中被淘汰/被覆盖的条目（均为 ``enabled=False``），
      供界面展示，不会被静默覆盖。

    规则（与设计文档第七节一致）
    ----------------------------
    * 相同 wrong + 相同 correct：累加 ``count``、提升 ``confidence``、刷新时间；
    * 相同 wrong + 不同 correct：手动条目优先；都是学习条目则保留原有条目，
      新候选进入 ``conflicts``；
    * 删除规则 vs 替换规则：替换优先；被淘汰的删除规则进入 ``conflicts``；
    * 只对 ``replace`` 且在 ``confidence >= auto_apply_threshold`` 时自动启用；
      ``delete`` 规则永远只进待确认，不自动启用；
    * 手动条目覆盖同 wrong 的学习条目（学习条目移入 ``conflicts``，不静默覆盖）。

    参数语义
    --------
    * ``manual_entries``：传 ``None`` 表示沿用条目表里 ``source="manual"`` 的旧条目；
      传列表（含空列表）表示以本次解析的手动术语表为准——设置页删掉的术语不会
      因为旧存储而“复活”，这是手动文本最高优先级的体现；
    * ``max_entries``：启用中的学习条目超过该数量时，按
      「更新时间 + 置信度」从低到高淘汰（被淘汰条目移入 ``conflicts``）。
    """
    started = time.perf_counter()
    now = _now_iso()
    source_entries = [item for item in list(entries or []) if isinstance(item, GlossaryEntry)]
    manual_provided = manual_entries is not None
    manual_source = [
        item for item in list(manual_entries or []) if isinstance(item, GlossaryEntry)
    ]
    input_candidates = [
        item for item in list(candidates or []) if isinstance(item, LearnCandidate)
    ]

    accepted = 0
    updated = 0
    pruned = 0
    conflicts: List[GlossaryEntry] = []

    # ---- 1. 手动条目（优先级最高；源文本条目覆盖旧的手动条目） ----
    manual_out: List[GlossaryEntry] = []
    manual_map: Dict[str, GlossaryEntry] = {}

    def _add_manual(entry: GlossaryEntry) -> None:
        copy = dataclass_replace(entry)
        copy.source = "manual"
        if not copy.created_at:
            copy.created_at = now
        if not copy.updated_at:
            copy.updated_at = now
        key = copy.wrong
        if key and key in manual_map:
            return  # 同一 wrong 的手动条目保留第一个
        manual_out.append(copy)
        if key:
            manual_map[key] = copy

    for entry in manual_source:
        _add_manual(entry)
    if not manual_provided:
        # 没本次解析结果时，沿用条目表里已存的手动条目
        for entry in source_entries:
            if entry.source == "manual":
                _add_manual(entry)

    # ---- 2. 已有学习条目（复制，避免修改调用方对象） ----
    learned_out: List[GlossaryEntry] = []
    learned_map: Dict[str, GlossaryEntry] = {}
    for entry in source_entries:
        if entry.source == "manual":
            continue
        copy = dataclass_replace(entry)
        if copy.wrong in learned_map:
            continue  # 历史脏数据：同 wrong 保留第一个
        learned_out.append(copy)
        learned_map[copy.wrong] = copy

    # 手动条目覆盖同 wrong 的学习条目：学习条目移入冲突区（不静默覆盖）
    for entry in list(learned_out):
        manual = manual_map.get(entry.wrong)
        if manual is None:
            continue
        learned_out.remove(entry)
        del learned_map[entry.wrong]
        displaced = dataclass_replace(entry)
        displaced.enabled = False
        conflicts.append(displaced)
        _log_event(
            logging.DEBUG,
            EVENT_CONFLICT,
            conflicts=1,
            wrong=entry.wrong,
            kept_correct=manual.correct,
            rejected_correct=entry.correct,
            reason="手动条目覆盖学习条目",
        )

    # ---- 3. 逐个合并新候选 ----
    for candidate in input_candidates:
        if not candidate.wrong:
            _log_filtered("候选缺少错误词", "", "merge")
            continue
        if candidate.kind == "replace" and not candidate.correct:
            _log_filtered("替换候选缺少正确词", candidate.wrong, "merge")
            continue
        if candidate.wrong == candidate.correct:
            continue

        # 3.1 手动条目优先
        manual = manual_map.get(candidate.wrong)
        if manual is not None:
            if manual.kind == candidate.kind and manual.correct == candidate.correct:
                manual.count += 1
                manual.updated_at = now
                updated += 1
            else:
                conflicts.append(_new_learned_entry(candidate, now, enabled=False))
                _conflict_event(
                    wrong=candidate.wrong,
                    kept_correct=manual.correct,
                    rejected_correct=candidate.correct,
                    reason="手动条目优先",
                    candidate=candidate,
                )
            continue

        existing = learned_map.get(candidate.wrong)

        # 3.2 新条目
        if existing is None:
            entry = _new_learned_entry(candidate, now)
            auto = (
                candidate.kind == "replace"
                and entry.confidence >= float(auto_apply_threshold)
            )
            entry.enabled = auto
            learned_out.append(entry)
            learned_map[entry.wrong] = entry
            if auto:
                accepted += 1
            continue

        # 3.3 同 wrong + 同 kind + 同 correct：重复累加
        if existing.kind == candidate.kind and existing.correct == candidate.correct:
            existing.count += 1
            base_conf = max(float(existing.confidence), float(candidate.confidence))
            existing.confidence = min(1.0, round(base_conf + CONFIDENCE_BUMP, 2))
            existing.updated_at = now
            if (
                not existing.enabled
                and existing.kind == "replace"
                and existing.confidence >= float(auto_apply_threshold)
            ):
                existing.enabled = True
                accepted += 1
            elif existing.enabled:
                updated += 1
            continue

        # 3.4 删除规则 vs 替换规则：替换优先（更具体）
        if existing.kind != candidate.kind:
            if candidate.kind == "replace" and existing.kind == "delete":
                learned_out.remove(existing)
                del learned_map[existing.wrong]
                displaced = dataclass_replace(existing)
                displaced.enabled = False
                conflicts.append(displaced)
                entry = _new_learned_entry(candidate, now)
                entry.enabled = (
                    candidate.kind == "replace"
                    and entry.confidence >= float(auto_apply_threshold)
                )
                learned_out.append(entry)
                learned_map[entry.wrong] = entry
                if entry.enabled:
                    accepted += 1
                _conflict_event(
                    wrong=candidate.wrong,
                    kept_correct=entry.correct,
                    rejected_correct=existing.correct,
                    reason="替换规则优先于删除规则",
                    candidate=candidate,
                )
            else:
                conflicts.append(_new_learned_entry(candidate, now, enabled=False))
                _conflict_event(
                    wrong=candidate.wrong,
                    kept_correct=existing.correct,
                    rejected_correct=candidate.correct,
                    reason="删除规则不自动覆盖替换规则",
                    candidate=candidate,
                )
            continue

        # 3.5 同 wrong + 不同 correct：保留已有（count 更高/先到），新候选进冲突
        conflicts.append(_new_learned_entry(candidate, now, enabled=False))
        _conflict_event(
            wrong=candidate.wrong,
            kept_correct=existing.correct,
            rejected_correct=candidate.correct,
            reason="同词不同纠正，保留已有条目",
            candidate=candidate,
        )

    # ---- 4. 条目数量上限：淘汰「最久未用 + 置信度最低」的启用条目 ----
    if max_entries and max_entries > 0:
        enabled_learned = [item for item in learned_out if item.enabled]
        if len(enabled_learned) > max_entries:
            ordered = sorted(
                enabled_learned,
                key=lambda item: (
                    item.updated_at or "",
                    float(item.confidence),
                    int(item.count),
                ),
            )
            for entry in ordered[: len(enabled_learned) - max_entries]:
                if entry in learned_out:
                    learned_out.remove(entry)
                learned_map.pop(entry.wrong, None)
                pruned += 1
                displaced = dataclass_replace(entry)
                displaced.enabled = False
                conflicts.append(displaced)
                _log_filtered("条目数量超限，已淘汰", entry.wrong, "merge")

    all_entries = manual_out + learned_out
    pending = [item for item in learned_out if not item.enabled]

    _log_event(
        logging.INFO,
        EVENT_MERGE,
        candidates=input_candidates,
        accepted=accepted,
        pending=len(pending),
        conflicts=len(conflicts),
        entries_before=len(source_entries),
        entries_after=len(all_entries),
        manual_count=len(manual_out),
        learned_count=len(learned_out),
        updated=updated,
        pruned=pruned,
        auto_apply_threshold=float(auto_apply_threshold),
        elapsed_ms=_elapsed_ms(started),
    )
    return all_entries, pending, conflicts


# ---------------------------------------------------------------------------
# 存储：data/glossary.json（原子写 + .bak）
# ---------------------------------------------------------------------------
def store_path() -> Path:
    """默认存储路径：与 ``config.json`` 同目录下的 ``glossary.json``。"""
    return config.config_path().parent / STORE_FILENAME


def load_store(path: Optional[Path] = None) -> List[GlossaryEntry]:
    """读取词语表；文件不存在返回空列表，损坏文件备份为 ``.broken`` 后返回空列表。"""
    target = Path(path) if path else store_path()
    try:
        if not target.exists():
            return []
        raw_text = target.read_text(encoding="utf-8")
    except OSError as exc:
        _log_event(logging.WARNING, "store_load", mode="error", error=_snippet(str(exc), 80))
        return []

    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        backup = target.with_suffix(target.suffix + ".broken")
        try:
            target.replace(backup)
        except OSError:
            pass
        _log_event(
            logging.WARNING,
            "store_load",
            mode="corrupt",
            error=_snippet(str(exc), 80),
            backup=str(backup),
        )
        return []

    if isinstance(payload, dict):
        items = payload.get("entries")
    else:  # 兼容直接存成数组的早期格式
        items = payload
    if not isinstance(items, list):
        return []

    entries: List[GlossaryEntry] = []
    for item in items:
        try:
            entry = GlossaryEntry.from_dict(item)
        except (TypeError, ValueError, KeyError):
            continue
        if entry.wrong or entry.correct:
            entries.append(entry)
    return entries


def save_store(entries: Sequence[GlossaryEntry], path: Optional[Path] = None) -> Path:
    """原子写入词语表（``.json`` + 旧文件备份 ``.json.bak``），返回实际路径。"""
    target = Path(path) if path else store_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STORE_VERSION,
        "updated_at": _now_iso(),
        "entries": [
            item.to_dict() if isinstance(item, GlossaryEntry) else dict(item)
            for item in (entries or [])
        ],
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=".glossary-", suffix=".json", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        if target.exists():
            backup = target.with_suffix(target.suffix + ".bak")
            try:
                shutil.copy2(target, backup)
            except OSError:
                pass
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    return target


# ---------------------------------------------------------------------------
# prompt 术语表 / 本地替换
# ---------------------------------------------------------------------------
def build_prompt_glossary(entries: Sequence[GlossaryEntry]) -> str:
    """生成发给 DeepSeek 的术语表文本（格式「正确：错误」）。

    只包含 ``enabled`` 的 ``replace`` 条目；只有正确写法的条目原样输出一行；
    按 wrong 去重，空条目跳过。没有任何条目时返回空字符串。
    """
    lines: List[str] = []
    seen: set = set()
    for entry in entries or []:
        if not isinstance(entry, GlossaryEntry):
            continue
        if not entry.enabled or entry.kind != "replace":
            continue
        correct = (entry.correct or "").strip()
        wrong = (entry.wrong or "").strip()
        if not correct:
            continue
        if wrong:
            if wrong in seen:
                continue
            seen.add(wrong)
            lines.append(f"{correct}：{wrong}")
        else:
            if correct in seen:
                continue
            seen.add(correct)
            lines.append(correct)
    return "\n".join(lines)


def _is_ascii_word(value: str) -> bool:
    """纯 ASCII 字母/数字规则（英文按单词边界匹配，避免替换词内子串）。"""
    return bool(value) and bool(re.fullmatch(r"[A-Za-z0-9]+", value))


def _build_replacement_map(
    entries: Sequence[GlossaryEntry],
) -> Tuple[Dict[str, str], List[List[str]]]:
    """收集启用规则并做一次传递闭包，返回 ``(映射, 环列表)``。

    例：A→B、B→C 的闭包结果是 A→C、B→C；检测到 A→B、B→A 这类环时，
    环上条目全部跳过并在日志里报 conflict（不会让替换陷入死循环）。
    """
    raw: Dict[str, str] = {}
    for entry in entries or []:
        if not isinstance(entry, GlossaryEntry) or not entry.enabled:
            continue
        wrong = (entry.wrong or "").strip()
        if not wrong:
            continue
        if entry.kind == "replace":
            correct = (entry.correct or "").strip()
            if not correct or correct == wrong:
                continue
            raw[wrong] = correct
        elif entry.kind == "delete":
            raw[wrong] = ""

    resolved: Dict[str, str] = {}
    cycles: List[List[str]] = []
    for word in raw:
        seen: set = set()
        path: List[str] = []
        current = word
        cyclic = False
        while current in raw:
            if current in seen:
                start = path.index(current)
                cycle = path[start:]
                if cycle and frozenset(cycle) not in [frozenset(item) for item in cycles]:
                    cycles.append(cycle)
                cyclic = True
                break
            seen.add(current)
            path.append(current)
            current = raw[current]
        if cyclic:
            continue
        if current != word:
            resolved[word] = current

    # 闭包后可能产生 A→A（A→B、B→A 已被环过滤；A→B、B→A 不会到这里）
    resolved = {key: value for key, value in resolved.items() if key != value}
    return resolved, cycles


def apply_replacements(text: str, entries: Sequence[GlossaryEntry]) -> str:
    """在本地应用词语表的替换/删除规则，返回处理后的文本。

    * 只应用 ``enabled`` 条目；``replace`` 把 wrong 换成 correct，``delete`` 删除 wrong；
    * 链式映射先做传递闭包（A→B、B→C 等价于 A→C、B→C），一遍扫描生效；
    * 检测到环（A→B、B→A）时跳过环上规则并记录 conflict 日志，绝不死循环；
    * 纯 ASCII 规则按单词边界匹配（``cat`` 不会命中 ``category``），中文按子串匹配；
    * 每次调用都会输出一条 apply 事件日志（只含计数/长度，不含正文）。
    """
    started = time.perf_counter()
    original = text or ""
    if not original:
        _log_event(logging.INFO, EVENT_APPLY, elapsed_ms=_elapsed_ms(started))
        return original

    mapping, cycles = _build_replacement_map(entries)
    if cycles:
        _log_event(
            logging.WARNING,
            EVENT_CONFLICT,
            conflicts=len(cycles),
            cycle_words=[list(cycle) for cycle in cycles],
            reason="检测到循环映射，已跳过",
            rules_used=len(mapping),
        )

    if not mapping:
        _log_event(
            logging.INFO,
            EVENT_APPLY,
            baseline_chars=len(original),
            edited_chars=len(original),
            rules_used=0,
            elapsed_ms=_elapsed_ms(started),
        )
        return original

    keys = sorted(mapping, key=lambda item: (-len(item), item))
    alternatives: List[str] = []
    for key in keys:
        escaped = re.escape(key)
        if _is_ascii_word(key):
            escaped = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        alternatives.append(escaped)
    pattern = re.compile("|".join(alternatives))

    stats = {"replacements": 0, "deletions": 0, "matched_chars": 0}

    def _replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        target = mapping.get(word, word)
        stats["replacements"] += 1
        stats["matched_chars"] += len(word)
        if target == "":
            stats["deletions"] += 1
        return target

    result = pattern.sub(_replace, original)
    matched_chars = int(stats["matched_chars"])
    _log_event(
        logging.INFO,
        EVENT_APPLY,
        baseline_chars=len(original),
        edited_chars=len(result),
        changed_chars=matched_chars,
        change_ratio=(matched_chars / len(original)) if original else 0.0,
        opcode_count=int(stats["replacements"]),
        replacements=int(stats["replacements"]),
        deletions=int(stats["deletions"]),
        rules_used=len(mapping),
        cycles=len(cycles),
        elapsed_ms=_elapsed_ms(started),
    )
    return result


def undo_candidates(
    entries: Sequence[GlossaryEntry],
    candidates: Iterable[LearnCandidate],
) -> List[GlossaryEntry]:
    """撤销最近一次学习：删除与候选完全匹配的「学习」条目。

    * 手动条目（``source="manual"``）永远保留；
    * 返回新的条目列表，并输出一条 ``undo`` 事件日志（供界面「撤销」按钮使用）。
    """
    started = time.perf_counter()
    targets = {
        (item.wrong, item.correct, item.kind)
        for item in (candidates or [])
        if isinstance(item, LearnCandidate)
    }
    source = [item for item in (entries or []) if isinstance(item, GlossaryEntry)]
    kept: List[GlossaryEntry] = []
    removed = 0
    for entry in source:
        if entry.source == "learned" and (
            entry.wrong,
            entry.correct,
            entry.kind,
        ) in targets:
            removed += 1
            continue
        kept.append(entry)
    _log_event(
        logging.INFO,
        EVENT_UNDO,
        undone=removed,
        entries_before=len(source),
        entries_after=len(kept),
        elapsed_ms=_elapsed_ms(started),
    )
    return kept


__all__ = [
    "LOGGER_NAME",
    "STORE_VERSION",
    "STORE_FILENAME",
    "DEFAULT_MAX_LEN",
    "DEFAULT_MAX_CHANGED_CHARS",
    "DEFAULT_MAX_CHANGE_RATIO",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_AUTO_APPLY_THRESHOLD",
    "DEFAULT_MAX_ENTRIES",
    "CONFIDENCE_REPLACE",
    "CONFIDENCE_DELETE",
    "CONFIDENCE_BUMP",
    "EVENT_EXTRACT",
    "EVENT_MERGE",
    "EVENT_APPLY",
    "EVENT_CONFLICT",
    "EVENT_UNDO",
    "EVENT_FILTER",
    "GlossaryEntry",
    "LearnCandidate",
    "ExtractionResult",
    "extract_candidates",
    "parse_manual_glossary",
    "merge_candidates",
    "load_store",
    "save_store",
    "store_path",
    "build_prompt_glossary",
    "apply_replacements",
    "undo_candidates",
]

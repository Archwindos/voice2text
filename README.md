# 语声成文 VoxScribe

> 说出来，就是文字。点一下（或按 **F2**）说一段话，几秒后得到整理好的、可直接复制的书面文字。
> Windows 桌面小工具 · 本地转写 · DeepSeek 校对 · 轻量离线可用。

<p align="center">
  <img src="docs/home-result.png" alt="语声成文 · 语音输入页" width="900">
</p>

---

## 一、它解决什么问题？

**不想打字的时候，直接说。** 口述笔记、聊天回复、会议发言、灵感记录……
说一段话，软件自动完成：

```
按 F2 说话  →  本地语音转写  →  DeepSeek 校对（去口头禅、改错别字、补标点）  →  一键复制
                 （音频不上传）        （只上传文字）
```

同时保留了一个 **文件转写** 页：会议录音、访谈、课堂、视频等长音频可以批量转写，
自动整理成 txt / Markdown / Word / SRT 字幕。

## 二、关于「DeepSeek 的语音模型」

DeepSeek 官方 API 目前**只提供文本模型**（`deepseek-chat`、`deepseek-reasoner`），
没有公开的语音识别（ASR）接口；DeepSeek App 里的语音输入使用的是第三方技术，不对外开放。

所以本项目这样分工，各取所长：

| 环节 | 用什么 | 为什么 |
| --- | --- | --- |
| 语音 → 文字 | **本地 faster-whisper**（OpenAI Whisper 的 CTranslate2 加速版） | 免费、离线、隐私；CPU int8 量化，轻量 |
| 文字 → 书面语 | **DeepSeek API（deepseek-chat）** | 中文书面表达质量高、能纠错别字、价格低 |

> 如果以后想接入云端语音识别，`voxscribe/asr.py` 是独立模块，替换掉即可。

## 三、功能特性

### 🎤 首页「语音输入」——主打功能

- **一键说话**：点大按钮或按 **F2** 开始录音，再按一次结束，自动转写 + 校对；
- **实时语音波形**：说话时能看到波形随音量跳动，不用猜「到底有没有录进去」；
- **自动整理**：默认走 DeepSeek「口述校对」——去掉"嗯、那个、就是说"，修正同音错别字、补标点、分段落；
- **结果可编辑**：转换完成的文字可以直接修改，再点「复制文字」或按 **Ctrl+Enter**；
- **自动复制**：可勾选「完成后自动复制到剪贴板」，说完直接去别处 Ctrl+V；
- **没有 Key 也能用**：自动降级为只做本地转写，完全免费、离线；
- **快捷键**：`F2` 开始/结束，`Esc` 取消录音或转换，`Ctrl+Enter` 复制结果。

### 📁 第二页「文件转写」

- 批量添加 / 拖拽音频（mp3、wav、m4a、aac、flac、ogg、opus、mp4、mov、mkv 等）；
- 4 种整理风格 + 口述校对，共 **5 种**：
  - **口述校对**：口语去掉口头禅、纠错别字、补标点（语音输入默认）
  - **口述成文**：口语转成通顺书面语，按语义分段
  - **会议纪要**：结构化输出议题、结论、待办事项
  - **逐字稿润色**：保留原话，只做最小清洗
  - **要点摘要**：压缩成要点列表
- **术语表**：人名、公司名、专业词按指定写法校正同音错字；
- 导出 txt / Markdown / Word（docx）/ SRT 字幕，附一份带时间戳的原始转写；
- 批量队列、进度、取消、失败跳过；
- 输出目录可选，重名自动加 `(2)` 不覆盖。

### 🪶 轻量与隐私

- 无 Electron / 无浏览器内核 / 无 PyTorch / 无 ffmpeg 可执行文件；
- CPU 默认最多 4 线程，不跑满机器；空闲 5 分钟自动卸载识别模型；
- 模型按需选择：tiny 75MB / base 145MB / small 484MB / medium 1.5GB；
- **实测**（本机 20 核 CPU、31GB 内存）：
  - 界面空载约 **55 MB** 内存；
  - small 模型转写峰值约 **406 MB**；medium 约 **937 MB**；空闲卸载后回落到约 100MB；
  - 13 秒音频：small 约 1.5 秒转写，medium 约 4 秒（首次加载另需 2~6 秒）；
- 语音识别全程在本机完成，**音频不上传**；只有 AI 整理时上传**文字**；
- API Key 用 Windows DPAPI 加密保存在本机，日志不记录 Key；
- 断网仍可使用「只转写」；模型只需下载一次。

## 四、快速开始

### 1. 启动

双击 **`run.bat`**。（依赖和识别模型已在本机装好、下好，无需再安装。）

> 如果换了电脑或重装：先双击 `setup.bat` 安装依赖（约 5-10 分钟），模型可在设置里选择后自动下载。

### 2. 配置 DeepSeek API Key（AI 整理需要）

1. 打开 [platform.deepseek.com](https://platform.deepseek.com/) → **API Keys** → 创建 Key（`sk-` 开头，只显示一次，请复制）；
2. 在软件里点右上角 **设置** → 粘贴 API Key → 点 **测试连接** → 保存。
   （本机已经配置好了，可跳过；想更换再操作。）

> 不配置 Key 也能用：软件会自动改为只做本地转写，完全免费、离线。

### 3. 开始说

1. 点击大按钮「**● 开始录音　F2**」，或直接按 **F2**；
2. 对着麦克风说话，波形会实时跳动；
3. 再点一次 / 再按 **F2** 结束录音；
4. 等 2~5 秒（会显示「正在转写… / DeepSeek 整理中…」），文字出现在下方；
5. 自动复制已开启时直接去粘贴；也可以修改后再点「复制文字」。

## 五、使用技巧与快捷键

| 操作 | 快捷键 / 位置 |
| --- | --- |
| 开始 / 结束录音 | `F2`（或点大按钮） |
| 取消录音 / 中止转换 | `Esc` |
| 复制转换结果 | `Ctrl+Enter`（或「复制文字」按钮） |
| 自动复制开关 | 结果区左下角「完成后自动复制到剪贴板」 |
| 长音频、会议录音 | 顶部切到「文件转写」页 |
| 只转写不整理 | 设置里清空 Key，或文件页取消勾选「启用 AI 整理」 |

- 结果框是**可编辑**的：转换后可以顺手改两个词再复制；
- 音频文件保存在 `data/recordings/`，忘记复制时还能回去找；
- 术语表（设置里）适合固定人名、公司名、专业词汇，能显著减少同音错字。

## 六、界面说明

**首页「语音输入」**（下图）：大录音按钮、实时波形、转换结果区。
说话时波形会随音量实时跳动，确认麦克风真的在收音。

<p align="center">
  <img src="docs/home-recording.png" alt="录音中的实时波形" width="900">
</p>

<p align="center">
  <img src="docs/home-idle.png" alt="语音输入页（空闲）" width="900">
</p>

**文件转写页**：左侧文件队列，右侧处理选项与结果预览。

<p align="center">
  <img src="docs/file-page.png" alt="文件转写页" width="900">
</p>

- 顶部导航切换「语音输入 / 文件转写」；右上角「设置」「帮助」；
- 底部状态栏：进度、当前状态、内存占用；
- 设置：API Key、Base URL / 模型名、术语表、CPU 线程数、空闲卸载、默认输出目录、测试连接。

## 七、模型怎么选？

| 模型 | 下载大小 | 转写内存峰值 | 中文准确率 | 速度（本机 CPU） | 建议 |
| --- | --- | --- | --- | --- | --- |
| `tiny` | ~75MB | ~130MB | 一般 | 最快 | 老电脑、只求有个稿子 |
| `base` | ~145MB | ~200MB | 够用 | 快 | 轻量优先 |
| **`small`** | ~484MB | ~406MB | **好** | 中等 | **默认推荐**，日常口述够用 |
| `medium` | ~1.5GB | ~937MB | 很好 | 较慢 | 对准确率要求高（本机已下载） |
| `large-v3` | ~3.1GB | ~2.5GB | 最好 | 很慢 | 一般不必在 CPU 上用 |

> 在「设置 → 识别模型」或文件页「识别模型」下拉里切换，切换后下一次转写自动加载。
> 模型缓存在 `data/models/`，删除即清理；已缓存的不会重复下载。
> 日常「语音输入」建议用 `small`（响应快）；重要录音可切 `medium`。

## 八、导出格式（文件转写页）

| 格式 | 内容 |
| --- | --- |
| `.txt` | 整理后的正文，纯文本 |
| `.md` | 带标题与元信息的 Markdown |
| `.docx` | Word 文档（标题 + 元信息 + 正文，可直接编辑） |
| `.srt` | 由原始转写时间戳生成的字幕 |
| `_原始转写.txt` | 有整理稿时额外保存的逐段原文（带时间戳），便于核对 |

文件名冲突时自动加 `(2)`、`(3)`，不会覆盖旧文件。

## 九、命令行用法（进阶 / 批量自动化）

```powershell
# 自检环境
.venv\Scripts\python.exe -m voxscribe --check

# 处理音频文件（默认开启 AI 整理）
.venv\Scripts\python.exe -m voxscribe -i D:\录音\会议.mp3 --style meeting

# 批量 + 指定输出格式与目录
.venv\Scripts\python.exe -m voxscribe -i D:\录音\*.m4a -o D:\整理稿 --formats txt,md,docx

# 只转写不整理（完全离线，不需要 API Key）
.venv\Scripts\python.exe -m voxscribe -i 访谈.wav --no-refine --print
```

参数：`-i/--input` 文件或通配符、`-o/--out` 输出目录、`--style` 风格、`--language` 语言、
`--model` 识别模型、`--formats` 导出格式、`--no-refine` 只转写、`--print` 终端打印正文、`--check` 自检。

## 十、配置项（`data/config.json`）

| 键 | 说明 | 默认 |
| --- | --- | --- |
| `asr_model` | 识别模型 | `small` |
| `cpu_threads` | 转写线程数，`0`=自动（最多 4） | `0` |
| `language` | 音频语言 | `zh` |
| `beam_size` | 识别搜索宽度 | `5` |
| `vad_filter` | 过滤静音段（更快更准） | `true` |
| `idle_unload_minutes` | 空闲多少分钟后卸载模型，`0`=不卸载 | `5` |
| `api_base_url` | DeepSeek 接口地址（可填兼容代理） | `https://api.deepseek.com` |
| `api_model` | 文本整理模型 | `deepseek-chat` |
| `api_key_enc` | 加密后的 API Key（勿手工修改） | 空 |
| `refine_by_default` | 文件页默认开启 AI 整理 | `true` |
| `default_style` | 文件页默认整理风格 | `oral` |
| `export_formats` | 默认导出格式 | `["txt","md"]` |
| `glossary` | 术语表 | 空 |
| `output_dir` | 输出目录，空=`outputs/` | 空 |
| `auto_copy` | 语音输入完成后自动复制 | `true` |

## 十一、隐私与安全

- 语音识别在本机完成，**音频文件不会上传**；
- 只有启用「AI 整理」时，**转写出的文字**才会发送到 DeepSeek 的 API；
- API Key 使用 Windows DPAPI 加密保存，日志中不会记录 Key；
- 断网时仍可使用「只转写」模式；模型下载需要联网（仅一次）；
- 所有数据都在本文件夹内（`data/`、`outputs/`），整个文件夹拷走即可迁移，删除即卸载。

## 十二、常见问题

**1. 说话时波形没反应？**
检查系统麦克风是否被其他程序占用；设置里确认没有设备静音；重新打开软件试试。
如果按钮是灰的，说明没检测到麦克风（右下角会有提示）。

**2. 按 F2 没反应？**
F2 是**软件窗口内**的快捷键，需要先点一下软件窗口让它在前台；也可以直接点大按钮。

**3. 第一次转换为什么慢？**
首次使用会加载识别模型（small 约 2 秒，medium 约 6 秒）；之后模型常驻，转换会快很多。
任务结束 5 分钟无操作会自动卸载模型释放内存，下次再自动加载。

**4. 模型下载失败 / 一直卡住？**
本机已缓存 small / tiny / medium，通常无需下载。换电脑时：
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
.venv\Scripts\python.exe tools\prefetch_model.py --model small
```

**5. DeepSeek 报 401 / 402 / 429？**
- 401：Key 错误或失效 → 设置里重新填写；
- 402：账户余额不足 → 到 platform.deepseek.com 充值；
- 429：请求频繁/额度限制 → 稍等再试（软件会自动退避重试）。

**6. 识别有明显的同音错字？**
- 换更大的模型（small → medium）；
- 在设置「术语表」里填人名、公司名、专业词；
- 「口述校对」会自动纠正常见的同音错字（如 在/再、的/得/地、帐/账、以下/一下），拿不准的会保留原文。

**7. 支持哪些音频格式？**
常见格式都支持（mp3/wav/m4a/aac/flac/ogg/opus/mp4/mov/mkv）。微信 `.silk` 需先转成 wav/mp3。

**8. 能完全离线使用吗？**
语音转写可以（没有 Key 时自动只转写），但模型首次需联网下载；AI 整理需要联网。

**9. 界面打不开 / 闪退？**
双击 `debug-run.bat` 查看错误信息，或双击 `diagnose.bat` 自检；日志在 `data/logs/app.log`。

## 十三、项目结构

```
voice2text/
├─ setup.bat               一键安装
├─ run.bat                 启动界面
├─ debug-run.bat           调试启动（带控制台）
├─ diagnose.bat            环境自检
├─ requirements.txt
├─ voxscribe/
│  ├─ __main__.py          入口（GUI / 命令行）
│  ├─ config.py            配置与路径（API Key 加密）
│  ├─ secure.py            Windows DPAPI 封装
│  ├─ models.py            数据结构契约（含 5 种整理风格）
│  ├─ errors.py            异常类型
│  ├─ asr.py               faster-whisper 本地转写
│  ├─ recorder.py          麦克风录音 + 实时音量
│  ├─ refine.py            DeepSeek 整理调用（分块/重试/取消）
│  ├─ prompts.py           5 种风格的提示词（含口述校对）
│  ├─ pipeline.py          转写→整理→导出 流水线
│  ├─ export.py            txt/md/docx/srt 导出
│  ├─ util.py              小工具
│  └─ gui/                 Tkinter 界面（语音输入页 + 文件转写页）
├─ tests/                  自动化测试（pytest）
├─ tools/                  安装/下载/截图辅助脚本
├─ data/                   配置、模型、录音、日志（运行时生成）
└─ outputs/                默认输出目录
```

## 十四、开发与测试

```powershell
# 安装开发依赖（pytest）
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 单元测试（默认跳过需要下载模型的端到端用例）
.venv\Scripts\python.exe -m pytest -q

# 全部测试：真实语音识别 + 完整流水线 + 图形界面端到端（需要 tiny 模型）
$env:VOXSCRIBE_RUN_ASR="1"
.venv\Scripts\python.exe -m pytest -q
```

> 开发提示：GUI 测试默认通过环境变量 `VOXSCRIBE_GUI_HIDDEN=1` 在**隐藏窗口**中运行，
> 不会在你的桌面上弹出调试窗口；修改界面后跑测试可以放心做别的事。

模块职责：`asr.py`（本地识别）、`recorder.py`（录音与波形）、`refine.py`（DeepSeek 整理）、
`prompts.py`（风格提示词）、`export.py`（写文件）、`pipeline.py`（编排）、`gui/`（界面）；
数据结构与异常定义在 `models.py` / `errors.py`。

## 十五、已知限制

- **没有说话人分离**：不会自动区分「谁在说」；
- F2 是软件内快捷键（非全局热键）：窗口在前台时生效；
- 暂不支持实时字幕（边说边出字），只支持录完立刻转换；
- medium 模型更准但更慢，日常口述建议 small；
- DeepSeek 整理按块进行，极长文本的块与块衔接处建议人工扫一眼。

## 十六、说明

- 本项目为个人效率工具，按「现状」提供；
- 使用 DeepSeek API 产生的费用由你的 DeepSeek 账户承担（短句整理通常每次几厘钱，以官网价格为准）；
- 请遵守录音相关法律法规，尊重他人隐私，录制前取得必要同意。

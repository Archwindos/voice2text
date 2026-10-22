# NVIDIA GPU vs Intel NPU：Whisper 语音转写实测

> 测试日期：2026-10-22 · 测试机：Lenovo Legion Y7000P（Intel Core Ultra 7 255HX + RTX 5060 Laptop 8GB + Intel AI Boost NPU）
> 测试目的：本软件（VoxScribe 语声成文）的语音转写应该跑在 NVIDIA GPU 上，还是 Intel NPU 上？

## 结论（TL;DR）

**用 NVIDIA GPU，NPU 不适合这个软件。**

- **速度**：同样一段 13 秒中文，NVIDIA GPU 热态 **0.78 秒**，NPU **2.46 秒**（慢约 3.2 倍）；50 秒长音频 GPU 2.89 秒，NPU 10.03 秒。
- **启动**：GPU 加载模型 2.8 秒；NPU 首次编译模型需要 **6~8 分钟**，之后有缓存也要 ~6.9 秒。
- **内存**：GPU 方案进程约 0.72GB；NPU 方案约 **4.1GB**（NPU 占用系统内存做权重/工作区）。
- **质量**：GPU 用 FP16，NPU 用 INT4；INT4 在长音频里出现过「轻量→清亮」等误识别，标点也更差。
- **意外结果**：在 OpenVINO 下，连 Intel CPU / iGPU 跑 Whisper medium INT4 都比 NPU 快（1.5~1.6 秒 vs 2.46 秒），说明 NPU 对 Whisper 这类编码器-解码器模型并不占优。
- NPU 唯一优势是**推理时 CPU 占用极低（约 6%）**、可能更省电；但口述转写是短时突发任务，省电收益远不如响应速度和启动体验重要。

## 测试方法

- 音频：16kHz 单声道 TTS（中文 13.2s / 21.2s / 50.4s，英文 15.8s）。
- 每种设备预热后跑 3 次取中位数（NPU 的首次编译单独统计）。
- GPU 路径：`faster-whisper` + CTranslate2，**medium FP16**（软件当前实际使用路径）。
- NPU 路径：Intel 官方 **OpenVINO GenAI WhisperPipeline** + `OpenVINO/whisper-medium-int4-ov`（NPU 只能跑低精度量化模型，选官方 INT4）。
- Intel iGPU / CPU 参照：同一 OpenVINO + INT4 模型。

## 实测数据

### 推理耗时（秒，越小越好）

| 方案 | 模型精度 | 13.2s 中文 | 21.2s 中文 | 50.4s 中文 | 15.8s 英文 |
| --- | --- | --- | --- | --- | --- |
| **NVIDIA RTX 5060 + faster-whisper** | medium FP16 | **0.78** | **1.15** | **2.89** | **0.91** |
| Intel NPU（AI Boost）+ OpenVINO | medium INT4 | 2.46 | 3.67 | 10.03 | 2.63 |
| Intel iGPU + OpenVINO | medium INT4 | 1.61 | 1.71 | 4.64 | 1.46 |
| Intel CPU + OpenVINO | medium INT4 | 1.54 | 2.05 | 5.38 | 1.64 |

### 模型加载 / 内存

| 方案 | 加载耗时 | 进程内存 | 推理时 CPU 占用 |
| --- | --- | --- | --- |
| NVIDIA GPU + faster-whisper | **2.8s** | ~0.72 GB | 单次推理 CPU 时间 0.7~4.2s |
| Intel NPU + OpenVINO | 首次 6~8 分钟编译；缓存后 ~6.9s | ~4.1 GB | CPU 时间 0.16~0.9s（~6%） |
| Intel iGPU + OpenVINO | 24.9s | ~1.7 GB | 1.4~4.8s |
| Intel CPU + OpenVINO | 4.3s | ~2.3 GB | 10~36s（吃满多核） |

> 补充：OpenVINO 在 NVIDIA dGPU 上也能跑（`GPU.1`），但其 NVIDIA 后端走的是 OpenCL，实测 6~16 秒，远不如 CUDA 优化路径，故不作为推荐。

## 为什么 NPU 反而慢？

1. **编译开销大**：NPU 需要把模型编译成设备专用 blob，medium 模型首次编译 6~8 分钟；GPU 的 CUDA kernel 是预先优化好的，加载权重即可推理。
2. **Whisper 是自回归解码**：解码器逐 token 串行生成，NPU 的并行算力用不上，层间调度/内存搬运反而成为瓶颈。
3. **量化差异**：NPU 只能跑 INT4/INT8，官方 INT4 模型直接换掉 FP16，精度与流畅度都略降。
4. **内存**：NPU 方案运行时占用约 4GB 系统内存，对"轻量软件"不友好。

## 建议

- **继续使用 NVIDIA GPU（faster-whisper medium FP16）**，这是本机上综合最快、最准、最省内存的方案。
- 不建议为 NPU 单独维护一套 OpenVINO 推理路径：收益为负（更慢 + 首次编译数分钟 + 精度下降 + 内存更高），只在"必须长时间离线低功耗转写"这类场景才有意义。
- 如果以后要优化功耗：可在**电池供电时**把模型降为 small，或空闲时卸载模型（本软件已支持），比切换到 NPU 简单得多。

## 原始数据

- `docs/bench-gpu.json`：NVIDIA GPU（faster-whisper）逐次记录
- `docs/bench-ov.json`：OpenVINO 在 NPU / iGPU / CPU / NVIDIA 上的逐次记录

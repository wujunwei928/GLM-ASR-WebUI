# GLM-ASR-WebUI

**[Readme in English](README.md)**

一个基于 **[GLM-ASR](https://github.com/zai-org/GLM-ASR)** 模型的现代化语音识别 Web 服务，拥有赛博朋克风格界面和长音频流式转录功能。

![版本](https://img.shields.io/badge/版本-0.0.1-blue)
![许可证](https://img.shields.io/badge/许可证-Apache%202.0-green)
![Python](https://img.shields.io/badge/python-3.12+-brightgreen)

## ✨ 特性

- 🎯 **高精度识别**: 采用 GLM-ASR-Nano-2512 模型，提供领先的语音识别能力
- 🚀 **流式 API**: 长音频实时转录进度反馈
- 🎨 **赛博朋克 UI**: 霓虹风格界面，配合动态粒子效果
- 🎙️ **多种输入方式**: 文件上传、URL 下载、实时录音
- 📦 **自动分块**: 长音频自动分割处理
- ⚡ **GPU 加速**: 支持 CUDA 推理加速
- 🔌 **REST API**: 清晰的 API 设计，易于集成

## 📸 预览

Web 界面拥有动态粒子背景和霓虹风格控件：

- **文件上传标签页**: 拖拽或点击上传音频文件
- **URL 输入标签页**: 通过直接 URL 转录音频
- **录音标签页**: 直接从麦克风录音

## 🛠️ 安装

### 前置要求

- Python 3.12 或更高版本
- 支持 CUDA 的 GPU（可选，用于加速推理）
- FFmpeg（系统级依赖，**仅用于以下功能**：
  - 识别音频时长
  - 分割超过 30 秒的长音频文件）

#### 安装 FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并添加到 PATH 环境变量。

### 安装步骤

1. **克隆仓库:**
```bash
git clone https://github.com/yourusername/GLM-ASR-WebUI.git
cd GLM-ASR-WebUI
```

2. **创建虚拟环境（推荐）:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

3. **安装依赖:**
```bash
pip install -r requirements.txt
```

## 🚀 使用

### 启动服务器

**开发模式（自动重载）:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**生产模式:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

Web 界面将可通过 `http://localhost:8000` 访问

### 使用 Web 界面

1. 打开浏览器访问 `http://localhost:8000`
2. 选择输入方式：
   - **文件上传**: 点击或拖拽音频文件（WAV、MP3、FLAC、OGG、M4A）
   - **URL 输入**: 输入音频文件的直接 URL
   - **实时录音**: 点击麦克风按钮开始录音
3. 点击"启动识别序列"（或相应按钮）
4. 实时查看转录进度
5. 查看完整转录结果

### API 使用

#### 健康检查
```bash
curl http://localhost:8000/health
```

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### 标准转录
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

**响应:**
```json
{
  "success": true,
  "text": "转录文本在这里...",
  "duration": 12.5,
  "error": null
}
```

#### 流式转录（推荐用于长音频）
```bash
curl -X POST http://localhost:8000/api/v1/transcribe-stream \
  -F "file=@long_audio.mp3" \
  -F "chunk_duration=30"
```

**流式响应 (NDJSON):**
```json
{"type": "info", "message": "音频时长: 125.40秒", "duration": 125.4}
{"type": "chunk", "chunk_index": 0, "total_chunks": 5, "text": "第一段文本...", "progress": 20.0}
{"type": "chunk", "chunk_index": 1, "total_chunks": 5, "text": "第二段文本...", "progress": 40.0}
{"type": "complete", "text": "完整转录文本...", "total_chunks": 5}
```

#### Python 示例

```python
import requests
import json

# 标准转录
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/transcribe',
        files={'file': f},
        data={'language': 'zh'}
    )
    result = response.json()
    print(result['text'])

# 流式转录
with open('long_audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/transcribe-stream',
        files={'file': f},
        data={'chunk_duration': 30},
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            print(f"类型: {data['type']}, 数据: {data}")
```

## 📡 API 端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/` | GET | Web 界面 |
| `/health` | GET | 健康检查和模型状态 |
| `/api/v1/transcribe` | POST | 标准转录（单次返回） |
| `/api/v1/transcribe-stream` | POST | 流式转录（推荐） |
| `/api/v1/model/info` | GET | 模型信息 |
| `/api/info` | GET | 服务信息 |
| `/docs` | GET | 交互式 API 文档（Swagger UI） |

## ⚙️ 配置

### 环境变量

可以在 `app.py` 中修改以下常量：

```python
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"  # 使用的模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 推理设备
```

### 分块参数

使用流式 API 时，可以指定：

- `chunk_duration`: 每个分块的时长（秒），默认 30

## 🏗️ 架构

### 后端技术栈
- **FastAPI**: 现代化、高性能的 Web 框架
- **Transformers**: Hugging Face 模型加载库
- **PyTorch**: 深度学习框架
- **ffmpeg-python**: 音频时长识别和长音频分块（>30秒）

### 前端技术栈
- **原生 JavaScript**: 无框架依赖
- **Canvas API**: 动态粒子背景系统
- **Fetch API**: 流式响应处理
- **MediaRecorder API**: 浏览器录音功能

## 🎯 技术细节

### 音频处理流程

1. **上传**: 音频文件保存到 `/tmp/glm_asr_uploads/`
2. **时长检测**: FFmpeg 提取音频时长（所有文件必需）
3. **分块**: 超过 30 秒的音频使用 FFmpeg 分割（可配置）
4. **转录**: 每个分块独立处理
5. **聚合**: 合并结果并返回

> **说明**: FFmpeg 仅用于元数据提取和音频分割。
> 实际的语音识别由 GLM-ASR 模型完成。

### 流式传输协议

流式 API 使用 **NDJSON**（换行符分隔的 JSON）格式：

```javascript
// 事件类型
"info"     // 初始信息（时长等）
"chunk"    // 单个分块结果（含进度）
"error"    // 分块处理错误
"complete" // 最终转录结果
```

### 模型推理参数

```python
# 固定的推理参数
max_new_tokens=256    # 最大输出长度
do_sample=False       # 使用贪婪解码
dtype=torch.bfloat16  # 精度设置，提升效率
```

## 🐛 故障排除

### 模型下载问题

首次运行会自动下载模型（约 500MB）。如果失败：

1. 检查网络连接
2. 设置 `HF_ENDPOINT` 环境变量使用镜像源：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 模型将缓存到 `~/.cache/huggingface/`

### CUDA 不可用

如果显示 `DEVICE = "cpu"` 但你有 GPU：

1. 验证 CUDA 安装：
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. 重新安装支持 CUDA 的 PyTorch：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### FFmpeg 错误

FFmpeg 仅用于识别音频时长和分割长音频（>30秒）。

确保 FFmpeg 已安装并可访问：
```bash
ffmpeg -version
```

如果遇到 FFmpeg 相关错误：
- 验证 FFmpeg 可以从命令行执行
- 检查音频文件格式是否被 FFmpeg 支持
- 短音频文件（<30秒）仅使用 FFmpeg 检测时长
- 长音频文件（>30秒）还使用 FFmpeg 进行音频分割

### 临时文件未清理

检查 `/tmp/glm_asr_*` 目录权限：
```bash
ls -la /tmp/glm_asr_uploads
ls -la /tmp/glm_asr_chunks
```

## 🔧 开发

### 运行测试

目前没有自动化测试。手动测试步骤：

1. 启动服务器
2. 测试健康检查: `curl http://localhost:8000/health`
3. 通过 Web 界面上传短音频文件
4. 上传长音频文件（> 2 分钟）测试分块功能

### 代码规范

本项目遵循 PEP 8 规范。贡献代码时应：

- 使用中文注释说明
- 遵循现有代码结构
- 包含详细的错误处理和日志记录
- 在 `finally` 块中清理临时文件

## 📊 性能

### 基准测试

不同硬件上的近似推理时间：

| 硬件 | 音频时长 | 推理时间 | 实时率 |
|----------|----------|------|------------------|
| RTX 3090 | 1 分钟 | ~6秒 | 0.1x |
| RTX 3060 | 1 分钟 | ~12秒 | 0.2x |
| CPU (i7) | 1 分钟 | ~180秒 | 3x |

*注：首次推理包含模型加载时间（约 10 秒）*

### 优化建议

- 尽可能使用 GPU
- 启用 CUDA 图以加速推理
- 根据需求调整 `chunk_duration`
- 使用流式 API 获得更好的用户体验

## 🤝 贡献

欢迎贡献！请：

1. Fork 本仓库
2. 创建特性分支
3. 提交你的更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **模型**: [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) by ZAI
- **框架**: [FastAPI](https://fastapi.tiangolo.com/)
- **UI 设计**: 灵感来自赛博朋克美学

## 📮 支持

如有问题、疑问或建议：

- 在 GitHub 上提 issue
- 查看 `/docs` 中的现有文档
- 阅读 `app.py` 中的代码注释

## 🔮 路线图

- [ ] 批量转录多个文件
- [ ] 支持说话人分离（Diarization）
- [ ] 导出字幕为 SRT/VTT 格式
- [ ] 添加身份认证和速率限制
- [ ] Docker 容器化部署
- [ ] WebSocket 实时转录支持

---

用 ❤️ 和霓虹灯光打造

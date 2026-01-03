# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

GLM-ASR-WebUI 是一个基于 **GLM-ASR-Nano-2512** 模型的语音识别 Web 服务，使用 FastAPI 构建，提供音频转录 REST API 和交互式 Web 界面。

## 核心架构

### 后端架构 (app.py)

- **框架**: FastAPI + Uvicorn
- **模型**: GLM-ASR-Nano-2512 (通过 Hugging Face transformers 加载)
- **设备支持**: CUDA (自动检测) 或 CPU
- **数据类型**: torch.bfloat16

**核心模块**:

1. **模型管理** (`load_model()`)
   - 懒加载模式，首次请求时加载
   - 全局缓存 `_model` 和 `_processor`
   - 支持从 ModelScope/Hugging Face 自动下载并缓存

2. **音频处理** (`split_audio()`, `get_audio_duration_ffmpeg()`)
   - 使用 ffmpeg-python 进行音频分割和元数据提取
   - 长音频自动分块（默认 30 秒/块）
   - 临时文件存储在 `/tmp/glm_asr_uploads` 和 `/tmp/glm_asr_chunks`

3. **推理流程** (`transcribe_chunk()`)
   - 异步分块转录，使用 `asyncio.to_thread` 避免阻塞
   - 每个分块独立处理，支持流式返回
   - 推理参数: `max_new_tokens=256`, `do_sample=False`

### 前端架构 (templates/index.html)

- **单页应用**: 纯原生 JavaScript，无外部框架依赖
- **UI 风格**: 霓虹赛博朋克风格，粒子背景动画
- **交互方式**: 三个标签页 - 文件上传、URL 链接、实时录音

**关键模块**:

1. **粒子系统** (`ParticleSystem` 类)
   - Canvas 绘制动态粒子背景
   - 粒子间连线效果
2. **流式 API 通信** (`sendStreamRequest()`)
   - Fetch API 读取 NDJSON 流
   - 实时更新进度条和分块结果
3. **录音功能** (`MediaRecorder`)
   - 浏览器原生录音，生成 WebM 格式

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/health` | GET | 健康检查 |
| `/api/v1/transcribe` | POST | 标准转录（单次返回） |
| `/api/v1/transcribe-stream` | POST | 流式转录（推荐，支持长音频） |
| `/api/v1/model/info` | GET | 模型信息 |
| `/api/info` | GET | 服务信息 |

## 常用开发命令

### 启动服务

```bash
# 开发模式（带热重载）
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 依赖管理

```bash
# 安装依赖
pip install -r requirements.txt

# 关键依赖
pip install torch>=2.9.1 torchaudio>=2.9.1 fastapi uvicorn jinja2 ffmpeg-python transformers
```

### 测试 API

```bash
# 健康检查
curl http://localhost:8000/health

# 流式转录测试
curl -X POST http://localhost:8000/api/v1/transcribe-stream \
  -F "file=@test.wav" \
  -F "chunk_duration=30"
```

## 代码规范要点

### 音频文件处理

- 始终验证 `file.content_type.startswith("audio/")`
- 临时文件必须清理：使用 `try...finally` 确保删除
- 分块时长参数: `chunk_duration` (秒)，默认 30 秒

### 流式响应格式

NDJSON (换行符分隔的 JSON)，每行一个 JSON 对象:

```json
{"type": "info", "message": "音频时长: 65.30 秒", "duration": 65.30}
{"type": "chunk", "chunk_index": 0, "total_chunks": 3, "text": "第一段文字", "progress": 33.33}
{"type": "complete", "text": "完整转录文本", "total_chunks": 3}
```

### 异步处理模式

- 使用 `asyncio.to_thread()` 将同步推理任务移到线程池
- 流式生成器使用 `AsyncGenerator[str, None]` 类型注解
- 避免在异步上下文中直接调用 `model.generate()`

### 错误处理

- 模型加载失败返回 `HTTPException(status_code=500)`
- 推理失败在流式响应中返回 `{"type": "error", ...}`
- 始终记录详细日志: `logger.error(..., exc_info=True)`

## 关键配置

### 全局变量 (app.py:48-53)

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
```

### 前端配置 (index.html:915-920)

```javascript
const CONFIG = {
    API_BASE_URL: '/api/v1',
    PARTICLE_COUNT: 80,
    PARTICLE_SPEED: 0.5,
    CONNECTION_DISTANCE: 150
};
```

## 常见问题排查

1. **模型加载慢**: 首次请求会自动下载模型，请检查网络连接和缓存目录
2. **CUDA 不可用**: 检查 PyTorch 版本和 GPU 驱动
3. **ffmpeg 错误**: 确保系统已安装 ffmpeg (`ffmpeg-python` 只是 Python 绑定)
4. **临时文件未清理**: 检查 `/tmp/glm_asr_*` 目录权限

## 文件结构

```
.
├── app.py              # FastAPI 主应用
├── requirements.txt    # Python 依赖
├── templates/
│   └── index.html      # Web 界面（单文件，含内联 CSS/JS）
└── web/static/         # 静态资源目录（可选）
```

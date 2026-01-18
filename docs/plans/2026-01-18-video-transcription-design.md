# 视频文件转录功能设计文档

**日期**: 2026-01-18
**版本**: 1.0
**状态**: 设计阶段

---

## 1. 概述

### 1.1 目标

为 GLM-ASR-WebUI 服务添加视频文件语音识别支持，允许用户上传本地视频文件或提供在线视频 URL，服务器自动提取音频并转录为文本。

### 1.2 范围

- ✅ 支持本地视频文件上传
- ✅ 支持在线视频 URL 下载
- ✅ 服务器端音频提取
- ✅ 复用现有转录 API
- ❌ 前端音频提取（已评估但未采用）

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   前端 UI   │────▶│  FastAPI    │────▶│ FFmpeg      │
│  (扩展)     │     │   后端      │     │ 音频提取    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  GLM-ASR    │
                    │  语音识别    │
                    └─────────────┘
```

### 2.2 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 处理位置 | 服务器端 | 无前端资源负担，更好的浏览器兼容性 |
| API 设计 | 复用现有端点 | 简洁统一，符合 YAGNI 原则 |
| UI 集成 | 扩展现有标签页 | 保持界面一致性 |

---

## 3. 后端实现

### 3.1 新增模块

#### `src/glm_asr/utils/video.py`

```python
"""视频处理工具函数"""

import logging
from pathlib import Path
import tempfile
import requests
import ffmpeg

logger = logging.getLogger(__name__)

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'flv', 'm4v']
MAX_VIDEO_SIZE_MB = 500
DOWNLOAD_TIMEOUT = 600  # 10分钟


def extract_audio_from_video(video_path: Path) -> Path:
    """
    从视频文件提取音频

    参数:
        video_path: 视频文件路径

    返回:
        提取的音频文件路径 (MP3格式)

    异常:
        RuntimeError: 音频提取失败
    """
    try:
        audio_path = video_path.with_suffix('.mp3')

        logger.info(f"开始提取音频: {video_path} -> {audio_path}")

        (
            ffmpeg.input(str(video_path))
            .output(
                str(audio_path),
                acodec='libmp3lame',  # MP3 编码器
                ac=1,                  # 单声道
                ar=16000,              # 16kHz 采样率（模型要求）
                q='2'                  # 高质量
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True)
        )

        logger.info(f"✅ 音频提取成功: {audio_path}")
        return audio_path

    except Exception as e:
        logger.error(f"音频提取失败: {str(e)}")
        raise RuntimeError(f"音频提取失败: {str(e)}") from e


def download_video(url: str, timeout: int = DOWNLOAD_TIMEOUT) -> Path:
    """
    下载在线视频到临时目录

    参数:
        url: 视频 URL
        timeout: 下载超时时间（秒）

    返回:
        下载的视频文件路径

    异常:
        RuntimeError: 下载失败
    """
    temp_dir = Path("/tmp/glm_asr_videos")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 从 URL 生成文件名
        filename = url.split('/')[-1][:50]
        if '.' not in filename or filename.split('.')[-1] not in SUPPORTED_VIDEO_FORMATS:
            filename = f"video_{hash(url) & 0x7FFFFFFF}.mp4"

        video_path = temp_dir / filename

        logger.info(f"开始下载视频: {url} -> {video_path}")

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # 检查文件大小
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > MAX_VIDEO_SIZE_MB:
                raise RuntimeError(f"视频文件过大: {size_mb:.1f}MB (最大 {MAX_VIDEO_SIZE_MB}MB)")

        # 下载文件
        with video_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ 视频下载成功: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f}MB)")
        return video_path

    except requests.Timeout:
        raise RuntimeError(f"下载超时（>{timeout}秒）")
    except requests.RequestException as e:
        raise RuntimeError(f"下载失败: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"视频处理失败: {str(e)}") from e


def is_video_file(content_type: str) -> bool:
    """判断是否为视频文件类型"""
    return content_type.startswith('video/')


def get_video_extension(filename: str) -> str:
    """获取视频文件扩展名"""
    return filename.split('.')[-1].lower() if '.' in filename else ''
```

### 3.2 API 端点修改

#### 修改 `/api/v1/transcribe-stream`

```python
@app.post("/api/v1/transcribe-stream")
async def transcribe_audio_stream(
    file: UploadFile = File(..., description="音频或视频文件"),
    chunk_duration: int = Form(30, description="分块时长(秒,默认 30)"),
):
    """
    音频/视频流式转录接口

    参数:
    - file: 音频或视频文件
    - chunk_duration: 分块时长(秒,默认 30)

    返回:
    - 流式 JSON 响应
    """

    # 修改验证：同时支持音频和视频
    is_video = file.content_type.startswith('video/')
    is_audio = file.content_type.startswith('audio/')

    if not (is_audio or is_video):
        return StreamingResponse(
            _iter_json([{"error": f"不支持的文件类型: {file.content_type}"}]),
            media_type="application/x-ndjson",
        )

    temp_video = None  # 视频临时文件
    temp_file = None   # 音频文件（可能是原音频或提取的音频）
    chunk_files = []   # 音频分块

    async def generate_transcription() -> AsyncGenerator[str, None]:
        nonlocal temp_video, temp_file, chunk_files

        try:
            # 加载模型
            model, processor = load_model()

            temp_dir = Path("/tmp/glm_asr_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # 保存上传的文件
            temp_file = temp_dir / f"{file.filename}"
            with temp_file.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # 如果是视频，提取音频
            if is_video:
                from glm_asr.utils.video import extract_audio_from_video

                logger.info(f"检测到视频文件，开始提取音频...")
                temp_video = temp_file  # 保存视频引用以便清理
                temp_file = extract_audio_from_video(temp_video)

            # 获取音频时长
            duration = get_audio_duration_ffmpeg(temp_file)
            if duration:
                logger.info(f"音频时长: {duration:.2f} 秒")
                yield json.dumps({
                    "type": "info",
                    "message": f"音频时长: {duration:.2f} 秒",
                    "duration": duration,
                }, ensure_ascii=False) + "\n"

            # 分割音频并转录（现有逻辑）
            chunk_files = split_audio(temp_file, chunk_duration)
            full_text = []

            for i, chunk_file in enumerate(chunk_files):
                result = await asyncio.to_thread(
                    transcribe_chunk, model, processor, chunk_file, i, len(chunk_files), DEVICE
                )

                if result["success"]:
                    full_text.append(result["text"])
                    yield json.dumps({
                        "type": "chunk",
                        "chunk_index": result["chunk_index"],
                        "total_chunks": result["total_chunks"],
                        "text": result["text"],
                        "progress": (result["chunk_index"] + 1) / result["total_chunks"] * 100,
                    }, ensure_ascii=False) + "\n"
                else:
                    yield json.dumps({
                        "type": "error",
                        "chunk_index": result["chunk_index"],
                        "total_chunks": result["total_chunks"],
                        "error": result.get("error", "转录失败"),
                    }, ensure_ascii=False) + "\n"

            # 返回完整结果
            yield json.dumps({
                "type": "complete",
                "text": " ".join(full_text),
                "total_chunks": len(chunk_files),
            }, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.error(f"转录失败: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": f"转录失败: {str(e)}"
            }, ensure_ascii=False) + "\n"

        finally:
            # 清理临时文件
            if temp_video and temp_video.exists():
                temp_video.unlink()
                logger.info(f"视频文件已删除: {temp_video}")

            if temp_file and temp_file.exists():
                temp_file.unlink()
                logger.info(f"音频文件已删除: {temp_file}")

            for chunk_file in chunk_files:
                try:
                    if chunk_file.exists():
                        chunk_file.unlink()
                except Exception:
                    pass

    return StreamingResponse(generate_transcription(), media_type="application/x-ndjson")
```

#### 新增 `/api/v1/transcribe-url`

```python
@app.post("/api/v1/transcribe-url")
async def transcribe_from_url(
    url: str = Form(..., description="音频或视频 URL"),
    chunk_duration: int = Form(30, description="分块时长(秒,默认 30)"),
):
    """
    通过 URL 转录音频/视频

    参数:
    - url: 音频或视频文件 URL
    - chunk_duration: 分块时长(秒,默认 30)

    返回:
    - 流式 JSON 响应
    """
    from glm_asr.utils.video import download_video, extract_audio_from_video

    temp_file = None
    chunk_files = []

    async def generate_transcription() -> AsyncGenerator[str, None]:
        nonlocal temp_file, chunk_files

        try:
            # 加载模型
            model, processor = load_model()

            # 下载文件
            yield json.dumps({
                "type": "info",
                "message": f"正在下载: {url}",
            }, ensure_ascii=False) + "\n"

            temp_file = download_video(url)

            # 判断是否为视频
            from glm_asr.utils.video import is_video_file
            import mimetypes

            content_type, _ = mimetypes.guess_type(str(temp_file))
            is_video = content_type and content_type.startswith('video/')

            if is_video:
                yield json.dumps({
                    "type": "info",
                    "message": "检测到视频文件，正在提取音频...",
                }, ensure_ascii=False) + "\n"

                video_file = temp_file
                temp_file = extract_audio_from_video(video_file)
                video_file.unlink()

            # 获取时长并转录（后续逻辑与 transcribe-stream 相同）
            duration = get_audio_duration_ffmpeg(temp_file)
            # ... 省略相同逻辑

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(generate_transcription(), media_type="application/x-ndjson")
```

---

## 4. 前端实现

### 4.1 文件选择器修改

```html
<!-- 三个标签页中的文件输入 -->
<input type="file" accept="audio/*,video/*" id="audioFile">
```

### 4.2 无需额外修改

- 前端调用 `/api/v1/transcribe-stream` 端点
- 后端自动判断文件类型
- 响应格式保持不变

---

## 5. 数据模型

### 5.1 扩展响应模型

```python
class TranscriptionResponse(BaseModel):
    success: bool
    text: str
    duration: float | None = None
    media_type: Literal["audio", "video"] = "audio"  # 新增
    error: str | None = None
```

---

## 6. 错误处理

### 6.1 错误类型

| 错误类型 | 处理方式 |
|----------|----------|
| 不支持的文件格式 | 返回 400 错误，提示支持的格式 |
| 视频文件过大 | 返回 413 错误，提示大小限制 |
| 音频提取失败 | 返回 500 错误，记录详细日志 |
| URL 下载超时 | 返回 408 错误，提示重试 |
| URL 无效 | 返回 400 错误，提示检查 URL |

### 6.2 流式响应错误

```json
{"type": "error", "error": "具体错误信息"}
```

---

## 7. 文件清理

### 7.1 临时文件路径

| 类型 | 路径 |
|------|------|
| 上传的视频 | `/tmp/glm_asr_uploads/` |
| 下载的视频 | `/tmp/glm_asr_videos/` |
| 提取的音频 | `/tmp/glm_asr_uploads/*.mp3` |
| 音频分块 | `/tmp/glm_asr_chunks/` |

### 7.2 清理时机

- 转录完成（成功或失败）
- 使用 `finally` 确保清理
- 记录清理日志

---

## 8. 测试计划

### 8.1 单元测试

```python
# tests/test_video_utils.py

def test_extract_audio_from_video():
    """测试音频提取"""
    video_path = Path("tests/fixtures/sample.mp4")
    audio_path = extract_audio_from_video(video_path)
    assert audio_path.exists()
    assert audio_path.suffix == '.mp3'

def test_download_video():
    """测试视频下载"""
    url = "https://example.com/sample.mp4"
    video_path = download_video(url)
    assert video_path.exists()

def test_unsupported_format():
    """测试不支持的格式"""
    with pytest.raises(RuntimeError):
        extract_audio_from_video(Path("test.xyz"))
```

### 8.2 集成测试

```python
def test_video_upload_transcription(client):
    """测试视频上传完整流程"""
    with open("tests/fixtures/sample.mp4", "rb") as f:
        response = client.post(
            "/api/v1/transcribe-stream",
            files={"file": ("sample.mp4", f, "video/mp4")}
        )
    assert response.status_code == 200

def test_url_transcription(client):
    """测试 URL 转录"""
    response = client.post(
        "/api/v1/transcribe-url",
        data={"url": "https://example.com/sample.mp4"}
    )
    assert response.status_code == 200
```

---

## 9. 实施优先级

### P0 - 必须实现
- [ ] 视频文件上传转录
- [ ] 音频提取功能
- [ ] 基本错误处理

### P1 - 应该实现
- [ ] 在线 URL 下载转录
- [ ] 文件大小限制
- [ ] 流式进度反馈

### P2 - 可以实现
- [ ] 大文件分块下载
- [ ] 断点续传
- [ ] 视频预览

---

## 10. 依赖更新

无需新增依赖，现有 `ffmpeg-python` 已足够。

---

## 附录 A：参考资料

- [FFmpeg 文档](https://ffmpeg.org/documentation.html)
- [ffmpeg-python GitHub](https://github.com/kkroening/ffmpeg-python)
- [GLM-ASR-Nano-2512 模型文档](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)

---

**文档结束**

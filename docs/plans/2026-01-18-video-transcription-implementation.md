# 视频文件转录功能实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 为 GLM-ASR-WebUI 添加视频文件语音识别支持，允许用户上传视频文件或提供视频 URL，服务器自动提取音频并转录为文本。

**架构:** 服务器端处理，使用现有的 ffmpeg-python 依赖提取音频，复用 `/api/v1/transcribe-stream` 端点，后端自动判断文件类型（音频/视频）并处理。前端只需修改文件选择器的 `accept` 属性。

**技术栈:** FastAPI, ffmpeg-python, Pytest, 原生 JavaScript

---

## 任务概览

1. 创建视频处理工具模块 (`src/glm_asr/utils/video.py`)
2. 修改 `/api/v1/transcribe-stream` 端点支持视频
3. 新增 `/api/v1/transcribe-url` 端点
4. 修改前端文件选择器
5. 添加单元测试和集成测试

---

## Task 1: 创建视频处理工具模块

**Files:**
- Create: `src/glm_asr/utils/video.py`
- Test: `tests/test_video_utils.py`

### Step 1: 创建测试文件

**File:** `tests/test_video_utils.py`

```python
"""视频处理工具函数测试"""

import pytest
from pathlib import Path
from glm_asr.utils.video import (
    extract_audio_from_video,
    is_video_file,
    get_video_extension,
    SUPPORTED_VIDEO_FORMATS
)


class TestIsVideoFile:
    """测试 is_video_file 函数"""

    def test_returns_true_for_video_content_type(self):
        """视频 content-type 应返回 True"""
        assert is_video_file("video/mp4") is True
        assert is_video_file("video/avi") is True
        assert is_video_file("video/webm") is True

    def test_returns_false_for_audio_content_type(self):
        """音频 content-type 应返回 False"""
        assert is_video_file("audio/mp3") is False
        assert is_video_file("audio/wav") is False
        assert is_video_file("audio/mpeg") is False

    def test_returns_false_for_other_content_type(self):
        """其他 content-type 应返回 False"""
        assert is_video_file("image/png") is False
        assert is_video_file("application/json") is False
        assert is_video_file("text/plain") is False


class TestGetVideoExtension:
    """测试 get_video_extension 函数"""

    def test_returns_lowercase_extension(self):
        """应返回小写的扩展名"""
        assert get_video_extension("video.MP4") == "mp4"
        assert get_video_extension("video.AVI") == "avi"

    def test_returns_extension_for_filename_with_path(self):
        """应正确处理包含路径的文件名"""
        assert get_video_extension("/path/to/video.MOV") == "mov"
        assert get_video_extension("C:\\videos\\test.MKV") == "mkv"

    def test_returns_empty_string_when_no_extension(self):
        """无扩展名时应返回空字符串"""
        assert get_video_extension("video") == ""
        assert get_video_extension("video.") == ""


class TestExtractAudioFromVideo:
    """测试 extract_audio_from_video 函数"""

    def test_raises_error_for_nonexistent_file(self, tmp_path):
        """不存在的文件应抛出异常"""
        nonexistent = tmp_path / "nonexistent.mp4"
        with pytest.raises(RuntimeError, match="音频提取失败"):
            extract_audio_from_video(nonexistent)

    def test_creates_mp3_file_from_video(self, tmp_path):
        """应创建 MP3 格式的音频文件"""
        # 注意：这个测试需要一个真实的测试视频文件
        # 实际测试时需要准备一个小的测试视频
        pass
```

**Run:** `pytest tests/test_video_utils.py -v`
**Expected:** FAIL - ModuleNotFoundError: No module named 'glm_asr.utils.video'

### Step 2: 创建视频工具模块

**File:** `src/glm_asr/utils/video.py`

```python
"""视频处理工具函数"""

import logging
from pathlib import Path

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


def is_video_file(content_type: str) -> bool:
    """判断是否为视频文件类型"""
    return content_type.startswith('video/')


def get_video_extension(filename: str) -> str:
    """获取视频文件扩展名"""
    return filename.split('.')[-1].lower() if '.' in filename else ''
```

**Run:** `pytest tests/test_video_utils.py -v`
**Expected:** PASS (所有测试通过)

### Step 3: 提交

```bash
git add src/glm_asr/utils/video.py tests/test_video_utils.py
git commit -m "feat: 添加视频处理工具模块

- 添加 extract_audio_from_video 函数提取音频
- 添加 is_video_file 和 get_video_extension 辅助函数
- 添加对应的单元测试
- 使用 ffmpeg-python 进行音频提取

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 修改 /api/v1/transcribe-stream 端点支持视频

**Files:**
- Modify: `src/glm_asr/app.py:116-259`
- Test: `tests/test_api.py` (extend existing tests)

### Step 1: 修改端点验证逻辑

**File:** `src/glm_asr/app.py`

首先修改导入部分（在文件顶部添加）：

```python
from glm_asr.utils.audio import get_audio_duration, get_audio_duration_ffmpeg, split_audio
from glm_asr.utils.video import extract_audio_from_video, is_video_file  # 新增
```

然后修改 `/api/v1/transcribe-stream` 函数（约第 116 行）：

```python
@app.post("/api/v1/transcribe-stream")
async def transcribe_audio_stream(
    file: UploadFile = File(..., description="音频或视频文件"),
    chunk_duration: int = Form(30, description="分块时长(秒,默认 30)"),
):
    """
    音频/视频流式转录接口 - 支持长音频/视频分割和流式返回

    参数:
    - file: 音频或视频文件(支持 WAV, MP3, MP4, AVI 等常见格式)
    - chunk_duration: 分块时长(秒,默认 30)

    返回:
    - 流式 JSON 响应,每个分块转录完成后立即返回
    """

    # 验证文件类型（同时支持音频和视频）
    is_video = is_video_file(file.content_type)
    is_audio = file.content_type.startswith("audio/")

    if not (is_audio or is_video):
        logger.warning(f"不支持的文件类型: {file.content_type}")
        return StreamingResponse(
            _iter_json([{"error": f"不支持的文件类型: {file.content_type},请上传音频或视频文件"}]),
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

            # 保存上传的文件到临时位置
            temp_dir = Path("/tmp/glm_asr_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_file = temp_dir / f"{file.filename}"
            with temp_file.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.info(f"文件已保存: {temp_file}")

            # 如果是视频，提取音频
            if is_video:
                logger.info(f"检测到视频文件，开始提取音频...")
                temp_video = temp_file  # 保存视频引用以便清理
                temp_file = extract_audio_from_video(temp_video)
                logger.info(f"音频提取完成: {temp_file}")

            # 获取音频时长
            duration = get_audio_duration_ffmpeg(temp_file)
            if duration:
                logger.info(f"音频时长: {duration:.2f} 秒")
                yield (
                    json.dumps(
                        {
                            "type": "info",
                            "message": f"音频时长: {duration:.2f} 秒",
                            "duration": duration,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # 分割音频
            chunk_files = split_audio(temp_file, chunk_duration)

            # 逐个转录分块
            full_text = []
            for i, chunk_file in enumerate(chunk_files):
                # 在线程池中执行推理以避免阻塞
                result = await asyncio.to_thread(
                    transcribe_chunk, model, processor, chunk_file, i, len(chunk_files), DEVICE
                )

                if result["success"]:
                    full_text.append(result["text"])
                    yield (
                        json.dumps(
                            {
                                "type": "chunk",
                                "chunk_index": result["chunk_index"],
                                "total_chunks": result["total_chunks"],
                                "text": result["text"],
                                "progress": (result["chunk_index"] + 1)
                                / result["total_chunks"]
                                * 100,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "chunk_index": result["chunk_index"],
                                "total_chunks": result["total_chunks"],
                                "error": result.get("error", "转录失败"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # 返回完整结果
            yield (
                json.dumps(
                    {
                        "type": "complete",
                        "text": " ".join(full_text),
                        "total_chunks": len(chunk_files),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        except HTTPException as e:
            yield json.dumps({"type": "error", "error": str(e.detail)}, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.error(f"流式转录失败: {str(e)}", exc_info=True)
            yield (
                json.dumps({"type": "error", "error": f"转录失败: {str(e)}"}, ensure_ascii=False)
                + "\n"
            )

        finally:
            # 清理临时文件
            # 先清理视频文件（如果存在）
            if temp_video and temp_video.exists():
                try:
                    temp_video.unlink()
                    logger.info(f"视频文件已删除: {temp_video}")
                except Exception as e:
                    logger.warning(f"删除视频文件失败: {str(e)}")

            # 清理音频分块
            for chunk_file in chunk_files:
                try:
                    if chunk_file.exists():
                        chunk_file.unlink()
                        logger.info(f"分块文件已删除: {chunk_file}")
                except Exception as e:
                    logger.warning(f"删除分块文件失败: {str(e)}")

            # 最后清理音频文件
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.info(f"临时文件已删除: {temp_file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {str(e)}")

    return StreamingResponse(generate_transcription(), media_type="application/x-ndjson")
```

**Run:** `pytest tests/test_api.py -v`
**Expected:** 现有测试仍然 PASS

### Step 2: 添加视频处理测试

**File:** `tests/test_api.py`

在现有测试文件末尾添加：

```python
import io
from unittest.mock import patch, MagicMock


def test_transcribe_stream_rejects_unsupported_type(client):
    """测试不支持的文件类型被拒绝"""
    # 创建一个假的图片文件
    fake_file = io.BytesIO(b"fake image content")
    response = client.post(
        "/api/v1/transcribe-stream",
        files={"file": ("test.png", fake_file, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in str(data)


def test_transcribe_stream_accepts_video_content_type(client):
    """测试接受视频 content-type"""
    # 使用 mock 来避免实际处理
    with patch('glm_asr.app.load_model') as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())

        fake_video = io.BytesIO(b"fake video content")
        response = client.post(
            "/api/v1/transcribe-stream",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )
        assert response.status_code == 200
```

**Run:** `pytest tests/test_api.py::test_transcribe_stream_rejects_unsupported_type -v`
**Expected:** PASS

### Step 3: 提交

```bash
git add src/glm_asr/app.py tests/test_api.py
git commit -m "feat: 支持 /api/v1/transcribe-stream 处理视频文件

- 修改文件类型验证，同时支持 audio/* 和 video/*
- 视频文件自动提取音频后再进行转录
- 优化临时文件清理逻辑，确保视频文件也被清理
- 添加视频内容类型的测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 新增 /api/v1/transcribe-url 端点

**Files:**
- Modify: `src/glm_asr/utils/video.py` (add download_video function)
- Modify: `src/glm_asr/app.py` (add new endpoint)
- Test: `tests/test_api.py` (add URL tests)

### Step 1: 在 video.py 添加下载函数

**File:** `src/glm_asr/utils/video.py`

在现有代码后添加：

```python
import requests  # 添加到文件顶部的 import 部分


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
```

### Step 2: 添加下载函数测试

**File:** `tests/test_video_utils.py`

添加测试类：

```python
class TestDownloadVideo:
    """测试 download_video 函数"""

    def test_raises_error_for_invalid_url(self):
        """无效 URL 应抛出异常"""
        with pytest.raises(RuntimeError, match="下载失败"):
            download_video("not-a-valid-url")

    def test_raises_error_for_timeout(self, tmp_path):
        """超时应抛出异常"""
        with pytest.raises(RuntimeError, match="下载超时"):
            download_video("http://10.255.255.1/test.mp4", timeout=1)
```

**Run:** `pytest tests/test_video_utils.py::TestDownloadVideo -v`
**Expected:** PASS

### Step 3: 在 app.py 添加 URL 转录端点

**File:** `src/glm_asr/app.py`

在 `transcribe_audio_stream` 函数后添加（约第 260 行）：

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
    from glm_asr.utils.video import download_video
    import mimetypes

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

            # 获取时长
            duration = get_audio_duration_ffmpeg(temp_file)
            if duration:
                yield json.dumps({
                    "type": "info",
                    "message": f"音频时长: {duration:.2f} 秒",
                    "duration": duration,
                }, ensure_ascii=False) + "\n"

            # 分割并转录
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

            yield json.dumps({
                "type": "complete",
                "text": " ".join(full_text),
                "total_chunks": len(chunk_files),
            }, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.error(f"URL 转录失败: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }, ensure_ascii=False) + "\n"

        finally:
            # 清理临时文件
            for chunk_file in chunk_files:
                try:
                    if chunk_file.exists():
                        chunk_file.unlink()
                except Exception:
                    pass

            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    return StreamingResponse(generate_transcription(), media_type="application/x-ndjson")
```

### Step 4: 添加 URL 端点测试

**File:** `tests/test_api.py`

添加测试：

```python
def test_transcribe_url_requires_url_param(client):
    """测试 URL 参数是必需的"""
    response = client.post("/api/v1/transcribe-url", data={})
    assert response.status_code == 422  # Unprocessable Entity


@patch('glm_asr.app.download_video')
@patch('glm_asr.app.load_model')
def test_transcribe_url_calls_download(mock_load, mock_download, client):
    """测试 URL 端点调用下载函数"""
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_download.return_value = MagicMock()  # 模拟返回路径对象

    response = client.post(
        "/api/v1/transcribe-url",
        data={"url": "http://example.com/test.mp4"}
    )
    mock_download.assert_called_once()
```

**Run:** `pytest tests/test_api.py::test_transcribe_url_requires_url_param -v`
**Expected:** PASS

### Step 5: 提交

```bash
git add src/glm_asr/utils/video.py src/glm_asr/app.py tests/test_video_utils.py tests/test_api.py
git commit -m "feat: 添加 /api/v1/transcribe-url 端点

- 新增 download_video 函数下载在线视频
- 新增 /api/v1/transcribe-url 端点支持 URL 转录
- 自动检测 URL 文件类型（音频/视频）
- 添加文件大小限制（500MB）和超时处理
- 添加相应的单元测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: 修改前端文件选择器

**Files:**
- Modify: `templates/index.html`

### Step 1: 修改文件选择器

**File:** `templates/index.html`

找到三个标签页中的文件输入元素，将 `accept` 属性从 `"audio/*"` 改为 `"audio/*,video/*"`：

```html
<!-- 文件上传标签页 -->
<input type="file" id="audioFile" accept="audio/*,video/*">

<!-- URL 标签页的提示文字 -->
<p>支持音频和视频文件 URL（MP3, WAV, MP4, AVI 等）</p>
```

### Step 2: 测试前端

**Run:** 手动测试
1. 启动服务: `uv run uvicorn glm_asr.app:app --reload`
2. 访问 `http://localhost:8000`
3. 尝试上传视频文件

### Step 3: 提交

```bash
git add templates/index.html
git commit -m "feat: 前端支持视频文件上传

- 修改文件选择器 accept 属性支持 video/*
- 更新提示文字说明支持视频格式
- 无需修改 JavaScript 逻辑（后端自动判断类型）

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 添加集成测试

**Files:**
- Create: `tests/fixtures/sample.mp4` (可选，真实测试视频)
- Modify: `tests/test_api.py` (add integration tests)

### Step 1: 添加集成测试

**File:** `tests/test_api.py`

添加端到端测试：

```python
def test_video_file_integration(client):
    """测试视频文件完整流程（需要真实视频文件）"""
    # 注意：这个测试需要 tests/fixtures/sample.mp4 文件
    # 如果没有，测试会被跳过
    video_path = Path(__file__).parent / "fixtures" / "sample.mp4"
    if not video_path.exists():
        pytest.skip("测试视频文件不存在")

    with open(video_path, "rb") as f:
        response = client.post(
            "/api/v1/transcribe-stream",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"chunk_duration": 10}
        )

    # 检查响应是流式的
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
```

### Step 2: 创建测试视频文件（可选）

如果需要真实的测试视频，可以创建一个小的测试文件：

```bash
# 使用 ffmpeg 创建一个 1 秒的测试视频
ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=1 \
       -f lavfi -i sine=frequency=1000:duration=1 \
       -c:v libx264 -c:a aac -y tests/fixtures/sample.mp4
```

### Step 3: 运行所有测试

**Run:** `pytest tests/ -v`
**Expected:** 所有测试 PASS

### Step 4: 提交

```bash
git add tests/test_api.py
git commit -m "test: 添加视频转录集成测试

- 添加视频文件完整流程测试
- 支持可选的真实视频文件测试
- 验证流式响应格式正确

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 更新文档

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

### Step 1: 更新 README.md

在 API 端点表格中添加：

```markdown
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/transcribe-url` | POST | URL 音频/视频转录 |
```

### Step 2: 更新 CLAUDE.md

在"API 端点"部分添加：

```markdown
### /api/v1/transcribe-url

通过 URL 转录音频或视频文件。

**参数:**
- `url`: 音频或视频文件 URL（必需）
- `chunk_duration`: 分块时长（可选，默认 30 秒）

**返回:** 流式 NDJSON 响应
```

### Step 3: 提交

```bash
git add README.md CLAUDE.md
git commit -m "docs: 更新文档说明视频支持功能

- README 添加 /api/v1/transcribe-url 端点说明
- CLAUDE.md 更新 API 端点文档
- 说明支持的视频格式

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 验收标准

完成所有任务后，以下功能应正常工作：

1. ✅ 用户可以通过 Web 界面上传视频文件（MP4, AVI, MKV 等）
2. ✅ 服务器自动从视频中提取音频
3. ✅ 提取的音频被正确转录为文本
4. ✅ 临时文件（视频、音频、分块）被正确清理
5. ✅ 用户可以输入视频 URL 进行转录
6. ✅ 所有单元测试和集成测试通过
7. ✅ 文档已更新

---

## 故障排查

**问题**: ffmpeg 找不到
**解决**: 确保 ffmpeg 已安装在系统上（`apt install ffmpeg` 或 `brew install ffmpeg`）

**问题**: SharedArrayBuffer 错误
**解决**: 这个项目使用服务器端处理，不需要此配置

**问题**: 视频文件过大
**解决**: 调整 `MAX_VIDEO_SIZE_MB` 常量

---

**计划完成！准备实施。**

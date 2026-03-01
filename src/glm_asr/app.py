"""
GLM-ASR FastAPI 语音识别服务

提供音频文件上传和语音识别的 REST API 接口
支持 Jinja2 模板渲染
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from glm_asr.models import HealthResponse, TranscriptionResponse
from glm_asr.services.asr import DEVICE, load_model, transcribe_chunk
from glm_asr.utils.audio import get_audio_duration, get_audio_duration_ffmpeg, split_audio
from glm_asr.utils.video import extract_audio_from_video, is_video_file

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 全局变量
app = FastAPI(
    title="GLM-ASR 语音识别服务",
    description="基于 GLM-ASR-Nano-2512 模型的音频转录 API",
    version="0.0.2",
)

# 配置 Jinja2 模板
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# 配置静态文件
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ 静态文件目录已挂载: {static_dir}")
else:
    logger.warning(f"⚠️ 静态文件目录不存在: {static_dir}")


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    from glm_asr.services.asr import MODEL_ID

    logger.info("=" * 60)
    logger.info("GLM-ASR FastAPI 服务启动中...")
    logger.info(f"模型 ID: {MODEL_ID}")
    logger.info(f"设备: {DEVICE}")
    logger.info("=" * 60)

    # 启动时预加载模型
    try:
        logger.info("正在预加载模型...")
        load_model()
        logger.info("✅ 模型已在启动时加载完成")
    except Exception as e:
        logger.error(f"⚠️ 启动时加载模型失败: {str(e)}")
        logger.warning("模型将在首次请求时重新尝试加载")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """根路径,返回 Web 界面"""
    from glm_asr.services.asr import MODEL_ID

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "GLM-ASR 语音识别服务",
            "version": "0.0.1",
            "model_id": MODEL_ID,
        },
    )


@app.get("/api/info")
async def api_info():
    """API 信息接口"""
    from glm_asr.services.asr import MODEL_ID

    return {
        "service": "GLM-ASR 语音识别服务",
        "version": "0.0.1",
        "model": MODEL_ID,
        "endpoints": {"health": "/health", "transcribe": "/api/v1/transcribe", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    from glm_asr.services.asr import _model, _processor

    model_loaded = _model is not None and _processor is not None

    return HealthResponse(
        status="healthy" if model_loaded else "initializing",
        model_loaded=model_loaded,
        device=DEVICE,
    )


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

            # 使用 UUID 生成唯一文件名，避免文件名过长错误
            file_ext = Path(file.filename).suffix if file.filename else ".mp3"
            temp_file = temp_dir / f"{uuid.uuid4().hex}{file_ext}"
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


def _iter_json(data_list: list) -> AsyncGenerator[str, None]:
    """将数据列表转换为 JSON 流"""
    for data in data_list:
        yield json.dumps(data, ensure_ascii=False) + "\n"


@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(..., description="音频文件(WAV, MP3 等格式)")):
    """
    音频转录接口

    参数:
    - file: 音频文件(支持 WAV, MP3 等常见格式)

    返回:
    - success: 是否成功
    - text: 转录文本
    - duration: 音频时长(秒)
    - error: 错误信息(如果失败)
    """

    # 验证文件类型
    if not file.content_type.startswith("audio/"):
        logger.warning(f"不支持的文件类型: {file.content_type}")
        return TranscriptionResponse(
            success=False, text="", error=f"不支持的文件类型: {file.content_type},请上传音频文件"
        )

    temp_file = None

    try:
        # 加载模型
        model, processor = load_model()

        # 保存上传的文件到临时位置
        temp_dir = Path("/tmp/glm_asr_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 使用 UUID 生成唯一文件名，避免文件名过长错误
        file_ext = Path(file.filename).suffix if file.filename else ".mp3"
        temp_file = temp_dir / f"{uuid.uuid4().hex}{file_ext}"
        with temp_file.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"文件已保存: {temp_file}")

        # 获取音频时长
        duration = get_audio_duration(temp_file)
        logger.info(f"音频时长: {duration:.2f} 秒" if duration else "无法获取音频时长")

        # 准备消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": str(temp_file),
                    },
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            }
        ]

        # 处理输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # 移动到设备
        inputs = inputs.to(DEVICE, dtype=torch.bfloat16)

        # 执行推理
        logger.info("开始推理...")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        # 解码结果
        transcript = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip()

        logger.info(f"转录成功: {transcript[:50]}...")

        return TranscriptionResponse(success=True, text=transcript, duration=duration)

    except HTTPException as e:
        # 重新抛出 HTTP 异常
        raise e

    except Exception as e:
        logger.error(f"转录失败: {str(e)}", exc_info=True)
        return TranscriptionResponse(success=False, text="", error=f"转录失败: {str(e)}")

    finally:
        # 清理临时文件
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"临时文件已删除: {temp_file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {str(e)}")


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


@app.get("/api/v1/model/info")
async def model_info():
    """获取模型信息"""
    from glm_asr.services.asr import MODEL_ID, _model, _processor

    model_loaded = _model is not None and _processor is not None

    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "model_loaded": model_loaded,
        "dtype": "torch.bfloat16",
        "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("glm_asr.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

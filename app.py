"""
GLM-ASR FastAPI 语音识别服务

提供音频文件上传和语音识别的 REST API 接口
支持 Jinja2 模板渲染
"""

import os
import logging
import asyncio
import json
from pathlib import Path
from typing import Optional, List, AsyncGenerator

import torch
import torchaudio
import ffmpeg
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
app = FastAPI(
    title="GLM-ASR 语音识别服务",
    description="基于 GLM-ASR-Nano-2512 模型的音频转录 API",
    version="0.0.1"
)

# 配置 Jinja2 模板
templates = Jinja2Templates(directory="templates")

# 配置静态文件 - 支持 static 目录下的示例音频文件
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ 静态文件目录已挂载: {static_dir}")
else:
    logger.warning(f"⚠️ 静态文件目录不存在: {static_dir}")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"

# 模型缓存
_model = None
_processor = None


class TranscriptionResponse(BaseModel):
    """转录响应模型"""
    success: bool
    text: str
    duration: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model_loaded: bool
    device: str


def load_model():
    """加载模型和处理器(懒加载,仅在首次请求时加载)

    支持从 ModelScope 或 Hugging Face 自动加载模型
    模型会自动缓存到系统默认缓存目录
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        logger.info("模型已加载,使用缓存")
        return _model, _processor

    try:
        logger.info(f"开始加载模型: {MODEL_ID}")
        logger.info(f"设备: {DEVICE}")

        # 直接加载模型（库会自动处理缓存）
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map=DEVICE
        )

        _model.eval()

        logger.info("✅ 模型加载成功")
        return _model, _processor

    except ImportError as e:
        logger.error(f"导入错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"模型加载失败: {str(e)}"
        )


def get_audio_duration(file_path: Path) -> float:
    """获取音频文件时长(秒)"""
    try:
        waveform, sample_rate = torchaudio.load(str(file_path))
        duration = waveform.shape[1] / sample_rate
        return duration
    except Exception as e:
        logger.warning(f"无法获取音频时长: {str(e)}")
        return None


def get_audio_duration_ffmpeg(file_path: Path) -> Optional[float]:
    """使用 ffmpeg 获取音频文件时长(秒)"""
    try:
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.warning(f"FFmpeg 无法获取音频时长: {str(e)}")
        return None


def split_audio(file_path: Path, chunk_duration: int = 30) -> List[Path]:
    """
    使用 ffmpeg 分割音频文件

    参数:
    - file_path: 音频文件路径
    - chunk_duration: 每个分块的时长(秒)

    返回:
    - 分割后的音频文件路径列表
    """
    try:
        # 获取音频总时长
        duration = get_audio_duration_ffmpeg(file_path)
        if duration is None:
            logger.error("无法获取音频时长，无法分割")
            return [file_path]

        # 如果音频时长小于等于分块时长，不需要分割
        if duration <= chunk_duration:
            logger.info(f"音频时长 {duration:.2f}s <= {chunk_duration}s，无需分割")
            return [file_path]

        logger.info(f"音频时长 {duration:.2f}s > {chunk_duration}s，开始分割...")

        # 计算分块数量
        num_chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)

        chunk_files = []
        temp_dir = Path("/tmp/glm_asr_chunks")
        temp_dir.mkdir(parents=True, exist_ok=True)

        base_name = file_path.stem

        for i in range(num_chunks):
            start_time = i * chunk_duration
            output_file = temp_dir / f"{base_name}_chunk_{i+1:03d}.wav"

            # 使用 ffmpeg 分割音频
            (
                ffmpeg
                .input(str(file_path), ss=start_time, t=chunk_duration)
                .output(str(output_file), acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )

            chunk_files.append(output_file)
            logger.info(f"已创建分块 {i+1}/{num_chunks}: {output_file.name}")

        logger.info(f"音频分割完成，共 {num_chunks} 个分块")
        return chunk_files

    except Exception as e:
        logger.error(f"音频分割失败: {str(e)}")
        return [file_path]


def transcribe_chunk(
    model,
    processor,
    chunk_file: Path,
    chunk_index: int,
    total_chunks: int,
    device: str
) -> dict:
    """
    转录单个音频分块 (同步函数，用于线程池执行)

    返回:
    - 转录结果字典
    """
    try:
        logger.info(f"正在转录分块 {chunk_index+1}/{total_chunks}: {chunk_file.name}")

        # 准备消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": str(chunk_file),
                    },
                    {
                        "type": "text",
                        "text": "Please transcribe this audio into text"
                    },
                ],
            }
        ]

        # 处理输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = inputs.to(device, dtype=torch.bfloat16)

        # 执行推理
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # 解码结果
        transcript = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        logger.info(f"分块 {chunk_index+1} 转录成功: {transcript[:50]}...")

        return {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "text": transcript,
            "success": True
        }

    except Exception as e:
        logger.error(f"分块 {chunk_index+1} 转录失败: {str(e)}")
        return {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "text": "",
            "success": False,
            "error": str(e)
        }


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
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
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "GLM-ASR 语音识别服务",
            "version": "0.0.1",
            "model_id": MODEL_ID
        }
    )


@app.get("/api/info")
async def api_info():
    """API 信息接口"""
    return {
        "service": "GLM-ASR 语音识别服务",
        "version": "0.0.1",
        "model": MODEL_ID,
        "endpoints": {
            "health": "/health",
            "transcribe": "/api/v1/transcribe",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    model_loaded = _model is not None and _processor is not None

    return HealthResponse(
        status="healthy" if model_loaded else "initializing",
        model_loaded=model_loaded,
        device=DEVICE
    )


@app.post("/api/v1/transcribe-stream")
async def transcribe_audio_stream(
    file: UploadFile = File(..., description="音频文件(WAV, MP3 等格式)"),
    chunk_duration: int = Form(30, description="分块时长(秒,默认 30)")
):
    """
    音频流式转录接口 - 支持长音频分割和流式返回

    参数:
    - file: 音频文件(支持 WAV, MP3 等常见格式)
    - chunk_duration: 分块时长(秒,默认 30)

    返回:
    - 流式 JSON 响应,每个分块转录完成后立即返回
    """

    # 验证文件类型
    if not file.content_type.startswith("audio/"):
        logger.warning(f"不支持的文件类型: {file.content_type}")
        return StreamingResponse(
            iter_json([{"error": f"不支持的文件类型: {file.content_type}"}]),
            media_type="application/x-ndjson"
        )

    temp_file = None
    chunk_files = []

    async def generate_transcription() -> AsyncGenerator[str, None]:
        nonlocal temp_file, chunk_files

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

            # 获取音频时长
            duration = get_audio_duration_ffmpeg(temp_file)
            if duration:
                logger.info(f"音频时长: {duration:.2f} 秒")
                yield json.dumps({
                    "type": "info",
                    "message": f"音频时长: {duration:.2f} 秒",
                    "duration": duration
                }, ensure_ascii=False) + "\n"

            # 分割音频
            chunk_files = split_audio(temp_file, chunk_duration)

            # 逐个转录分块
            full_text = []
            for i, chunk_file in enumerate(chunk_files):
                # 在线程池中执行推理以避免阻塞
                result = await asyncio.to_thread(
                    transcribe_chunk,
                    model,
                    processor,
                    chunk_file,
                    i,
                    len(chunk_files),
                    DEVICE
                )

                if result["success"]:
                    full_text.append(result["text"])
                    yield json.dumps({
                        "type": "chunk",
                        "chunk_index": result["chunk_index"],
                        "total_chunks": result["total_chunks"],
                        "text": result["text"],
                        "progress": (result["chunk_index"] + 1) / result["total_chunks"] * 100
                    }, ensure_ascii=False) + "\n"
                else:
                    yield json.dumps({
                        "type": "error",
                        "chunk_index": result["chunk_index"],
                        "total_chunks": result["total_chunks"],
                        "error": result.get("error", "转录失败")
                    }, ensure_ascii=False) + "\n"

            # 返回完整结果
            yield json.dumps({
                "type": "complete",
                "text": " ".join(full_text),
                "total_chunks": len(chunk_files)
            }, ensure_ascii=False) + "\n"

        except HTTPException as e:
            yield json.dumps({
                "type": "error",
                "error": str(e.detail)
            }, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.error(f"流式转录失败: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": f"转录失败: {str(e)}"
            }, ensure_ascii=False) + "\n"

        finally:
            # 清理临时文件
            for chunk_file in chunk_files:
                try:
                    if chunk_file.exists():
                        chunk_file.unlink()
                        logger.info(f"分块文件已删除: {chunk_file}")
                except Exception as e:
                    logger.warning(f"删除分块文件失败: {str(e)}")

            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.info(f"临时文件已删除: {temp_file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {str(e)}")

    return StreamingResponse(
        generate_transcription(),
        media_type="application/x-ndjson"
    )


def iter_json(data_list: list) -> AsyncGenerator[str, None]:
    """将数据列表转换为 JSON 流"""
    for data in data_list:
        yield json.dumps(data, ensure_ascii=False) + "\n"


@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件(WAV, MP3 等格式)")
):
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
            success=False,
            text="",
            error=f"不支持的文件类型: {file.content_type},请上传音频文件"
        )

    temp_file = None

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
                    {
                        "type": "text",
                        "text": "Please transcribe this audio into text"
                    },
                ],
            }
        ]

        # 处理输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = inputs.to(DEVICE, dtype=torch.bfloat16)

        # 执行推理
        logger.info("开始推理...")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # 解码结果
        transcript = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        logger.info(f"转录成功: {transcript[:50]}...")

        return TranscriptionResponse(
            success=True,
            text=transcript,
            duration=duration
        )

    except HTTPException as e:
        # 重新抛出 HTTP 异常
        raise e

    except Exception as e:
        logger.error(f"转录失败: {str(e)}", exc_info=True)
        return TranscriptionResponse(
            success=False,
            text="",
            error=f"转录失败: {str(e)}"
        )

    finally:
        # 清理临时文件
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"临时文件已删除: {temp_file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {str(e)}")


@app.get("/api/v1/model/info")
async def model_info():
    """获取模型信息"""
    model_loaded = _model is not None and _processor is not None

    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "model_loaded": model_loaded,
        "dtype": "torch.bfloat16",
        "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

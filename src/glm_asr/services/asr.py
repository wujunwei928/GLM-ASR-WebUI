"""ASR 服务模块"""
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"

# 模型缓存
_model = None
_processor = None


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
        raise e
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        raise e


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

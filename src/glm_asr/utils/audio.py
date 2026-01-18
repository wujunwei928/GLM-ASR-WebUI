"""音频处理工具函数"""

import logging
from pathlib import Path

import ffmpeg
import torchaudio

logger = logging.getLogger(__name__)


def get_audio_duration(file_path: Path) -> float | None:
    """获取音频文件时长(秒)"""
    try:
        waveform, sample_rate = torchaudio.load(str(file_path))
        duration = waveform.shape[1] / sample_rate
        return duration
    except Exception as e:
        logger.warning(f"无法获取音频时长: {str(e)}")
        return None


def get_audio_duration_ffmpeg(file_path: Path) -> float | None:
    """使用 ffmpeg 获取音频文件时长(秒)"""
    try:
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        logger.warning(f"FFmpeg 无法获取音频时长: {str(e)}")
        return None


def split_audio(file_path: Path, chunk_duration: int = 30) -> list[Path]:
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
                ffmpeg.input(str(file_path), ss=start_time, t=chunk_duration)
                .output(str(output_file), acodec="pcm_s16le", ac=1, ar=16000)
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

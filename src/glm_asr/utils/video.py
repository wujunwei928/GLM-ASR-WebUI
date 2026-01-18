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

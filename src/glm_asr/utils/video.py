"""视频处理工具函数"""

import logging
from pathlib import Path

import ffmpeg
import requests

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

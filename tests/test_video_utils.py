"""视频处理工具函数测试"""

import pytest
from pathlib import Path
from glm_asr.utils.video import (
    extract_audio_from_video,
    is_video_file,
    get_video_extension,
    download_video,
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


class TestDownloadVideo:
    """测试 download_video 函数"""

    def test_raises_error_for_invalid_url(self):
        """无效 URL 应抛出异常"""
        with pytest.raises(RuntimeError, match="下载失败"):
            download_video("not-a-valid-url")

    def test_raises_error_for_timeout(self):
        """超时应抛出异常"""
        with pytest.raises(RuntimeError, match="下载超时"):
            download_video("http://10.255.255.1/test.mp4", timeout=1)

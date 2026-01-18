"""pytest 配置和 fixtures"""

import pytest
from fastapi.testclient import TestClient

from glm_asr import app


@pytest.fixture
def client():
    """测试客户端"""
    return TestClient(app)


@pytest.fixture
def sample_audio_path(tmp_path):
    """创建测试用的音频文件"""
    audio_path = tmp_path / "test.wav"
    return audio_path

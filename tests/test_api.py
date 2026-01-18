"""API 端点测试"""

import io
from unittest.mock import patch, MagicMock


def test_root(client):
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_check(client):
    """测试健康检查"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "model_loaded" in data


def test_api_info(client):
    """测试 API 信息"""
    response = client.get("/api/info")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "GLM-ASR 语音识别服务"
    assert "endpoints" in data


def test_model_info(client):
    """测试模型信息"""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200

    data = response.json()
    assert "model_id" in data
    assert "device" in data
    assert "model_loaded" in data


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


def test_transcribe_url_requires_url_param(client):
    """测试 URL 参数是必需的"""
    response = client.post("/api/v1/transcribe-url", data={})
    assert response.status_code == 422  # Unprocessable Entity


@patch('glm_asr.utils.video.download_video')
@patch('glm_asr.app.load_model')
def test_transcribe_url_calls_download(mock_load, mock_download, client):
    """测试 URL 端点调用下载函数"""
    from pathlib import Path
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_download.return_value = Path("/tmp/test.mp4")  # 模拟返回路径对象

    response = client.post(
        "/api/v1/transcribe-url",
        data={"url": "http://example.com/test.mp4"}
    )
    mock_download.assert_called_once()
    assert response.status_code == 200

"""API 端点测试"""
from glm_asr.models import HealthResponse


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

"""ASR 服务测试"""

from glm_asr.services.asr import DEVICE, MODEL_ID


def test_device_config():
    """测试设备配置"""
    assert DEVICE in ["cuda", "cpu"]


def test_model_id():
    """测试模型 ID"""
    assert MODEL_ID == "zai-org/GLM-ASR-Nano-2512"


def test_imports():
    """测试模块导入"""
    from glm_asr.services.asr import load_model, transcribe_chunk

    assert callable(load_model)
    assert callable(transcribe_chunk)

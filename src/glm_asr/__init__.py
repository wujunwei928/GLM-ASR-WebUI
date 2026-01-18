"""
GLM-ASR-WebUI 包

基于 GLM-ASR-Nano-2512 模型的语音识别 Web 服务
"""

__version__ = "0.0.1"

from .app import app

__all__ = ["app", "__version__"]

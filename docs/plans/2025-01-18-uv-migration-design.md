# GLM-ASR-WebUI 迁移至 uv 设计文档

**日期**: 2025-01-18
**作者**: Claude Code
**状态**: 设计阶段

---

## 概述

将 GLM-ASR-WebUI 项目从 pip + requirements.txt 迁移至 uv 包管理器，采用渐进式迁移策略，分三个阶段完成。迁移后同时保留 pip 兼容性，确保部署不受影响。

### 目标

- 使用 uv 提升依赖安装速度
- 采用 pyproject.toml 现代化项目规范
- 重构为 src 布局，遵循 Python 包最佳实践
- 添加开发工具链（pytest、ruff、pre-commit）
- 保留 pip 兼容性，支持现有部署流程

---

## 第一阶段：添加 pyproject.toml

**目标**: 最小改动，仅添加新文件，不修改现有代码结构。

### 新建文件

#### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "glm-asr-webui"
version = "0.0.1"
description = "基于 GLM-ASR-Nano-2512 的语音识别 Web 服务"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "wujunwei" }
]

dependencies = [
    "torch>=2.9.1",
    "torchaudio>=2.9.1",
    "torchcodec>=0.9.0",
    "transformers@git+https://github.com/huggingface/transformers",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "jinja2>=3.1.2",
    "python-multipart>=0.0.6",
    "aiofiles>=23.0.0",
    "ffmpeg-python>=0.2.0",
    "accelerate>=0.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "pre-commit>=4.0.0",
]

[project.scripts]
glm-asr = "glm_asr:app"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "pre-commit>=4.0.0",
]
```

#### `uv.lock`

由 uv 自动生成，添加到 git 版本控制。

### 保留文件

- `requirements.txt` - 继续维护，与 pyproject.toml 同步

### 工作流变更

```bash
# 本地开发
uv sync                    # 同步依赖
uv run python app.py       # 运行应用

# 部署（不变）
pip install -r requirements.txt
```

### 同步机制

手动或通过 CI/CD 自动同步：
```bash
uv pip compile pyproject.toml -o requirements.txt
```

---

## 第二阶段：重构为 src 布局

**目标**: 将代码移入 `src/glm_asr/` 目录，采用标准 Python 包结构。

### 新目录结构

```
src/
└── glm_asr/
    ├── __init__.py           # 包初始化，导出 FastAPI app
    ├── app.py                # 从根目录移入
    ├── models.py             # Pydantic 模型（新增，从 app.py 提取）
    ├── services/
    │   ├── __init__.py
    │   └── asr.py           # ASR 服务逻辑（从 app.py 提取）
    └── utils/
        ├── __init__.py
        └── audio.py         # 音频处理工具（从 app.py 提取）

templates/          # 模板目录原地保留
static/             # 静态文件原地保留
resources/          # 资源文件原地保留
tests/              # 测试目录（第三阶段添加）
```

### 模块拆分

#### `models.py` - Pydantic 模型

```python
from pydantic import BaseModel
from typing import Optional

class TranscriptionResponse(BaseModel):
    """转录响应模型"""
    success: bool
    text: str
    duration: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model_loaded: bool
    device: str
```

#### `utils/audio.py` - 音频处理

```python
from pathlib import Path
from typing import List, Optional
import torch
import torchaudio
import ffmpeg

def get_audio_duration(file_path: Path) -> Optional[float]:
    """获取音频文件时长(秒)"""
    # ... 从 app.py 移入

def get_audio_duration_ffmpeg(file_path: Path) -> Optional[float]:
    """使用 ffmpeg 获取音频文件时长"""
    # ... 从 app.py 移入

def split_audio(file_path: Path, chunk_duration: int = 30) -> List[Path]:
    """使用 ffmpeg 分割音频文件"""
    # ... 从 app.py 移入
```

#### `services/asr.py` - ASR 服务

```python
import torch
from transformers import AutoModel, AutoProcessor
from pathlib import Path

# 全局模型缓存
_model = None
_processor = None

def load_model():
    """加载模型和处理器"""
    # ... 从 app.py 移入

def transcribe_chunk(model, processor, chunk_file: Path, ...):
    """转录单个音频分块"""
    # ... 从 app.py 移入
```

#### `app.py` - 主应用

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from glm_asr.models import TranscriptionResponse, HealthResponse
from glm_asr.services.asr import load_model
from glm_asr.utils.audio import split_audio, get_audio_duration_ffmpeg

app = FastAPI(title="GLM-ASR 语音识别服务", version="0.0.1")

# 配置模板和静态文件
templates = Jinja2Templates(directory="templates")
static_dir = Path(__file__).parent.parent.parent / "static"
# ...
```

#### `__init__.py` - 包导出

```python
from glm_asr.app import app

__all__ = ["app"]
__version__ = "0.0.1"
```

### 运行命令更新

```bash
# 旧方式（仍然支持）
python app.py
uvicorn app:app --reload

# 新方式（推荐）
uv run python -m glm_asr
uv run glm-asr
uv run uvicorn glm_asr:app --reload
```

---

## 第三阶段：添加开发工具链

**目标**: 配置代码质量工具、测试支持和 pre-commit 钩子。

### 新增配置文件

#### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: uv run pytest
        language: system
        pass_filenames: false
        always_run: true
```

#### `pyproject.toml` 扩展

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
asyncio_mode = "auto"
addopts = [
    "--cov=src/glm_asr",
    "--cov-report=term-missing",
    "--cov-report=html",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
```

### 测试目录

```
tests/
├── __init__.py
├── conftest.py              # pytest fixtures
├── test_api.py              # API 端点测试
└── test_asr_service.py      # ASR 服务测试
```

#### `conftest.py`

```python
import pytest
from fastapi.testclient import TestClient
from glm_asr import app

@pytest.fixture
def client():
    """测试客户端"""
    return TestClient(app)

@pytest.fixture
def mock_audio_file(tmp_path):
    """模拟音频文件"""
    audio_path = tmp_path / "test.wav"
    # 创建测试音频文件
    return audio_path
```

### 开发命令

```bash
# 安装 pre-commit 钩子（首次）
uv run pre-commit install

# 手动运行所有钩子
uv run pre-commit run --all-files

# 运行测试
uv run pytest

# 代码检查
uv run ruff check src/

# 格式化代码
uv run ruff format src/

# 提交时自动运行
git commit -m "..."
```

### CI/CD 同步脚本

#### `scripts/export-requirements.sh`

```bash
#!/bin/bash
# 导出 requirements.txt 供部署使用
uv pip compile pyproject.toml -o requirements.txt
```

添加到 `.github/workflows/ci.yml`（如果使用 GitHub Actions）：

```yaml
- name: Export requirements.txt
  run: |
    pip install uv
    uv pip compile pyproject.toml -o requirements.txt

- name: Upload requirements.txt
  uses: actions/upload-artifact@v4
  with:
    name: requirements
    path: requirements.txt
```

---

## 迁移检查清单

### 第一阶段

- [ ] 创建 `pyproject.toml`
- [ ] 运行 `uv sync` 验证依赖安装
- [ ] 测试 `uv run python app.py` 正常运行
- [ ] 手动/自动同步 `requirements.txt`
- [ ] 提交第一阶段变更

### 第二阶段

- [ ] 创建 `src/glm_asr/` 目录
- [ ] 拆分代码到 `models.py`, `services/asr.py`, `utils/audio.py`
- [ ] 更新 `app.py` 导入路径
- [ ] 创建 `__init__.py` 导出 app
- [ ] 测试 `uv run python -m glm_asr` 正常运行
- [ ] 验证所有 API 端点正常工作
- [ ] 提交第二阶段变更

### 第三阶段

- [ ] 创建 `.pre-commit-config.yaml`
- [ ] 运行 `uv run pre-commit install`
- [ ] 创建 `tests/` 目录和测试文件
- [ ] 配置 `pyproject.toml` 工具设置
- [ ] 运行 `uv run pytest` 验证测试
- [ ] 运行 `uv run pre-commit run --all-files`
- [ ] 配置 CI/CD 自动导出 requirements.txt
- [ ] 提交第三阶段变更

---

## 回滚方案

每个阶段独立提交，可随时回滚：

```bash
# 回滚到迁移前
git revert <commit-hash>

# 或直接 reset
git reset --hard <before-migration-commit>
```

---

## 预期收益

- **速度提升**: uv 依赖解析和安装速度比 pip 快 10-100 倍
- **现代化**: pyproject.toml 是 Python 社区标准
- **代码质量**: pre-commit 确保代码风格一致
- **可测试性**: src 布局更适合编写测试
- **兼容性**: 保留 requirements.txt，现有部署不受影响

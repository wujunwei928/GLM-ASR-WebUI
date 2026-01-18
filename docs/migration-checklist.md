# UV 迁移检查清单

> **迁移日期**: 2026-01-18
> **迁移类型**: 渐进式迁移（pip → uv）
> **项目**: GLM-ASR-WebUI

---

## ✅ 阶段 1：基础配置

### pyproject.toml 创建

- [x] 创建 `pyproject.toml` 文件
- [x] 配置 `build-system`（hatchling）
- [x] 添加项目元数据（name, version, description）
- [x] 配置 `dependencies`（从 requirements.txt 迁移）
- [x] 配置 `dev-dependencies`（开发依赖）
- [x] 配置工具设置（pytest, ruff, coverage）
- [x] 设置 `allow-direct-references = true`（支持 git+https 依赖）

**关键依赖**:
```toml
torch>=2.9.1
torchaudio>=2.9.1
transformers @ git+https://github.com/huggingface/transformers.git
fastapi>=0.100.0
uvicorn[standard]>=0.30.0
```

### UV 初始化

- [x] 运行 `uv sync` 同步依赖
- [x] 生成 `uv.lock` 锁文件
- [x] 创建 `.venv` 虚拟环境
- [x] 验证依赖安装成功

### requirements.txt 同步

- [x] 创建 `scripts/export-requirements.sh`
- [x] 运行 `uv pip compile pyproject.toml -o requirements.txt`
- [x] 验证 requirements.txt 与 pyproject.toml 一致

---

## ✅ 阶段 2：项目结构重构

### src 布局创建

- [x] 创建 `src/glm_asr/` 目录
- [x] 创建 `src/glm_asr/__init__.py`
- [x] 创建 `src/glm_asr/models.py`（Pydantic 模型）
- [x] 创建 `src/glm_asr/services/asr.py`（ASR 服务）
- [x] 创建 `src/glm_asr/utils/audio.py`（音频工具）

### 代码模块化

**models.py** - 数据模型：
- [x] `TranscriptionResponse`
- [x] `HealthResponse`
- [x] `ModelInfoResponse`
- [x] `ServiceInfoResponse`

**services/asr.py** - ASR 服务：
- [x] `load_model()` - 模型加载
- [x] `transcribe_chunk()` - 音频转录
- [x] 全局配置：DEVICE, MODEL_ID

**utils/audio.py** - 音频处理：
- [x] `get_audio_duration()` - 获取时长
- [x] `get_audio_duration_ffmpeg()` - FFmpeg 时长
- [x] `split_audio()` - 音频分割

### 主应用重构

- [x] 移动 `app.py` → `src/glm_asr/app.py`
- [x] 更新导入路径（使用新的模块结构）
- [x] 配置 `PROJECT_ROOT` 路径计算
- [x] 配置 Jinja2 模板路径
- [x] 删除根目录旧的 `app.py`

---

## ✅ 阶段 3：开发工具链

### 测试配置

- [x] 创建 `tests/` 目录
- [x] 创建 `tests/__init__.py`
- [x] 创建 `tests/conftest.py`（pytest fixtures）
- [x] 创建 `tests/test_api.py`（API 测试）
- [x] 创建 `tests/test_asr_service.py`（服务测试）
- [x] 配置 `[tool.pytest.ini_options]`（pytest 设置）

**测试覆盖**:
- [x] 根路径测试
- [x] 健康检查测试
- [x] API 信息测试
- [x] 模型信息测试
- [x] ASR 服务配置测试

### 代码质量工具

**Ruff 配置** (`[tool.ruff]`):
- [x] 行长度：100
- [x] 目标 Python 版本：py312
- [x] 启用规则：F, E, W, I, N, UP, B, C4, DTZ, T10, EM, ISC, ICN, G, PIE, PYI, RSE, RET, SIM, TID, TCH, ARG, PTH, PL, TRY, NPY, RUF
- [x] 忽略：E501（行长度由 formatter 处理）

**Ruff Format**:
- [x] 配置 `[tool.ruff.format]`
- [x] 设置 `quote-style = "double"`
- [x] 设置 `indent-style = "space"`

### Pre-commit 钩子

- [x] 创建 `.pre-commit-config.yaml`
- [x] 配置 ruff（自动修复）
- [x] 配置 ruff-format
- [x] 配置 pre-commit-hooks（trailing-whitespace, end-of-file-fixer, yaml, toml, merge-conflict, debug-statements）
- [x] 配置 pytest（自动运行测试）

### 文档更新

- [x] 更新 `README.md`
  - [x] 添加 uv 安装说明
  - [x] 更新启动命令（`glm_asr.app:app`）
  - [x] 添加 uv 和 pip 两种安装方式

---

## ✅ 阶段 4：验证与清理

### 功能验证

- [x] 运行所有测试：`uv run pytest tests/ -v --cov`
- [x] 运行 pre-commit：`uv run pre-commit run --all-files`
- [x] 测试应用启动：`uv run uvicorn glm_asr.app:app`
- [x] 测试 API 端点：`/health`, `/api/info`

### 代码质量验证

- [x] 所有测试通过（7/7）
- [x] Pre-commit 检查通过（自动修复 23 个问题）
- [x] 代码格式化完成（8 个文件）

### Git 提交

- [x] 创建功能分支：`feature/uv-migration`
- [x] 提交阶段 1 变更
- [x] 提交阶段 2 变更
- [x] 提交阶段 3 变更
- [x] 提交 pre-commit 修复
- [x] 合并到 main 分支
- [x] 清理功能分支和工作树

---

## 📊 迁移统计

| 指标 | 迁移前 | 迁移后 |
|------|--------|--------|
| 依赖管理 | requirements.txt | pyproject.toml + uv.lock |
| 项目结构 | 平铺布局 | src 布局 |
| 测试框架 | 无 | pytest + coverage |
| 代码质量 | 无 | ruff + pre-commit |
| 开发工具 | pip | uv run |
| 模块数量 | 1 个文件 | 5 个模块 |
| 测试数量 | 0 | 7 个测试 |

---

## 🎯 后续建议

### 短期（立即）

1. **安装 pre-commit**（开发者本地）：
   ```bash
   pre-commit install
   ```

2. **验证 CUDA 环境**（如果使用 GPU）：
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### 中期（1-2 周）

1. **增加测试覆盖率**：
   - 当前 34%，目标 80%+
   - 添加音频处理测试
   - 添加流式 API 测试

2. **替换 FastAPI on_event**：
   - `@app.on_event("startup")` 已弃用
   - 迁移到 `lifespan` 事件处理器

### 长期（1-3 月）

1. **添加 CI/CD**：
   - GitHub Actions 自动化测试
   - 自动化代码质量检查

2. **添加 Docker 支持**：
   - 创建 Dockerfile
   - 支持 Docker Compose 部署

---

## 🚀 常用命令速查

### 开发命令

```bash
# 安装依赖
uv sync

# 运行服务（开发模式）
uv run uvicorn glm_asr.app:app --reload

# 运行服务（生产模式）
uv run uvicorn glm_asr.app:app --host 0.0.0.0 --port 8000 --workers 4

# 运行测试
uv run pytest tests/ -v --cov

# 代码检查
uv run ruff check .
uv run ruff format .

# 运行 pre-commit
uv run pre-commit run --all-files

# 同步 requirements.txt
./scripts/export-requirements.sh
```

### 验证命令

```bash
# 健康检查
curl http://localhost:8000/health

# API 信息
curl http://localhost:8000/api/info

# 模型信息
curl http://localhost:8000/api/v1/model/info
```

---

## ✍️ 迁移记录

**日期**: 2026-01-18
**执行者**: Claude (subagent-driven-development)
**审查方式**: 两阶段审查（规范合规 → 代码质量）
**迁移模式**: 渐进式（4 个阶段）

**合并记录**:
- 特性分支：`feature/uv-migration`
- 合并提交：`1aaabbd`（style: 通过 pre-commit 自动修复代码格式和 lint 问题）

---

## ✅ 验收标准

所有验收标准已达成：

- [x] pyproject.toml 配置完整
- [x] uv.lock 锁文件生成
- [x] src 布局重构完成
- [x] 代码模块化完成
- [x] 测试框架集成（pytest）
- [x] 代码质量工具集成（ruff, pre-commit）
- [x] requirements.txt 保持同步
- [x] 所有测试通过
- [x] 文档更新完成
- [x] 功能验证通过

**迁移状态**: ✅ 完成

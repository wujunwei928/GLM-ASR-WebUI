# 脚本说明

## export-requirements.sh

从 `pyproject.toml` 导出 `requirements.txt`，用于部署环境。

```bash
./scripts/export-requirements.sh
```

## CI/CD 集成

在 CI/CD 流程中自动运行此脚本，确保 `requirements.txt` 与 `pyproject.toml` 同步。

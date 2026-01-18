#!/bin/bash
# 导出 requirements.txt 供部署使用

set -e

echo "正在从 pyproject.toml 导出 requirements.txt..."
uv pip compile pyproject.toml -o requirements.txt
echo "✅ requirements.txt 已更新"

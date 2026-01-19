#!/bin/bash

# Скрипт для повного очищення кешу перед запуском dev режиму

echo "🧹 Очищення всіх кешів..."

# Очищення Python кешу
echo "  • Очищення Python __pycache__..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Очищення Node кешу
echo "  • Очищення Node node_modules/.cache..."
rm -rf node_modules/.cache 2>/dev/null

# Очищення Vite кешу
echo "  • Очищення Vite кешу..."
rm -rf .vite 2>/dev/null

# Очищення Electron cache
echo "  • Очищення Electron кешу..."
rm -rf ~/Library/Caches/atlastrinity* 2>/dev/null

# Очищення зображень STT/TTT
echo "  • Очищення тимчасових файлів..."
rm -rf ~/.config/atlastrinity/screenshots/*.png 2>/dev/null

# Вбивство завислих процесів
echo "  • Вбивство завислих процесів (port 8000, MCP servers)..."
# Вбиваємо все на порту 8000/8088 (brain.server)
lsof -ti :8000 -ti :8088 | xargs kill -9 2>/dev/null || true
# Вбиваємо основні MCP сервери за маскою
pkill -9 -f vibe_server 2>/dev/null || true
pkill -9 -f memory_server 2>/dev/null || true
pkill -9 -f graph_server 2>/dev/null || true
pkill -9 -f mcp-server 2>/dev/null || true
pkill -9 -f macos-use 2>/dev/null || true
pkill -9 -f brain.server 2>/dev/null || true

echo "✅ Кеші очищені!"

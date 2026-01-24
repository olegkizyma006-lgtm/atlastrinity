#!/bin/bash

# Fresh Install Test Script
# Видаляє ВСЕ та симулює нову установку

set -e  # Exit on error

echo "🧹 =========================================="
echo "   FRESH INSTALL SIMULATION"
echo "   Це видалить ВСІ локальні налаштування!"
echo "=========================================="
echo ""

# Handle arguments
AUTO_YES=false
if [[ "$1" == "--yes" ]]; then
    AUTO_YES=true
fi

# Confirmation helper
confirm() {
    if [ "$AUTO_YES" = true ]; then
        return 0
    fi
    read -p "$1 (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Check for active virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "⚠️  You are currently in an ACTIVATED virtual environment: $VIRTUAL_ENV"
    if ! confirm "❓ Do you want to continue anyway?"; then
        echo "❌ Aborted. Please deactivate and restart."
        exit 1
    fi
fi

# Confirm
if ! confirm "⚠️  This will DELETE ALL local configuration and environments. Continue?"; then
    echo "❌ Cancelled"
    exit 1
fi

# 0. Backup Prompt
echo "🛡️  Backup Check"
if [ "$AUTO_YES" = true ] || confirm "❓ Create database backup before wiping?"; then
    echo "📦 Backing up databases..."
    python3 scripts/setup_dev.py --backup
    if [ $? -eq 0 ]; then
        echo "✅ Backup completed successfully."
    else
        echo "❌ Backup failed! Aborting to prevent data loss."
        exit 1
    fi
else
    echo "⚠️  Skipping backup. Hope you know what you are doing!"
fi

echo ""
echo "📦 Крок 1/8: Видалення Python venv..."
if [ -d ".venv" ]; then
    rm -rf .venv || sudo rm -rf .venv
    echo "✅ .venv видалено"
else
    echo "ℹ️  .venv не існує"
fi

echo ""
echo "📦 Крок 2/8: Видалення node_modules + lockfile..."
if [ -d "node_modules" ]; then
    rm -rf node_modules || sudo rm -rf node_modules
    echo "✅ node_modules видалено"
else
    echo "ℹ️  node_modules не існує"
fi

if [ -f "package-lock.json" ]; then
    rm -f package-lock.json
    echo "✅ package-lock.json видалено"
else
    echo "ℹ️  package-lock.json не існує"
fi

echo ""
echo "📦 Крок 3/8: Видалення Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || find . -type d -name "__pycache__" -exec sudo rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || find . -type f -name "*.pyc" -exec sudo rm -f {} + 2>/dev/null || true
echo "✅ Python cache видалено"

echo ""
echo "📦 Крок 4/8: Видалення build артефактів..."
rm -rf dist/ release/ dist_venv/ .vite/ || sudo rm -rf dist/ release/ dist_venv/ .vite/
echo "✅ Build artifacts видалено"

echo ""
echo "📦 Крок 5/8: Видалення Swift компіляції..."
if [ -d "vendor/mcp-server-macos-use/.build" ]; then
    rm -rf vendor/mcp-server-macos-use/.build || sudo rm -rf vendor/mcp-server-macos-use/.build
    echo "✅ Swift .build видалено"
else
    echo "ℹ️  Swift .build не існує"
fi

echo ""
echo "📦 Крок 6/8: Видалення глобальної конфігурації..."

DELETE_MODELS="n"
if [ -d "$HOME/.config/atlastrinity/models" ]; then
    echo ""
    echo "❓ Бажаєте видалити AI моделі (TTS/STT)? (Заощадить ~3GB трафіку якщо залишити)"
    if confirm "   Видалити моделі?"; then
        DELETE_MODELS="y"
        echo "   -> Моделі буде видалено."
    else
        DELETE_MODELS="n"
        echo "   -> Моделі буде збережено."
    fi
fi

if [ -d "$HOME/.config/atlastrinity" ]; then
    if [ "$DELETE_MODELS" == "n" ] && [ -d "$HOME/.config/atlastrinity/models" ]; then
        # Preserve models
        TEMP_MODELS="/tmp/atlastrinity_models_backup"
        rm -rf "$TEMP_MODELS"
        mv "$HOME/.config/atlastrinity/models" "$TEMP_MODELS"
        
        rm -rf "$HOME/.config/atlastrinity" || sudo rm -rf "$HOME/.config/atlastrinity"
        
        # Recreate and restore
        mkdir -p "$HOME/.config/atlastrinity"
        mv "$TEMP_MODELS" "$HOME/.config/atlastrinity/models"
        echo "✅ ~/.config/atlastrinity видалено (Models збережено)"
    else
        rm -rf "$HOME/.config/atlastrinity" || sudo rm -rf "$HOME/.config/atlastrinity"
        echo "✅ ~/.config/atlastrinity видалено (Models теж видалено)"
    fi
else
    echo "ℹ️  ~/.config/atlastrinity не існує"
fi



echo ""
echo "📦 Крок 7/8: Видалення Electron cache..."
if [ -d "$HOME/Library/Application Support/atlastrinity" ]; then
    rm -rf "$HOME/Library/Application Support/atlastrinity"
    echo "✅ Electron userData видалено"
else
    echo "ℹ️  Electron userData не існує"
fi

echo ""
echo "📦 Крок 8/8: Очищення логів та кешів..."
rm -f brain_start.log *.log
find . -name ".DS_Store" -delete 2>/dev/null || true
echo "✅ Логи та .DS_Store видалено"

echo ""
echo "🎉 =========================================="
echo "   ОЧИЩЕННЯ ЗАВЕРШЕНО!"
echo "=========================================="
echo ""
echo "Тепер запустіть:"
echo "  1️⃣  python scripts/setup_dev.py"
echo "  2️⃣  npm run dev"
echo ""
echo "Очікуваний результат:"
echo "  ✅ Відновлення баз даних з backups/"
echo "  ✅ Створення .venv"
echo "  ✅ Встановлення Python пакетів"
echo "  ✅ Встановлення NPM пакетів"
echo "  ✅ Компіляція Swift macos-use"
echo "  ✅ Завантаження моделей (Whisper, TTS)"
echo "  ✅ Ініціалізація баз даних"
echo ""

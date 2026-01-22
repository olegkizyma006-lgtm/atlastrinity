#!/bin/bash

# AtlasTrinity Custom Build Script for macOS 26.3 Beta
# This script handles spoofing deactivation and sets correct SDK paths.

echo "🔱 Starting custom build for macOS 26.3 (Darwin 25.3.0)..."

# 1. Temporarily disable spoofing if tools are present
if command -v locale_spoof &> /dev/null; then
    echo "Disabling locale_spoof..."
    sudo locale_spoof --disable
fi

# 2. Clear spoofing environment variables
echo "Clearing spoofing environment variables..."
unset MACOSX_DEPLOYMENT_TARGET
unset SDKROOT
unset SYSTEM_VERSION_COMPAT

# 3. Set correct parameters for macOS 26.3
export MACOSX_DEPLOYMENT_TARGET=26.3
# Use the actual Xcode path found: /Applications/Xcode-beta.app
export SDKROOT=/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk

echo "Deployment Target: $MACOSX_DEPLOYMENT_TARGET"
echo "SDK Root: $SDKROOT"

# 4. Prepare portable dependencies
echo "Syncing standalone venv for distribution (rsync with dereference)..."
# Use rsync for incremental updates. -a=archive, -L=copy links as files (standalone), --delete=clean old files
rsync -aL --delete .venv/ dist_venv/

# 4.1 Ensure models are initialized
echo "Checking models in ~/.config/atlastrinity/..."
if [ ! -f "$HOME/.config/atlastrinity/models/whisper/large-v3.pt" ] && [ ! -d "$HOME/.config/atlastrinity/models/faster-whisper/models--deepdml--faster-whisper-large-v3-ct2" ]; then
    echo "⚠️  Whisper large-v3 model not found. Will be downloaded on first run."
fi
if [ ! -f "$HOME/.config/atlastrinity/models/tts/model.pth" ]; then
    echo "⚠️  TTS model not found. Will be downloaded on first run."
fi

# 5. Run the build
echo "Running npm run build..."
npm run build

if [[ "$1" == "--fast" ]]; then
    echo "🚀 FAST MODE: Building unpacked application (dir) only..."
    npx electron-builder --mac --arm64 --dir
else
    echo "📦 PRODUCTION MODE: Building DMG installer..."
    npx electron-builder --mac --arm64
fi

# 5. Restore spoofing if needed (optional, uncomment if desired)
# if command -v locale_spoof &> /dev/null; then
#     echo "Re-enabling locale_spoof..."
#     sudo locale_spoof --enable
# fi

echo "✅ Custom build complete!"

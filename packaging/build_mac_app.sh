#!/usr/bin/env bash
# packaging/build_mac_app.sh — build DocumentAgent.app, a double-clickable
# native Mac app bundle wrapping a real Python venv (Streamlit + embedded
# Apple MLX local model). No terminal or pip required to *run* the built app;
# building it does need a compatible Python 3 on this machine.
#
# Usage: bash packaging/build_mac_app.sh
# Output: dist/DocumentAgent.app

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="DocumentAgent"
DIST_DIR="$ROOT/dist"
BUNDLE="$DIST_DIR/${APP_NAME}.app"
RES="$BUNDLE/Contents/Resources"
APP_SRC="$RES/app"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: this app bundle only supports Apple Silicon macOS (Darwin/arm64)." >&2
    exit 1
fi

# ── Pick a compatible Python interpreter ────────────────────────────────────
# mlx/mlx-vlm publish wheels for cp310-cp314 on macOS 14+/arm64. Prefer
# whichever interpreter is already on this machine (zero extra setup); a
# pinned python3.12/3.13 is a reasonable fallback if a dependency ever lacks
# a wheel for the default interpreter.
BUILD_PY=""
for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        BUILD_PY="$candidate"
        break
    fi
done
if [[ -z "$BUILD_PY" ]]; then
    echo "error: no python3 interpreter found. Install one with: brew install python@3.12" >&2
    exit 1
fi
echo "Using interpreter: $(command -v "$BUILD_PY") ($("$BUILD_PY" --version))"

# ── Fresh bundle layout ──────────────────────────────────────────────────────
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$APP_SRC"

# ── 1. Copy application source ──────────────────────────────────────────────
echo "Copying application source..."
cp "$ROOT"/app.py "$ROOT"/common.py "$ROOT"/nav.py "$ROOT"/local_model.py \
   "$ROOT"/agent.py "$ROOT"/storage.py "$ROOT"/requirements.txt "$ROOT"/.env.example \
   "$APP_SRC/"
cp -R "$ROOT/views" "$APP_SRC/views"
cp -R "$ROOT/api" "$APP_SRC/api"
cp -R "$ROOT/.streamlit" "$APP_SRC/.streamlit"
[[ -d "$ROOT/samples" ]] && cp -R "$ROOT/samples" "$APP_SRC/samples"

# ── 2. Build a real venv and install dependencies ───────────────────────────
echo "Creating venv (this can take a few minutes — installs mlx, mlx-vlm, streamlit)..."
"$BUILD_PY" -m venv "$RES/venv"
"$RES/venv/bin/pip" install --upgrade pip --quiet
"$RES/venv/bin/pip" install --quiet -r "$ROOT/requirements.txt"

# ── 3. Info.plist ────────────────────────────────────────────────────────────
cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key><string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key><string>com.documentagent.app</string>
    <key>CFBundleVersion</key><string>1.0.0</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundleExecutable</key><string>${APP_NAME}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
</dict>
</plist>
PLIST

# ── 4. Launcher executable ───────────────────────────────────────────────────
cat > "$BUNDLE/Contents/MacOS/${APP_NAME}" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
PORT=8501
HEALTH_URL="http://localhost:${PORT}/_stcore/health"

cd "$RES/app"
source "$RES/venv/bin/activate"

STREAMLIT_PID=""
if ! curl -fs "$HEALTH_URL" >/dev/null 2>&1; then
    streamlit run app.py --server.headless true --server.port "$PORT" \
        --browser.gatherUsageStats false >>"$HOME/Library/Logs/DocumentAgent/launcher.log" 2>&1 &
    STREAMLIT_PID=$!
    trap '[[ -n "$STREAMLIT_PID" ]] && kill "$STREAMLIT_PID" 2>/dev/null || true' EXIT

    for _ in $(seq 1 60); do
        curl -fs "$HEALTH_URL" >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

open "http://localhost:${PORT}"

if [[ -n "$STREAMLIT_PID" ]]; then
    wait "$STREAMLIT_PID" 2>/dev/null || true
fi
LAUNCHER
chmod +x "$BUNDLE/Contents/MacOS/${APP_NAME}"

mkdir -p "$HOME/Library/Logs/DocumentAgent"

# ── 5. Ad-hoc codesign so the bundle runs locally ───────────────────────────
codesign --force --deep -s - "$BUNDLE"

echo ""
echo "Built: $BUNDLE"
echo "Run it with: open \"$BUNDLE\""
echo "(First launch: right-click → Open, since this is an ad-hoc signed, unnotarized build.)"

FROM node:22.23.1-bookworm-slim
RUN npm install --prefix /opt/qwen-code @qwen-code/qwen-code@0.20.0
RUN mkdir -p /opt/qwen-code/bin /opt/qwen-code/node/bin \
    && cp /usr/local/bin/node /opt/qwen-code/node/bin/node \
    && printf '#!/bin/sh\nROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"\nexport QWEN_CODE_LAUNCHER_PATH="$ROOT/bin/qwen"\nexec "$ROOT/node/bin/node" "$ROOT/node_modules/@qwen-code/qwen-code/scripts/cli-entry.js" "$@"\n' > /opt/qwen-code/bin/qwen \
    && chmod +x /opt/qwen-code/bin/qwen

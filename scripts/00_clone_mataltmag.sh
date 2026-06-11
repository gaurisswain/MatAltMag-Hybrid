#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_URL="${MATALTMAG_REPO_URL:-https://github.com/zfgao66/MatAltMag.git}"
if [ -d external/MatAltMag/.git ]; then
  echo "external/MatAltMag already contains a git checkout"
else
  rm -f external/MatAltMag/.gitkeep
  git clone "$REPO_URL" external/MatAltMag
fi

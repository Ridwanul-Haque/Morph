#!/usr/bin/env bash
# Fetch robot models from MuJoCo Menagerie at a pinned commit:
#   unitree_go1 (BSD-3-Clause), franka_emika_panda (Apache-2.0), ufactory_xarm7 (BSD-3-Clause)
set -euo pipefail
cd "$(dirname "$0")/.."
REV=c96a32d28fb5da84da38c1da4d749e7a13212855
DEST=third_party/mujoco_menagerie
if [ ! -d "$DEST/.git" ]; then
  git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DEST"
fi
git -C "$DEST" sparse-checkout set unitree_go1 franka_emika_panda ufactory_xarm7
git -C "$DEST" fetch -q --depth 1 origin "$REV" || true
git -C "$DEST" checkout -q "$REV"
echo "Menagerie (go1, panda, xarm7) at $(git -C "$DEST" rev-parse --short HEAD); licenses in each model folder"

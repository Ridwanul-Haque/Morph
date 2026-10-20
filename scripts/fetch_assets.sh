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
MODELS="unitree_go1 franka_emika_panda ufactory_xarm7"
if [ "${1:-}" = "--extra" ]; then  # robots Morph was never configured for, used to test `morph inspect`
  MODELS="$MODELS kinova_gen3 universal_robots_ur5e ufactory_lite6 trs_so_arm100 agilex_piper unitree_go2 anybotics_anymal_c boston_dynamics_spot unitree_h1 unitree_g1 skydio_x2 hello_robot_stretch shadow_hand wonik_allegro"
fi
git -C "$DEST" sparse-checkout set $MODELS
git -C "$DEST" fetch -q --depth 1 origin "$REV" || true
git -C "$DEST" checkout -q "$REV"
echo "Menagerie ($MODELS) at $(git -C "$DEST" rev-parse --short HEAD); licenses in each model folder"

# Morph

**Transfer the outcome of a demonstrated robot task, not the robot's motion, to a robot with a different body.**

Morph runs a demonstration on a source robot in MuJoCo and records what actually happened to the object. It turns that
into a robot-independent, user-confirmed goal (`TaskSpec`: where the block must end up and which way its arrow must point).
It then checks whether a target robot can plausibly achieve that goal, picks a strategy that fits the target's real
capabilities, executes it in physics, and reports the *measured* result: success, or the specific reason for failure.
NVIDIA Nemotron (via Nebius Token Factory) interprets the instruction, selects among implemented strategies and
diagnoses failures. The simulator, not the language model, decides what is feasible and whether it worked.

Status as of 2026-09-28 (measured with the commands under "Reproduce the results"):

| What | Status |
|---|---|
| Franka Emika Panda → UFactory xArm7 (Menagerie models): Panda-trained task-space policy, zero-shot on the xArm7 | **48/50** held-out |
| Same Panda-trained policy in joint space / raw joint replay on the xArm7 (baselines) | 0/50 / 0/50 |
| Generated arm A → arm B (49% longer upper arm): task-space policy / joint policy / replay | 47/50 / 33/50 / 0/50 |
| Source arm → arm-free mobile pusher (closed-loop push planning, contact only) | **26/30**; baselines 2/30 and 2/30 |
| Source arm → Unitree Go1 (no arm): learned walking (RL, trained locally in ~20 min) + foot pushing | **12/16** held-out, 0 falls (hand-written crawl: 0/8) |
| Unsupported request (face-up flip) | reported as `unsupported` with reasons |
| Nemotron (`nvidia/nemotron-3-super-120b-a12b` via Token Factory) | live; intent benchmark Nemotron 8/8 vs rule 4/8 |
| Visual demo | `morph view` (live 3D window) and `morph showcase` (4-robot video) |

## Quick start

Tested on Ubuntu 24.04 (WSL2), Python 3.12. Running everything needs only a CPU. Training the Go1 walking policy
needs an NVIDIA GPU, but the trained policy is included.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
./scripts/fetch_assets.sh              # Unitree Go1 MJCF from MuJoCo Menagerie (BSD-3-Clause), pinned commit
cp .env.example .env                   # then put your Nebius Token Factory key in .env (git-ignored)
```

If your shell sets `PYTHONPATH` (e.g. ROS), run commands with `env -u PYTHONPATH`. Rendering uses `MUJOCO_GL=glfw`
(needs a display; on WSL2 WSLg provides one). For headless servers install OSMesa/EGL and set `MUJOCO_GL=egl`.

### See it (visual demo)

```bash
.venv/bin/morph view --robot franka_panda              # live 3D window: Panda does the demonstration
.venv/bin/morph view --robot ufactory_xarm7 --policy   # Panda-trained policy running on the xArm7
.venv/bin/morph view --robot pusher                    # arm-free mobile pusher reaches the same goal
.venv/bin/morph view --robot go1                       # Unitree Go1: learned walking + foot pushing
.venv/bin/morph showcase                               # renders runs/showcase/showcase.mp4 (4 robots, 2x2)
```

The viewer is MuJoCo's native window (drag to orbit, right-drag to pan, scroll to zoom). `--seed N` picks a
held-out scenario, `--speed 2` plays faster, and `--once` stops after one episode.

### The showcase (one command)

```bash
.venv/bin/morph check-llm      # verifies the key and that the Nemotron model id is served (GET /v1/models)
.venv/bin/morph transfer       # Panda demo -> Nemotron -> confirm -> assess -> strategy -> xArm7 -> verify -> export
.venv/bin/morph transfer --source arm_A --target arm_B    # the generated different-length arm pair
.venv/bin/morph transfer --target pusher   # same demo -> arm-free mobile pusher (closed-loop pushing)
.venv/bin/morph transfer --target go1      # same demo -> Go1 quadruped (learned walking + foot pushes)
.venv/bin/morph transfer --interactive     # confirm/edit goal x, y, yaw and tolerances at a prompt
.venv/bin/morph transfer --flip            # face-up flip request: should come back "unsupported"
```

Each run writes `runs/<timestamp>_*/` with: `source/` (the executed demo trace), `taskspec_draft.json` and
`taskspec_confirmed.json` (per-field provenance), `capability.json`, `strategy.json`, `target/` traces, `diagnosis*.json`,
`verify.json`, `inference.jsonl` (every Nemotron call: prompt version, model, latency, tokens, raw and parsed reply; no
key), `export/plan.json`, `report.md`, and `replay_side_by_side.mp4`.

**Offline baseline (not hackathon compliant):** `morph transfer --no-llm` makes no Nebius call and uses deterministic
rules. It is labelled `OFFLINE BASELINE` in the console, report and JSON.

### Reproduce the results

Nothing in the results table is shipped as a file; every number is re-measured on your machine by these commands.
Results are written to `artifacts/evidence/` (git-ignored), which the strategy selector then reads.

```bash
.venv/bin/python -m pytest -q                                  # 28 tests, ~15 s
.venv/bin/morph eval -n 50 --source franka_panda               # Panda -> xArm7 / arm B transfer table (~4 min)
.venv/bin/morph eval -n 50 --source arm_A                      # arm A -> arm B (~1.5 min)
.venv/bin/python -c "from pathlib import Path; from morph.evaluation import evaluate_pusher, evaluate_go1_learned, evaluate_go1; o=Path('artifacts/evidence'); evaluate_pusher(30, out=o); evaluate_go1_learned(16, out=o); evaluate_go1(8, out=o)"
.venv/bin/python -c "from morph.intent_bench import run_bench; run_bench()"   # Nemotron vs rule (needs the API key)
.venv/bin/morph spike                                          # Go1 foot-contact feasibility check + video
.venv/bin/morph train --source franka_panda                    # retrain the arm policies from scratch (~10 min CPU each)
```

### What runs live vs. what is loaded

* **Simulated live, every time:** the source demonstration, every target robot's motion, all grasps and pushes (by
  contact physics only: no welds, no teleporting), and the success check, which reads the block's final pose from
  the simulator. Nothing is replayed from a recording.
* **Loaded from disk:** only trained network weights. `artifacts/policies/franka_panda/` was trained on the Panda
  alone (see `trained_on` in its `.json`); the xArm7 and arm B run it zero-shot through their own IK.
  `artifacts/go1_walk/` is the Go1 walking policy (training log in `train_meta.json`). Both can be retrained with the
  commands above; without the Go1 file, Morph falls back to the hand-written crawl gait.
* **Nemotron** sees measured object poses, the instruction, the robot's capability description and, if you've run
  `morph eval`, the held-out success counts. It chooses among implemented strategies; the simulator decides whether
  they worked.

### Train the Go1 walking policy (optional; the trained policy is committed)

```bash
python3 -m venv .venv-rl && .venv-rl/bin/pip install "jax[cuda12]==0.11.2" && .venv-rl/bin/pip install -r requirements-rl.lock
.venv-rl/bin/python scripts/train_go1_walk.py --steps 200_000_000 --envs 4096   # ~20 min on an RTX 5070 Laptop GPU
.venv/bin/python scripts/eval_go1_walk.py                                       # sim-to-sim check in Morph's scene
```

## Compute and cost

| item | measured | cost |
|---|---|---|
| Nemotron calls (Token Factory) | ~3.4–4.2k tokens per `morph transfer` run (2 calls) | ~$0.002 per run (at $0.30 / $0.90 per 1M input/output tokens) |
| Arm policy training (BC/DAgger) | CPU, ~3–10 min per policy | $0 |
| Go1 walking policy (RL, 200M steps) | local RTX 5070 Laptop GPU, 20.5 min | $0 |
| Same on **Nebius AI Cloud** (option, not used) | an L40S from $1.55/h or an H100 at $3.85/h (rising to $4.50/h on 2026-10-01) | about $1–3 per training run; parallel sweeps for harder robots scale linearly |

## How NVIDIA and Nebius are used

* **Model:** NVIDIA Nemotron (default `nvidia/nemotron-3-super-120b-a12b`, configurable via `MORPH_NEMOTRON_MODEL`),
  an NVIDIA open model, called at runtime through the **Nebius Token Factory** OpenAI-compatible API
  (`https://api.tokenfactory.nebius.com/v1/`, bearer `NEBIUS_API_KEY`).
* **Three bounded calls per run**, none in the control loop:
  1. `interpret_task`: instruction + measured object change → proposed goal, tolerances, orientation mode (e.g.
     "arrow pointing north" overrides a sloppy 68° demo; "turn it over" → `face_up_flip` → `unsupported`).
  2. `select_strategy`: choose one of the **implemented** strategies using recorded held-out evidence. An invented id is
     rejected.
  3. `diagnose`: classify a failed trial from structured evidence and propose one revision from a declared space.
     Out-of-space revisions are rejected.
* Every reply is JSON-schema validated (pydantic), sanity-checked against the scene, logged, and shown with its provenance.
  Timeouts, 429/5xx retries with backoff, one bounded re-ask on invalid JSON, and a clearly labelled fallback on failure.
* **Ablation:** `morph/intent_bench.py` scores the target's *measured* final pose against the operator's intended goal
  for 8 fixed cases, with and without Nemotron. Result: Nemotron 8/8, rule-based baseline 4/8.
  The rule baseline trusts the demonstration literally; Nemotron recovers intent the demo alone can't express
  ("pointing north" when the demo stopped at 68°, "same place but facing east", "turn it over" → unsupported).
* **Where Token Factory helped:** an OpenAI-compatible endpoint meant the reasoning layer is one small HTTP client
  (`morph/reasoning.py`) with no SDK or GPU of our own. Swapping Nemotron sizes is one environment variable
  (`MORPH_NEMOTRON_MODEL`, checked at startup against `GET /v1/models`). Calls take ~8–16 s and ~2k tokens, which is
  cheap enough (~$0.002 per transfer run) to call Nemotron on every run and after every failed attempt.
* Every call is logged to `runs/<run>/inference.jsonl` (model, latency, tokens, prompt version, raw and parsed reply).

## Robots and assets

| Robot | Source | License | Controller |
|---|---|---|---|
| **Franka Emika Panda** (source) | [MuJoCo Menagerie `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) @ `c96a32d` | Apache-2.0 | our DLS IK (7-DoF, posture null-space), top-down TCP, wrist-range-aware grasp axis; gripper servo gain x10 (force limit unchanged); gravity compensation |
| **UFactory xArm7** (target arm) | [Menagerie `ufactory_xarm7`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7) @ `c96a32d` | BSD-3-Clause | same generic adapter; its own linkage gripper; gravity compensation |
| **Unitree Go1** (arm-free quadruped) | [Menagerie `unitree_go1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1) @ `c96a32d` | BSD-3-Clause | our analytic leg IK, crawl gait, three-leg stance, foot push; servo kp raised to 200 (torque limits unchanged) |
| mobile pusher | generated MJCF: holonomic base (slide x/y, hinge yaw, 30 N / 6 Nm limits), 0.023 m tip | Apache-2.0 | our tip-pose lead controller, travel around the object, push primitive |
| arm A, arm B | generated MJCF (`morph/scene.py`), 5-DoF + parallel gripper; B has a 49% longer upper arm | Apache-2.0 | analytic IK + Cartesian lead controller |

The shared object is a 5 x 5 x 6 cm block with an arrow on top, sized for the Panda's 8 cm gripper.

## What this is not

A simulation workbench. It reads simulator ground truth (no perception), has no sim-to-real validation, and the exported
plan is valid only for the MuJoCo models in this repository. It is not a real-robot controller.

## Third-party code and licenses

This repository is Apache-2.0 (`LICENSE`). It builds on:

* [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0) and MuJoCo MJX/Warp.
* [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) robot models, fetched by
  `scripts/fetch_assets.sh` at a pinned commit (not stored in this repo): `franka_emika_panda` (Apache-2.0),
  `ufactory_xarm7` (BSD-3-Clause, © UFACTORY), `unitree_go1` (BSD-3-Clause, © Unitree Robotics).
* [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Apache-2.0): its Go1 model is included in
  `third_party/playground_go1/` with its license, and its `Go1Joystick` task was used to train the walking policy.
* Brax, JAX, Flax (Apache-2.0); PyTorch, NumPy, httpx (BSD-3-Clause); pydantic (MIT); imageio (BSD-2-Clause);
  Pillow (MIT-CMU).
* NVIDIA Nemotron via Nebius Token Factory (called over the API; no model weights in this repo).

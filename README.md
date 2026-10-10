# Morph

**Describe a task once, in plain words. Any robot does it in MuJoCo, with NVIDIA Nemotron reasoning at every step.**

You give Morph a scene (any MuJoCo file with one object to move: a parcel, a ball, …), pick robots (MuJoCo
Menagerie models or your own MJCF, URDF or xacro), and type what should happen: *"throw the parcel into the bin from
1 m north of it"*, *"knock the parcel over"*, *"lift it 10 cm and hold it there"*. Morph then:

1. **Understands the goal.** Nemotron turns the sentence into a checklist of measurable conditions (where the object
   must end up, which way up, when: at the end, at the moment the robot lets go, the whole time, the last N seconds).
   A second Nemotron pass reviews it for wrong meanings, and Morph checks that it is consistent and physically possible
   in this scene.
2. **Understands the robot.** It reads the robot file itself: base (bolted down, legged, wheeled, flying), arms, legs,
   gripper and how far it opens, each part a program can move, reach, and joint limits. Torque motors become servos;
   floating models are stood on the floor.
3. **Does the task with a program Nemotron writes.** Nemotron writes a short recipe from building blocks for that
   body. Morph runs it in physics, measures what happened at every step (*"the fingers closed on nothing: the parcel
   is 3 cm from the point between them"*), and Nemotron fixes it, up to 5 times.
4. **Learns a robust skill from it, when that's wanted or needed.** Reinforcement learning on the GPU (MuJoCo
   Warp + Brax PPO, thousands of copies at once, varied physics). It starts from the working program and learns
   corrections on top of it; between rounds Nemotron reads what the robot actually did and rewrites the plan and
   reward terms, Eureka-style.
5. **Exports it.** The learned policy as NumPy weights plus a standalone runner, a **ROS 2 package** (a node for the
   real robot plus a simulated stand-in), and for scripted arm runs a **MoveIt** (moveit_py) script.

The simulator, not the language model, decides whether anything worked: every result is measured on the physics.

## Results

Measured on 2026-10-03 with the commands under [Reproduce](#reproduce). *n* tries each from slightly different
starts (object ±2 cm, ±10°, varied mass and friction).

**Programs, one sentence in, Nemotron on** (`morph do`, 4 tasks × 6 robots):

| Task | Franka Panda | UFactory xArm7 | UR5e (no gripper) | SO-ARM100 (5-joint wrist, 41 cm reach) | Unitree Go1 | Unitree Go2 |
|---|---|---|---|---|---|---|
| Put the parcel into the bin | **5/5** (ready-made skill) | **5/5** (ready-made skill) | can't lift *(reported)* | 0/5: bin out of reach | can't lift *(reported)* | can't lift *(reported)* |
| Lift it 10 cm and hold it | 2/5 | **5/5** | can't lift *(reported)* | 0/5 | can't lift *(reported)* | can't lift *(reported)* |
| Knock it over | **5/5** | **5/5** | 0/5 (tilted it 34° at best) | 0/5 | 2/5 | 0/5 |
| Push it 20 cm east | **5/5** | **5/5** | **4/5** | 0/5 (11 cm at best) | **4/5** | **4/5** |

"Can't lift" cells are physically impossible for that body (no gripper, or no arm), and Morph says so before trying.
Of the 18 possible cells, 12 succeeded at least once and 10 succeeded 4/5 or 5/5; every program was written by
Nemotron within 5 attempts, from the sentence alone. Programs: 2–5 Nemotron calls each, about $0.01 per robot.

**Learning on top of a program**, end to end with Nemotron (`morph learn`, xArm7, "lift the parcel 10 cm off the
ground and hold it there", 10 tries): Nemotron's third program worked 8/8 on the arm it was written for, but
replayed move for move in the learner's GPU physics its grasp slips (that physics has no anti-slip contact solver).
Copied as is: **0/10**. After one practice round of corrections (10M steps, easy-first, about 8 min on a laptop
GPU): **10/10**.

With a hand-written program (no Nemotron), the copy alone already succeeded: lift 10/10, put it in the bin 10/10.

**Why learn when the program already works:** the lift skill with the parcel knocked mid-task (it slides about 4 cm,
a quarter of the way through), 10 tries each:

| Knock | Program alone (hand-written) | Learned skill | Nemotron's program alone | Learned from it |
|---|---|---|---|---|
| 0.7 m/s at 20% of the way | 3/10 | **8/10** | 0/10 | **8/10** |
| 1.0 m/s at 20% | 1/10 | **6/10** | 0/10 | 1/10 |
| 0.7 m/s at 35% | 3/10 | **9/10** | 0/10 | **9/10** |

A program follows fixed moves; the learned skill sees where the object is and corrects. The app runs this test with
one click (Robot tab → *Test it: knock the parcel mid-task*).

**Understanding requests:** `morph goal-bench`, 27 varied requests judged on example outcomes (throwing from a spot,
gentle placing, "first the shelf, then the bin", lying on its side by the wall, …): **Nemotron 27/27**. Morph's own rule
reader, without Nemotron: 13/26 on the earlier 26 cases.

## Quick start

Tested on Ubuntu 24.04 (WSL2), Python 3.12, an RTX 5070 Laptop GPU (8 GB) for learning. Programs and everything
except learning run on a CPU.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
./scripts/fetch_assets.sh --extra      # MuJoCo Menagerie robots at a pinned commit (licenses in each folder)
cp .env.example .env                   # then put your Nebius Token Factory key in .env (git-ignored)
.venv/bin/morph app                    # opens http://localhost:8765
```

For learning, a second environment with JAX on CUDA:
`python3 -m venv .venv-rl && .venv-rl/bin/pip install "jax[cuda12]==0.11.2" && .venv-rl/bin/pip install -r requirements-rl.lock`.
If your shell sets `PYTHONPATH` (e.g. ROS), run commands with `env -u PYTHONPATH`.

In the app: pick a scene (or upload your own), tick robots, type a goal, and press Run. The Robot tab shows what
Morph found in each robot, where it stands (move it by typing or clicking on the view), the plan, and *Learn this
task* / *Make it robust*. The view orbits, pans and zooms (⟲ resets it).

From the command line:

```bash
.venv/bin/morph do examples/parcel "put the parcel into the bin" --robot third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml
.venv/bin/morph do examples/ball "put the ball into the bin"          # a real sphere: it rolls
.venv/bin/morph learn examples/parcel "lift the parcel 10 cm and hold it" --robot <model.xml> --rounds 4
.venv/bin/morph export runs/learned/<robot>__<goal>                    # policy + runner + ROS 2 package
.venv/bin/morph inspect <model.xml|.urdf|.xacro>                       # what Morph sees in a robot
.venv/bin/morph goal-bench                                             # does it understand varied requests?
```

## How it works

### The goal: a checklist, not a target pose

Conditions Morph can measure on the simulation: inside / on top of / touching / near / beside a fixture (or not),
at a position, facing a heading, which face is up, lifted to a height, moved N cm in a direction, away from something,
plus any formula over measured quantities (`x`, `y`, `z`, `tilt_deg`, `speed`, `spin_dps`, `held`, `dist_to('bin')`,
…). Each can say *when* it must hold: `end`, `release` (throwing, dropping), `always`, `ever`, or `last` (the last N
seconds without a break: keep it spinning, keep it balanced). "Held" is decided from real contacts: touched from two
opposite sides and nothing else carrying it.

### Programs: building blocks for each kind of body

| Body | Building blocks | Runs on |
|---|---|---|
| Arm with a parallel gripper | `move_hand` (hand pointing down, lined up with the object), `open_hand`, `close_hand`, `throw_to` (Morph does the wind-up and swing and corrects the aim from where practice throws land), `wait` | the arm's own IK and gripper, grip strength found by a test pick |
| Any robot that stands still (arms without a gripper, hands, other grippers) | `move` *any part* to a point, pointing any way (`down`, `east`, `object`, or a vector), fingers turned across the object; `push`; `grip`; `joints`; `home`; `wait` | damped least-squares IK on the learner's own model; joint-space swing when the straight path hits a joint limit; grip strength found by measuring the squeeze |
| Legged robots with a learned gait (Go1, Go2) | `walk_to`, `turn_to`, `foot_push` (low slides it, high pushes near its top), `bump` (with the body), `push_object_to` (Morph plans a series of pushes) | the robot's walking policy, three-leg stance for the foot push |

All of them are reactive: `until: "touch"` stops a move on contact, and `repeat_if(condition, back_to, times)` goes
back if something didn't work ("if `held == 0` after lifting: grasp again"). Numbers can be formulas over where things
are when the step starts (`obj_x`, `obj_top + 0.05`, `bin_x`, `hand_z`, …), so a program follows the object.
Programs are JSON data, never code. Working programs are cached, so a rerun makes no new Nemotron call.

### Learning: start from the program, learn corrections

* **Corrections on top of the program** (residual learning): the learner's action is the program's action at that
  moment plus a learned correction, so a policy that outputs nothing *is* the program. The correction network starts
  at exactly zero and is first fitted to the corrections the program needed from other starts, on slightly noisy inputs
  so it stays steady near the recorded path.
* **Keep copying while practising** (DAPG, Rajeswaran et al. 2018): an extra term in PPO's loss keeps a pull towards
  the program's actions, so practice refines the skill instead of slowly forgetting it.
* **Easy-first practice** (reverse curriculum): early rounds start from moments late in the program (already holding
  it, nearly there) and each round moves the starts back. If it's stuck at the same step twice, most practice starts
  where it got closest on that step.
* **The steps a person would take** (Nemotron, per body: which parts, in what order) are the reward: progress
  towards the current step, and a step only counts after it has held for 0.2 s.
* **Eureka-style revision:** after each round Morph replays the policy, measures where it gets stuck and what it
  actually does, and sends that to Nemotron. Nemotron rewrites the steps and can add weighted **reward terms**
  (checked formulas such as `closure * (touch_dist < 0.03)`), and sees how each term behaved in training. When progress
  stalls it proposes several different plans; each practises briefly and the best is kept.
* **You can coach it too:** while it practises, type what you see it doing wrong in the Learning tab ("it grabs the
  top edge and it slips; grab it lower"), or rate the practice try you just watched 👍 / 👎. At the end of the round
  (or right away, keeping what it learned so far) Nemotron reads your words, and compares what the tries you liked
  did with the ones you didn't (preference feedback, Christiano et al. 2017), together with Morph's measurements, and
  changes the steps and reward terms. Your input is advice: the measurements still decide what is kept. When you give
  it, or when the copied program doesn't work, practice is allowed to move further away from the program.
* **A critic that sees more** (asymmetric actor-critic, Pinto et al. 2018): during training, the part that judges its
  moves also sees how well it really grips, how far through the steps it is, and each practice copy's true object
  weight and friction. The exported policy only uses what a robot can sense.
* **Fair judging and no lost progress:** 10 tries per round, and the best policy so far is always kept.
* **Out of reach:** a bolted-down robot that never gets near the object because it stands too far away is moved
  closer, once (not if you placed it yourself).
* Varied physics per practice copy (friction, object mass, motor strength, a one-step delay, sensor noise).

Arm programs and general programs are recorded on the learner's own model, or translated joint by joint, so the copy
matches. Learning from legged programs isn't supported yet: legged robots learn from scratch or use their program.

### Exports

* **Standalone:** `policy.npz` (NumPy weights), `policy.json` (exactly what it reads and outputs, joint order, rates,
  the program it corrects), `model.xml` with its meshes, and `run_policy.py`.
* **ROS 2** (`ament_python`): `policy_node` subscribes to `joint_states` and the object pose (and odometry for mobile
  robots) and publishes `JointTrajectory` or `Float64MultiArray` commands; `sim_bridge` is a MuJoCo stand-in that speaks
  the same topics; `ros2 launch <pkg> policy.launch.py sim:=true`. Tested: the node rebuilds the training observation
  exactly, and the closed loop reaches the goal through ROS topics.
* **MoveIt:** for scripted arm runs, a `moveit_py` script with the grasp and place waypoints in the robot's base frame.

## Compute and cost

| Item | Measured | Cost |
|---|---|---|
| Nemotron (Token Factory, $0.30 / $0.90 per 1M input / output tokens, read live from `/v1/models`) | goal + review: 2 calls; a program: 1 call per attempt; a learning revision: 1 call | about $0.002–0.01 per call; a full `morph do` with programs: a few cents |
| Learning one skill (60M practice steps, 4 rounds) | local RTX 5070 Laptop, 512–1024 copies at once | about 30–40 min, no bill (about $0.05 of electricity) |
| Same on **Nebius AI Cloud** (option) | H100 / H200 / B200; prices change on 2026-10-01, Morph shows the right one by date | about $1–3 per skill; Morph prepares the job, you start it |

## How NVIDIA and Nebius are used

* **Model:** NVIDIA Nemotron (`nvidia/nemotron-3-super-120b-a12b`, configurable with `MORPH_NEMOTRON_MODEL`), an
  NVIDIA open model, called through the **Nebius Token Factory** OpenAI-compatible API.
* **What Nemotron does** (none of it in the control loop):
  1. reads the request into a goal checklist, and reviews its own reading for wrong meanings;
  2. plans the steps a person would take with this body;
  3. writes programs from building blocks and fixes them from Morph's step-by-step report;
  4. coaches learning: reads what the robot actually did, rewrites the steps and reward terms, proposes alternative
     plans when progress stalls.
* Every reply is schema-validated (pydantic), every formula is parsed by Morph's own safe evaluator (no `eval`), and
  invented names, unknown building blocks or impossible numbers are sent back with the reason. A wrong answer costs a
  retry, never a crash. Every call is logged with model, latency and tokens (`runs/app_inference.jsonl`), never the key.
* **GPU:** learning runs on NVIDIA GPUs through JAX and MuJoCo Warp (local, or a prepared Nebius AI Cloud job).

## Related work

* **Eureka** (Ma et al., NVIDIA 2023): an LLM writes reward code and improves it from training statistics; it reached
  expert-level rewards for dexterous skills such as pen spinning, with expert-prepared environments and large GPU
  budgets. Morph's revision loop is Eureka-like, but Nemotron writes checked formulas rather than code, and Morph aims at
  everyday tasks from a sentence, for any robot file, on one GPU.
* **Code as Policies** (Liang et al. 2022), **RoboGen** (Wang et al. 2023), **GenSim** (Wang et al. 2023): language
  models writing robot programs, and using them to make training data. Morph's programs follow this idea.
* **Deep RL from human preferences** (Christiano et al. 2017), **asymmetric actor-critic** (Pinto et al. 2018),
  **DAPG** (Rajeswaran et al. 2018), **residual policy learning** (Silver et al. 2018; Johannink et al. 2019),
  **reverse curriculum** (Florensa et al. 2017), **DART** (Laskey et al. 2017): the learning methods above.

## Robots and assets

Any MJCF, URDF or xacro (ROS packages next to it are found). Tested with these MuJoCo Menagerie models
(`scripts/fetch_assets.sh --extra`, pinned commit `c96a32d`): Franka Panda, UFactory xArm7 and Lite6, AgileX Piper,
Kinova Gen3, UR5e, SO-ARM100, Unitree Go1, Go2, H1 and G1, ANYmal C, Spot with arm, Skydio X2, Hello Robot Stretch.
Walking policies are included for the Go1 (`artifacts/go1_walk/`) and the Go2 (`artifacts/walk/go2/`); `morph app`
can train one for another legged robot.

## What this is not, and known limits

* A simulation workbench: it reads simulator ground truth (no perception) and has no real-robot validation. The ROS 2
  export is meant as a starting point to fine-tune on hardware.
* Programs can't express everything: dexterous in-hand skills (spinning a pen), balancing, catching. Those need
  learning from scratch, which is much less reliable and needs far more compute.
* Humanoids and robots without a walking policy get no programs yet; learning from legged programs isn't supported.
* Objects are boxes, spheres, upright cylinders or capsules (other shapes are treated as their bounding box).
* A program's accuracy is limited by its open-loop moves (pushing to within 2 cm is hard); that's what learning
  corrections are for.

## Reproduce

```bash
.venv/bin/python -m pytest -q                       # 150 tests, ~1 min (CPU)
.venv/bin/morph goal-bench                          # request understanding: Nemotron vs rules (~$0.07)
.venv/bin/morph do examples/parcel "<task>" --robot <model> --run 5 --yes   # each cell of the programs table
```

The learning tables use `morph learn` (or the app) on the xArm7, and the knock test is `learn_task.robustness`.

## The original transfer pipeline

Morph started as demonstration transfer: a Panda demonstrates, Morph records what happened to the object and has a
robot with a different body reach the same outcome. It's still here (`morph transfer`, `morph view`, `morph showcase`,
`morph eval`):

| What (measured 2026-09-28) | Result |
|---|---|
| Panda-trained task-space policy, zero-shot on the xArm7 | 48/50 (joint-space / raw replay baselines: 0/50, 0/50) |
| Generated arm A → arm B (49% longer upper arm): task-space / joint policy / replay | 47/50 / 33/50 / 0/50 |
| Arm → arm-free mobile pusher (closed-loop push planning) | 26/30 (baselines 2/30, 2/30) |
| Arm → Unitree Go1: learned walking (RL, ~20 min) + foot pushing | 12/16, 0 falls (hand-written crawl: 0/8) |

```bash
.venv/bin/morph transfer                     # Panda demo -> Nemotron -> confirm -> strategy -> xArm7 -> verify
.venv/bin/morph view --robot go1             # live 3D window
.venv/bin/morph eval -n 50 --source franka_panda
```

## Third-party code and licenses

This repository is Apache-2.0 (`LICENSE`). It builds on:

* [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0), MuJoCo MJX and Warp.
* [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) robot models, fetched by
  `scripts/fetch_assets.sh` at a pinned commit (not stored in this repo); each model folder holds its license.
* [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Apache-2.0): its Go1 model is included in
  `third_party/playground_go1/` with its license, and its `Go1Joystick` task was used to train the Go1 walking policy.
* Brax, JAX, Flax, optax (Apache-2.0); PyTorch, NumPy, httpx (BSD-3-Clause); pydantic (MIT); imageio (BSD-2-Clause);
  Pillow (MIT-CMU); xacro (BSD-3-Clause, optional).
* NVIDIA Nemotron via Nebius Token Factory (called over the API; no model weights in this repo).

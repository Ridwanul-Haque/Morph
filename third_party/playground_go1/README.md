Unitree Go1 model as used by MuJoCo Playground's `Go1Joystick*` tasks
(https://github.com/google-deepmind/mujoco_playground, `mujoco_playground/_src/locomotion/go1/xmls/go1_mjx_fullcollisions.xml`,
playground 0.2.0, Apache-2.0, see LICENSE). Only change: `meshdir` points at our pinned MuJoCo Menagerie
`unitree_go1/assets` (BSD-3-Clause). Morph deploys the learned walking policy on this full-collision model so the
legs and body physically collide with the block; the policy itself was trained on Playground's feet-only variant.

"""Scripted expert reliability on any arm: python scripts/batch_expert.py N arm_A arm_B franka_panda ufactory_xarm7"""
import collections
import sys
import time

import numpy as np

from morph.arm_skills import expert_action, run_episode
from morph.learning import sample_scenario
from morph.robots.registry import make_arm
from morph.sim import SimulationUnstable
from morph.task import default_spec

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for name in sys.argv[2:] or ["arm_A", "arm_B"]:
    arm = make_arm(name)
    ok = unst = 0
    errs, reasons, t0 = [], collections.Counter(), time.time()
    for i in range(N):
        sc = sample_scenario(i)
        arm.reset(i, sc.block, sc.params)
        try:
            r = run_episode(arm, default_spec(*sc.goal), expert_action)
        except SimulationUnstable:
            unst += 1
            continue
        ok += r.outcome.success
        errs.append(r.outcome.pos_err_m)
        if not r.outcome.success:
            reasons[(r.phases[-1][1], r.outcome.reasons[0].split(" (")[0][:40])] += 1
    print(f"{name}: success {ok}/{N} unstable {unst} median pos err {np.median(errs) * 100:.2f} cm "
          f"wall {time.time() - t0:.0f}s failures {dict(reasons)}", flush=True)

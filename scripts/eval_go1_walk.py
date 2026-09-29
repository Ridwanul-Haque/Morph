"""Sim-to-sim check of the learned Go1 walking policy inside Morph's own CPU MuJoCo scene (full collisions,
full solver): does it stand, track velocity commands and turn without falling?"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from morph.robots.go1_learned import DEFAULT_POLICY, LearnedGo1
from morph.sim import PhysicalParams

policy = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_POLICY
g = LearnedGo1(policy)
tests = [("stand", (0.0, 0.0, 0.0)), ("forward 0.5", (0.5, 0.0, 0.0)), ("backward 0.3", (-0.3, 0.0, 0.0)),
         ("sideways 0.3", (0.0, 0.3, 0.0)), ("turn left 0.8", (0.0, 0.0, 0.8)), ("turn right 0.8", (0.0, 0.0, -0.8))]
rows = []
for name, cmd in tests:
    g.reset(0, (5, 5, 0), PhysicalParams())
    g.walk((0, 0, 0), 1.0)
    p0, th0, _ = g.base()
    vx, vy, wz, tilt = [], [], [], []
    for _ in range(int(4.0 / 0.02)):
        g.step(cmd)
        lv = g._sensor("local_linvel")
        vx.append(lv[0]); vy.append(lv[1]); wz.append(g._sensor("gyro")[2])
        tilt.append(math.degrees(g.base()[2]))
        if g.fallen():
            break
    rows.append({"test": name, "cmd": cmd, "fell": g.fallen(), "mean_vx": round(float(np.mean(vx[50:])), 3),
                 "mean_vy": round(float(np.mean(vy[50:])), 3), "mean_wz": round(float(np.mean(wz[50:])), 3),
                 "max_tilt_deg": round(max(tilt), 1)})
    print(rows[-1], flush=True)
ok = all(not r["fell"] for r in rows)
print(json.dumps({"all_upright": ok}))
Path("artifacts/go1_walk").mkdir(parents=True, exist_ok=True)
Path("artifacts/go1_walk/sim2sim_check.json").write_text(json.dumps(rows, indent=2))

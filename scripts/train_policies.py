"""Train the joint-space and task-space arm policies (same as `morph train`)."""

import argparse
from pathlib import Path

from morph.learning import train_dagger

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/policies")
    ap.add_argument("--kinds", nargs="+", default=["task", "joint"])
    ap.add_argument("--n-expert", type=int, default=150)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--n-per-iter", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    for k in a.kinds:
        train_dagger(k, Path(a.out), n_expert=a.n_expert, n_iter=a.n_iter, n_per_iter=a.n_per_iter, epochs=a.epochs)

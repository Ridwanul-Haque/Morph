"""Side-by-side replay video: source and target renders from the actual
simulation runs, annotated with measured goal error and contact events."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(frame: np.ndarray, title: str, lines: list[str], color=(255, 255, 255)) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 26 + 18 * len(lines)], fill=(0, 0, 0, 150))
    d.text((8, 4), title, fill=color, font=_font(17))
    for i, ln in enumerate(lines):
        d.text((8, 26 + 18 * i), ln, fill=(230, 230, 230), font=_font(14))
    return np.asarray(im)


def goal_err_lines(block: dict, goal: tuple[float, float, float], extra: str = "") -> list[str]:
    pe = math.hypot(block["x"] - goal[0], block["y"] - goal[1]) * 100
    ye = abs(math.degrees(math.remainder(block["yaw"] - goal[2], 2 * math.pi)))
    return [f"goal error: {pe:5.1f} cm  {ye:5.1f} deg" + (f"   {extra}" if extra else "")]


def side_by_side(left: list[np.ndarray], right: list[np.ndarray], out: Path, fps: int = 17) -> Path:
    n = max(len(left), len(right))
    frames = []
    for i in range(n):
        a = left[min(i, len(left) - 1)]
        b = right[min(i, len(right) - 1)]
        h = max(a.shape[0], b.shape[0])
        pad = lambda x: np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0)))
        frames.append(np.concatenate([pad(a), np.full((h, 6, 3), 20, np.uint8), pad(b)], axis=1))
    import imageio.v2 as imageio
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, macro_block_size=2)
    return out


def annotate_episode(frames: list[np.ndarray], trace: list[dict], goal, title: str, every: int = 3,
                     strategy: str = "") -> list[np.ndarray]:
    out = []
    for i, f in enumerate(frames):
        t = trace[min(i * every, len(trace) - 1)] if trace else None
        lines = []
        if t:
            lines = goal_err_lines(t["block"], goal, f"t={t['t']:.1f}s")
            if "phase" in t:
                lines.append(f"phase: {t['phase']}" + (f"   strategy: {strategy}" if strategy else ""))
        out.append(annotate(f, title, lines))
    return out

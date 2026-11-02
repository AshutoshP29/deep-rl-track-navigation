"""Shared track data for the SAC approach.

Re-uses the track loading, road mask and lidar from ppo_checkpoints/race_circuit.py, and adds the
closed centerline that the progress reward and observations are measured against.
"""
import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "ppo_checkpoints"))  # Appended so this folder's modules win name clashes

from race_circuit import HEIGHT, START_POS, WIDTH, is_on_road, load_track  # noqa: E402

CENTERLINE_CSV = HERE / "centerline.csv"


class Centerline:
    """Closed polyline around the track. Point 0 sits on the start/finish line and points run in
    driving order. s is the arc length from point 0, so s wraps back to 0 at `length`."""

    def __init__(self, points, half_widths):
        self.points = np.asarray(points, dtype=np.float64)
        self.half_widths = np.asarray(half_widths, dtype=np.float64)

        seg = np.roll(self.points, -1, axis=0) - self.points
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.tangents = seg / self.seg_len[:, None]
        self.s = np.concatenate(([0.0], np.cumsum(self.seg_len)[:-1]))
        self.length = float(self.seg_len.sum())

        # Signed curvature per point: change in heading per pixel, + = turning left on screen
        heading = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        dheading = np.angle(np.exp(1j * (heading - np.roll(heading, 1))))
        self.curvature = -dheading / self.seg_len

    def __len__(self):
        return len(self.points)

    def project(self, x, y, hint=None, window=30):
        """Nearest point on the centerline to (x, y).

        Returns (segment index, s, signed lateral offset). Offset is + to the left of the driving
        direction. `hint` limits the search to `window` segments either side of a previous index.
        """
        n = len(self.points)
        idx = np.arange(n) if hint is None else (hint + np.arange(-window, window + 1)) % n

        a = self.points[idx]
        t_vec = self.tangents[idx]
        rel = np.array([x, y]) - a
        along = np.clip((rel * t_vec).sum(axis=1), 0.0, self.seg_len[idx])
        closest = a + t_vec * along[:, None]
        dist = np.hypot(*(np.array([x, y]) - closest).T)

        k = int(np.argmin(dist))
        i = int(idx[k])
        tx, ty = self.tangents[i]
        rx, ry = rel[k]
        # Screen y points down, so "left of travel" is the cross product's negative side
        side = -np.sign(tx * ry - ty * rx) or 1.0
        return i, (self.s[i] + along[k]) % self.length, side * dist[k]

    def index_at(self, s):
        return int(np.searchsorted(self.s, s % self.length, side="right") - 1)

    def point_at(self, s):
        i = self.index_at(s)
        return self.points[i] + self.tangents[i] * (s % self.length - self.s[i])

    def curvature_at(self, s, smooth=40.0):
        """Mean curvature over the `smooth` pixels around arc length s."""
        lo, hi = self.index_at(s - smooth / 2), self.index_at(s + smooth / 2)
        if hi < lo:
            vals = np.concatenate((self.curvature[lo:], self.curvature[: hi + 1]))
        else:
            vals = self.curvature[lo : hi + 1]
        return float(vals.mean())


@lru_cache(maxsize=1)
def load_centerline():
    if not CENTERLINE_CSV.exists():
        raise SystemExit(f"{CENTERLINE_CSV.name} is missing. Run `python centerline.py` to build it.")
    with open(CENTERLINE_CSV, newline="") as f:
        rows = [tuple(map(float, row)) for row in list(csv.reader(f))[1:]]
    rows = np.array(rows)
    return Centerline(rows[:, :2], rows[:, 2])


__all__ = ["HEIGHT", "WIDTH", "START_POS", "Centerline", "is_on_road", "load_centerline", "load_track"]

"""Builds centerline.csv from the road mask. Run once after changing the track image:

    python centerline.py            # writes centerline.csv and centerline_preview.png

1. Distance-to-edge map of the road mask (repeated erosion).
2. Dijkstra from each checkpoint gate to the next, where steps near the edge cost far more,
   so the path follows the middle of the road.
3. Smooth, resample every SPACING pixels, and read each point's half-width (distance to the
   nearest edge) from a full-resolution distance map.
"""
import csv
import heapq
import math

import numpy as np
import pygame

from track import CENTERLINE_CSV, HEIGHT, HERE, WIDTH, is_on_road, load_track

from car_env import GATES  # ppo_checkpoints is on sys.path via track.py

SPACING = 8.0  # Pixels between centerline points
DOWNSAMPLE = 2  # Dijkstra runs on a half-resolution grid


def edge_distance(mask):
    """Approximate distance (in grid cells) from each True cell to the nearest False cell."""
    dist = np.zeros(mask.shape, dtype=np.float32)
    cur = mask.copy()
    level = 0
    while cur.any():
        level += 1
        dist[cur] = level
        nxt = cur.copy()
        nxt[1:, :] &= cur[:-1, :]
        nxt[:-1, :] &= cur[1:, :]
        nxt[:, 1:] &= cur[:, :-1]
        nxt[:, :-1] &= cur[:, 1:]
        if level % 2 == 0:  # Alternate 4- and 8-neighbour erosion for a rounder metric
            nxt[1:, 1:] &= cur[:-1, :-1]
            nxt[:-1, :-1] &= cur[1:, 1:]
            nxt[1:, :-1] &= cur[:-1, 1:]
            nxt[:-1, 1:] &= cur[1:, :-1]
        cur = nxt
    return dist


def nearest_cell(mask, x, y):
    cells = np.argwhere(mask)
    k = np.argmin((cells[:, 0] - x) ** 2 + (cells[:, 1] - y) ** 2)
    return int(cells[k][0]), int(cells[k][1])


def shortest_path(cost, start, goal):
    """Dijkstra on an 8-connected grid. cost[x, y] is the price of entering a cell (inf = blocked)."""
    w, h = cost.shape
    best = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    steps = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    while heap:
        d, node = heapq.heappop(heap)
        if node == goal:
            break
        if d > best[node]:
            continue
        x, y = node
        for dx, dy, length in steps:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and cost[nx, ny] < np.inf:
                nd = d + length * cost[nx, ny]
                if nd < best.get((nx, ny), np.inf):
                    best[(nx, ny)] = nd
                    prev[(nx, ny)] = node
                    heapq.heappush(heap, (nd, (nx, ny)))
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def resample_closed(points, spacing):
    closed = np.vstack((points, points[:1]))
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(int(s[-1] // spacing), 3)
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(targets, s, closed[:, k]) for k in range(2)])


def smooth_closed(points, window=5):
    kernel = np.ones(window) / window
    pad = window // 2
    wrapped = np.vstack((points[-pad:], points, points[:pad]))
    return np.column_stack([np.convolve(wrapped[:, k], kernel, mode="valid") for k in range(2)])


def build(road):
    grid = road[::DOWNSAMPLE, ::DOWNSAMPLE]
    dist = edge_distance(grid)
    # Strongly prefer cells far from the edge so the path rides the ridge of the distance map
    cost = np.where(grid, 1.0 + 400.0 / np.maximum(dist, 1.0) ** 2, np.inf)

    mids = [((a[0] + b[0]) / 2 / DOWNSAMPLE, (a[1] + b[1]) / 2 / DOWNSAMPLE) for a, b in GATES]
    cells = [nearest_cell(grid, x, y) for x, y in mids]

    path = []
    for i in range(len(cells)):
        leg = shortest_path(cost, cells[i], cells[(i + 1) % len(cells)])
        path.extend(leg[:-1])
    points = np.array(path, dtype=np.float64) * DOWNSAMPLE

    points = resample_closed(smooth_closed(resample_closed(points, 2.0), 15), SPACING)
    full_dist = edge_distance(road)
    half = full_dist[points[:, 0].astype(int), points[:, 1].astype(int)].astype(np.float64)
    assert all(is_on_road(road, x, y) for x, y in points), "centerline left the road"

    # Put point 0 on the start/finish gate (gate 0)
    (ax, ay), (bx, by) = GATES[0]
    start = np.argmin(np.hypot(points[:, 0] - (ax + bx) / 2, points[:, 1] - (ay + by) / 2))
    return np.roll(points, -start, axis=0), np.roll(half, -start)


def save_preview(track, points, half, path):
    surf = track.copy()
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangents /= np.hypot(*tangents.T)[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    for side, color in ((1, (255, 220, 0)), (-1, (0, 200, 255))):
        edge = points + normals * half[:, None] * side
        pygame.draw.lines(surf, color, True, edge.tolist(), 2)
    pygame.draw.lines(surf, (255, 0, 255), True, points.tolist(), 2)
    pygame.draw.circle(surf, (255, 0, 0), points[0], 8)
    pygame.image.save(surf, str(path))


def main():
    track, road = load_track()
    points, half = build(road)

    with open(CENTERLINE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "half_width"])
        writer.writerows((f"{x:.2f}", f"{y:.2f}", f"{w:.2f}") for (x, y), w in zip(points, half))

    preview = HERE / "centerline_preview.png"
    save_preview(track, points, half, preview)

    length = np.hypot(*np.diff(np.vstack((points, points[:1])), axis=0).T).sum()
    print(f"{len(points)} points, lap length {length:.0f} px, "
          f"half-width {half.min():.0f}-{half.max():.0f} px (mean {half.mean():.0f})")
    print(f"Wrote {CENTERLINE_CSV.name} and {preview.name}")


if __name__ == "__main__":
    main()

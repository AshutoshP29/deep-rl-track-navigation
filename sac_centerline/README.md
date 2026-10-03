# SAC + Centerline Progress (planned)

Not implemented yet. The plan:

- **Algorithm:** SAC with continuous actions, steering ∈ [-1, 1] and throttle ∈ [-1, 1].
- **Reward:** progress `Δs` along a closed centerline polyline, a small time penalty, a crash penalty, and an optional lateral-offset penalty `c1 · (d / half_width)²`. Training starts with a small `c1` and anneals it to 0 so the agent can find its own racing line.
- **Physics:** a grip limit, where the maximum turn rate shrinks as speed rises (`v² · curvature ≤ MAX_LAT_ACCEL`). This makes wide-entry / apex / wide-exit lines worthwhile.
- **Observations:** lidar, speed, heading error to the track tangent, lateral offset, and upcoming curvature.
- **Extras:** randomised start position along the centerline, and termination when progress stalls.

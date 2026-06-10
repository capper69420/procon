# procon

1. algorithm saijruulna(fps)
2. tatah, stickman igg huduguungu bailgah jishee
3. unah hurd tootsooloh
4. nuuriig tanij hucilturugc hemjdgee saijruulah

1. mashin surgah ai \muuguntsur bakter hugts\
2. zahialga gargah \ heregtei tehnik tuhuurumjiinhuu\
3. bakter mugntsr zergee haraad ilruuleh
4. prompt:Act as a senior Computer Vision and Python engineer specializing in real-time edge analytics.
Write a complete, production-ready Python module for real-time fall detection using YOLOv8-pose keypoints.

Requirements:
Input:
- YOLOv8-pose keypoints tensor/array in format [17, 3] representing [x_pixel, y_pixel, confidence].
- Image dimensions (width, height) to normalize spatial features and ensure scale invariance.
- Monotonically increasing float timestamp (in seconds) for true time-delta tracking.

Implement a heuristic, multi-gated mathematical pipeline tracking:
1. Shoulder midpoint & Hip midpoint calculation.
2. Trunk angle estimation: Vector angle of the spine relative to the vertical Y-axis using atan2.
3. Vertical velocity & acceleration calculation using rolling history windows and true time deltas (dt).
4. Body height collapse percentage: Current bounding-box/skeleton height compared to a rolling baseline of the person standing.
5. Bounding box aspect ratio tracking (Width/Height tracking over time).
6. Center of mass estimation (approximated via hip/shoulder distribution).
7. Post-fall immobility detection: Flagging if the person remains in a fallen state with near-zero acceleration for X seconds.
8. Temporal majority voting: A rolling frame buffer to smooth out single-frame keypoint jitter.

Create a `FallDetector` class with the following interface:
- __init__(config: FallDetectorConfig)
- update(keypoints: np.ndarray, timestamp: float, img_shape: tuple) -> dict
- _calculate_features()
- _detect_fall()
- reset()

Output structure returned by update():
{
    "fall_detected": bool,
    "confidence": float,        # Average detection confidence of critical trunk keypoints
    "trunk_angle": float,       # In degrees
    "velocity": float,          # Normalized units per second
    "acceleration": float,      # Normalized units per second^2
    "body_height_ratio": float, # Current height vs standing baseline
    "immobile": bool            # True if motionless post-fall
}

Constraints & Code Style:
- Pure Python 3.10+, NumPy, and standard libraries only (No extra deep learning layers).
- Real-time CPU optimized (avoid intensive per-frame loops; use NumPy vector operations).
- Include strict type hints, Google-style docstrings, and robust error handling for missing/occluded keypoints (low confidence).
- Configuration thresholds (angle bounds, velocity limits, window sizes) must be cleanly contained in a @dataclass.

Provide the complete code in a single file that can be directly dropped into an active OpenCV/YOLOv8 loop.

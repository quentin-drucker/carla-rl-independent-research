"""
test2___loop_utils.py

Goal:
- Hold small per-tick loop utilities for test2 scripts.
- Specifically: safely pull the newest LiDAR frame from a queue and reuse last frame if none arrived.
"""

import queue


def get_latest_lidar_frame(lidar_queue, *, last_lidar_frame, last_lidar_tick, t):
    """
    Step logic (same as your in-main code):
    - Drain the queue to get the most recent frame available right now.
    - If no new frame arrived this tick, keep using last_lidar_frame.
    - Track last_lidar_tick for debugging/analysis.

    Returns:
      lidar_frame, last_lidar_frame, last_lidar_tick
    """
    # Step 1: grab newest lidar frame; if none arrived this tick, reuse last
    new_frame = None
    try:
        while True:
            new_frame = lidar_queue.get_nowait()
    except queue.Empty:
        pass

    if new_frame is not None:
        last_lidar_frame = new_frame
        last_lidar_tick = t

    lidar_frame = last_lidar_frame
    return lidar_frame, last_lidar_frame, last_lidar_tick
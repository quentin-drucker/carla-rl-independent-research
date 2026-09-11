"""
load_town4.py

Forces CARLA to load Town04 (non-Opt version) so the renderer window
actually shows the correct map visually.

Run this BEFORE eval_sac.py whenever you want to watch training.

Usage:
    python load_town4.py
    python eval_sac.py
"""

import sys
import os

CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
for p in [os.path.join(CARLA_ROOT, "PythonAPI"),
          os.path.join(CARLA_ROOT, "PythonAPI", "carla")]:
    if p not in sys.path:
        sys.path.append(p)

import carla

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)

print("Loading Town04 (this takes ~30 seconds)...")
client.load_world("Town04")
world = client.get_world()
print(f"Done. Current map: {world.get_map().name}")
print("You can now run: python eval_sac.py")

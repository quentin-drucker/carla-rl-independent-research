# Manual CARLA driving cheat sheet

Open two PowerShell windows.

## Normal CARLA physics

**Window 1 — start CARLA:**

```powershell
& "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe"
```

**Window 2 — after CARLA loads, open manual driving:**

```powershell
& "C:\Users\qdruc\Projects\Carla Project\venv\Scripts\python.exe" "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\PythonAPI\examples\manual_control.py" --filter vehicle.tesla.model3
```

## Chrono physics

Start CARLA with Chrono enabled:

```powershell
& "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe" --chrono
```

Then run:

```powershell
& "C:\Users\qdruc\Projects\Carla Project\venv\Scripts\python.exe" "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\PythonAPI\examples\manual_control_chrono.py" --filter vehicle.tesla.model3
```

Click the Pygame window. Use `W`/`S` for throttle/brake, `A`/`D` to steer,
and `Ctrl+O` to toggle Chrono/default physics. Press `Esc` to close the manual
client cleanly, then close CARLA.

The stock `manual_control_chrono.py` only prints `o pressed` on `Ctrl+O`, which does
not say which physics is now active. Quentin's local CARLA install has a patched copy
(not part of this repository) that prints `CHRONO PHYSICS ENABLED` /
`DEFAULT PHYSICS ENABLED` instead. On an unpatched install, count your presses: odd =
Chrono, even = default.

Chrono is for exploration only. The research backend is CARLA's default physics (Week 4
decision; see `MASTER_CARLA_RESEARCH_SUMMARY.md`).

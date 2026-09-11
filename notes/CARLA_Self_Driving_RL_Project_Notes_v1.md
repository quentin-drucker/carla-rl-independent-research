# CARLA Self-Driving RL Project Notes (v1)

These notes consolidate the setup, scope, scenario design rules, and debugging guidance for training a single ego vehicle in **CARLA 0.9.16** using **reinforcement learning (RL)** with a **discrete action space**.

---

## 1) Project Goal (Research Focus)

This project is **not** aiming for:
- perfect driving
- human-level autonomy

This project **is** aiming for:
- measurable safety improvements
- parameterized rare scenarios
- controlled comparisons

The emphasis is on **research structure**, not a flashy demo.

---

## 2) RL Guardrails (v1 Scope)

### In scope
- Single ego vehicle
- Discrete action space
- RGB camera observation
- Rule-based traffic for other vehicles

### Explicitly out of scope (for v1)
- Multi-agent RL
- Continuous control
- Sensor fusion
- End-to-end autonomy

These may (or may not) be future extensions once the v1 pipeline works reliably.

---

## 3) CARLA Mental Model (How the system works)

### Server–Client Architecture
- `CarlaUE4.exe` = **simulation server** (physics + rendering)
- Python scripts = **client** (logic + control + training)

The server does nothing by itself — **everything is driven by Python**.

### CARLA handles
- Physics
- Sensors
- Maps
- Actors (vehicles, pedestrians)

### Python handles
- Spawning actors
- Vehicle control
- Scenario logic
- Reward calculation
- Training loops

---

## 4) Scenario Design Rules (Very important)

### Core design choice
- Scenario-driven episodes (**NOT** free driving)
- One hazard per episode
- Reset world every episode

### Each scenario must define
- Ego spawn point
- Hazard trigger condition
- Success condition
- Failure condition

### Parameterize everything (avoid hardcoding)
Try to define these as variables/config values instead of hard-coded constants:
- Distances
- Speeds
- Timing
- Road friction
- Occlusion
- Etc. (these are just some examples--but may not consider all of them for all scenarios)

This makes experiments fair, repeatable, and easy to compare.

---

## 5) Setup Quick Reference (CARLA + Python)

### Python / virtual environment
- CARLA 0.9.16 requires **Python 3.12.x**
- **Do NOT** use Python 3.14 with CARLA
- Always activate your virtual environment before running scripts

```powershell
# Project root
cd "C:\Users\qdruc\OneDrive\Desktop\Carla Project"

# Activate venv
.\venv\Scripts\activate
```

### CARLA server
- Start `CarlaUE4.exe` **BEFORE** running Python scripts
- Default connection: `localhost:2000`

### Sanity check
```powershell
python -c "import carla; client=carla.Client('localhost',2000); print(client.get_world().get_map().name)"
```

---

## 6) Standard Workflow (Common Commands)

### Standard workflow
1. Open `CarlaUE4.exe` (in the project folder @ Desktop/CARLA_0.9.16/)
2. Activate virtual environment
3. Run Python script

### Activate venv
```powershell
.\venv\Scripts\activate
```

### Run a script
```powershell
python main.py
```

### Stop everything
- Close the CARLA window  
**OR**
- Press `Ctrl + C` in terminal

---

## 7) How to Activate the Python Virtual Environment (venv)

```powershell
# Navigate to the project root
cd "C:\Users\qdruc\OneDrive\Desktop\Carla Project"

# Activate the virtual environment
.\venv\Scripts\activate

# Verify Python version (should be 3.12.x (I'm currently using 3.12.10))
python --version

# Quick CARLA sanity check
python -c "import carla; print('carla import OK')"
```

---

## 8) CARLA Pitfalls & Tips

### Common pitfalls
- Forgetting to activate venv
- Using the wrong Python version
- Running Python before the CARLA server starts
- Not destroying actors on reset
- CARLA version / wheel mismatch

### Performance tips
- Start with Town01 or Town03
- Limit NPC vehicles during training
- Lower graphics settings if needed

---

## 9) Debugging Checklist

### Cannot connect to CARLA
- Is `CarlaUE4.exe` running?
- Correct port (2000)?
- Firewall blocking localhost?

### `import carla` fails
- Is venv active?
- Python version = 3.12.x?
- CARLA wheel installed in this venv?

### CARLA crashes
- GPU driver issues
- Too many actors
- VRAM exhaustion

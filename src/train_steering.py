"""train_steering.py

Train a SAC steering policy in CarlaAEBEnv's steering mode (Phase 2, Week 5
plan step 5). One run trains ONE commitment version; run it once with
--commit lock_at_onset and once with --commit per_tick to compare them.

Training scenarios (MASTER, "Phase 2 design decisions" 7): the test26
stationary pedestrian (steer_brake_baseline.baseline_scenario_config) with
speed uniform in 30-50 mph and onset TTC uniform in 0.3-1.5 s, sampled per
episode from a seeded generator. Week 7: --walker samples uniformly from
the given pedestrian cases (e.g. the dynamic pair right_cross /
right_turn_back), with --ttc-range, --lock-delay-s and --road-right-limit-m;
the defaults reproduce the Week 5 setup.

The approach to the pedestrian is fast-forwarded: the environment ignores
the policy before onset, so those ticks are stepped inside reset() with a
dummy action until one second before the hazard starts
(rl_steering.policy_should_act). Every transition SAC learns from is then a
decision that can change the outcome. The dummy ticks earn no reward
(comfort is charged only after onset; contact is impossible that far out).

SAC settings are train_sac.py's (v3), unchanged, so the steering result is
not confounded by new hyperparameters.

Long-run pattern (as test27): progress.log with one line per episode,
episodes.csv, checkpoints with the replay buffer, --resume, --relaunch-carla.

Usage (CARLA running, default physics; from src/ in the venv):
    python -u -X utf8 train_steering.py --commit lock_at_onset --steps 50000
    python -u -X utf8 train_steering.py --resume runs/steering_train/<run dir>
    add --relaunch-carla to restart a crashed CARLA server (renderless) and continue

Evidence (git-ignored):
    src/runs/steering_train/<timestamp>_steering_train_<commit>_default/
        manifest.json, progress.log, episodes.csv, checkpoints/, final_model.zip
"""

import argparse
import csv
import dataclasses
import datetime
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

import gymnasium
import numpy as np

from physics_backend import BACKEND_DEFAULT, create_run_dir
from rl_steering import STEER_COMMIT_MODES, SteeringRewardConfig, policy_should_act

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(SRC_DIR, "runs", "steering_train")
MPH_RANGE = (30.0, 50.0)
TTC_RANGE_S = (0.3, 1.5)
HAND_OVER_LEAD_S = 1.0
SAC_KWARGS = dict(buffer_size=100_000, batch_size=512, learning_rate=3e-4, tau=0.005, ent_coef="auto")  # = train_sac v3
EPISODE_COLUMNS = ["episode", "timesteps", "walker", "mph", "ttc_s", "steps", "reward", "contact", "contact_speed_mps",
                   "min_clearance_m", "went_off_road", "u_applied", "final_lateral_m", "final_speed_mps",
                   "missing_passage_ticks", "wall_s"]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, cwd=SRC_DIR).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAC steering training (see docstring)")
    p.add_argument("--commit", choices=STEER_COMMIT_MODES)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=5_000)
    p.add_argument("--walker", nargs="+", default=["stationary"],
                   help="pedestrian cases sampled uniformly (steer_brake_baseline WALKER_CASES / DYNAMIC_WALKER_CASES)")
    p.add_argument("--walker-speed", type=float, help="walking speed in m/s (default: the scenario default)")
    p.add_argument("--walker-change-s", type=float, help="change time of a dynamic walker (default: its default)")
    p.add_argument("--ttc-range", type=float, nargs=2, default=list(TTC_RANGE_S), metavar=("LO", "HI"))
    p.add_argument("--lock-delay-s", type=float, default=0.0, help="lock_at_onset locks this long after onset")
    p.add_argument("--road-right-limit-m", type=float, help="cap the passage's right road edge (default: whole road)")
    p.add_argument("--resume", metavar="RUN_DIR", help="continue a run from its latest checkpoint")
    p.add_argument("--relaunch-carla", action="store_true",
                   help="if CARLA stops answering, restart it renderless (default physics, no --chrono) and continue")
    args = p.parse_args(argv)
    if not (args.commit or args.resume):
        p.error("give --commit for a new run, or --resume")
    return args


class SkipApproach(gymnasium.Wrapper):
    """Fast-forwards the approach inside reset() (see the module docstring)
    and samples a fresh training scenario for every episode."""

    def __init__(self, env, rng, scenario=None):
        super().__init__(env)
        self.rng, self.scenario = rng, scenario or {}

    def reset(self, **kwargs):
        self.env.unwrapped.cfg = sample_scenario(self.rng, **self.scenario)
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        while not policy_should_act(dist_travelled_m=base._dist_m, encounter_distance_m=base.cfg.encounter_distance_m,
                                    trigger_ttc_s=base.cfg.trigger_ttc_s, speed_mps=info.get("ego_speed_mps", 0.0),
                                    lead_s=HAND_OVER_LEAD_S):
            obs, _, terminated, truncated, info = self.env.step(np.zeros(2, dtype=np.float32))
            if info["triggered"] or terminated or truncated:
                raise RuntimeError(f"approach fast-forward overshot onset: {info}")
        return obs, info

    def step(self, action):
        # The scenario goes into info here: SB3 auto-resets a finished episode
        # before callbacks run, so env.cfg is already the next one by then.
        obs, reward, terminated, truncated, info = self.env.step(action)
        cfg = self.env.unwrapped.cfg
        info.update(scenario_mph=cfg.target_mph, scenario_ttc_s=cfg.trigger_ttc_s, scenario_run_id=cfg.run_id)
        return obs, reward, terminated, truncated, info


def sample_scenario(rng, *, walkers=("stationary",), ttc_range=TTC_RANGE_S, walker_speed_mps=None,
                    walker_change_s=None):
    """One training scenario. With the defaults this draws exactly what Week 5 drew
    (mph, then TTC), so old seeds reproduce; a walker is drawn only if there is a choice."""
    from steer_brake_baseline import baseline_scenario_config

    mph, ttc = rng.uniform(*MPH_RANGE), rng.uniform(*ttc_range)
    walker = walkers[0] if len(walkers) == 1 else rng.choice(list(walkers))
    tag = "" if walker == "stationary" else f"_{walker}"
    return baseline_scenario_config(mph, ttc, walker=walker, walker_speed_mps=walker_speed_mps,
                                    walker_change_s=walker_change_s, run_id=f"train_{mph:.1f}mph_ttc{ttc:.2f}{tag}")


def _make_callbacks(run_dir, log, start_time, checkpoint_every):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class EpisodeLog(BaseCallback):
        """One progress.log line and one episodes.csv row per finished episode."""

        def __init__(self):
            super().__init__()
            self.ep_reward, self.ep_steps, self.recent = 0.0, 0, []
            self.n_episodes = _count_rows(os.path.join(run_dir, "episodes.csv"))

        def _on_step(self):
            self.ep_reward += float(self.locals["rewards"][0])
            self.ep_steps += 1
            if not self.locals["dones"][0]:
                return True
            info = self.locals["infos"][0]
            self.n_episodes += 1
            self.recent = (self.recent + [self.ep_reward])[-20:]
            row = {
                "episode": self.n_episodes, "timesteps": self.num_timesteps,
                "walker": info["scenario_run_id"].split("_", 3)[3] if info["scenario_run_id"].count("_") >= 3 else "stationary",
                "mph": round(info["scenario_mph"], 2),
                "ttc_s": round(info["scenario_ttc_s"], 3), "steps": self.ep_steps, "reward": round(self.ep_reward, 3),
                "contact": info["collision"], "contact_speed_mps": round(info["ego_speed_mps"], 3) if info["collision"] else None,
                "min_clearance_m": info["min_clearance_m"], "went_off_road": info["went_off_road"],
                "u_applied": round(info["u_applied"], 3), "final_lateral_m": round(info["ego_lateral_m"], 3),
                "final_speed_mps": round(info["ego_speed_mps"], 3), "missing_passage_ticks": info["missing_passage_ticks"],
                "wall_s": round(time.time() - start_time, 1),
            }
            _append_csv(os.path.join(run_dir, "episodes.csv"), row)
            clr = "-" if row["min_clearance_m"] is None else f"{row['min_clearance_m']:.2f} m"
            result = (f"HIT {row['contact_speed_mps']:.1f} m/s" if row["contact"] else f"clear {clr}") + \
                     (" OFF-ROAD" if row["went_off_road"] else "")
            log(f"[ep {self.n_episodes:4d}] {self.num_timesteps:7d} steps | {row['mph']:4.1f} mph ttc {row['ttc_s']:.2f} | "
                f"u {row['u_applied']:.2f} | {result:22s} | reward {self.ep_reward:8.1f} | "
                f"mean of last {len(self.recent):2d}: {sum(self.recent) / len(self.recent):8.1f} | "
                f"{(time.time() - start_time) / 60:5.1f} min")
            self.ep_reward, self.ep_steps = 0.0, 0
            return True

    return [EpisodeLog(), CheckpointCallback(save_freq=checkpoint_every, save_path=os.path.join(run_dir, "checkpoints"),
                                             name_prefix="steering", save_replay_buffer=True)]


def _count_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _append_csv(path, row):
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def _latest_checkpoint(run_dir):
    """(model path, replay buffer path, timesteps) of the newest checkpoint, or None."""
    found = []
    for path in glob.glob(os.path.join(glob.escape(run_dir), "checkpoints", "steering_*_steps.zip")):
        m = re.search(r"steering_(\d+)_steps\.zip$", path)
        buffer = os.path.join(run_dir, "checkpoints", f"steering_replay_buffer_{m.group(1)}_steps.pkl")
        if m and os.path.isfile(buffer):
            found.append((int(m.group(1)), path, buffer))
    if not found:
        return None
    steps, path, buffer = max(found)
    return path, buffer, steps


def main(argv=None):
    args = _parse_args(argv)
    from stable_baselines3 import SAC
    from carla_aeb_env import CarlaAEBEnv
    from test27___fixed_profile_rescore import _carla_port_open, _relaunch_carla

    if args.resume:
        run_dir = args.resume
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest = json.load(f)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = create_run_dir(runs_root=RUNS_ROOT, timestamp=timestamp,
                                 run_family=f"steering_train_{args.commit}", backend=BACKEND_DEFAULT)
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        manifest = {
            "created_at_iso": datetime.datetime.now().isoformat(), "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "git_tracked_changes": dirty,
            "git_dirty": bool(dirty), "physics_backend": BACKEND_DEFAULT, "steer_commit": args.commit,
            "u_bounds": [0.0, 1.0], "total_steps": args.steps, "seed": args.seed,
            "checkpoint_every": args.checkpoint_every, "mph_range": MPH_RANGE, "ttc_range_s": list(args.ttc_range),
            "scenario": "steer_brake_baseline.baseline_scenario_config (Town04)", "walkers": args.walker,
            "walker_speed_mps": args.walker_speed, "walker_change_s": args.walker_change_s,
            "lock_delay_s": args.lock_delay_s, "road_right_limit_m": args.road_right_limit_m,
            "hand_over_lead_s": HAND_OVER_LEAD_S, "sac": SAC_KWARGS,
            "reward": dataclasses.asdict(SteeringRewardConfig()),
            "python": sys.version,
        }
        with open(os.path.join(run_dir, "manifest.json"), "x") as f:
            json.dump(manifest, f, indent=2)
    commit, total, seed = manifest["steer_commit"], manifest["total_steps"], manifest["seed"]
    if args.resume and _latest_checkpoint(run_dir) is None:
        raise SystemExit(f"--resume: no checkpoint with a replay buffer in {run_dir}; refusing to start over there")
    if os.path.isfile(os.path.join(run_dir, "final_model.zip")):
        raise SystemExit(f"{run_dir} already has final_model.zip; nothing to do")
    log_path = os.path.join(run_dir, "progress.log")
    start_time = time.time()

    def log(line):
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def make_env(timesteps_done):
        # Reseeding from the step count makes a resumed run deterministic too, though not
        # identical to an uninterrupted one.
        # .get(): Week 5 manifests predate these keys and mean the stationary defaults.
        env = CarlaAEBEnv(steer_commit=commit, lock_delay_s=manifest.get("lock_delay_s", 0.0),
                          road_right_limit_m=manifest.get("road_right_limit_m"))
        scenario = dict(walkers=manifest.get("walkers", ["stationary"]),
                        ttc_range=manifest.get("ttc_range_s", TTC_RANGE_S),
                        walker_speed_mps=manifest.get("walker_speed_mps"),
                        walker_change_s=manifest.get("walker_change_s"))
        return SkipApproach(env, random.Random(seed * 1_000_003 + timesteps_done), scenario)

    env, model, attempts = None, None, 0
    while True:
        try:
            ckpt = _latest_checkpoint(run_dir)
            env = make_env(0 if ckpt is None else ckpt[2])
            if ckpt is None:
                model = SAC("MlpPolicy", env, verbose=0, seed=seed, **SAC_KWARGS)
                log(f"[train] new run: {commit}, {total} steps, seed {seed} | evidence: {run_dir}")
            else:
                model = SAC.load(ckpt[0], env=env)
                model.load_replay_buffer(ckpt[1])
                log(f"[train] resumed {commit} from {os.path.basename(ckpt[0])} ({model.num_timesteps} steps done)")
            # ponytail: a crash loses the steps since the last checkpoint, and their episodes.csv rows stay
            # (numbering continues). Fine for smoke runs; checkpoint more often for long ones.
            callbacks = _make_callbacks(run_dir, log, start_time, manifest["checkpoint_every"])
            remaining = total - model.num_timesteps
            if remaining > 0:
                model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
            model.save(os.path.join(run_dir, "final_model"))
            log(f"[train] DONE {model.num_timesteps} steps | final model: {os.path.join(run_dir, 'final_model.zip')}")
            return 0
        except KeyboardInterrupt:
            log(f"[train] interrupted; resume with --resume {run_dir}")
            raise
        except Exception as exc:  # a CARLA crash must not lose the run
            attempts += 1
            log(f"[train] FAILED ({type(exc).__name__}: {exc}); attempt {attempts}")
            if not args.relaunch_carla or attempts >= 3:
                log(f"[train] STOPPING; resume with --resume {run_dir}")
                return 1
            if not _carla_port_open():
                _relaunch_carla(lambda line: log(line.replace("[test27]", "[train]")))
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
                env = None


if __name__ == "__main__":
    sys.exit(main())

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physics_backend import (  # noqa: E402
    BACKEND_CHRONO,
    BACKEND_DEFAULT,
    CHRONO_STATUS_API_CALL_COMPLETED,
    CHRONO_STATUS_FAILED,
    CHRONO_STATUS_NOT_ATTEMPTED,
    CHRONO_STATUS_NOT_REQUESTED,
    RECORDED_CHRONO,
    RECORDED_CHRONO_ENABLE_FAILED,
    RECORDED_CHRONO_INVALIDATED_BY_COLLISION,
    RECORDED_DEFAULT,
    SMOKE_SCHEDULE,
    SMOKE_SCHEDULE_VERSION,
    BackendConfigError,
    build_chrono_config,
    check_initial_state,
    cmdline_has_chrono_flag,
    collect_template_references,
    create_run_dir,
    phase_tick_count,
    resolve_recorded_backend,
    run_preflight,
    schedule_to_dicts,
    strip_json_comments,
    summarize_server_processes,
    validate_backend_name,
    validate_chrono_config,
)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _make_fake_install(root):
    """Minimal stand-in for CARLA_0.9.16 with the same sedan template layout
    and the same kind of nested references the real Sedan_Vehicle.json has."""
    base = os.path.join(root, "Co-Simulation", "Chrono", "Vehicles")
    _write_json(os.path.join(base, "sedan/vehicle/Sedan_Vehicle.json"), {
        "Chassis": {"Input File": "sedan/chassis/Sedan_Chassis.json"},
        "Axles": [{"Left Brake Input File": "sedan/brake/Sedan_BrakeSimple.json",
                   "Right Brake Input File": "sedan/brake/Sedan_BrakeSimple.json"}],
    })
    _write_json(os.path.join(base, "sedan/chassis/Sedan_Chassis.json"),
                {"Mesh": "sedan/sedan_chassis_vis.obj"})
    _write_json(os.path.join(base, "sedan/brake/Sedan_BrakeSimple.json"), {"Max": 4000})
    _write_json(os.path.join(base, "sedan/powertrain/Sedan_SimpleMapPowertrain.json"), {"Type": "x"})
    _write_json(os.path.join(base, "sedan/tire/Sedan_TMeasyTire.json"),
                {"Visualization": {"Mesh Filename Left": "sedan/sedan_tire.obj"}})
    for mesh in ("sedan/sedan_chassis_vis.obj", "sedan/sedan_tire.obj"):
        with open(os.path.join(base, mesh), "w") as f:
            f.write("o mesh\n")
    return base


class BackendNameTests(unittest.TestCase):
    def test_accepts_the_two_supported_backends(self):
        self.assertEqual(validate_backend_name("default"), "default")
        self.assertEqual(validate_backend_name("chrono"), "chrono")

    def test_rejects_anything_else(self):
        for bad in ("Chrono", "physx", "", "carsim"):
            with self.assertRaises(BackendConfigError):
                validate_backend_name(bad)


class ChronoConfigTests(unittest.TestCase):
    def test_base_dir_defaults_under_carla_root(self):
        cfg = build_chrono_config(carla_root=os.path.join("C:", "CARLA"))
        self.assertEqual(cfg.base_dir, os.path.join("C:", "CARLA", "Co-Simulation", "Chrono", "Vehicles"))

    def test_uses_supplied_sedan_templates_and_official_substeps(self):
        cfg = build_chrono_config(carla_root="x")
        self.assertEqual(cfg.vehicle_json, "sedan/vehicle/Sedan_Vehicle.json")
        self.assertEqual(cfg.powertrain_json, "sedan/powertrain/Sedan_SimpleMapPowertrain.json")
        self.assertEqual(cfg.tire_json, "sedan/tire/Sedan_TMeasyTire.json")
        self.assertEqual(cfg.max_substeps, 5000)
        self.assertAlmostEqual(cfg.max_substep_delta_time_s, 0.002)

    def test_api_base_path_is_absolute_with_trailing_separator(self):
        cfg = build_chrono_config(carla_root="x", base_dir="some_dir")
        self.assertTrue(os.path.isabs(cfg.api_base_path))
        self.assertTrue(cfg.api_base_path.endswith(os.sep))


class StripJsonCommentsTests(unittest.TestCase):
    def test_removes_line_and_block_comments(self):
        text = '{\n  // note\n  "a": 1, /* x */ "b": 2\n}'
        self.assertEqual(json.loads(strip_json_comments(text)), {"a": 1, "b": 2})

    def test_keeps_comment_markers_inside_strings(self):
        text = '{"url": "http://x/y", "s": "a /* not */ b", "q": "say \\"//hi\\""}'
        self.assertEqual(json.loads(strip_json_comments(text)),
                         {"url": "http://x/y", "s": "a /* not */ b", "q": 'say "//hi"'})


class CollectTemplateReferencesTests(unittest.TestCase):
    def test_finds_nested_json_and_obj_references_once_each(self):
        refs = collect_template_references({
            "a": {"Input File": "x/a.json"},
            "b": [{"L": "x/b.json", "R": "x/b.json"}, "x/mesh.obj"],
            "name": "not a file", "number": 3,
        })
        self.assertEqual(refs, ["x/a.json", "x/b.json", "x/mesh.obj"])


class ValidateChronoConfigTests(unittest.TestCase):
    def test_valid_install_returns_hashes_for_top_level_and_nested_files(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_install(root)
            report = validate_chrono_config(build_chrono_config(carla_root=root))
        self.assertEqual(set(report["templates"]), {"vehicle_json", "powertrain_json", "tire_json"})
        for entry in report["templates"].values():
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertIn("sedan/chassis/Sedan_Chassis.json", report["referenced_files"])
        self.assertIn("sedan/brake/Sedan_BrakeSimple.json", report["referenced_files"])
        # Found via the chassis sub-template, i.e. recursion works.
        self.assertIn("sedan/sedan_chassis_vis.obj", report["referenced_files"])
        self.assertIn("sedan/sedan_tire.obj", report["referenced_files"])
        self.assertEqual(report["max_substeps"], 5000)

    def test_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_install(root)
            report = validate_chrono_config(build_chrono_config(carla_root=root))
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_missing_carla_root_and_base_dir_fail(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = build_chrono_config(carla_root=os.path.join(root, "nope"))
            with self.assertRaises(BackendConfigError) as ctx:
                validate_chrono_config(cfg)
        self.assertIn("CARLA root directory not found", str(ctx.exception))
        self.assertIn("Chrono base directory not found", str(ctx.exception))

    def test_every_missing_top_level_template_is_reported_together(self):
        with tempfile.TemporaryDirectory() as root:
            base = _make_fake_install(root)
            os.remove(os.path.join(base, "sedan/powertrain/Sedan_SimpleMapPowertrain.json"))
            os.remove(os.path.join(base, "sedan/tire/Sedan_TMeasyTire.json"))
            with self.assertRaises(BackendConfigError) as ctx:
                validate_chrono_config(build_chrono_config(carla_root=root))
        self.assertIn("powertrain_json", str(ctx.exception))
        self.assertIn("tire_json", str(ctx.exception))

    def test_missing_nested_reference_fails_before_any_live_call(self):
        with tempfile.TemporaryDirectory() as root:
            base = _make_fake_install(root)
            os.remove(os.path.join(base, "sedan/brake/Sedan_BrakeSimple.json"))
            with self.assertRaises(BackendConfigError) as ctx:
                validate_chrono_config(build_chrono_config(carla_root=root))
        self.assertIn("Sedan_BrakeSimple.json", str(ctx.exception))

    def test_unparseable_template_fails(self):
        with tempfile.TemporaryDirectory() as root:
            base = _make_fake_install(root)
            with open(os.path.join(base, "sedan/tire/Sedan_TMeasyTire.json"), "w") as f:
                f.write("{ not json")
            with self.assertRaises(BackendConfigError) as ctx:
                validate_chrono_config(build_chrono_config(carla_root=root))
        self.assertIn("not readable JSON", str(ctx.exception))

    def test_commented_template_like_supplied_tmeasy_tire_is_accepted(self):
        # The real Sedan_TMeasyTire.json uses // comments (RapidJSON-style),
        # found live 2026-10-01; strict json.load rejected it.
        with tempfile.TemporaryDirectory() as root:
            base = _make_fake_install(root)
            with open(os.path.join(base, "sedan/tire/Sedan_TMeasyTire.json"), "w") as f:
                f.write('{\n  "Name": "Sedan TMeasy Tire",\n\n  // ------\n  // Tire design (REQUIRED)\n'
                        '  "Design": {"Radius": 0.33}, /* block */\n'
                        '  "Visualization": {"Mesh Filename Left": "sedan/sedan_tire.obj"}\n}\n')
            report = validate_chrono_config(build_chrono_config(carla_root=root))
        self.assertIn("sedan/sedan_tire.obj", report["referenced_files"])

    def test_validation_does_not_modify_templates(self):
        with tempfile.TemporaryDirectory() as root:
            base = _make_fake_install(root)
            path = os.path.join(base, "sedan/vehicle/Sedan_Vehicle.json")
            before = Path(path).read_bytes()
            validate_chrono_config(build_chrono_config(carla_root=root))
            self.assertEqual(Path(path).read_bytes(), before)


class ServerChronoFlagTests(unittest.TestCase):
    def test_detects_flag_in_real_windows_cmdlines(self):
        # Exactly the two command lines observed live on 2026-10-01.
        self.assertTrue(cmdline_has_chrono_flag(
            '"C:\\Users\\qdruc\\OneDrive\\Desktop\\CARLA_0.9.16\\CarlaUE4.exe" --chrono '))
        self.assertTrue(cmdline_has_chrono_flag(
            '"C:\\...\\CarlaUE4-Win64-Shipping.exe" CarlaUE4 --chrono'))

    def test_rejects_missing_or_lookalike_flags(self):
        self.assertFalse(cmdline_has_chrono_flag('"C:\\CARLA\\CarlaUE4.exe" -quality-level=Low'))
        self.assertFalse(cmdline_has_chrono_flag('"C:\\Chrono\\CarlaUE4.exe" --chronology'))
        self.assertFalse(cmdline_has_chrono_flag(""))

    def test_summary_distinguishes_unqueried_from_absent(self):
        self.assertIsNone(summarize_server_processes(None)["any_has_chrono_flag"])
        self.assertFalse(summarize_server_processes([])["any_has_chrono_flag"])
        summary = summarize_server_processes([
            {"pid": 1, "name": "CarlaUE4.exe", "cmdline": "CarlaUE4.exe --chrono"},
        ])
        self.assertTrue(summary["queried"])
        self.assertTrue(summary["any_has_chrono_flag"])

    def test_summary_tolerates_missing_cmdline(self):
        summary = summarize_server_processes([{"pid": 1, "name": "CarlaUE4.exe", "cmdline": None}])
        self.assertFalse(summary["any_has_chrono_flag"])


class RunPreflightTests(unittest.TestCase):
    _CHRONO_PROC = [{"pid": 1, "name": "CarlaUE4.exe", "cmdline": "CarlaUE4.exe --chrono"}]
    _PLAIN_PROC = [{"pid": 1, "name": "CarlaUE4.exe", "cmdline": "CarlaUE4.exe"}]

    def test_default_backend_needs_no_chrono_files(self):
        cfg, report, summary = run_preflight(
            backend="default", carla_root="does/not/exist", chrono_base_dir=None,
            host="localhost", process_query=lambda: self._PLAIN_PROC)
        self.assertIsNone(cfg)
        self.assertIsNone(report)
        self.assertFalse(summary["any_has_chrono_flag"])

    def test_chrono_passes_with_valid_files_and_flag(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_install(root)
            cfg, report, _ = run_preflight(
                backend="chrono", carla_root=root, chrono_base_dir=None,
                host="localhost", process_query=lambda: self._CHRONO_PROC)
        self.assertEqual(cfg.max_substeps, 5000)
        self.assertIn("templates", report)

    def test_chrono_refused_without_server_flag(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_install(root)
            with self.assertRaises(BackendConfigError) as ctx:
                run_preflight(backend="chrono", carla_root=root, chrono_base_dir=None,
                              host="localhost", process_query=lambda: self._PLAIN_PROC)
        self.assertIn("--chrono", str(ctx.exception))

    def test_chrono_refused_when_processes_cannot_be_queried(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_install(root)
            with self.assertRaises(BackendConfigError):
                run_preflight(backend="chrono", carla_root=root, chrono_base_dir=None,
                              host="localhost", process_query=lambda: None)

    def test_chrono_refused_on_remote_host(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_install(root)
            with self.assertRaises(BackendConfigError):
                run_preflight(backend="chrono", carla_root=root, chrono_base_dir=None,
                              host="10.0.0.5", process_query=lambda: self._CHRONO_PROC)

    def test_bad_paths_fail_before_process_flag_matters(self):
        with self.assertRaises(BackendConfigError):
            run_preflight(backend="chrono", carla_root="does/not/exist", chrono_base_dir=None,
                          host="localhost", process_query=lambda: self._CHRONO_PROC)


class ResolveRecordedBackendTests(unittest.TestCase):
    def test_default_run_is_labelled_default(self):
        self.assertEqual(resolve_recorded_backend(
            requested_backend=BACKEND_DEFAULT, chrono_enable_status=CHRONO_STATUS_NOT_REQUESTED,
            collision_detected=False), RECORDED_DEFAULT)

    def test_default_run_with_chrono_status_is_rejected(self):
        with self.assertRaises(BackendConfigError):
            resolve_recorded_backend(
                requested_backend=BACKEND_DEFAULT,
                chrono_enable_status=CHRONO_STATUS_API_CALL_COMPLETED, collision_detected=False)

    def test_clean_chrono_run_is_labelled_chrono(self):
        self.assertEqual(resolve_recorded_backend(
            requested_backend=BACKEND_CHRONO, chrono_enable_status=CHRONO_STATUS_API_CALL_COMPLETED,
            collision_detected=False), RECORDED_CHRONO)

    def test_chrono_never_labelled_chrono_unless_enable_completed(self):
        for status in (CHRONO_STATUS_FAILED, CHRONO_STATUS_NOT_ATTEMPTED, CHRONO_STATUS_NOT_REQUESTED):
            for collided in (False, True):
                label = resolve_recorded_backend(
                    requested_backend=BACKEND_CHRONO, chrono_enable_status=status,
                    collision_detected=collided)
                self.assertEqual(label, RECORDED_CHRONO_ENABLE_FAILED)

    def test_collision_invalidates_chrono_label(self):
        self.assertEqual(resolve_recorded_backend(
            requested_backend=BACKEND_CHRONO, chrono_enable_status=CHRONO_STATUS_API_CALL_COMPLETED,
            collision_detected=True), RECORDED_CHRONO_INVALIDATED_BY_COLLISION)


class ScheduleTests(unittest.TestCase):
    def test_schedule_exercises_accel_coast_steer_and_brake(self):
        self.assertTrue(any(p.throttle > 0 for p in SMOKE_SCHEDULE))
        self.assertTrue(any(p.throttle == 0 and p.brake == 0 and p.steer == 0 for p in SMOKE_SCHEDULE))
        self.assertTrue(any(p.steer > 0 for p in SMOKE_SCHEDULE))
        self.assertTrue(any(p.steer < 0 for p in SMOKE_SCHEDULE))
        self.assertEqual(SMOKE_SCHEDULE[-1].name, "brake")
        self.assertTrue(SMOKE_SCHEDULE[-1].end_when_stopped)

    def test_braking_starts_after_a_pure_coast_with_wheel_centred(self):
        coast, brake = SMOKE_SCHEDULE[-2], SMOKE_SCHEDULE[-1]
        self.assertEqual((coast.throttle, coast.brake, coast.steer), (0.0, 0.0, 0.0))
        self.assertEqual((brake.throttle, brake.steer), (0.0, 0.0))

    def test_schedule_version_is_recorded(self):
        self.assertEqual(SMOKE_SCHEDULE_VERSION, 2)

    def test_steering_is_modest_and_net_symmetric(self):
        steers = [p.steer * p.duration_s for p in SMOKE_SCHEDULE]
        self.assertAlmostEqual(sum(steers), 0.0)
        self.assertTrue(all(abs(p.steer) <= 0.1 for p in SMOKE_SCHEDULE))

    def test_controls_are_within_carla_ranges(self):
        for p in SMOKE_SCHEDULE:
            self.assertTrue(0.0 <= p.throttle <= 1.0)
            self.assertTrue(0.0 <= p.brake <= 1.0)
            self.assertTrue(-1.0 <= p.steer <= 1.0)
            self.assertFalse(p.throttle > 0 and p.brake > 0, p.name)

    def test_tick_counts_at_fixed_dt(self):
        self.assertEqual(phase_tick_count(SMOKE_SCHEDULE[0], 0.02), 200)
        self.assertEqual([phase_tick_count(p, 0.02) for p in SMOKE_SCHEDULE],
                         [200, 50, 50, 50, 250])

    def test_schedule_serializes_for_manifest(self):
        dicts = schedule_to_dicts(SMOKE_SCHEDULE)
        self.assertEqual(json.loads(json.dumps(dicts)), dicts)
        self.assertEqual(dicts[0]["name"], "accelerate")


class InitialStateTests(unittest.TestCase):
    def test_at_rest_and_level_passes(self):
        self.assertEqual(check_initial_state(speed_mps=0.0, roll_deg=0.2, pitch_deg=-0.5), [])

    def test_moving_or_tilted_vehicle_is_reported(self):
        problems = check_initial_state(speed_mps=0.5, roll_deg=4.0, pitch_deg=-5.0)
        self.assertEqual(len(problems), 3)


class CreateRunDirTests(unittest.TestCase):
    def test_creates_backend_labelled_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path = create_run_dir(runs_root=os.path.join(root, "physics"), timestamp="20261001_120000",
                                  run_family="chrono_smoke", backend="chrono")
            self.assertTrue(os.path.isdir(path))
            self.assertTrue(path.endswith("20261001_120000_chrono_smoke_chrono"))

    def test_refuses_to_reuse_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as root:
            kwargs = dict(runs_root=root, timestamp="t", run_family="f", backend="default")
            create_run_dir(**kwargs)
            with self.assertRaises(FileExistsError):
                create_run_dir(**kwargs)

    def test_rejects_unknown_backend(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(BackendConfigError):
                create_run_dir(runs_root=root, timestamp="t", run_family="f", backend="physx")


if __name__ == "__main__":
    unittest.main()

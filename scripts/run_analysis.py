#!/usr/bin/env python3
"""Two-pass Photonic Cavity Analysis Orchestrator.

This script coordinates a two-stage workflow:

1. Scout stage: run a broadband simulation (or use existing results) to locate the
   resonance frequency and quality factor using the existing Q-factor tools.
2. Lock-in stage: build a narrowband simulation at the detected resonance and run
   all detailed analyses (mode volume, polarization, near-/far-field, collection
   efficiency).

Geometry generation and analysis modules remain unchanged; this script simply
reuses them in a structured workflow with a small CLI for convenience.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

warnings.filterwarnings("ignore")

# Ensure repository root is on sys.path so we can import the top-level `modules` package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.simulation_setup import SimulationSetup  # noqa: E402
from modules.simulation_runner import SimulationRunner  # noqa: E402
from modules.q_factor_analysis import analyze_q_factor  # noqa: E402
from modules.mode_volume_analysis import analyze_mode_volume  # noqa: E402
from modules.polarization_analysis import analyze_polarization  # noqa: E402
from modules.nearfield_analysis import analyze_nearfield  # noqa: E402
from modules.farfield_analysis import analyze_farfield  # noqa: E402
from modules.collection_efficiency_analysis import analyze_collection_efficiency  # noqa: E402

# Physical constant (shared with analysis modules)
C0 = 299_792_458.0  # Speed of light in vacuum (m/s)


@dataclass
class StageOptions:
    """Configuration for an individual workflow stage."""

    name: str
    run_time_ps: float
    bandwidth_rel: float
    simulation_path: Path
    results_path: Path
    summary_path: Path
    task_name: str
    run_simulation: bool = False
    force_rerun: bool = False
    estimate_cost: bool = False


@dataclass
class WorkflowConfig:
    """Top-level configuration for the two-stage workflow."""

    thickness_um: float = 0.14
    initial_wavelength_um: float = 0.650
    stage: str = "all"
    n_bg: float = 1.0
    NA: float = 0.9
    resonance_wavelength_um: Optional[float] = None
    scout: Optional[StageOptions] = None
    lockin: Optional[StageOptions] = None
    cavity_gds: Optional[str] = None
    holes_gds: Optional[str] = None
    sidewall_angle_deg: float = 0.0
    trapezoid_slices: int = 1
    auto_confirm: bool = False


def format_thickness_tag(value: float) -> str:
    """Convert a thickness value into a compact string for filenames."""

    return f"{value:.3f}".rstrip("0").rstrip(".")


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for *path* if needed."""

    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/Path objects into JSON-friendly values."""

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Persist *data* as pretty-printed JSON."""

    ensure_parent_dir(path)
    with path.open("w") as fh:
        json.dump(sanitize_for_json(data), fh, indent=2)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON data from *path* if it exists."""

    if path is None or not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)


def build_default_workflow_config(
    thickness_um: float = 0.14,
    initial_wavelength_um: float = 0.650,
) -> WorkflowConfig:
    """Create a workflow configuration with sensible defaults."""

    tag = format_thickness_tag(thickness_um)
    scout = StageOptions(
        name="scout",
        run_time_ps=12.0,
        bandwidth_rel=0.12,
        simulation_path=(REPO_ROOT / f"data/simulations/simulation_scout_q_only_{tag}um.json"),
        results_path=(REPO_ROOT / f"data/results/results_scout_q_only_{tag}um.hdf5"),
        summary_path=(REPO_ROOT / f"data/summaries/scout_summary_{tag}um.json"),
        task_name=f"photonic_cavity_scout_{tag}um",
    )
    lockin = StageOptions(
        name="lockin",
        run_time_ps=8.0,
        bandwidth_rel=0.02,
        simulation_path=(REPO_ROOT / f"data/simulations/simulation_lockin_full_{tag}um.json"),
        results_path=(REPO_ROOT / f"data/results/results_lockin_full_{tag}um.hdf5"),
        summary_path=(REPO_ROOT / f"data/summaries/lockin_summary_{tag}um.json"),
        task_name=f"photonic_cavity_lockin_{tag}um",
    )
    return WorkflowConfig(
        thickness_um=thickness_um,
        initial_wavelength_um=initial_wavelength_um,
        scout=scout,
        lockin=lockin,
    )


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    """Create a workflow configuration from CLI arguments."""

    config = build_default_workflow_config(
        thickness_um=args.thickness_um,
        initial_wavelength_um=args.initial_wavelength_um,
    )
    config.stage = args.stage
    config.n_bg = args.n_bg
    config.NA = args.na
    config.resonance_wavelength_um = args.resonance_wavelength_um
    config.cavity_gds = args.cavity_gds
    config.holes_gds = args.holes_gds
    config.sidewall_angle_deg = args.sidewall_angle_deg
    config.trapezoid_slices = args.trapezoid_slices
    config.auto_confirm = getattr(args, 'auto_confirm', False)

    # Stage-specific overrides
    config.scout.run_time_ps = args.scout_run_time_ps
    config.lockin.run_time_ps = args.lockin_run_time_ps
    config.scout.bandwidth_rel = args.scout_bandwidth
    config.lockin.bandwidth_rel = args.lockin_bandwidth

    def _resolve_to_repo_root(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        pp = Path(p)
        return pp if pp.is_absolute() else (REPO_ROOT / pp)

    if args.scout_results:
        config.scout.results_path = _resolve_to_repo_root(args.scout_results)
    if args.lockin_results:
        config.lockin.results_path = _resolve_to_repo_root(args.lockin_results)
    if args.scout_sim:
        config.scout.simulation_path = _resolve_to_repo_root(args.scout_sim)
    if args.lockin_sim:
        config.lockin.simulation_path = _resolve_to_repo_root(args.lockin_sim)
    if args.scout_summary:
        config.scout.summary_path = _resolve_to_repo_root(args.scout_summary)
    if args.lockin_summary:
        config.lockin.summary_path = _resolve_to_repo_root(args.lockin_summary)

    # Simulation execution flags
    # Default: run simulations unless explicitly disabled via CLI
    want_run = True if args.run_sims is None else args.run_sims
    if want_run:
        if config.stage in {"all", "scout"}:
            config.scout.run_simulation = True
            config.scout.estimate_cost = args.estimate_cost
        if config.stage in {"all", "lockin"}:
            config.lockin.run_simulation = True
            config.lockin.estimate_cost = args.estimate_cost
    if args.force_rerun:
        if config.stage in {"all", "scout"}:
            config.scout.force_rerun = True
        if config.stage in {"all", "lockin"}:
            config.lockin.force_rerun = True

    return config


def extract_float(value: Any) -> Optional[float]:
    """Safely convert *value* to float if possible."""

    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.atleast_1d(value)
        if arr.size == 0:
            return None
        value = arr.flat[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def run_scout_stage(config: WorkflowConfig) -> Dict[str, Any]:
    """Execute the broadband scout analysis stage."""

    stage = config.scout
    print("=" * 80)
    print("🔎 SCOUT STAGE — broadband resonance search")
    print("=" * 80)

    ensure_parent_dir(stage.simulation_path)
    ensure_parent_dir(stage.results_path)
    ensure_parent_dir(stage.summary_path)

    setup = SimulationSetup(
        thickness_um=config.thickness_um,
        wavelength_um=config.initial_wavelength_um,
        source_bandwidth_rel=stage.bandwidth_rel,
        cavity_gds=config.cavity_gds,
        holes_gds=config.holes_gds,
        sidewall_angle_deg=config.sidewall_angle_deg,
        trapezoid_slices=config.trapezoid_slices,
    )
    simulation = setup.create_q_scout_simulation(run_time_ps=stage.run_time_ps)
    setup.save_simulation(simulation, str(stage.simulation_path))

    if stage.run_simulation:
        runner = SimulationRunner(task_name=stage.task_name)
        runner.run_simulation(
            simulation=simulation,
            results_path=str(stage.results_path),
            force_rerun=stage.force_rerun,
            estimate_cost_first=stage.estimate_cost,
            auto_confirm=config.auto_confirm,
            expected_monitors=["probe"],
        )
    else:
        if stage.results_path.exists():
            print(f"✓ Using existing scout results at {stage.results_path}")
        else:
            raise FileNotFoundError(
                f"Scout results file not found: {stage.results_path}. Run the scout "
                "simulation (e.g. with modules/simulation_runner.py) before continuing."
            )

    q_results = analyze_q_factor(
        data_path=str(stage.results_path),
        wavelength_um=config.initial_wavelength_um,
        save_results=False,
    )

    # Use the selected resonance index from the analysis results
    sel_idx = int(q_results.get("selected_index", 0))

    # Safely coerce arrays/lists and select the chosen mode
    def _at_index(arr_like, idx):
        if arr_like is None:
            return None
        arr = np.atleast_1d(np.asarray(arr_like))
        if arr.size == 0:
            return None
        idx = max(0, min(int(idx), arr.size - 1))
        return float(arr[idx])

    freq_hz = _at_index(q_results.get("frequencies_hz"), sel_idx)
    q_factor = _at_index(q_results.get("q_factors"), sel_idx)
    tau_ps = None
    if "decay_times_s" in q_results:
        tau_ps_val = _at_index(q_results.get("decay_times_s"), sel_idx)
        if tau_ps_val is not None:
            tau_ps = float(tau_ps_val) * 1e12

    if freq_hz is None or q_factor is None:
        raise RuntimeError("Q-factor analysis did not return valid frequency/Q values.")

    resonance_wavelength_um = C0 / freq_hz * 1e6
    summary = {
        "stage": stage.name,
        "simulation_file": str(stage.simulation_path),
        "results_file": str(stage.results_path),
        "run_time_ps": stage.run_time_ps,
        "bandwidth_rel": stage.bandwidth_rel,
        "resonance_frequency_thz": freq_hz / 1e12,
        "resonance_wavelength_um": resonance_wavelength_um,
        "q_factor": q_factor,
        "decay_time_ps": tau_ps,
        "num_modes": int(q_results.get("num_modes", 1)),
    }
    save_json(summary, stage.summary_path)

    print(f"✓ Resonance located at {resonance_wavelength_um:.6f} µm (f0 = {freq_hz / 1e12:.6f} THz)")
    print(f"✓ Estimated quality factor Q = {q_factor:,.0f}")
    if tau_ps is not None:
        print(f"✓ Decay time τ ≈ {tau_ps:.3f} ps")
    print(f"✓ Scout summary written to {stage.summary_path}")

    return {"summary": summary, "raw": q_results}


def run_lockin_stage(config: WorkflowConfig, resonance_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the narrowband lock-in analysis stage."""

    stage = config.lockin
    print("=" * 80)
    print("🎯 LOCK-IN STAGE — narrowband analysis")
    print("=" * 80)

    ensure_parent_dir(stage.simulation_path)
    ensure_parent_dir(stage.results_path)
    ensure_parent_dir(stage.summary_path)

    lockin_wavelength = config.resonance_wavelength_um or resonance_summary.get("resonance_wavelength_um")
    if lockin_wavelength is None:
        raise RuntimeError(
            "Resonance wavelength unavailable. Run the scout stage first or provide "
            "--resonance-wavelength-um."
        )

    lockin_frequency_thz = C0 / (lockin_wavelength * 1e-6) / 1e12
    print(f"Using lock-in wavelength {lockin_wavelength:.6f} µm ({lockin_frequency_thz:.6f} THz)")

    # Resolve GDS inputs once and pass through explicitly
    cavity_gds = getattr(config, 'cavity_gds', None)
    holes_gds = getattr(config, 'holes_gds', None)

    setup = SimulationSetup(
        thickness_um=config.thickness_um,
        wavelength_um=lockin_wavelength,
        source_bandwidth_rel=stage.bandwidth_rel,
        cavity_gds=cavity_gds,
        holes_gds=holes_gds,
        sidewall_angle_deg=config.sidewall_angle_deg,
        trapezoid_slices=config.trapezoid_slices,
    )
    simulation = setup.create_simulation(run_time_ps=stage.run_time_ps)
    setup.save_simulation(simulation, str(stage.simulation_path))

    if stage.run_simulation:
        runner = SimulationRunner(task_name=stage.task_name)
        runner.run_simulation(
            simulation=simulation,
            results_path=str(stage.results_path),
            force_rerun=stage.force_rerun,
            estimate_cost_first=stage.estimate_cost,
            auto_confirm=config.auto_confirm,
            expected_monitors=[
                "probe",
                "flux",
                "field_near",
                "farfield_cartesian",
                "farfield_kspace",
                "farfield_angles",
                "fld_3d_box",
            ],
        )
    else:
        if stage.results_path.exists():
            print(f"✓ Using existing lock-in results at {stage.results_path}")
        else:
            raise FileNotFoundError(
                f"Lock-in results file not found: {stage.results_path}. Run the lock-in "
                "simulation before launching the analyses."
            )

    results: Dict[str, Any] = {}
    q_for_lockin = extract_float(resonance_summary.get("q_factor"))
    if q_for_lockin is None:
        print("⚠️  Q-factor unavailable; Purcell metrics may be limited.")

    summary: Dict[str, Any] = {
        "stage": stage.name,
        "simulation_file": str(stage.simulation_path),
        "results_file": str(stage.results_path),
        "run_time_ps": stage.run_time_ps,
        "bandwidth_rel": stage.bandwidth_rel,
        "lockin_wavelength_um": lockin_wavelength,
        "lockin_frequency_thz": lockin_frequency_thz,
        "q_factor": q_for_lockin,
    }

    # Mode volume analysis ---------------------------------------------------
    try:
        mv_results = analyze_mode_volume(
            data_path=str(stage.results_path),
            cavity_gds=cavity_gds,
            holes_gds=holes_gds,
            thickness_um=config.thickness_um,
            wavelength_um=lockin_wavelength,
            Q=q_for_lockin,
            save_results=True,
            create_plots=True,
        )
        results["mode_volume"] = mv_results
        summary["mode_volume_um3"] = extract_float(mv_results.get("effective_mode_volume_um3"))
        summary["purcell_factor"] = extract_float(mv_results.get("purcell_factor"))
        print("✓ Mode volume analysis completed")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Mode volume analysis failed: {exc}")
        results["mode_volume_error"] = str(exc)

    # Polarization / far-field ------------------------------------------------
    try:
        pol_results = analyze_polarization(
            data_path=str(stage.results_path),
            wavelength_um=lockin_wavelength,
            NA=config.NA,
            n_bg=config.n_bg,
            save_results=True,
            create_plots=True,
            save_prefix="polarization",
        )
        results["polarization"] = pol_results
        summary["dolp"] = extract_float(pol_results.DoLP_avg)
        summary["docp"] = extract_float(pol_results.DoCP_avg)
        summary["dop"] = extract_float(pol_results.DoP_avg)
        summary["psi_deg"] = extract_float(np.degrees(pol_results.psi_avg))
        summary["chi_deg"] = extract_float(np.degrees(pol_results.chi_avg))
        print("✓ Polarization analysis completed")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Polarization analysis failed: {exc}")
        results["polarization_error"] = str(exc)

    # Near-field --------------------------------------------------------------
    try:
        nf_results = analyze_nearfield(
            data_path=str(stage.results_path),
            monitor_name="field_near",
            wavelength_um=lockin_wavelength,
            cavity_gds=cavity_gds,
            holes_gds=holes_gds,
            save_results=True,
            create_plots=True,
        )
        results["nearfield"] = nf_results
        summary["nearfield_confinement_um2"] = extract_float(
            nf_results.get("confinement", {}).get("confinement_area_um2")
        )
        summary["nearfield_mode_area_um2"] = extract_float(
            nf_results.get("mode_parameters", {}).get("mode_area_um2")
        )
        print("✓ Near-field analysis completed")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Near-field analysis failed: {exc}")
        results["nearfield_error"] = str(exc)

    # Far-field radiation patterns -------------------------------------------
    try:
        ff_results = analyze_farfield(
            data_path=str(stage.results_path),
            monitor_names=["farfield_kspace", "farfield_angles"],
            wavelength_um=lockin_wavelength,
            NA=config.NA,
            n_bg=config.n_bg,
            save_results=True,
            create_plots=True,
        )
        results["farfield"] = ff_results
        print("✓ Far-field analysis completed")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Far-field analysis failed: {exc}")
        results["farfield_error"] = str(exc)

    # Collection efficiency --------------------------------------------------
    try:
        ce_results = analyze_collection_efficiency(
            data_path=str(stage.results_path),
            monitor_names=["farfield_kspace", "farfield_angles"],
            wavelength_um=lockin_wavelength,
            NA=config.NA,
            n_bg=config.n_bg,
            save_results=True,
            create_plots=False,
        )
        results["collection_efficiency"] = ce_results
        overall = ce_results.get("overall", {}) if isinstance(ce_results, dict) else {}
        summary["collection_efficiency"] = extract_float(overall.get("overall_efficiency"))
        print("✓ Collection efficiency analysis completed")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Collection efficiency analysis failed: {exc}")
        results["collection_efficiency_error"] = str(exc)

    save_json(summary, stage.summary_path)
    print(f"✓ Lock-in summary written to {stage.summary_path}")

    return {"summary": summary, "analyses": results}


def run_workflow(config: WorkflowConfig) -> Dict[str, Any]:
    """Run the requested workflow stages and return collected results."""

    results: Dict[str, Any] = {}
    resonance_info: Optional[Dict[str, Any]] = None

    if config.stage in {"all", "scout"}:
        scout_output = run_scout_stage(config)
        results["scout"] = scout_output
        resonance_info = scout_output["summary"]
        config.resonance_wavelength_um = resonance_info["resonance_wavelength_um"]

    if config.stage in {"all", "lockin"}:
        if resonance_info is None:
            resonance_info = load_json(config.scout.summary_path) or {}
            if config.resonance_wavelength_um is not None:
                resonance_info.setdefault("resonance_wavelength_um", config.resonance_wavelength_um)
        lockin_output = run_lockin_stage(config, resonance_info)
        results["lockin"] = lockin_output

    print("\n" + "=" * 80)
    print("📋 WORKFLOW SUMMARY")
    print("=" * 80)
    if "scout" in results:
        scout = results["scout"]["summary"]
        print(
            f"Scout: λ0 = {scout['resonance_wavelength_um']:.6f} µm, "
            f"Q = {scout['q_factor']:,.0f}"
        )
    if "lockin" in results:
        lock = results["lockin"]["summary"]
        mv = lock.get("mode_volume_um3")
        purcell = lock.get("purcell_factor")
        ce = lock.get("collection_efficiency")
        mv_str = f"{mv:.3f} µm³" if mv is not None else "n/a"
        purcell_str = f"{purcell:.2f}" if purcell is not None else "n/a"
        ce_str = f"{ce*100:.1f}%" if ce is not None else "n/a"
        lock_lambda = lock.get("lockin_wavelength_um")
        lambda_str = f"{lock_lambda:.6f} µm" if lock_lambda is not None else "n/a"
        print(f"Lock-in: λ = {lambda_str}, Veff = {mv_str}, Fp = {purcell_str}, η = {ce_str}")

    print("\n🏁 WORKFLOW COMPLETE")
    return results


def run_complete_analysis(config: Optional[WorkflowConfig] = None) -> Dict[str, Any]:
    """Public API to run the complete analysis workflow."""

    if config is None:
        config = build_default_workflow_config()
    if config.scout is None or config.lockin is None:
        defaults = build_default_workflow_config(
            thickness_um=config.thickness_um,
            initial_wavelength_um=config.initial_wavelength_um,
        )
        if config.scout is None:
            config.scout = defaults.scout
        if config.lockin is None:
            config.lockin = defaults.lockin
    return run_workflow(config)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    thickness_um = 0.136 # default thickness
    cavity_gds = "gds/Cavity_Fab.gds"
    holes_gds = "gds/Holes_Fab.gds"

    parser = argparse.ArgumentParser(
        description="Run the two-pass photonic cavity analysis workflow.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "scout", "lockin"],
        default="all",
        help="Which stage(s) to run.",
    )
    parser.add_argument("--thickness-um", type=float, default=thickness_um, help="Device thickness in micrometers.")
    parser.add_argument(
        "--initial-wavelength-um",
        type=float,
        default=0.630,
        help="Initial wavelength guess for the scout simulation (µm).",
    )
    parser.add_argument(
        "--resonance-wavelength-um",
        type=float,
        default=None,
        help="Override the lock-in wavelength (µm) when running the lock-in stage only.",
    )
    parser.add_argument("--na", type=float, default=0.6, help="Collection numerical aperture for far-field analyses.")
    parser.add_argument("--sidewall-angle-deg", type=float, default=15.6, help="Trapezoid sidewall angle in degrees (0 for rectangle).")
    parser.add_argument("--trapezoid-slices", type=int, default=10, help="Number of vertical slices to approximate trapezoid (>=2 enables).")
    parser.add_argument("--n-bg", type=float, default=1.0, help="Background refractive index.")
    parser.add_argument("--scout-run-time-ps", type=float, default=12.0, help="Scout simulation run time (ps).")
    parser.add_argument("--lockin-run-time-ps", type=float, default=8.0, help="Lock-in simulation run time (ps).")
    parser.add_argument("--scout-bandwidth", type=float, default=0.12, help="Relative bandwidth Δf/f0 for the scout source.")
    parser.add_argument("--lockin-bandwidth", type=float, default=0.02, help="Relative bandwidth Δf/f0 for the lock-in source.")
    parser.add_argument("--scout-results", type=str, help="Path to scout simulation results (HDF5).")
    parser.add_argument("--lockin-results", type=str, help="Path to lock-in simulation results (HDF5).")
    parser.add_argument("--scout-sim", type=str, help="Output path for the scout simulation JSON file.")
    parser.add_argument("--lockin-sim", type=str, help="Output path for the lock-in simulation JSON file.")
    parser.add_argument("--scout-summary", type=str, help="Output path for the scout summary JSON file.")
    parser.add_argument("--lockin-summary", type=str, help="Output path for the lock-in summary JSON file.")
    parser.add_argument("--cavity-gds", type=str, default=cavity_gds, help="Cavity GDS file name or path (e.g., Cavity_Design.gds)")
    parser.add_argument("--holes-gds", type=str, default=holes_gds, help="Holes GDS file name or path (e.g., Holes_Design.gds)")
    parser.add_argument(
        "--run-sims",
        action="store_true",
        default=None,
        help="Run simulations (default). Use --no-run-sims to skip running.",
    )
    parser.add_argument(
        "--no-run-sims",
        dest="run_sims",
        action="store_false",
        help="Do not submit simulations; only use existing results.",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        default=True,
        help="Estimate simulation cost before submission (default).",
    )
    parser.add_argument(
        "--no-estimate-cost",
        dest="estimate_cost",
        action="store_false",
        help="Skip cost estimation before running simulations.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running simulations even if results already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    """Entry point when running as a script."""

    args = parse_args(argv)
    config = config_from_args(args)
    run_complete_analysis(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

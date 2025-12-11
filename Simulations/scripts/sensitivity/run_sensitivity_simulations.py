#!/usr/bin/env python3
"""
Run 1D cutline sensitivity simulations (very_cheap preset) with per-point GDS.

For each of width, radius, thickness cutlines at variations [-20,-10,0,10,20]%
(reusing center), generate scaled GDS via gds/reconstruct_cavity.py, build a
minimal scout sim (probe-only) with runtime=7 ps and min_steps_per_wvl=13, run
on Tidy3D, and save:
- simulations JSON: data/simulations/
- results HDF5: data/results/
- Q summaries JSON: data/summaries/

Usage:
  python scripts/sensitivity/run_sensitivity_simulations.py [--parallel N] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import tidy3d as td

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.simulation_setup import SimulationSetup
from modules.simulation_runner import SimulationRunner
from modules.q_factor_analysis import analyze_q_factor, ResonanceConfig

# Use non-interactive backend and disable plotting in Q analysis (CI/serial-safe)
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass
try:
    ResonanceConfig.do_plot = False  # type: ignore
except Exception:
    pass


@dataclass
class Point:
    param: str
    variation: int
    width_um: float
    radius_um: float
    thickness_um: float


def sobol_percentages() -> List[int]:
    return [-30, -15, 0, 15, 30]


def build_points(center_width=0.314, center_radius=0.043, center_thickness=0.136) -> List[Point]:
    vars = sobol_percentages()
    points: List[Point] = []
    # center
    points.append(Point("center", 0, center_width, center_radius, center_thickness))
    # width
    for v in vars:
        if v == 0:
            continue
        points.append(Point("width", v, center_width * (1 + v / 100), center_radius, center_thickness))
    # radius
    for v in vars:
        if v == 0:
            continue
        points.append(Point("radius", v, center_width, center_radius * (1 + v / 100), center_thickness))
    # thickness
    for v in vars:
        if v == 0:
            continue
        points.append(Point("thickness", v, center_width, center_radius, center_thickness * (1 + v / 100)))
    return points


def run_reconstruct(bottom_width_um: float, radius_um: float, out_tag: str) -> Tuple[Path, Path]:
    scaled_dir = REPO_ROOT / "gds" / "scaled"
    scaled_dir.mkdir(parents=True, exist_ok=True)
    cav_out = scaled_dir / f"cavity_{out_tag}.gds"
    hol_out = scaled_dir / f"holes_{out_tag}.gds"
    # Idempotent: if outputs already exist, reuse to avoid race on temp files in reconstruct script
    if cav_out.exists() and hol_out.exists():
        return cav_out, hol_out
    cmd = [
        sys.executable, str(REPO_ROOT / "gds" / "reconstruct_cavity.py"),
        "--cavity-design", "gds/Cavity_Design.gds",
        "--holes-design", "gds/Holes_Design.gds",
        "--output", str(cav_out),
        "--holes-fab", str(hol_out),
        "--target-height", f"{bottom_width_um}",
        "--target-hole-radius", f"{radius_um}",
        "--save", "--no-show", "--overwrite",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"reconstruct_cavity failed: {res.stderr}\nCmd: {' '.join(cmd)}")
    return cav_out, hol_out


def build_scout_sim(cav_gds: Path, hol_gds: Path, thickness_um: float, wavelength_um: float = 0.62,
                    run_time_ps: float = 7.0, min_steps_per_wvl: int = 13,
                    sidewall_angle_deg: float = 15.6, trapezoid_slices: int = 12) -> td.Simulation:
    setup = SimulationSetup(
        thickness_um=float(thickness_um),
        wavelength_um=float(wavelength_um),
        cavity_gds=str(cav_gds),
        holes_gds=str(hol_gds),
        sidewall_angle_deg=sidewall_angle_deg,
        trapezoid_slices=trapezoid_slices,
    )
    return setup.create_q_scout_simulation(run_time_ps=run_time_ps, min_steps_per_wvl=min_steps_per_wvl)


def run_point(p: Point, wavelength_um=0.62, run_time_ps=7.0, min_steps_per_wvl=13, dry_run=False, estimate_only: bool = False) -> Dict:
    tag = f"{p.param}_{p.variation:+d}_w{p.width_um:.3f}_r{p.radius_um:.3f}_t{p.thickness_um:.3f}"
    tag = tag.replace("+", "p").replace("-", "m")
    cav_gds, hol_gds = run_reconstruct(p.width_um, p.radius_um, out_tag=tag)

    sim = build_scout_sim(
        cav_gds, hol_gds, p.thickness_um, wavelength_um=wavelength_um,
        run_time_ps=run_time_ps, min_steps_per_wvl=min_steps_per_wvl,
        sidewall_angle_deg=15.6, trapezoid_slices=12,
    )

    sims_dir = REPO_ROOT / "data" / "simulations"
    sims_dir.mkdir(parents=True, exist_ok=True)
    sim_path = sims_dir / f"sim_{tag}.json"
    td.Simulation.to_file(sim, str(sim_path))

    results_dir = REPO_ROOT / "data" / "results" / "sensitivity_v2"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{tag}.hdf5"

    if dry_run:
        return {
            "tag": tag,
            "sim_path": str(sim_path),
            "results_path": str(results_path),
        }

    runner = SimulationRunner(task_name=f"cutline_{p.param}_{p.variation:+d}")
    if estimate_only:
        credits = runner.estimate_cost(sim)
        return {
            "tag": tag,
            "sim_path": str(sim_path),
            "results_path": str(results_path),
            "credits": float(credits) if credits is not None else None,
        }
    data = runner.run_simulation(simulation=sim, results_path=str(results_path), force_rerun=False, estimate_cost_first=True, auto_confirm=True, verbose=False)

    summary_dir = REPO_ROOT / "data" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"q_summary_{tag}.json"
    qres = analyze_q_factor(str(results_path), monitor_name="probe", field="Ey", wavelength_um=wavelength_um, save_results=False)

    # Persist Q summary
    import json
    with summary_path.open("w") as fh:
        json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in qres.items()}, fh, indent=2)

    import numpy as _np
    q_arr = _np.asarray(qres.get("q_factors", []))
    q_top = float(q_arr.max()) if q_arr.size else float("nan")

    return {
        "tag": tag,
        "sim_path": str(sim_path),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "q_top": q_top,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1, help="Max parallel jobs")
    ap.add_argument("--dry-run", action="store_true", help="Build inputs only, no submission")
    ap.add_argument("--estimate-only", action="store_true", help="Estimate credits only (no run)")
    args = ap.parse_args()

    pts = build_points()
    print(f"Planned jobs: {len(pts)} (center + 4 per cutline)\n")

    # Prebuild all GDS serially to avoid temp-file races inside reconstruct_cavity
    for p in pts:
        tag = f"{p.param}_{p.variation:+d}_w{p.width_um:.3f}_r{p.radius_um:.3f}_t{p.thickness_um:.3f}".replace("+", "p").replace("-", "m")
        try:
            run_reconstruct(p.width_um, p.radius_um, out_tag=tag)
        except Exception as exc:
            print(f"❌ GDS build failed for {p.param} {p.variation:+d}: {exc}")

    if args.dry_run:
        for p in pts:
            info = run_point(p, dry_run=True)
            print(f"- {info['tag']} -> sim={info['sim_path']} results={info['results_path']}")
        return

    results: List[Dict] = []
    if args.parallel <= 1:
        # Run serially to avoid Rich live progress conflicts
        for p in pts:
            try:
                res = run_point(p, estimate_only=args.estimate_only)
                results.append(res)
                if args.estimate_only:
                    print(f"✓ Estimated {res['tag']} (credits={res.get('credits','-')})")
                else:
                    print(f"✓ Completed {res['tag']} (Q_top={res.get('q_top','n/a')})")
            except Exception as exc:
                print(f"❌ Failed {p.param} {p.variation:+d}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            futs = {ex.submit(run_point, p, 0.62, 7.0, 13, False, args.estimate_only): p for p in pts}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    if args.estimate_only:
                        print(f"✓ Estimated {res['tag']} (credits={res.get('credits','-')})")
                    else:
                        print(f"✓ Completed {res['tag']} (Q_top={res.get('q_top','n/a')})")
                except Exception as exc:
                    print(f"❌ Failed {p.param} {p.variation:+d}: {exc}")

    print("\nAll done. Outputs are under data/simulations, data/results, data/summaries.")


if __name__ == "__main__":
    main()



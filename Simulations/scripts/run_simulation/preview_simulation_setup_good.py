#!/usr/bin/env python3
"""
Preview Simulation Setup

Builds a Tidy3D simulation using the existing SimulationSetup module and
renders the geometry/monitor layout figure, without running any simulations
or analyses. Useful for quickly checking geometry, padding, and monitors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path

# Ensure repository root on path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.simulation_setup import SimulationSetup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Tidy3D simulation setup (no run)")
    parser.add_argument("--mode", choices=["scout", "full"], default="scout",
                        help="Which setup to preview: minimal 'scout' (probe only) or 'full' monitors")
    parser.add_argument("--thickness-um", type=float, default=0.14,
                        help="Cavity thickness in micrometers")
    parser.add_argument("--wavelength-um", type=float, default=0.650,
                        help="Center wavelength in micrometers")
    parser.add_argument("--bandwidth-rel", type=float, default=0.12,
                        help="Relative source bandwidth Δf/f₀")
    parser.add_argument("--run-time-ps", type=float, default=12.0,
                        help="Simulation run time in picoseconds (for metadata only)")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Optional path to save the simulation JSON")
    parser.add_argument("--sidewall-angle-deg", type=float, default=15.6,
                        help="Trapezoid sidewall angle in degrees (0 for rectangle)")
    parser.add_argument("--num-slices", type=int, default=10,
                        help="Number of vertical slices to approximate trapezoid (>=2 enables)")
    parser.add_argument("--cavity-gds", type=str, default="gds/Cavity_Fab.gds",
                        help="Override cavity GDS (e.g., gds/Cavity_Fab.gds)")
    parser.add_argument("--holes-gds", type=str, default="gds/Holes_Fab.gds",
                        help="Override holes GDS (e.g., gds/Holes_Fab.gds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup = SimulationSetup(
        thickness_um=args.thickness_um,
        wavelength_um=args.wavelength_um,
        source_bandwidth_rel=args.bandwidth_rel,
        cavity_gds=args.cavity_gds,
        holes_gds=args.holes_gds,
        sidewall_angle_deg=args.sidewall_angle_deg,
        trapezoid_slices=args.num_slices,
    )

    if args.mode == "scout":
        sim = setup.create_q_scout_simulation(run_time_ps=args.run_time_ps)
        default_name = f"simulation_scout_q_only_{args.thickness_um:.2f}um.json"
    else:
        sim = setup.create_simulation(run_time_ps=args.run_time_ps)
        default_name = f"simulation_lockin_full_{args.thickness_um:.2f}um.json"

    # Default to repo-root/data/simulations; anchor relative overrides
    if args.save_json:
        save_path = Path(args.save_json)
        if not save_path.is_absolute():
            save_path = _REPO_ROOT / save_path
    else:
        save_path = _REPO_ROOT / 'data' / 'simulations' / default_name
    setup.save_simulation(sim, str(save_path))

    # Render and save the visualization figure
    setup.visualize_simulation(sim)


if __name__ == "__main__":
    main()


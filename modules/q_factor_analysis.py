#!/usr/bin/env python3
"""Q-factor analysis leveraging Tidy3D ResonanceFinder.

This module reimplements the old ringdown-based analysis but now uses
Tidy3D's resonance utilities to estimate resonant frequency(ies) and Q.

Public API intentionally mirrors the old module's `analyze_q_factor` so
that callers (e.g., `scripts/run_analysis.py`) continue to work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d.plugins.resonance import resonance as _res_mod  # for type access

try:
    # Tidy3D ResonanceFinder plugin (Tidy3D >= 2.5.x; path may vary by version)
    from tidy3d.plugins.resonance import ResonanceFinder
except Exception:  # pragma: no cover - optional dependency guard
    ResonanceFinder = None  # type: ignore

from .plot_style import apply_theme, PALETTE


C0 = 299_792_458.0  # m/s


@dataclass
class ResonanceConfig:
    """Configuration for the resonance finder-based analysis."""

    wavelength_um: float = 0.62
    # Relative frequency window around initial guess (e.g., ±12%)
    rel_window: float = 0.12
    # Initial number of trial frequencies used by the resonance finder
    init_num_freqs: int = 400
    # Regularization / conditioning parameter (implementation dependent)
    rcond: float = 1e-4
    # Plot controls
    do_plot: bool = True
    # Tail visualization controls
    guard_cycles: int = 8
    max_modes_for_fit: int = 1
    # Fractional window within the tail used to solve amplitudes (for alignment)
    fit_window_lo: float = 0.50
    fit_window_hi: float = 0.90


def _wavelength_to_frequency_hz(wavelength_um: float) -> float:
    wavelength_m = wavelength_um * 1e-6
    return C0 / wavelength_m


def _compute_freq_window_hz(center_hz: float, rel_window: float) -> Tuple[float, float]:
    span = abs(center_hz) * rel_window
    return max(center_hz - span, 0.0), center_hz + span


def _extract_time_signal(sim_data: td.SimulationData, monitor_name: str = "probe", field: str = "Ey") -> Tuple[np.ndarray, np.ndarray]:
    if monitor_name not in sim_data.monitor_data:
        raise KeyError(f"Monitor '{monitor_name}' not found in data")
    mdat = sim_data[monitor_name]

    arr = getattr(mdat, field, None)
    if arr is None:
        raise RuntimeError(f"Field '{field}' not present in monitor '{monitor_name}'.")

    # Try best-effort extraction of time vector
    t = None
    try:
        t0 = getattr(arr, "t", None)
        if t0 is not None:
            t = t0.values if hasattr(t0, "values") else np.asarray(t0)
    except Exception:
        t = None
    if t is None and hasattr(mdat, "t"):
        t0 = getattr(mdat, "t")
        t = t0.values if hasattr(t0, "values") else np.asarray(t0)
    if t is None and hasattr(arr, "coords") and ("t" in arr.coords):
        t0 = arr.coords["t"]
        t = t0.values if hasattr(t0, "values") else np.asarray(t0)
    if t is None:
        if hasattr(arr, "dims") and "t" in getattr(arr, "dims", []):
            t = np.arange(arr.sizes["t"], dtype=float)
        else:
            raise RuntimeError("Could not determine time axis for the probe monitor.")

    # Extract the field at the monitor center if addressable, otherwise squeeze
    try:
        e = arr.sel(x=0.0, y=0.0, z=0.0).values
    except Exception:
        e = np.asarray(arr).squeeze()

    t = np.atleast_1d(np.asarray(t, float))
    e = np.atleast_1d(np.real(np.asarray(e, float)))

    if t.ndim != 1 or e.ndim != 1:
        raise RuntimeError(f"Unexpected shapes: t.shape={t.shape}, e.shape={e.shape}")
    if t.size != e.size:
        if e.size > t.size and e.size % t.size == 0:
            e = e.reshape(t.size, -1)[:, 0]
        elif t.size > e.size and t.size % e.size == 0:
            t = t[: e.size]
        else:
            raise RuntimeError(f"Length mismatch: len(t)={t.size}, len(e)={e.size}")
    return t, e


def _plot_ringdown_like_old(t_all: np.ndarray, y_all: np.ndarray, i0: int, model_env: Optional[np.ndarray], f_mode: float, a_mode: float) -> None:
    """Publication-quality single-axes semilog envelope plot (OLD-like)."""
    apply_theme()
    plt.figure(figsize=(12.5, 6.0))
    from scipy.signal import hilbert as _hilbert
    env_full = np.abs(_hilbert(y_all)) + 1e-30

    # Colors
    c_full = PALETTE[0]
    c_tail = PALETTE[1]
    c_model = PALETTE[3]
    c_mark = PALETTE[2]

    # Full envelope
    plt.semilogy(t_all * 1e12, env_full, color=c_full, alpha=0.8, linewidth=1.4, label='Envelope |E(t)|')

    # Tail used (from i0)
    plt.semilogy(t_all[i0:] * 1e12, env_full[i0:], color=c_tail, linewidth=1.8, label='Tail segment')

    # Model envelope if provided
    if model_env is not None and model_env.size == env_full.size:
        plt.semilogy(t_all * 1e12, np.maximum(model_env, 1e-30), linestyle='--', color=c_model, linewidth=2.2, label='Single-mode fit')

    # Mark i0
    plt.axvline(t_all[i0] * 1e12, color=c_mark, linestyle=':', linewidth=1.4, label='Tail start')

    # Labels
    plt.xlabel("Time, t (ps)", fontsize=13)
    plt.ylabel("Field envelope |E(t)| (arb. units)", fontsize=13)
    plt.title("Cavity ringdown and single-mode fit", fontsize=14)

    # Improve ticks and grid for publication
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.25)

    # Annotation for Mode 1 only
    tau_ps = (1.0 / max(a_mode, 1e-30)) * 1e12
    Q_val = np.pi * f_mode / max(a_mode, 1e-30)
    lambda_nm = (C0 / f_mode) * 1e9
    text = fr"Mode 1: Q={Q_val:,.0f}, $\tau$={tau_ps:.2f} ps, $\lambda$={lambda_nm:.1f} nm"
    plt.gca().text(0.98, 0.98, text, ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, color='k', bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    plt.legend(frameon=False, fontsize=11, loc='lower right')
    plt.tight_layout(pad=1.0)
    repo_root = Path(__file__).resolve().parents[1]
    fig_dir = repo_root / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / 'q_factor_analysis.png'
    plt.savefig(str(out_png), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Q-factor plot saved to {out_png}")


def _results_dict(
    t: np.ndarray,
    y: np.ndarray,
    freq_hz: np.ndarray,
    alpha_hz: np.ndarray,
    amplitudes: Optional[np.ndarray],
    selection: int,
    wavelength_um: float,
    model: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    # Convert to arrays
    f = np.atleast_1d(np.asarray(freq_hz, float))
    a = np.atleast_1d(np.asarray(alpha_hz, float))
    A = None if amplitudes is None else np.asarray(amplitudes)
    tau = 1.0 / np.maximum(a, 1e-30)
    Q = np.pi * f / np.maximum(a, 1e-30)
    coverage = (t[-1] - t[0]) / np.maximum(tau, 1e-30)

    return {
        "t_full": t,
        "y_full": y,
        "t_tail": t,  # with ResonanceFinder we use the full segment
        "z_tail": y,   # keep key for backward compatibility
        "model": model,
        "residual_fraction": None,
        "frequencies_hz": f,
        "decay_rates_hz": a,
        "amplitudes": A,
        "decay_times_s": tau,
        "q_factors": Q,
        "coverage": coverage,
        "num_modes": int(f.size),
        "selected_index": int(selection),
        "analysis_parameters": {
            "wavelength_um": wavelength_um,
            "method": "ResonanceFinder",
        },
    }


def _save_results_json(results: Dict[str, Any], filename: str = "q_factor_results.json") -> Path:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return {"real": obj.real.tolist(), "imag": obj.imag.tolist(), "dtype": "complex"}
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    out = _convert(results)
    path = Path(filename)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        if len(path.parts) == 1:
            path = repo_root / "data" / "summaries" / path.name
        else:
            path = repo_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(out, fh, indent=2)
    return path


def analyze_q_factor(
    data_path: str,
    monitor_name: str = "probe",
    field: str = "Ey",
    wavelength_um: float = 0.62,
    quality_preset: str = "medium",  # preserved for compatibility; not used directly
    save_results: bool = True,
) -> Dict[str, Any]:
    """Analyze Q-factor(s) using Tidy3D ResonanceFinder.

    Parameters mirror the previous API. `quality_preset` is accepted for
    compatibility with workflow scripts but is not used by ResonanceFinder.
    """

    # Load simulation data
    sim_data = td.SimulationData.from_file(data_path)

    # Try to use native Tidy3D data containers for ResonanceFinder first
    mdat = None
    if monitor_name in sim_data.monitor_data:
        mdat = sim_data[monitor_name]
    # Prepare numpy signal as fallback and for plotting
    t, y = _extract_time_signal(sim_data, monitor_name=monitor_name, field=field)

    # Build resonance search window around wavelength guess
    f0 = _wavelength_to_frequency_hz(wavelength_um)
    # Reuse scout/legacy default window ±12% to match project expectations
    rel_window = ResonanceConfig.rel_window
    if isinstance(rel_window, property):  # defensive (shouldn't happen)
        rel_window = 0.12
    f_lo, f_hi = _compute_freq_window_hz(f0, rel_window)  # Hz

    # Run ResonanceFinder if available; otherwise fall back to crude estimator
    freq_hz: np.ndarray
    alpha_hz: np.ndarray
    amplitudes: Optional[np.ndarray]

    try:
        if ResonanceFinder is not None:
            rf = ResonanceFinder(freq_window=(f_lo, f_hi), init_num_freqs=ResonanceConfig.init_num_freqs, rcond=ResonanceConfig.rcond)
            last_err: Optional[Exception] = None
            res = None  # type: ignore
            # 1) Preferred: pass ScalarFieldTimeDataArray for the desired component
            try:
                comp_da = getattr(mdat, field) if mdat is not None else None
                if comp_da is not None and isinstance(comp_da, _res_mod.ScalarFieldTimeDataArray):
                    res = rf.run_scalar_field_time(signal=comp_da)
            except Exception as _e:
                last_err = _e
                res = None
            # 2) Next: pass the full FieldTimeData object
            if res is None:
                try:
                    if mdat is not None and isinstance(mdat, _res_mod.FieldTimeData):
                        res = rf.run(signals=mdat)
                except Exception as _e:
                    last_err = _e
                    res = None
            # 3) Finally: attempt array-based fallbacks (older code paths)
            if res is None:
                for attempt in (
                    lambda: rf.run_scalar_field_time(field_time=y, t=t, component=field),
                    lambda: rf.run_scalar_field_time(field_time=y, t=t),
                    lambda: rf.run(signals={field: y}, t=t),
                    lambda: rf.run(signals={field: y}),
                    lambda: rf.run(signals=[y]),
                ):
                    try:
                        res = attempt()
                        break
                    except Exception as _e:
                        last_err = _e
                        res = None
            if res is None and last_err is not None:
                raise last_err

            # Convert to a convenient structure; Dataset -> arrays (with robust key fallbacks)
            try:
                def _ds_has(d, k):
                    try:
                        return k in d
                    except Exception:
                        return False
                def _arr_for(ds, candidates):
                    for k in candidates:
                        if _ds_has(ds, k):
                            return np.asarray(ds[k].values)
                        if hasattr(ds, "coords") and k in getattr(ds, "coords"):
                            return np.asarray(ds.coords[k].values)
                    raise KeyError(candidates)
                freq_hz = _arr_for(res, ("frequency", "freq", "f", "frequency_hz"))
                # decay rate variable
                try:
                    alpha_hz = _arr_for(res, ("decay_rate", "alpha", "gamma", "decay_rate_hz"))
                except KeyError:
                    # derive from Q if needed later
                    alpha_hz = None  # type: ignore
                # amplitude optional
                try:
                    amplitudes = _arr_for(res, ("amplitude", "A"))
                except KeyError:
                    amplitudes = None
                # Q variable or compute
                try:
                    q_vals = _arr_for(res, ("Q", "q"))
                except KeyError:
                    if alpha_hz is None:
                        raise
                    q_vals = np.pi * freq_hz / np.maximum(alpha_hz, 1e-30)
                if alpha_hz is None:
                    alpha_hz = np.pi * freq_hz / np.maximum(q_vals, 1e-30)
            except Exception:
                # As a fallback, try DataFrame adapter
                df = res.to_dataframe()
                # Frequency
                for k in ("frequency", "freq", "f", "frequency_hz"):
                    if k in df.columns:
                        freq_hz = df[k].to_numpy()
                        break
                else:
                    raise KeyError("frequency in ResonanceFinder dataframe")
                # Q
                q_vals = None
                for k in ("Q", "q"):
                    if k in df.columns:
                        q_vals = df[k].to_numpy()
                        break
                # Decay rate
                alpha_hz = None
                for k in ("decay_rate", "alpha", "gamma", "decay_rate_hz"):
                    if k in df.columns:
                        alpha_hz = df[k].to_numpy()
                        break
                if alpha_hz is None and q_vals is not None:
                    alpha_hz = np.pi * freq_hz / np.maximum(q_vals, 1e-30)
                if q_vals is None and alpha_hz is not None:
                    q_vals = np.pi * freq_hz / np.maximum(alpha_hz, 1e-30)
                if q_vals is None or alpha_hz is None:
                    raise KeyError("Missing Q and decay_rate in ResonanceFinder output")
                amplitudes = df["amplitude"].to_numpy() if "amplitude" in df.columns else None
        else:
            raise RuntimeError("ResonanceFinder plugin unavailable")
    except Exception as exc:
        print(f"⚠️  ResonanceFinder failed ({exc}); using minimal FFT fallback.")
        # Minimal FFT-based fallback
        dt = float(np.mean(np.diff(t)))
        X = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, d=dt)
        k = int(np.argmax(np.abs(X)))
        freq_hz = np.atleast_1d(float(f[k]))
        from scipy.signal import hilbert
        env = np.abs(hilbert(y)) + 1e-30
        slope = -np.polyfit(t - t[0], np.log(env), 1)[0]
        alpha_hz = np.atleast_1d(float(max(slope, 1e9)))
        amplitudes = None
        q_vals = np.pi * freq_hz / np.maximum(alpha_hz, 1e-30)
        # Minimal FFT-based fallback (no plugin available)
        dt = float(np.mean(np.diff(t)))
        X = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, d=dt)
        k = int(np.argmax(np.abs(X)))
        freq_hz = np.atleast_1d(float(f[k]))
        # crude ringdown estimate: slope of log-envelope
        from scipy.signal import hilbert
        env = np.abs(hilbert(y)) + 1e-30
        slope = -np.polyfit(t - t[0], np.log(env), 1)[0]
        alpha_hz = np.atleast_1d(float(max(slope, 1e9)))
        amplitudes = None
        q_vals = np.pi * freq_hz / np.maximum(alpha_hz, 1e-30)

    # Select Mode 1 = highest-Q mode
    idx = int(np.argmax(q_vals)) if np.size(q_vals) else 0

    # Build an analytic model from Mode 1 only
    model = None
    try:
        # Use the selected frequency to define a tail start (peak + guard cycles)
        from scipy.signal import hilbert as _hilbert
        z_full = _hilbert(y).astype(complex)
        env_full = np.abs(z_full)
        i_peak = int(np.argmax(env_full))
        f_primary = float(np.atleast_1d(freq_hz)[idx])
        guard_T = float(ResonanceConfig.guard_cycles) / max(f_primary, 1e9)
        t_after = t[i_peak] + guard_T
        i0 = int(np.searchsorted(t, t_after, side='left'))
        i0 = min(max(i0, 0), len(t) - 2)

        t_tail = t[i0:]
        z_tail = z_full[i0:]

        # Use only the selected mode (Mode 1)
        f_sel = np.atleast_1d(freq_hz)[[idx]]
        a_sel = np.atleast_1d(alpha_hz)[[idx]]

        # Build basis on the tail segment
        tt_full = t_tail - t_tail[0]
        E_full = np.exp((1j * 2 * np.pi * f_sel[:, None] - a_sel[:, None]) * tt_full[None, :]).T

        # Use a sub-window [50%, 90%] of the tail for solving amplitudes
        N = tt_full.size
        i_lo = int(max(0, min(N - 2, np.floor(ResonanceConfig.fit_window_lo * N))))
        i_hi = int(max(i_lo + 1, min(N, np.floor(ResonanceConfig.fit_window_hi * N))))
        tt = tt_full[i_lo:i_hi]
        E = E_full[i_lo:i_hi, :]
        z_seg = z_tail[i_lo:i_hi]

        # Weighted LS emphasizing later times in the fit window
        M = tt.size
        w = np.linspace(0.5, 1.0, M)
        Ew = (w[:, None] * E)
        zw = w * z_seg

        from numpy.linalg import lstsq as _lstsq
        A_est, *_ = _lstsq(Ew, zw, rcond=None)

        # Prune to strongest amplitudes if more than 1 kept
        if A_est.size > 1:
            order_A = np.argsort(np.abs(A_est))[::-1]
            top = min(ResonanceConfig.max_modes_for_fit, A_est.size)
            A_est = A_est[order_A[:top]]
            f_sel = f_sel[order_A[:top]]
            a_sel = a_sel[order_A[:top]]
            E = np.exp((1j * 2 * np.pi * f_sel[:, None] - a_sel[:, None]) * tt[None, :]).T
            Ew = (w[:, None] * E)
            A_est, *_ = _lstsq(Ew, zw, rcond=None)

        # Reconstruct model over the full tail using the estimated amplitude
        model_tail = E_full @ A_est
        model_env_full = np.full_like(env_full, np.nan, dtype=float)
        model_env_full[i0:] = np.abs(model_tail)
        model = model_env_full
    except Exception as exc:
        # Non-fatal; continue without model overlay
        model = None

    # Optional plotting (OLD-style ringdown plot)
    try:
        if ResonanceConfig.do_plot:
            # Recompute tail start index i0 consistent with model construction
            from scipy.signal import hilbert as _hilbert
            z_full = _hilbert(y).astype(complex)
            env_full = np.abs(z_full)
            i_peak = int(np.argmax(env_full))
            f_primary = float(np.atleast_1d(freq_hz)[idx])
            guard_T = float(ResonanceConfig.guard_cycles) / max(f_primary, 1e9)
            t_after = t[i_peak] + guard_T
            i0 = int(np.searchsorted(t, t_after, side='left'))
            i0 = min(max(i0, 0), len(t) - 2)
            _plot_ringdown_like_old(t, y, i0=i0, model_env=model, f_mode=float(np.atleast_1d(freq_hz)[idx]), a_mode=float(np.atleast_1d(alpha_hz)[idx]))
    except Exception as exc:  # pragma: no cover - plotting should not break analysis
        print(f"⚠️  Plotting failed: {exc}")

    results = _results_dict(
        t=t,
        y=y,
        freq_hz=freq_hz,
        alpha_hz=alpha_hz,
        amplitudes=amplitudes,
        selection=idx,
        wavelength_um=wavelength_um,
        model=model,
    )

    if save_results:
        out_path = _save_results_json(results)
        print(f"✓ Q-factor analysis results saved to {out_path}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python q_factor_analysis.py <data_file.hdf5> [monitor_name] [field]")
        sys.exit(1)
    data_file = sys.argv[1]
    monitor_name = sys.argv[2] if len(sys.argv) > 2 else "probe"
    field = sys.argv[3] if len(sys.argv) > 3 else "Ey"
    res = analyze_q_factor(
        data_path=data_file,
        monitor_name=monitor_name,
        field=field,
        save_results=True,
    )
    print("\n✅ Q-factor analysis completed!")
    q = np.atleast_1d(res.get("q_factors", []))
    f = np.atleast_1d(res.get("frequencies_hz", []))
    print(f"Top Q: {q.max() if q.size else 'n/a'} at {f[np.argmax(q)]/1e12:.3f} THz" if q.size else "No modes found.")



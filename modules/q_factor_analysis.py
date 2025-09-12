#!/usr/bin/env python3
"""
Q-Factor Analysis Module

This module provides comprehensive analysis tools for computing quality factors
from time-domain electromagnetic simulations of photonic cavities.

Key Features:
- Robust time-domain signal extraction
- Multi-mode ringdown fitting with variable projection
- Configurable quality presets for different simulation accuracies
- Comprehensive diagnostic output and visualization
- Integration with Tidy3D simulation data
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from scipy.signal import hilbert, stft, welch
from scipy.optimize import curve_fit, least_squares
from numpy.linalg import lstsq
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Physical constants
C0 = 299792458.0  # Speed of light in vacuum (m/s)


class QFactorAnalyzer:
    """
    Comprehensive Q-factor analyzer for photonic cavity simulations.
    """
    
    def __init__(self, 
                 wavelength_um: float = 0.62,
                 quality_preset: str = "medium"):
        """
        Initialize Q-factor analyzer.
        
        Args:
            wavelength_um: Analysis wavelength in micrometers
            quality_preset: Quality preset ('coarse', 'medium', 'fine')
        """
        self.wavelength_um = wavelength_um
        self.quality_preset = quality_preset
        self.quality_presets = self._setup_quality_presets()
        self.ringdown_config = self._setup_ringdown_config()
        
    def _setup_quality_presets(self) -> Dict[str, Dict]:
        """Setup quality presets for different simulation accuracies."""
        return {
            "coarse": dict(
                min_steps_per_wvl=10,
                pml_layers=8,
                rel_bandwidth_A=0.12,
                rel_bandwidth_B=0.03,
                time_interval=8,
            ),
            "medium": dict(
                min_steps_per_wvl=18,
                pml_layers=10,
                rel_bandwidth_A=0.10,
                rel_bandwidth_B=0.02,
                time_interval=5,
            ),
            "fine": dict(
                min_steps_per_wvl=24,
                pml_layers=12,
                rel_bandwidth_A=0.08,
                rel_bandwidth_B=0.015,
                time_interval=3,
            ),
        }
    
    def _setup_ringdown_config(self) -> Dict:
        """Setup ringdown analysis configuration."""
        return dict(
            K_max=2,
            use_weighted_LS=True,
            weight_floor=0.25,
            guard_cycles=8,
            enforce_decaying=True,
            Nf=121, Na=81,
            df_rel_span=0.010,
            df_min_Hz=5e10,
            alpha_lo_Hz=5e10,
            alpha_hi_Hz=8e12,
            alpha_span_lo=0.4,
            alpha_span_hi=3.0,
            residual_improve_min=0.12,
            refine_df_bound_thz=0.8,
            refine_alpha_lo=3e10,
            refine_alpha_hi=1e13,
            energy_eps=1e-30,
            verbose=True,
            do_plot=True,
            apodize_for_plot=False,
        )
    
    def extract_time_and_signal(self, probe_ds: Any, field: str = "Ey") -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time and signal data from probe dataset.
        
        Args:
            probe_ds: Tidy3D probe dataset
            field: Field component to extract
            
        Returns:
            Tuple of (time_array, field_array)
        """
        arr = getattr(probe_ds, field, None)
        if arr is None:
            raise RuntimeError(f"Field '{field}' not present in probe dataset.")

        # Try to get the time coordinate
        t = None
        try:
            t = getattr(arr, "t", None)
            if t is not None:
                t = t.values if hasattr(t, "values") else np.asarray(t)
        except Exception:
            t = None

        # Fallback: dataset-level time
        if (t is None) and hasattr(probe_ds, "t"):
            t0 = getattr(probe_ds, "t")
            t = t0.values if hasattr(t0, "values") else np.asarray(t0)

        # Fallback: coords on the array
        if (t is None) and hasattr(arr, "coords") and ("t" in arr.coords):
            t0 = arr.coords["t"]
            t = t0.values if hasattr(t0, "values") else np.asarray(t0)

        # Last resort: create synthetic time index
        if t is None:
            if hasattr(arr, "dims") and "t" in arr.dims:
                t = np.arange(arr.sizes["t"], dtype=float)
            else:
                raise RuntimeError("Could not find time axis for the probe monitor.")

        # Get the field data
        try:
            e = arr.sel(x=0.0, y=0.0, z=0.0).values
        except Exception:
            e = np.asarray(arr).squeeze()

        t = np.atleast_1d(np.asarray(t, dtype=float))
        e = np.atleast_1d(np.real(np.asarray(e, dtype=float)))

        if t.ndim != 1 or e.ndim != 1:
            raise RuntimeError(f"Unexpected shapes: t.shape={t.shape}, e.shape={e.shape}")

        if t.size != e.size:
            # Handle size mismatches
            if e.size > t.size and e.size % t.size == 0:
                e = e.reshape(t.size, -1)[:, 0]
            elif t.size > e.size and t.size % e.size == 0:
                t = t[:e.size]
            else:
                raise RuntimeError(f"Length mismatch: len(t)={t.size}, len(e)={e.size}")

        return t, e
    
    def fft_peak_hz(self, x: np.ndarray, t_like: np.ndarray) -> float:
        """
        Estimate peak frequency from signal using FFT.
        
        Args:
            x: Signal array
            t_like: Time array
            
        Returns:
            Peak frequency in Hz
        """
        t_like = np.asarray(t_like, float)
        dt = float(np.mean(np.diff(t_like)))
        
        if np.allclose(np.diff(t_like), dt, rtol=1e-6, atol=0):
            if np.iscomplexobj(x):
                X = np.fft.fft(x)
                f = np.fft.fftfreq(len(t_like), d=dt)
                mask = f > 0
                f_pos, X_pos = f[mask], X[mask]
                return float(f_pos[np.argmax(np.abs(X_pos))])
            else:
                X = np.fft.rfft(x)
                f = np.fft.rfftfreq(len(t_like), d=dt)
                return float(f[np.argmax(np.abs(X))])
        
        # Non-uniform grid → phase-slope method
        phi = np.unwrap(np.angle(x))
        p = np.polyfit(t_like, phi, 1)
        return float(p[0] / (2*np.pi))
    
    def cut_tail_strict(self, t_all: np.ndarray, y_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """
        Cut signal to ensure strictly decaying tail.
        
        Args:
            t_all: Full time array
            y_all: Full signal array
            
        Returns:
            Tuple of (t_tail, y_tail, start_index, reason)
        """
        z_full = hilbert(y_all)
        env = np.abs(z_full)
        i_peak = int(np.argmax(env))
        f_est = max(self.fft_peak_hz(z_full, t_all), 1e9)
        guard_T = self.ringdown_config['guard_cycles'] / f_est
        
        t_after = t_all[i_peak] + guard_T
        i0 = int(np.searchsorted(t_all, t_after, side='left'))
        i0 = min(max(i0, 0), len(t_all)-2)
        
        return t_all[i0:], y_all[i0:], i0, "peak+guard"
    
    def seeds_from_tail(self, t: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract initial parameter seeds from the tail.
        
        Args:
            t: Time array
            z: Complex signal array
            
        Returns:
            Tuple of (frequency, frequency_mad, decay_rate)
        """
        phi = np.unwrap(np.angle(z))
        p = np.polyfit(t, phi, 1)
        f_med = float(p[0] / (2*np.pi))
        gph = np.gradient(phi, t) / (2*np.pi)
        mad_f = float(np.median(np.abs(gph - np.median(gph)))) if len(gph) > 4 else 0.0

        env = np.abs(z) + 1e-30
        i_start = int(0.4*len(env))
        tloc = t - t[i_start]
        a = -np.polyfit(tloc[i_start:], np.log(env[i_start:]), 1)[0]
        alpha_est = float(max(1e9, a))
        
        return f_med, mad_f, alpha_est
    
    def build_grids(self, f_seed: float, alpha_seed: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build frequency and decay rate grids for fitting.
        
        Args:
            f_seed: Initial frequency estimate
            alpha_seed: Initial decay rate estimate
            
        Returns:
            Tuple of (frequency_grid, decay_rate_grid)
        """
        df = max(self.ringdown_config['df_min_Hz'], abs(f_seed)*self.ringdown_config['df_rel_span'])
        f_grid = np.linspace(f_seed - df, f_seed + df, self.ringdown_config['Nf'])
        
        if alpha_seed > 5e9:
            alo = max(self.ringdown_config['alpha_lo_Hz'], alpha_seed*self.ringdown_config['alpha_span_lo'])
            ahi = min(self.ringdown_config['alpha_hi_Hz'], alpha_seed*self.ringdown_config['alpha_span_hi'])
            a_grid = np.linspace(alo, ahi, self.ringdown_config['Na'])
        else:
            a_grid = np.geomspace(max(self.ringdown_config['alpha_lo_Hz'], 1e10), 
                                self.ringdown_config['alpha_hi_Hz'], 
                                self.ringdown_config['Na'])
        
        return f_grid, a_grid
    
    def fit_one_mode_grid(self, t: np.ndarray, z: np.ndarray, 
                         f_grid: np.ndarray, a_grid: np.ndarray, 
                         weighted: bool = True) -> Dict:
        """
        Fit a single mode using grid search.
        
        Args:
            t: Time array
            z: Complex signal array
            f_grid: Frequency grid
            a_grid: Decay rate grid
            weighted: Whether to use weighted least squares
            
        Returns:
            Dictionary with fit results
        """
        tt = t - t[0]
        N = len(tt)
        w = np.linspace(self.ringdown_config['weight_floor'], 1.0, N) if weighted else np.ones(N)
        
        def energy(w, z): 
            return float(np.sum(np.abs(w*z)**2))
        
        Ed = energy(w, z)
        best = dict(frac=1e9)
        
        for a in a_grid:
            decay = np.exp(-a*tt)
            for f in f_grid:
                E = decay * np.exp(1j*2*np.pi*f*tt)
                num = np.vdot(w*E, w*z)
                den = np.vdot(w*E, w*E) + 1e-30
                A = num/den
                yhat = E*A
                frac = float(energy(w, z - yhat)/max(Ed, 1e-30))
                if frac < best['frac']:
                    best = dict(f=f, alpha=a, A=A, model=yhat, frac=frac)
        
        return best
    
    def greedy_add_modes(self, t: np.ndarray, z: np.ndarray, best1: Dict,
                        f_grid: np.ndarray, a_grid: np.ndarray, 
                        weighted: bool = True) -> Tuple[int, np.ndarray, Dict, Optional[Dict], float]:
        """
        Greedily add a second mode if it improves the fit.
        
        Args:
            t: Time array
            z: Complex signal array
            best1: First mode fit results
            f_grid: Frequency grid
            a_grid: Decay rate grid
            weighted: Whether to use weighted least squares
            
        Returns:
            Tuple of (num_modes, combined_model, mode1, mode2, residual_fraction)
        """
        if self.ringdown_config['K_max'] <= 1:
            return 1, best1['model'], best1, None, best1['frac']
        
        tt = t - t[0]
        N = len(tt)
        w = np.linspace(self.ringdown_config['weight_floor'], 1.0, N) if weighted else np.ones(N)
        z_res = z - best1['model']
        best2 = self.fit_one_mode_grid(t, z_res, f_grid, a_grid, weighted=weighted)
        
        improved = (best1['frac'] - best2['frac']) > self.ringdown_config['residual_improve_min'] * max(best1['frac'], 1e-12)
        
        if not improved:
            return 1, best1['model'], best1, None, best1['frac']
        
        model2 = best1['model'] + best2['model']
        
        def energy(w, z): 
            return float(np.sum(np.abs(w*z)**2))
        
        frac2 = float(energy(w, z - model2)/max(energy(w, z), 1e-30))
        return 2, model2, best1, best2, frac2
    
    def refine_variable_projection(self, t: np.ndarray, z: np.ndarray, 
                                 picks: List[Dict], weighted: bool = True) -> Dict:
        """
        Refine fit using variable projection method.
        
        Args:
            t: Time array
            z: Complex signal array
            picks: List of mode fit results
            weighted: Whether to use weighted least squares
            
        Returns:
            Dictionary with refined fit results
        """
        tt = t - t[0]
        N = len(tt)
        w = np.linspace(self.ringdown_config['weight_floor'], 1.0, N) if weighted else np.ones(N)
        z_w = w*z

        f0s = np.array([picks[0]['f']] + ([picks[1]['f']] if len(picks)>1 and picks[1] else []), float)
        a0s = np.array([picks[0]['alpha']] + ([picks[1]['alpha']] if len(picks)>1 and picks[1] else []), float)
        K = len(f0s)

        p0 = np.concatenate([f0s, np.log(np.maximum(a0s, 1e9))])

        df_bound = self.ringdown_config['refine_df_bound_thz']*1e12
        f_lo = f0s - df_bound; f_hi = f0s + df_bound
        a_lo = np.full(K, self.ringdown_config['refine_alpha_lo'])
        a_hi = np.full(K, self.ringdown_config['refine_alpha_hi'])
        lb = np.concatenate([f_lo, np.log(a_lo)])
        ub = np.concatenate([f_hi, np.log(a_hi)])

        def residual(p):
            f = p[:K]; a = np.exp(p[K:])
            E = np.exp((1j*2*np.pi*f[:,None] - a[:,None]) * tt[None,:]).T
            Ew = (w[:,None] * E)
            A, *_ = lstsq(Ew, z_w, rcond=None)
            r = z_w - Ew @ A
            return np.concatenate([r.real, r.imag])

        out = least_squares(residual, p0, bounds=(lb, ub), max_nfev=100, 
                          xtol=1e-12, ftol=1e-12, gtol=1e-12)
        p = out.x
        f = p[:K]; a = np.exp(p[K:])
        E = np.exp((1j*2*np.pi*f[:,None] - a[:,None]) * tt[None,:]).T
        Ew = (w[:,None]*E)
        A, *_ = lstsq(Ew, w*z, rcond=None)
        model = E @ A
        
        def energy(w, z): 
            return float(np.sum(np.abs(w*z)**2))
        
        frac = float(energy(w, z - model)/max(energy(w, z), 1e-30))
        
        return dict(f=f, a=a, A=A, model=model, frac=frac, 
                   nfev=out.nfev, cost=out.cost, success=out.success)
    
    def analyze_ringdown(self, data: td.SimulationData, 
                        monitor_name: str = "probe",
                        field: str = "Ey") -> Dict:
        """
        Perform comprehensive ringdown analysis.
        
        Args:
            data: Tidy3D simulation data
            monitor_name: Name of probe monitor
            field: Field component to analyze
            
        Returns:
            Dictionary with analysis results
        """
        print("="*70)
        print("🔬 Q-FACTOR RINGDOWN ANALYSIS")
        print("="*70)
        
        # Extract time and signal data
        if monitor_name not in data.monitor_data:
            raise KeyError(f"Monitor '{monitor_name}' not found in data")
        
        probe_data = data[monitor_name]
        t_all, y_all = self.extract_time_and_signal(probe_data, field)
        
        print(f"✓ Extracted signal data")
        print(f"  - Time points: {len(t_all)}")
        print(f"  - Duration: {t_all[-1]*1e12:.2f} ps")
        print(f"  - Time step: {(t_all[1]-t_all[0])*1e12:.3f} ps")
        
        # Cut to strictly decaying tail
        t, y, i0, why = self.cut_tail_strict(t_all, y_all)
        if len(t) < 16:
            raise RuntimeError("Tail too short after cut; extend run_time or move t_safe later.")
        
        z = hilbert(y).astype(complex)
        Tseg = float(t[-1]-t[0])
        
        print(f"✓ Prepared analysis segment")
        print(f"  - Tail start: {why} (index {i0})")
        print(f"  - Tail duration: {Tseg*1e12:.3f} ps")
        print(f"  - Tail points: {len(t)}")
        
        # Extract initial seeds
        f_med, f_mad, a_seed = self.seeds_from_tail(t, z)
        f_guess = C0 / (self.wavelength_um * 1e-6)  # Use wavelength-based guess
        
        print(f"✓ Extracted initial seeds")
        print(f"  - Frequency seed: {f_med/1e12:.6f} THz")
        print(f"  - Decay rate seed: {a_seed:.3e} 1/s")
        print(f"  - Wavelength guess: {f_guess/1e12:.6f} THz")
        
        # Build grids
        f_grid, a_grid = self.build_grids(f_guess, a_seed)
        print(f"✓ Built search grids")
        print(f"  - Frequency grid: {len(f_grid)} points")
        print(f"  - Decay rate grid: {len(a_grid)} points")
        
        # Grid fit
        best1 = self.fit_one_mode_grid(t, z, f_grid, a_grid, 
                                     weighted=self.ringdown_config['use_weighted_LS'])
        f1, a1, A1, frac1 = best1['f'], best1['alpha'], best1['A'], best1['frac']
        tau1 = 1.0/a1; Q1 = np.pi*f1/a1
        cov1 = Tseg / max(tau1, 1e-30)
        
        print(f"✓ Single mode fit complete")
        print(f"  - Frequency: {f1/1e12:.6f} THz")
        print(f"  - Decay time: {tau1*1e12:.3f} ps")
        print(f"  - Q-factor: {Q1:,.0f}")
        print(f"  - Residual: {100*frac1:.2f}%")
        print(f"  - Coverage: {cov1:.2f} e-folds")
        
        # Greedy add second mode
        K_eff, model_grid, pick1, pick2, frac_grid = self.greedy_add_modes(
            t, z, best1, f_grid, a_grid, 
            weighted=self.ringdown_config['use_weighted_LS'])
        
        print(f"✓ Multi-mode analysis complete")
        print(f"  - Effective modes: {K_eff}")
        print(f"  - Grid residual: {100*frac_grid:.2f}%")
        
        # Refine with variable projection
        picks = [pick1, pick2] if K_eff == 2 else [pick1]
        ref = self.refine_variable_projection(t, z, picks, 
                                            weighted=self.ringdown_config['use_weighted_LS'])
        
        f_ref, a_ref, A_ref, frac_ref = ref['f'], ref['a'], ref['A'], ref['frac']
        tau_ref = 1.0/a_ref; Q_ref = np.pi*f_ref/a_ref
        cov_ref = Tseg / np.maximum(tau_ref, 1e-30)
        
        print(f"✓ Variable projection refinement complete")
        for k in range(len(f_ref)):
            print(f"  - Mode {k+1}: f={f_ref[k]/1e12:.6f} THz, τ={tau_ref[k]*1e12:.3f} ps, "
                  f"Q={Q_ref[k]:,.0f}, coverage={cov_ref[k]:.2f}")
        print(f"  - Final residual: {100*frac_ref:.2f}%")
        
        # Create plots if requested
        if self.ringdown_config['do_plot']:
            self._plot_ringdown_analysis(t_all, y_all, t, z, ref, i0)
        
        # Compile results
        results = {
            't_full': t_all,
            'y_full': y_all,
            't_tail': t,
            'z_tail': z,
            'model': ref['model'],
            'residual_fraction': frac_ref,
            'frequencies_hz': f_ref,
            'decay_rates_hz': a_ref,
            'amplitudes': A_ref,
            'decay_times_s': tau_ref,
            'q_factors': Q_ref,
            'coverage': cov_ref,
            'num_modes': len(f_ref),
            'analysis_parameters': {
                'wavelength_um': self.wavelength_um,
                'quality_preset': self.quality_preset,
                'tail_start_index': i0,
                'tail_duration_ps': Tseg * 1e12
            }
        }
        
        return results
    
    def _plot_ringdown_analysis(self, t_all: np.ndarray, y_all: np.ndarray,
                               t: np.ndarray, z: np.ndarray, ref: Dict, i0: int) -> None:
        """Create ringdown analysis plots."""
        apply_theme()
        plt.figure(figsize=(12, 5.5))

        # Colors from shared palette
        c_full = PALETTE[0]   # blue
        c_tail = PALETTE[1]   # orange
        c_model = PALETTE[3]  # red
        c_mark = PALETTE[2]   # green

        # Compute envelopes
        env_full = np.abs(hilbert(y_all)) + 1e-30
        env_tail = np.abs(z) + 1e-30

        # Plot full-signal envelope
        plt.semilogy(t_all*1e12, env_full, color=c_full, alpha=0.7,
                     linewidth=1.0, label='Full signal envelope |E|(t)')

        # Plot tail segment used for fitting
        plt.semilogy(t*1e12, env_tail, color=c_tail, linewidth=1.6,
                     label='Tail used for fit')

        # Plot fitted model envelope
        plt.semilogy(t*1e12, np.abs(ref['model']), linestyle='--', color=c_model,
                     linewidth=2.0, label=f'Fitted model (K={len(ref["f"])})')

        # Mark tail start
        plt.axvline(t_all[i0]*1e12, color=c_mark, linestyle=':', linewidth=1.2,
                    label='Tail start index')

        # Labels and title
        plt.xlabel("Time (ps)")
        plt.ylabel("|E|(t) [arb.] (log scale)")
        plt.title("Q-factor ringdown: tail fit with variable projection")

        # Annotation with per-mode Q and tau
        f = np.atleast_1d(ref['f'])
        a = np.atleast_1d(ref['a'])
        tau_ps = (1.0/np.maximum(a, 1e-30)) * 1e12
        Q_vals = np.pi * f / np.maximum(a, 1e-30)
        # Calculate resonance wavelength(s) in nanometers: lambda_nm = (c / f) * 1e9
        lambda_nm = (C0 / np.atleast_1d(f)) * 1e9
        lines = [
            fr"Mode {k+1}: Q={Q_vals[k]:,.0f}, $\tau$={tau_ps[k]:.2f} ps, $\lambda$={lambda_nm[k]:.1f} nm"
            for k in range(len(f))
        ]
        text = "\n".join(lines)
        # Place text in upper-right inside axes
        plt.gca().text(0.98, 0.98, text,
                       ha='right', va='top', transform=plt.gca().transAxes,
                       fontsize=10, color='k', bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

        plt.legend(frameon=False, fontsize=10, loc='lower right')
        plt.tight_layout()
        plt.savefig('q_factor_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Ringdown analysis plot saved to q_factor_analysis.png")
    
    def save_results(self, results: Dict, filename: str = "q_factor_results.json") -> None:
        """Save analysis results to file."""
        import json
        
        def convert_for_json(obj):
            """Convert objects to JSON-serializable format."""
            if isinstance(obj, np.ndarray):
                # Handle complex arrays
                if np.iscomplexobj(obj):
                    return {
                        'real': obj.real.tolist(),
                        'imag': obj.imag.tolist(),
                        'dtype': 'complex'
                    }
                else:
                    return obj.tolist()
            elif isinstance(obj, (np.complex128, np.complex64, complex)):
                return {
                    'real': float(obj.real),
                    'imag': float(obj.imag),
                    'dtype': 'complex'
                }
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Convert results to JSON-serializable format
        json_results = convert_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Q-factor analysis results saved to {filename}")


def analyze_q_factor(data_path: str,
                    monitor_name: str = "probe",
                    field: str = "Ey",
                    wavelength_um: float = 0.62,
                    quality_preset: str = "medium",
                    save_results: bool = True) -> Dict:
    """
    Convenience function to analyze Q-factor from simulation data.
    
    Args:
        data_path: Path to simulation data file
        monitor_name: Name of probe monitor
        field: Field component to analyze
        wavelength_um: Analysis wavelength
        quality_preset: Quality preset for analysis
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    data = td.SimulationData.from_file(data_path)
    
    # Create analyzer
    analyzer = QFactorAnalyzer(wavelength_um=wavelength_um, quality_preset=quality_preset)
    
    # Perform analysis
    results = analyzer.analyze_ringdown(data, monitor_name=monitor_name, field=field)
    
    # Save results if requested
    if save_results:
        analyzer.save_results(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python q_factor_analysis.py <data_file.hdf5> [monitor_name] [field]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    monitor_name = sys.argv[2] if len(sys.argv) > 2 else "probe"
    field = sys.argv[3] if len(sys.argv) > 3 else "Ey"
    
    results = analyze_q_factor(
        data_path=data_file,
        monitor_name=monitor_name,
        field=field,
        save_results=True
    )
    
    print("\n✅ Q-factor analysis completed!")
    print(f"Q-factors: {results['q_factors']}")
    print(f"Frequencies: {results['frequencies_hz']/1e12:.3f} THz")

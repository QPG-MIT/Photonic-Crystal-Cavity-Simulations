#!/usr/bin/env python3
"""
Polarization Analysis Module

This module provides comprehensive analysis of far-field polarization properties
from electromagnetic simulations of photonic cavities.

Key Features:
- Stokes parameter calculation from far-field data
- Degree of polarization analysis (linear, circular, total)
- Collection efficiency calculations for different NA values
- Back focal plane (BFP) visualization
- Integration with Tidy3D simulation data
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from numpy import pi, trapezoid
from pathlib import Path
from dataclasses import dataclass
from .plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Physical constants
C0 = 299_792_458.0  # Speed of light in vacuum (m/s)


@dataclass
class FarfieldSummary:
    """Container for far-field polarization analysis results."""
    theta: np.ndarray
    phi: np.ndarray
    NA_mask: np.ndarray
    per_angle: dict
    S_avg: dict
    DoLP_avg: float
    DoCP_avg: float
    DoP_avg: float
    psi_avg: float
    chi_avg: float
    NA: float
    n_bg: float
    theta_max: float


class PolarizationAnalyzer:
    """
    Analyzer for far-field polarization properties of photonic cavities.
    """
    
    def __init__(self, wavelength_um: float = 0.62):
        """
        Initialize polarization analyzer.
        
        Args:
            wavelength_um: Analysis wavelength in micrometers
        """
        self.wavelength_um = wavelength_um
        self.f0_center = C0 / (wavelength_um * 1e-6)
    
    def _nearest_freq_index(self, da: Any, f0: float) -> int:
        """Find nearest frequency index in dataset."""
        f = np.array(da.f) if "f" in da.dims else np.array(da.coords.get("f", [f0]))
        return int(np.argmin(np.abs(f - f0)))
    
    def _get_coords(self, da: Any, names: Tuple[str, str] = ("theta", "phi")) -> List[Optional[np.ndarray]]:
        """Extract coordinate arrays from dataset."""
        out = []
        for nm in names:
            if nm in da.coords:
                out.append(np.array(da.coords[nm], float))
            else:
                out.append(None)
        return out
    
    def _sph_basis_from_angles(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unit vectors in spherical coordinates (physics convention).
        
        Args:
            theta: Polar angle array
            phi: Azimuthal angle array
            
        Returns:
            Tuple of (e_theta, e_phi) unit vectors
        """
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        e_theta = np.stack([ct*cp, ct*sp, -st], axis=0)
        e_phi = np.stack([-sp, cp, np.zeros_like(theta)], axis=0)
        return e_theta, e_phi
    
    def _cart_to_spherical(self, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
                          theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project Cartesian field components onto spherical basis.
        
        Args:
            Ex, Ey, Ez: Cartesian field components
            theta, phi: Spherical coordinate arrays
            
        Returns:
            Tuple of (E_theta, E_phi) spherical field components
        """
        e_theta, e_phi = self._sph_basis_from_angles(theta[None, ...], phi[None, ...])
        Evec = np.stack([Ex, Ey, Ez], axis=0)
        E_theta = np.sum(Evec * e_theta, axis=0)
        E_phi = np.sum(Evec * e_phi, axis=0)
        return E_theta, E_phi
    
    def _stokes_from_jones(self, Et: np.ndarray, Ep: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Stokes parameters from Jones vector components.
        
        Args:
            Et, Ep: Spherical field components (E_theta, E_phi)
            
        Returns:
            Dictionary with Stokes parameters and derived quantities
        """
        S0 = np.abs(Et)**2 + np.abs(Ep)**2
        S1 = np.abs(Et)**2 - np.abs(Ep)**2
        S2 = 2.0 * np.real(Et * np.conj(Ep))
        S3 = 2.0 * np.imag(Et * np.conj(Ep))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            DoLP = np.sqrt(S1**2 + S2**2) / np.maximum(S0, 1e-30)
            DoCP = np.abs(S3) / np.maximum(S0, 1e-30)
            DoP = np.sqrt(S1**2 + S2**2 + S3**2) / np.maximum(S0, 1e-30)
            psi = 0.5 * np.arctan2(S2, S1)  # orientation angle
            chi = 0.5 * np.arcsin(np.clip(S3/np.maximum(S0, 1e-30), -1, 1))  # ellipticity angle
        
        return dict(S0=S0, S1=S1, S2=S2, S3=S3, DoLP=DoLP, DoCP=DoCP, DoP=DoP, psi=psi, chi=chi)
    
    def _solid_angle_weights(self, theta: np.ndarray) -> np.ndarray:
        """Calculate solid angle weights for integration."""
        return np.sin(theta)
    
    def analyze_farfield_polarization(self, data: td.SimulationData,
                                    monitor_name_angles: str = "farfield_angles",
                                    monitor_name_cart: str = "farfield_cartesian",
                                    n_bg: float = 1.0,
                                    NA: float = 0.65) -> FarfieldSummary:
        """
        Analyze far-field polarization properties.
        
        Args:
            data: Tidy3D simulation data
            monitor_name_angles: Name of angle monitor
            monitor_name_cart: Name of Cartesian monitor
            n_bg: Background refractive index
            NA: Numerical aperture for analysis
            
        Returns:
            FarfieldSummary with analysis results
        """
        print("="*70)
        print("🔬 FAR-FIELD POLARIZATION ANALYSIS")
        print("="*70)
        
        mdict = getattr(data, "monitor_data", data)
        Et = Ep = theta = phi = None
        
        # Try to get angle monitor first (preferred)
        if monitor_name_angles in mdict:
            print(f"✓ Using angle monitor: {monitor_name_angles}")
            md = mdict[monitor_name_angles]
            targ = getattr(md, "E_theta", None) or getattr(md, "Etheta", None)
            if targ is not None:
                i = self._nearest_freq_index(targ, self.f0_center)
                Et = np.array(targ.isel(f=i), dtype=complex)
                Ep_field = getattr(md, "E_phi", None) or getattr(md, "Ephi", None)
                if Ep_field is not None:
                    Ep = np.array(Ep_field.isel(f=i), dtype=complex)
                theta, phi = self._get_coords(targ)
                
                # Handle extra dimensions and unit conversion
                Et = np.squeeze(Et)
                Ep = np.squeeze(Ep)
                if theta is not None and np.nanmax(theta) > 2*np.pi:
                    theta = np.deg2rad(theta)
                if phi is not None and np.nanmax(phi) > 2*np.pi:
                    phi = np.deg2rad(phi)
        
        # Fallback: convert from Cartesian projection
        if (Et is None or Ep is None) and (monitor_name_cart in mdict):
            print(f"✓ Using Cartesian monitor: {monitor_name_cart}")
            md = mdict[monitor_name_cart]
            Ex_da = getattr(md, "Ex", None)
            if Ex_da is None:
                raise RuntimeError("Cartesian monitor present but Ex missing.")
            
            i = self._nearest_freq_index(Ex_da, self.f0_center)
            Ex = np.array(md.Ex.isel(f=i), dtype=complex)
            Ey = np.array(md.Ey.isel(f=i), dtype=complex)
            Ez = np.array(md.Ez.isel(f=i), dtype=complex)
            
            # Build θ,φ from ux,uy
            ux = np.array(md.ux) if hasattr(md, "ux") else None
            uy = np.array(md.uy) if hasattr(md, "uy") else None
            if ux is None or uy is None:
                raise RuntimeError("Need ux, uy on Cartesian monitor to reconstruct θ, φ.")
            
            UY, UX = np.meshgrid(uy, ux, indexing='xy')
            s = np.sqrt(UX**2 + UY**2)
            s = np.clip(s, 0, 0.999999)
            theta = np.arcsin(s)
            phi = np.mod(np.arctan2(UY, UX), 2*np.pi)
            Et, Ep = self._cart_to_spherical(Ex, Ey, Ez, theta, phi)
        
        if Et is None or Ep is None or theta is None or phi is None:
            raise KeyError(
                "Could not find usable far-field data. "
                "Make sure your results include angle monitor (E_theta/E_phi) "
                "or Cartesian projection with ux/uy and Ex/Ey/Ez."
            )
        
        print(f"✓ Extracted far-field data")
        print(f"  - Field shape: {Et.shape}")
        print(f"  - Theta range: {np.degrees(theta.min()):.1f}° to {np.degrees(theta.max()):.1f}°")
        print(f"  - Phi range: {np.degrees(phi.min()):.1f}° to {np.degrees(phi.max()):.1f}°")
        
        # Calculate Stokes parameters
        pol = self._stokes_from_jones(Et, Ep)
        
        # Build 2D angle grids
        if theta.ndim == 1 and phi.ndim == 1:
            theta2d, phi2d = np.meshgrid(theta, phi, indexing='ij')
        else:
            theta2d, phi2d = theta, phi
        
        # NA cone analysis
        sin_theta = np.sin(theta2d)
        na_max = NA / max(n_bg, 1e-9)
        na_mask = sin_theta <= na_max
        
        print(f"✓ Calculated NA cone")
        print(f"  - NA: {NA}")
        print(f"  - Background index: {n_bg}")
        print(f"  - Theta max: {np.degrees(np.arcsin(na_max)):.1f}°")
        
        # Solid angle weights and NA-averaged Stokes
        w_theta = self._solid_angle_weights(theta2d)
        W = w_theta * na_mask
        S0, S1, S2, S3 = pol["S0"], pol["S1"], pol["S2"], pol["S3"]
        
        Savg = dict(
            S0=np.sum(W * S0),
            S1=np.sum(W * S1),
            S2=np.sum(W * S2),
            S3=np.sum(W * S3),
        )
        
        DoLP_avg = np.sqrt(Savg["S1"]**2 + Savg["S2"]**2) / max(Savg["S0"], 1e-30)
        DoCP_avg = np.abs(Savg["S3"]) / max(Savg["S0"], 1e-30)
        DoP_avg = np.sqrt(Savg["S1"]**2 + Savg["S2"]**2 + Savg["S3"]**2) / max(Savg["S0"], 1e-30)
        psi_avg = 0.5 * np.arctan2(Savg["S2"], Savg["S1"])
        chi_avg = 0.5 * np.arcsin(np.clip(Savg["S3"]/max(Savg["S0"], 1e-30), -1, 1))
        
        print(f"✓ Calculated polarization metrics")
        print(f"  - DoLP: {DoLP_avg:.3f}")
        print(f"  - DoCP: {DoCP_avg:.3f}")
        print(f"  - DoP: {DoP_avg:.3f}")
        print(f"  - Psi: {np.degrees(psi_avg):.1f}°")
        print(f"  - Chi: {np.degrees(chi_avg):.1f}°")
        
        return FarfieldSummary(
            theta=theta2d, phi=phi2d, NA_mask=na_mask, per_angle=pol, S_avg=Savg,
            DoLP_avg=DoLP_avg, DoCP_avg=DoCP_avg, DoP_avg=DoP_avg,
            psi_avg=psi_avg, chi_avg=chi_avg, NA=NA, n_bg=n_bg,
            theta_max=np.arcsin(na_max)
        )
    
    def calculate_collection_efficiency(self, res: FarfieldSummary) -> float:
        """
        Calculate collection efficiency for the given NA.
        
        Args:
            res: FarfieldSummary results
            
        Returns:
            Collection efficiency (0-1)
        """
        theta2d, phi2d = res.theta, res.phi
        S0 = res.per_angle["S0"]
        w = np.sin(theta2d)
        
        P_na = trapezoid(trapezoid(S0 * w * res.NA_mask, phi2d, axis=1), theta2d[:, 0], axis=0)
        P_all = trapezoid(trapezoid(S0 * w, phi2d, axis=1), theta2d[:, 0], axis=0)
        
        return float(P_na / (P_all + 1e-30))
    
    def summarize_polarization(self, res: FarfieldSummary) -> Dict[str, Any]:
        """
        Generate a text summary of the polarization state.
        
        Args:
            res: FarfieldSummary results
            
        Returns:
            Dictionary with polarization summary
        """
        S = res.S_avg
        DoLP, DoCP, DoP = res.DoLP_avg, res.DoCP_avg, res.DoP_avg
        psi_deg = np.degrees(res.psi_avg)
        chi_deg = np.degrees(res.chi_avg)
        
        # Determine handedness
        handed = "right-circular" if S["S3"] > 0 else "left-circular"
        if abs(S["S3"]) < 1e-12:
            handed = "negligible circular"
        
        # Classify polarization state
        if DoP < 0.15:
            label = "mostly unpolarized"
        elif DoCP > DoLP:
            label = f"mostly {handed}"
        else:
            label = f"weakly linear (ψ ≈ {psi_deg:.1f}°, χ ≈ {chi_deg:.1f}°)"
        
        return dict(
            DoP=DoP, DoLP=DoLP, DoCP=DoCP, 
            psi_deg=psi_deg, chi_deg=chi_deg, 
            label=label, handedness=handed
        )
    
    def plot_theta_phi_heatmaps(self, res: FarfieldSummary, save_prefix: str = "polarization") -> None:
        """Plot theta-phi heatmaps of polarization properties."""
        print(f"\n📊 Creating theta-phi heatmaps...")
        apply_theme()
        
        theta_deg = np.rad2deg(res.theta)
        phi_deg = np.rad2deg(res.phi)
        S0 = res.per_angle["S0"]
        DoLP = res.per_angle["DoLP"]
        DoCP = res.per_angle["DoCP"]
        
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
        
        # Use the consistent colormap from plot_style
        consistent_cmap = mono_cmap
        
        # S0 (power)
        im1 = axs[0].pcolormesh(phi_deg, theta_deg, S0/np.max(S0), 
                               shading="auto", cmap=consistent_cmap)
        axs[0].set_title("S0 (power, normalized)")
        axs[0].set_xlabel("φ (deg)")
        axs[0].set_ylabel("θ (deg)")
        plt.colorbar(im1, ax=axs[0], label="S0 (norm.)")
        axs[0].contour(phi_deg, theta_deg, res.NA_mask, levels=[0.5], colors="w", linewidths=1.0)
        
        # DoLP
        im2 = axs[1].pcolormesh(phi_deg, theta_deg, DoLP, 
                               shading="auto", cmap=consistent_cmap)
        axs[1].set_title("DoLP")
        axs[1].set_xlabel("φ (deg)")
        axs[1].set_ylabel("θ (deg)")
        plt.colorbar(im2, ax=axs[1], label="DoLP")
        axs[1].contour(phi_deg, theta_deg, res.NA_mask, levels=[0.5], colors="w")
        
        # DoCP
        im3 = axs[2].pcolormesh(phi_deg, theta_deg, DoCP, 
                               shading="auto", cmap=consistent_cmap)
        axs[2].set_title("DoCP")
        axs[2].set_xlabel("φ (deg)")
        axs[2].set_ylabel("θ (deg)")
        plt.colorbar(im3, ax=axs[2], label="DoCP")
        axs[2].contour(phi_deg, theta_deg, res.NA_mask, levels=[0.5], colors="w")
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_heatmaps.png", dpi=160, bbox_inches="tight")
        plt.show()
        print(f"✓ Theta-phi heatmaps saved to {save_prefix}_heatmaps.png")
    
    def plot_azimuthal_averages(self, res: FarfieldSummary, save_prefix: str = "polarization") -> None:
        """Plot azimuthal averages of polarization properties."""
        print(f"\n📊 Creating azimuthal averages...")
        # Use default matplotlib style for linear plots (common Jupyter style)
        plt.style.use('default')
        
        S0 = res.per_angle["S0"]
        S1 = res.per_angle["S1"]
        S2 = res.per_angle["S2"]
        S3 = res.per_angle["S3"]
        theta2d, phi2d = res.theta, res.phi
        w = np.sin(theta2d)
        
        # Integrate over φ for each θ
        S0_t = trapezoid(S0 * w, phi2d, axis=1)
        S1_t = trapezoid(S1 * w, phi2d, axis=1)
        S2_t = trapezoid(S2 * w, phi2d, axis=1)
        S3_t = trapezoid(S3 * w, phi2d, axis=1)
        
        DoLP_t = np.sqrt(S1_t**2 + S2_t**2) / np.maximum(S0_t, 1e-30)
        DoCP_t = np.abs(S3_t) / np.maximum(S0_t, 1e-30)
        DoP_t = np.sqrt(S1_t**2 + S2_t**2 + S3_t**2) / np.maximum(S0_t, 1e-30)
        
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(np.rad2deg(theta2d[:, 0]), DoLP_t, label="DoLP", color=PALETTE[0])
        plt.plot(np.rad2deg(theta2d[:, 0]), DoCP_t, label="DoCP", color=PALETTE[1])
        plt.plot(np.rad2deg(theta2d[:, 0]), DoP_t, label="DoP", linestyle="--", color=PALETTE[2])
        plt.axvline(np.rad2deg(res.theta_max), color="k", ls=":", label="θ_max (NA)")
        plt.xlabel("θ (deg)")
        plt.ylabel("degree")
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_azimuthal.png", dpi=160, bbox_inches="tight")
        plt.show()
        print(f"✓ Azimuthal averages saved to {save_prefix}_azimuthal.png")
    
    def plot_bfp_intensity(self, res: FarfieldSummary, save_prefix: str = "polarization", 
                          scatter: bool = True) -> None:
        """Plot back focal plane intensity with NA circle."""
        print(f"\n📊 Creating back focal plane plot...")
        apply_theme()
        
        theta2d, phi2d = res.theta, res.phi
        S0 = res.per_angle["S0"]
        ux = np.sin(theta2d) * np.cos(phi2d)
        uy = np.sin(theta2d) * np.sin(phi2d)
        I = S0 / (np.max(S0) + 1e-30)
        
        # Use the consistent colormap from plot_style
        bfp_cmap = mono_cmap
        
        plt.figure(figsize=(5.5, 5.2))
        if scatter:
            plt.scatter(ux.flatten(), uy.flatten(), c=I.flatten(), s=8, cmap=bfp_cmap)
        else:
            # Interpolate to regular grid
            from scipy.interpolate import griddata
            grid_n = 256
            gx = np.linspace(-1, 1, grid_n)
            gy = np.linspace(-1, 1, grid_n)
            GX, GY = np.meshgrid(gx, gy, indexing="xy")
            GI = griddata((ux.flatten(), uy.flatten()), I.flatten(), (GX, GY),
                         method="linear", fill_value=np.nan)
            im = plt.imshow(GI, extent=[-1, 1, -1, 1], origin="lower", cmap=bfp_cmap)
            plt.colorbar(im, label="$S_0$ (norm.)")
        
        circle = plt.Circle((0, 0), res.NA/res.n_bg, color="w", fill=False, lw=1.2)
        plt.gca().add_artist(circle)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("$u_x$ = sin$\\theta$ cos$\\phi$")
        plt.ylabel("$u_y$ = sin$\\theta$ sin$\\phi$")
        plt.title("Back Focal Plane (normalized)")
        if scatter:
            plt.colorbar(label="$S_0$ (norm.)")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_bfp.png", dpi=160, bbox_inches="tight")
        plt.show()
        print(f"✓ Back focal plane plot saved to {save_prefix}_bfp.png")
    
    def plot_bfp_polarization_quiver(self, res: FarfieldSummary, save_prefix: str = "polarization",
                                   step: int = 4, scale: float = 0.075) -> None:
        """Plot polarization orientation and ellipticity in BFP."""
        print(f"\n📊 Creating polarization quiver plot...")
        apply_theme()
        
        theta2d, phi2d = res.theta, res.phi
        ux = np.sin(theta2d) * np.cos(phi2d)
        uy = np.sin(theta2d) * np.sin(phi2d)
        psi = res.per_angle["psi"]
        chi = res.per_angle["chi"]
        
        skip = (slice(None, None, step), slice(None, None, step))
        ux_s, uy_s = ux[skip], uy[skip]
        psi_s, chi_s = psi[skip], chi[skip]
        dx = scale * np.cos(psi_s)
        dy = scale * np.sin(psi_s)
        
        # Use the consistent colormap from plot_style for quiver plot
        quiver_cmap = bipolar_cmap
        
        plt.figure(figsize=(6, 6))
        plt.quiver(ux_s, uy_s, dx, dy, np.tanh(2*chi_s), angles="xy",
                  scale_units="xy", scale=1, cmap=quiver_cmap, width=0.004)
        plt.colorbar(label="tanh(2χ) ~ ellipticity")
        circle = plt.Circle((0, 0), res.NA/res.n_bg, color="k", fill=False, lw=1.0)
        plt.gca().add_artist(circle)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("BFP polarization: orientation (arrows) & ellipticity (color)")
        plt.xlabel("$u_x$")
        plt.ylabel("$u_y$")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_bfp_quiver.png", dpi=160, bbox_inches="tight")
        plt.show()
        print(f"✓ Polarization quiver plot saved to {save_prefix}_bfp_quiver.png")
    
    def save_results(self, res: FarfieldSummary, filename: str = "polarization_results.json") -> None:
        """Save analysis results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results = {
            'DoLP_avg': float(res.DoLP_avg),
            'DoCP_avg': float(res.DoCP_avg),
            'DoP_avg': float(res.DoP_avg),
            'psi_avg_deg': float(np.degrees(res.psi_avg)),
            'chi_avg_deg': float(np.degrees(res.chi_avg)),
            'NA': float(res.NA),
            'n_bg': float(res.n_bg),
            'theta_max_deg': float(np.degrees(res.theta_max)),
            'S_avg': {k: float(v) for k, v in res.S_avg.items()},
            'wavelength_um': float(self.wavelength_um)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Polarization analysis results saved to {filename}")


def analyze_polarization(data_path: str,
                        wavelength_um: float = 0.62,
                        NA: float = 0.65,
                        n_bg: float = 1.0,
                        monitor_name_angles: str = "farfield_angles",
                        monitor_name_cart: str = "farfield_cartesian",
                        save_results: bool = True,
                        create_plots: bool = True,
                        save_prefix: str = "polarization") -> FarfieldSummary:
    """
    Convenience function to analyze polarization from simulation data.
    
    Args:
        data_path: Path to simulation data file
        wavelength_um: Analysis wavelength in micrometers
        NA: Numerical aperture for analysis
        n_bg: Background refractive index
        monitor_name_angles: Name of angle monitor
        monitor_name_cart: Name of Cartesian monitor
        save_results: Whether to save results to file
        create_plots: Whether to create visualization plots
        save_prefix: Prefix for output files
        
    Returns:
        FarfieldSummary with analysis results
    """
    # Load data
    data = td.SimulationData.from_file(data_path)
    
    # Create analyzer
    analyzer = PolarizationAnalyzer(wavelength_um=wavelength_um)
    
    # Perform analysis
    results = analyzer.analyze_farfield_polarization(
        data, 
        monitor_name_angles=monitor_name_angles,
        monitor_name_cart=monitor_name_cart,
        n_bg=n_bg,
        NA=NA
    )
    
    # Calculate collection efficiency
    eta = analyzer.calculate_collection_efficiency(results)
    print(f"✓ Collection efficiency: {100*eta:.2f}%")
    
    # Generate summary
    summary = analyzer.summarize_polarization(results)
    print(f"✓ Polarization state: {summary['label']}")
    
    # Create plots if requested
    if create_plots:
        analyzer.plot_theta_phi_heatmaps(results, save_prefix=save_prefix)
        analyzer.plot_azimuthal_averages(results, save_prefix=save_prefix)
        analyzer.plot_bfp_intensity(results, save_prefix=save_prefix)
        analyzer.plot_bfp_polarization_quiver(results, save_prefix=save_prefix)
    
    # Save results if requested
    if save_results:
        analyzer.save_results(results, f"{save_prefix}_results.json")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python polarization_analysis.py <data_file.hdf5> [NA] [n_bg]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    NA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
    n_bg = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    results = analyze_polarization(
        data_path=data_file,
        NA=NA,
        n_bg=n_bg,
        save_results=True,
        create_plots=True
    )
    
    print("\n✅ Polarization analysis completed!")
    print(f"DoLP: {results.DoLP_avg:.3f}")
    print(f"DoCP: {results.DoCP_avg:.3f}")
    print(f"DoP: {results.DoP_avg:.3f}")

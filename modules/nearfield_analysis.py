#!/usr/bin/env python3
"""
Near-Field Analysis Module for Photonic Cavity Simulations

This module provides comprehensive near-field analysis including:
- Field confinement analysis
- Mode parameters calculation
- Field quality metrics
- Near-field visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from pathlib import Path
import warnings
from scipy.optimize import curve_fit
from typing import Dict, Optional

try:
    from .plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap
except ImportError:
    from plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class NearFieldAnalyzer:
    """
    Comprehensive near-field analyzer for Tidy3D simulations
    """

    def __init__(self, wavelength_um: float = 0.62):
        """
        Initialize the near-field analyzer

        Args:
            wavelength_um: Analysis wavelength in micrometers
        """
        self.wavelength_um = wavelength_um
        self.results = {}

    # ------------------------------ Public API ------------------------------

    def analyze_nearfield(
        self,
        data: td.SimulationData,
        monitor_name: str = "fld_xy_narrow",
        save_results: bool = True,
        create_plots: bool = True,
    ) -> Dict:
        """
        Perform comprehensive near-field analysis

        Args:
            data: Tidy3D simulation data
            monitor_name: Name of the field monitor
            save_results: Whether to save results to file
            create_plots: Whether to create visualization plots

        Returns:
            Dictionary with analysis results
        """
        print("=" * 70)
        print("🔬 NEAR-FIELD ANALYSIS")
        print("=" * 70)

        # ------------------ Load field data (robust dims handling) ------------------
        if monitor_name not in data.monitor_data:
            raise KeyError(f"Monitor '{monitor_name}' not found in data")

        field_data = data[monitor_name]

        # First frequency sample
        Ex_da = field_data.Ex.isel(f=0)
        Ey_da = field_data.Ey.isel(f=0)
        Ez_da = field_data.Ez.isel(f=0)

        # Squeeze to 2D and FORCE ('y','x') ordering
        if "z" in Ex_da.dims:
            Ex_da = Ex_da.isel(z=0)
            Ey_da = Ey_da.isel(z=0)
            Ez_da = Ez_da.isel(z=0)

        # Ensure consistent 2D shape (Ny, Nx) and dims ('y','x')
        Ex_da = Ex_da.transpose("y", "x")
        Ey_da = Ey_da.transpose("y", "x")
        Ez_da = Ez_da.transpose("y", "x")

        # Numpy arrays (Ny, Nx)
        Ex2 = np.asarray(Ex_da.values)
        Ey2 = np.asarray(Ey_da.values)
        Ez2 = np.asarray(Ez_da.values)

        # Coordinates that MATCH the arrays: x is length Nx, y is length Ny
        x = np.asarray(Ex_da.coords["x"].values)   # Nx
        y = np.asarray(Ex_da.coords["y"].values)   # Ny

        # Convert to microns if coordinates look like meters
        x, y = self._ensure_microns(x, y)

        # Intensity (Ny, Nx) — aligned with (rows=y, cols=x)
        I = (np.abs(Ex2)**2 + np.abs(Ey2)**2 + np.abs(Ez2)**2)

        # Sanity check (now this will pass regardless of original dim order)
        assert I.shape == (len(y), len(x)), (I.shape, len(y), len(x))

        print(f"Field array shapes: Ex2={Ex2.shape}, Ey2={Ey2.shape}, Ez2={Ez2.shape}, I={I.shape}")
        print(f"Coordinate lengths: x={len(x)} (Nx), y={len(y)} (Ny)")
        print(f"  - X range: {x.min():.3f} to {x.max():.3f} µm")
        print(f"  - Y range: {y.min():.3f} to {y.max():.3f} µm")

        # Simple power proxy
        P = np.real(Ex2*np.conj(Ex2) + Ey2*np.conj(Ey2) + Ez2*np.conj(Ez2))
        print(f"  - Max intensity: {np.max(I):.2e}")
        print(f"  - Total power (sum of |E|^2): {np.sum(P):.2e}")

        # ----------------------------- Analyses --------------------------------
        confinement_results = self._analyze_field_confinement(I, x, y)
        mode_results = self._calculate_mode_parameters(I)
        quality_results = self._calculate_field_quality_metrics(I, Ex2, Ey2, Ez2)

        # Combine results
        results = {
            "field_intensity": I,
            "field_power": P,
            "coordinates": {"x": x, "y": y},
            "field_components": {"Ex": Ex2, "Ey": Ey2, "Ez": Ez2},
            "confinement": confinement_results,
            "mode_parameters": mode_results,
            "quality_metrics": quality_results,
        }

        # ----------------------------- Plots -----------------------------------
        if create_plots:
            self._create_nearfield_plots(I, x, y, Ex2, Ey2, Ez2)

        if save_results:
            self._save_results(results)

        self.results = results
        return results

    # ------------------------------ Internals ------------------------------

    @staticmethod
    def _ensure_microns(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Ensure coordinates are in microns. If the span along either axis is "small",
        we assume coordinates are in meters and convert to µm.
        """
        span_x = float(x.max() - x.min()) if x.size else 0.0
        span_y = float(y.max() - y.min()) if y.size else 0.0
        # Heuristic: if EITHER axis span is < 1e-3 (likely in meters), convert BOTH to µm.
        # This prevents mismatched units across axes.
        if max(span_x, span_y) < 1e-3:
            return x * 1e6, y * 1e6
        return x, y

    def _analyze_field_confinement(self, I: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Analyze field confinement using Gaussian fitting over 1D profiles.
        I: (Ny, Nx) with rows=y, cols=x
        """
        print("\n📊 Field confinement analysis:")

        I_x_profile = I.sum(axis=0)  # length Nx, vs x
        I_y_profile = I.sum(axis=1)  # length Ny, vs y

        def gaussian(xx, A, x0, sigma):
            return A * np.exp(-((xx - x0) / sigma) ** 2)

        def _fit_profile(coord, prof, p0_sigma):
            coord_f = np.asarray(coord, dtype=np.float64)
            prof_f = np.asarray(prof, dtype=np.float64)
            valid = np.isfinite(coord_f) & np.isfinite(prof_f)
            if valid.sum() < 3:
                return None
            coord_f = coord_f[valid]
            prof_f = prof_f[valid]
            try:
                popt, _ = curve_fit(
                    gaussian,
                    coord_f,
                    prof_f,
                    p0=[float(prof_f.max()), float(coord_f.mean()), p0_sigma],
                    maxfev=2000,
                )
                sigma = abs(popt[2])
                return 2 * np.sqrt(2) * sigma  # 1/e^2 width
            except Exception:
                return None

        w_x = _fit_profile(x, I_x_profile, 1.0)
        w_y = _fit_profile(y, I_y_profile, 0.5)

        # Fallback (1/e^2 threshold width) if fit fails
        def _fallback_width(coord, prof):
            vmax = prof.max()
            idx = np.where(prof >= vmax / np.e**2)[0]
            if idx.size > 0:
                return float(coord[idx[-1]] - coord[idx[0]])
            return np.nan

        if w_x is None:
            w_x = _fallback_width(x, I_x_profile)
        if w_y is None:
            w_y = _fallback_width(y, I_y_profile)

        confinement_area = w_x * w_y
        aspect_ratio = w_x / w_y if w_y != 0 else np.nan

        print(f"  - 1/e² width (x): {w_x:.3f} µm")
        print(f"  - 1/e² width (y): {w_y:.3f} µm")
        print(f"  - Confinement area: {confinement_area:.3f} µm²")
        print(f"  - Aspect ratio (x/y): {aspect_ratio:.2f}")

        return {
            "width_x_um": float(w_x),
            "width_y_um": float(w_y),
            "confinement_area_um2": float(confinement_area),
            "aspect_ratio": float(aspect_ratio),
        }

    def _calculate_mode_parameters(self, I: np.ndarray) -> Dict:
        """
        Calculate mode area (A_eff) and simple effective parameters
        """
        print("\n📐 Mode parameters:")
        total_power = float(np.sum(I))
        max_intensity = float(np.max(I))
        mode_area = total_power / max_intensity if max_intensity > 0 else np.nan

        wavelength_um = self.wavelength_um
        mode_area_lambda2 = mode_area / (wavelength_um**2)

        # Simple placeholder
        n_eff = 2.4

        print(f"  - Mode area: {mode_area:.3f} µm²")
        print(f"  - Mode area (λ²): {mode_area_lambda2:.3f}")
        print(f"  - Effective index (rough): {n_eff:.2f}")

        return {
            "mode_area_um2": float(mode_area),
            "mode_area_lambda2": float(mode_area_lambda2),
            "effective_index": float(n_eff),
        }

    def _calculate_field_quality_metrics(
        self,
        I: np.ndarray,
        Ex2: np.ndarray,
        Ey2: np.ndarray,
        Ez2: np.ndarray,
    ) -> Dict:
        """
        Calculate simple field quality metrics
        """
        print("\n📈 Field quality metrics:")
        field_uniformity = float(np.std(I) / np.mean(I))

        Ex_mag, Ey_mag, Ez_mag = np.abs(Ex2), np.abs(Ey2), np.abs(Ez2)
        total_mag = Ex_mag + Ey_mag + Ez_mag
        Ex_fraction = float(np.sum(Ex_mag) / np.sum(total_mag))
        Ey_fraction = float(np.sum(Ey_mag) / np.sum(total_mag))
        Ez_fraction = float(np.sum(Ez_mag) / np.sum(total_mag))

        I_sorted = np.sort(I.ravel())[::-1]
        k = max(1, int(0.1 * len(I_sorted)))
        concentration = float(np.sum(I_sorted[:k]) / np.sum(I_sorted))

        print(f"  - Field uniformity (std/mean): {field_uniformity:.3f}")
        print(f"  - Ex/Ey/Ez fractions: {Ex_fraction:.3f} / {Ey_fraction:.3f} / {Ez_fraction:.3f}")
        print(f"  - Field concentration (top 10%): {concentration:.3f}")

        return {
            "field_uniformity": field_uniformity,
            "polarization_fractions": {
                "Ex": Ex_fraction,
                "Ey": Ey_fraction,
                "Ez": Ez_fraction,
            },
            "field_concentration": concentration,
        }

    # ------------------------------ Plotting ------------------------------

    def _create_nearfield_plots(
        self,
        I: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        Ex2: np.ndarray,
        Ey2: np.ndarray,
        Ez2: np.ndarray,
    ):
        """
        Create near-field visualization plots showing only field intensity, Ex real, and Ey real
        """
        print("\n📊 Creating near-field analysis plots...")
        apply_theme()

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        # Compute edge-aware extents to avoid visual shrink/stretch on coarse/nonuniform grids
        def _edges_from_centers(coords: np.ndarray) -> (float, float):
            if coords.size == 0:
                return 0.0, 1.0
            if coords.size == 1:
                return float(coords[0] - 0.5), float(coords[0] + 0.5)
            diffs = np.diff(coords)
            step = float(np.median(diffs))
            return float(coords.min() - 0.5 * step), float(coords.max() + 0.5 * step)

        x_ext_min, x_ext_max = _edges_from_centers(x)
        y_ext_min, y_ext_max = _edges_from_centers(y)

        # Calculate colormap limits for better visibility (crop to show 1-99th percentile)
        vmin_linear = np.percentile(I, 0.1)
        vmax_linear = np.percentile(I, 99.9)

        # Prepare coordinate mesh for non-uniform grids
        XX, YY = np.meshgrid(x, y, indexing="xy")  # (Ny, Nx)

        # 1) Field Intensity (use pcolormesh to respect non-uniform spacing)
        im1 = axes[0].pcolormesh(
            XX,
            YY,
            I,
            shading="auto",
            cmap=mono_cmap,
            vmin=vmin_linear,
            vmax=vmax_linear,
        )
        axes[0].set_aspect("equal")
        axes[0].set_title("Field Intensity", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("x (µm)")
        axes[0].set_ylabel("y (µm)")
        axes[0].grid(False)
        plt.colorbar(im1, ax=axes[0], label="Intensity (arb. units)", shrink=0.8, aspect=8)

        # Overlay structure outline
        self._add_structure_outline_nearfield(axes[0], x, y)

        # 2) Ex (real) - with symmetric colormap limits
        ex_real = np.real(Ex2)
        vmax_ex = np.max(np.abs(ex_real))
        vmin_ex = -vmax_ex
        
        im2 = axes[1].pcolormesh(
            XX,
            YY,
            ex_real,
            shading="auto",
            cmap=bipolar_cmap,
            vmin=vmin_ex,
            vmax=vmax_ex,
        )
        axes[1].set_aspect("equal")
        axes[1].set_title("Ex (real)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("x (µm)")
        axes[1].set_ylabel("y (µm)")
        axes[1].grid(False)
        plt.colorbar(im2, ax=axes[1], label="Ex (V/m)", shrink=0.8, aspect=8)
        
        # Overlay structure outline in dark grey
        self._add_structure_outline_nearfield(axes[1], x, y, color="darkgrey")

        # 3) Ey (real) - with symmetric colormap limits
        ey_real = np.real(Ey2)
        vmax_ey = np.max(np.abs(ey_real))
        vmin_ey = -vmax_ey
        
        im3 = axes[2].pcolormesh(
            XX,
            YY,
            ey_real,
            shading="auto",
            cmap=bipolar_cmap,
            vmin=vmin_ey,
            vmax=vmax_ey,
        )
        axes[2].set_aspect("equal")
        axes[2].set_title("Ey (real)", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("x (µm)")
        axes[2].set_ylabel("y (µm)")
        axes[2].grid(False)
        plt.colorbar(im3, ax=axes[2], label="Ey (V/m)", shrink=0.8, aspect=8)
        
        # Overlay structure outline in dark grey
        self._add_structure_outline_nearfield(axes[2], x, y, color="darkgrey")

        plt.tight_layout()
        plt.savefig("nearfield_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("✓ Near-field analysis plots saved to 'nearfield_analysis.png'")

    def _create_individual_field_plot(self, I: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Create an individual plot for the field intensity with structure outline."""
        print("\n📊 Creating individual field intensity plot...")
        apply_theme()

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(
            I,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap=mono_cmap,
            aspect="equal",
            alpha=1,
        )
        self._add_structure_outline_nearfield(ax, x, y)
        #self._plot_gds_outline_raw(ax)

        ax.set_title("Near-Field Intensity Distribution", fontsize=16, fontweight="bold")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.grid(False)
        cbar = plt.colorbar(im, ax=ax, label="Field Intensity (arb. units)")
        cbar.ax.tick_params(labelsize=10)

        # Add quick stats
        ax.text(
            0.02,
            0.98,
            f"Max Intensity: {np.max(I):.2e}\nTotal Power: {np.sum(I):.2e}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )

        plt.tight_layout()
        plt.savefig("nearfield_intensity_individual.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("✓ Individual field intensity plot saved to 'nearfield_intensity_individual.png'")

    # ------------------------------ Overlays ------------------------------

    def _build_eps_mask_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cavity_gds: str = "Cavity.gds",
        holes_gds: str = "Holes.gds",
        hole_layer: int = 0,
        hole_dtype: int = 0,
        n_core: float = 2.414,
        n_clad: float = 1.0,
    ) -> np.ndarray:
        """
        Build a 2D permittivity mask on the monitor grid (rows=y, cols=x).
        Assumes x,y are in µm to match GDS units.
        """
        import gdstk
        from matplotlib.path import Path as MPLPath

        # Grid with rows=y, cols=x so eps shape matches I (Ny, Nx)
        Y, X = np.meshgrid(y, x, indexing="ij")  # (Ny, Nx)
        Ny, Nx = Y.shape

        eps = np.full((Ny, Nx), n_clad**2, dtype=float)

        # Cavity polygons
        cav = gdstk.read_gds(str(cavity_gds))
        top = cav.top_level()[0]
        scale = cav.unit / 1e-6  # -> µm
        polys_cav = []
        for poly in getattr(top, "polygons", []):
            verts_um = np.array(poly.points, float) * scale
            polys_cav.append(MPLPath(verts_um))

        pts = np.column_stack([X.ravel(), Y.ravel()])  # (Ny*Nx, 2) in µm
        in_core = np.zeros(pts.shape[0], dtype=bool)
        for pth in polys_cav:
            in_core |= pth.contains_points(pts)
        in_core = in_core.reshape(Ny, Nx)

        # Subtract holes if present
        if Path(holes_gds).exists():
            hol = gdstk.read_gds(str(holes_gds))
            top_h = hol.top_level()[0]
            scale_h = hol.unit / 1e-6
            in_holes = np.zeros((Ny, Nx), dtype=bool)
            for poly in getattr(top_h, "polygons", []):
                if (poly.layer, poly.datatype) == (hole_layer, hole_dtype):
                    verts_um = np.array(poly.points, float) * scale_h
                    pth = MPLPath(verts_um)
                    in_holes |= pth.contains_points(pts).reshape(Ny, Nx)
            in_core &= ~in_holes

        eps[in_core] = n_core**2
        return eps  # (Ny, Nx)

    def _add_structure_outline_nearfield(
        self, ax, x: np.ndarray, y: np.ndarray, n_core=2.414, n_clad=1.0, color="white"
    ):
        """
        Add structure outline using the mask-on-grid approach (perfectly registered to the image).
        """
        try:
            from skimage import measure

            eps = self._build_eps_mask_xy(x, y, n_core=n_core, n_clad=n_clad)
            thr = 0.5 * (n_core**2 + n_clad**2)
            contours = measure.find_contours(eps, thr)

            # Map contour (row, col) -> (y, x) physical coords
            for c in contours:
                # Use subpixel interpolation to avoid shrinking due to integer truncation
                r = np.clip(c[:, 0], 0.0, len(y) - 1.0)
                cidx = np.clip(c[:, 1], 0.0, len(x) - 1.0)
                idx_y = np.arange(len(y), dtype=float)
                idx_x = np.arange(len(x), dtype=float)
                yy = np.interp(r, idx_y, y)
                xx = np.interp(cidx, idx_x, x)
                ax.plot(xx, yy, color=color, linewidth=1.2, alpha=1)
        except Exception as e:
            print(f"Warning: Could not add structure outline: {e}")

    def _plot_gds_outline_raw(
        self,
        ax,
        *,
        cavity_gds: str = "Cavity.gds",
        holes_gds: str = "Holes.gds",
        hole_layer: int = 0,
        hole_dtype: int = 0,
    ):
        """
        Draw GDS polygons directly in µm to show the full device, even outside the monitor window.
        Also expands axis limits to include the full GDS extents.
        """
        try:
            import gdstk

            def _plot_cell(cell, scale, color="w", lw=0.1, alpha=1, layer_filter=None):
                xs, ys = [], []
                for poly in getattr(cell, "polygons", []):
                    if layer_filter is not None and (poly.layer, poly.datatype) != layer_filter:
                        continue
                    verts = np.array(poly.points, float) * scale  # µm
                    verts = np.vstack([verts, verts[0]])  # close polygon
                    ax.plot(verts[:, 0], verts[:, 1], color=color, linewidth=lw, alpha=alpha)
                    xs.append(verts[:, 0])
                    ys.append(verts[:, 1])
                return xs, ys

            xs_all, ys_all = [], []

            # Cavity
            cav = gdstk.read_gds(str(cavity_gds))
            top = cav.top_level()[0]
            s = cav.unit / 1e-6
            xs, ys = _plot_cell(top, s, color="w", lw=0.1, alpha=1)
            xs_all += xs
            ys_all += ys

            # Holes
            if Path(holes_gds).exists():
                hol = gdstk.read_gds(str(holes_gds))
                top_h = hol.top_level()[0]
                sh = hol.unit / 1e-6
                xs, ys = _plot_cell(
                    top_h,
                    sh,
                    color="w",
                    lw=0.1,
                    alpha=1,
                    layer_filter=(hole_layer, hole_dtype),
                )
                xs_all += xs
                ys_all += ys

            if xs_all:
                gxmin = float(np.min([x.min() for x in xs_all]))
                gxmax = float(np.max([x.max() for x in xs_all]))
                gymin = float(np.min([y.min() for y in ys_all]))
                gymax = float(np.max([y.max() for y in ys_all]))
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.set_xlim(min(xmin, gxmin), max(xmax, gxmax))
                ax.set_ylim(min(ymin, gymin), max(ymax, gymax))
        except Exception as e:
            print(f"Warning: Could not draw raw GDS outline: {e}")

    # ------------------------------ Save results ------------------------------

    def _save_results(self, results: Dict):
        """Save analysis results to JSON file (skip large arrays)."""
        import json

        json_results = {}
        for key, value in results.items():
            if key in ["field_intensity", "field_power", "field_components"]:
                continue
            elif key == "coordinates":
                json_results[key] = {
                    "x_range": [float(results["coordinates"]["x"].min()), float(results["coordinates"]["x"].max())],
                    "y_range": [float(results["coordinates"]["y"].min()), float(results["coordinates"]["y"].max())],
                }
            else:
                json_results[key] = self._to_json(value)

        filename = "nearfield_analysis_results.json"
        with open(filename, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"✓ Near-field analysis results saved to {filename}")

    @staticmethod
    def _to_json(obj):
        """Convert numpy types to JSON-serializable"""
        if isinstance(obj, dict):
            return {k: NearFieldAnalyzer._to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [NearFieldAnalyzer._to_json(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return obj


# ------------------------------ Convenience wrapper ------------------------------

def analyze_nearfield(
    data_path: str,
    monitor_name: str = "fld_xy_narrow",
    wavelength_um: float = 0.62,
    save_results: bool = True,
    create_plots: bool = True,
) -> Dict:
    """
    Convenience function to analyze near-field from simulation data
    """
    data = td.SimulationData.from_file(data_path)
    analyzer = NearFieldAnalyzer(wavelength_um=wavelength_um)
    results = analyzer.analyze_nearfield(
        data=data,
        monitor_name=monitor_name,
        save_results=save_results,
        create_plots=create_plots,
    )
    return results


if __name__ == "__main__":
    # Example usage
    results = analyze_nearfield(
        data_path="results_0.14um.hdf5",
        monitor_name="fld_xy_narrow",
        wavelength_um=0.62,
        save_results=True,
        create_plots=True,
    )

    print("\nNear-field analysis completed!")
    print(f"Confinement area: {results['confinement']['confinement_area_um2']:.3f} µm²")
    print(f"Mode area: {results['mode_parameters']['mode_area_um2']:.3f} µm²")

#!/usr/bin/env python3
"""
Mode Volume Analysis Module

This module provides tools for computing the effective mode volume (V_eff) 
from 3D electromagnetic field data in photonic cavities.

Key Features:
- 3D field integration with proper material boundaries
- GDS-based geometry reconstruction for accurate permittivity mapping
- Mode volume calculation using standard cavity QED formulas
- Purcell factor computation for quantum emitters
- Integration with Tidy3D simulation data
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import gdstk
from pathlib import Path
from matplotlib.path import Path as MPLPath
from typing import Dict, Tuple, Optional, Any
import warnings

from .plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Physical constants
C0 = 299_792_458.0  # Speed of light in vacuum (m/s)


def _sellmeier_diamond(wavelength_um: float) -> float:
    """
    Calculate refractive index of diamond using a two-term Sellmeier equation.
    Wavelength must be in micrometers.
    Ref: A. M. Zaitsev, "Optical Properties of Diamond: A Review,"
         Phys. Status Solidi A 172, 15 (1999). (Table 1, natural diamond)
    """
    lambda_sq_um = wavelength_um**2
    # Sellmeier coefficients for natural diamond (matching simulation_setup.py)
    B1, C1 = 0.3306, 0.175**2
    B2, C2 = 4.3356, 0.106**2

    n_sq = 1.0 + B1 * lambda_sq_um / (lambda_sq_um - C1) + B2 * lambda_sq_um / (lambda_sq_um - C2)
    return np.sqrt(n_sq)


class ModeVolumeAnalyzer:
    """
    Analyzer for computing effective mode volume from 3D field data.
    """
    
    def __init__(self, 
                 cavity_gds: str = "gds/Cavity.gds",
                 holes_gds: str = "gds/Holes.gds",
                 thickness_um: float = 0.14,
                 wavelength_um: float = 0.62,
                 hole_layer: int = 0,
                 hole_dtype: int = 0,
                 n_core: float = 2.414,
                 n_clad: float = 1.0):
        """
        Initialize mode volume analyzer.
        
        Args:
            cavity_gds: Path to cavity GDS file
            holes_gds: Path to holes GDS file
            thickness_um: Slab thickness in micrometers
            wavelength_um: Analysis wavelength in micrometers
            hole_layer: GDS layer for holes
            hole_dtype: GDS datatype for holes
            n_core: Core refractive index
            n_clad: Cladding refractive index
        """
        self.cavity_gds = Path(cavity_gds)
        self.holes_gds = Path(holes_gds)
        # Resolve relative paths to repo root so notebooks/scripts work consistently
        repo_root = Path(__file__).resolve().parents[1]
        if not self.cavity_gds.is_absolute():
            self.cavity_gds = (repo_root / self.cavity_gds).resolve()
        if not self.holes_gds.is_absolute():
            self.holes_gds = (repo_root / self.holes_gds).resolve()
        self.thickness_um = thickness_um
        self.wavelength_um = wavelength_um
        self.hole_layer = hole_layer
        self.hole_dtype = hole_dtype
        self.n_core = n_core
        self.n_clad = n_clad
        
    def load_field_data(self, data: td.SimulationData, 
                       monitor_name: str = "fld_3d_box") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load 3D field data from simulation results.
        
        Args:
            data: Tidy3D simulation data
            monitor_name: Name of 3D field monitor
            
        Returns:
            Tuple of (Ex, Ey, Ez, x, y, z) field arrays and coordinates
        """
        if monitor_name not in data.monitor_data:
            raise KeyError(f"Monitor '{monitor_name}' not found in data")
        
        field_data = data[monitor_name]
        
        # Get field components
        Ex = field_data.Ex.values
        Ey = field_data.Ey.values
        Ez = field_data.Ez.values
        
        # Get coordinates (already in micrometers)
        x = field_data.Ex.coords['x'].values  # Coordinates are already in micrometers
        y = field_data.Ex.coords['y'].values  # Coordinates are already in micrometers
        z = field_data.Ex.coords['z'].values  # Coordinates are already in micrometers
        
        # Handle frequency dimension if present
        if Ex.ndim == 4:  # (x, y, z, f) -> (423, 107, 69, 1)
            Ex = Ex[:, :, :, 0]  # Take first frequency
            Ey = Ey[:, :, :, 0]
            Ez = Ez[:, :, :, 0]
        
        # Reorder from (x, y, z) to (z, y, x) to match the original implementation
        # Original shape: (423, 107, 69) -> (x, y, z)
        # Target shape: (69, 107, 423) -> (z, y, x)
        Ex = np.transpose(Ex, (2, 1, 0))  # (x, y, z) -> (z, y, x)
        Ey = np.transpose(Ey, (2, 1, 0))  # (x, y, z) -> (z, y, x)
        Ez = np.transpose(Ez, (2, 1, 0))  # (x, y, z) -> (z, y, x)
        
        print(f"✓ Loaded 3D field data")
        print(f"  - Field shape: {Ex.shape}")
        print(f"  - X range: {x.min():.3f} to {x.max():.3f} µm")
        print(f"  - Y range: {y.min():.3f} to {y.max():.3f} µm")
        print(f"  - Z range: {z.min():.3f} to {z.max():.3f} µm")
        
        return Ex, Ey, Ez, x, y, z
    
    def create_permittivity_map(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, field_shape: tuple = None) -> np.ndarray:
        """
        Create 3D permittivity map based on GDS geometry.
        This method follows the original implementation approach.
        
        Args:
            x, y, z: Coordinate arrays in meters
            field_shape: Shape of the field data (z, y, x)
            
        Returns:
            3D permittivity array with shape (z, y, x)
        """
        print("\n🔧 Creating permittivity map from GDS geometry...")
        
        # Load cavity GDS to get core boundaries (robust handling)
        if not Path(self.cavity_gds).exists():
            raise RuntimeError(f"Cavity GDS not found: {self.cavity_gds}")
        try:
            lib_cavity = gdstk.read_gds(str(self.cavity_gds))
        except Exception as exc:
            raise RuntimeError(f"Failed to read cavity GDS '{self.cavity_gds}': {exc}")
        gds_scale_cavity = lib_cavity.unit / 1e-6  # µm / user-unit
        tops_cavity = lib_cavity.top_level()
        if not tops_cavity:
            raise RuntimeError("No top-level cells found in cavity GDS")
        cavity_cell = tops_cavity[0]
        
        # Get cavity bounding box
        bbox = cavity_cell.bounding_box()
        if bbox is None:
            raise RuntimeError("Could not determine cavity bounding box")
        
        xmin_um, ymin_um = np.array(bbox[0]) * gds_scale_cavity  # µm
        xmax_um, ymax_um = np.array(bbox[1]) * gds_scale_cavity  # µm
        # Field coordinates are already in micrometers, so keep GDS coordinates in micrometers
        xmin_m, xmax_m = xmin_um, xmax_um  # Keep in micrometers
        ymin_m, ymax_m = ymin_um, ymax_um  # Keep in micrometers
        
        print(f"  - Cavity bounds: x[{xmin_um:.3f}, {xmax_um:.3f}] µm, "
              f"y[{ymin_um:.3f}, {ymax_um:.3f}] µm")
        
        # Load holes GDS (optional; proceed without if missing/unreadable)
        hole_paths = []
        if Path(self.holes_gds).exists():
            try:
                lib_holes = gdstk.read_gds(str(self.holes_gds))
                gds_scale_holes = lib_holes.unit / 1e-6  # µm / user-unit
                tops_holes = lib_holes.top_level()
                if tops_holes:
                    hole_cell = tops_holes[0]
                    for poly in getattr(hole_cell, "polygons", []):
                        if (poly.layer, poly.datatype) == (self.hole_layer, self.hole_dtype):
                            verts_um = np.array(poly.points, float) * gds_scale_holes  # µm
                            hole_paths.append(MPLPath(verts_um))  # Keep in micrometers
                else:
                    print("Warning: No top-level cells found in holes GDS; proceeding without holes")
            except Exception as exc:
                print(f"Warning: Failed to read holes GDS '{self.holes_gds}': {exc}. Proceeding without holes.")
        else:
            print(f"Warning: Holes GDS not found at '{self.holes_gds}'. Proceeding without holes.")
        
        print(f"  - Found {len(hole_paths)} hole polygons")
        
        # Get field shape (should be (z, y, x) after transpose)
        Nz, Ny, Nx = field_shape
        print(f"  - Field shape: (z={Nz}, y={Ny}, x={Nx})")
        
        # Build XY mask (Ny, Nx) - this matches the original implementation
        XX, YY = np.meshgrid(x, y, indexing="xy")  # (Ny, Nx)
        mask_core = (XX >= xmin_m) & (XX <= xmax_m) & (YY >= ymin_m) & (YY <= ymax_m)
        
        # Handle holes
        if hole_paths:
            pts = np.column_stack([XX.ravel(), YY.ravel()])
            mask_holes = np.zeros(pts.shape[0], dtype=bool)
            for hp in hole_paths:
                mask_holes |= hp.contains_points(pts)
            mask_holes = mask_holes.reshape(YY.shape)
        else:
            mask_holes = np.zeros_like(mask_core, dtype=bool)
        
        mask_xy = mask_core & (~mask_holes)  # (Ny, Nx)
        
        # Extrude across slab thickness (−t/2..+t/2)
        t_half = 0.5 * self.thickness_um  # micrometers
        mask_z = (z >= -t_half) & (z <= +t_half)  # (Nz,)
        mask_3d = mask_z[:, None, None] & mask_xy[None, :, :]  # (Nz, Ny, Nx)
        
        # Build epsilon
        eps = np.where(mask_3d, self.n_core**2, self.n_clad**2).astype(float)
        
        print(f"  - Permittivity map created: {eps.shape}")
        print(f"  - Core volume fraction: {np.sum(mask_3d) / eps.size:.3f}")
        
        return eps
    
    def compute_mode_volume(self, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
                          x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          eps: np.ndarray) -> float:
        """
        Compute effective mode volume from 3D field data.
        
        Args:
            Ex, Ey, Ez: Electric field components
            x, y, z: Coordinate arrays
            eps: 3D permittivity array
            
        Returns:
            Effective mode volume in m³
        """
        print("\n📊 Computing effective mode volume...")
        
        # Calculate volume element (coordinates are in micrometers)
        dx = float(np.mean(np.diff(x))) if x.size > 1 else (x.max() - x.min()) / max(Ex.shape[2], 1)
        dy = float(np.mean(np.diff(y))) if y.size > 1 else (y.max() - y.min()) / max(Ex.shape[1], 1)
        if z.size > 1:
            dz = float(np.mean(np.diff(z)))
        else:
            # If monitor had only a single z slice, distribute across the slab
            dz = self.thickness_um / max(Ex.shape[0], 1)
        
        dV = dx * dy * dz  # Volume element in µm³
        
        print(f"  - Volume element: {dV*1e18:.3f} µm³")
        
        # Compute |E|² weighted by permittivity
        E_squared = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        eps_E_squared = eps * E_squared
        
        # Find maximum value
        max_eps_E_squared = np.max(eps_E_squared)
        if max_eps_E_squared <= 0:
            raise ValueError("Maximum field intensity is zero or negative")
        
        # Compute mode volume: V_eff = ∫ ε|E|² dV / max(ε|E|²)
        # dV is in µm³, so convert to m³ for final result
        numerator = np.sum(eps_E_squared) * dV * 1e-18  # Convert µm³ to m³
        V_eff = numerator / max_eps_E_squared
        
        print(f"  - Maximum ε|E|²: {max_eps_E_squared:.2e}")
        print(f"  - Total weighted energy: {numerator:.2e}")
        print(f"  - Effective mode volume: {V_eff:.3e} m³ = {V_eff*1e18:.3f} µm³")
        
        return V_eff
    
    def compute_purcell_factor(self, V_eff: float, Q: float, 
                              wavelength_um: Optional[float] = None) -> float:
        """
        Compute Purcell factor for a quantum emitter.
        
        Args:
            V_eff: Effective mode volume in m³
            Q: Quality factor
            wavelength_um: Wavelength in micrometers (uses class default if None)
            
        Returns:
            Purcell factor
        """
        if wavelength_um is None:
            wavelength_um = self.wavelength_um
        
        # Calculate diamond refractive index using Sellmeier equation
        n_emitter = _sellmeier_diamond(wavelength_um)
        
        # Convert wavelength to meters
        wavelength = wavelength_um * 1e-6
        
        # Purcell factor formula: F = (3/4π²) * (λ/n)³ * (Q/V_eff)
        F = (3.0 / (4.0 * np.pi**2)) * (wavelength / n_emitter)**3 * (Q / V_eff)
        
        print(f"\n🎯 Purcell Factor Analysis")
        print(f"  - Wavelength: {wavelength_um:.3f} µm")
        print(f"  - Emitter environment index (Sellmeier): {n_emitter:.6f}")
        print(f"  - Quality factor: {Q:.0f}")
        print(f"  - Mode volume: {V_eff*1e18:.3f} µm³")
        print(f"  - Purcell factor: {F:.2f}")
        
        return F
    
    def analyze_mode_volume(self, data: td.SimulationData,
                          monitor_name: str = "fld_3d_box",
                          Q: Optional[float] = None) -> Dict:
        """
        Perform complete mode volume analysis.
        
        Args:
            data: Tidy3D simulation data
            monitor_name: Name of 3D field monitor
            Q: Quality factor (optional, for Purcell calculation)
            
        Returns:
            Dictionary with analysis results
        """
        print("="*70)
        print("🔬 MODE VOLUME ANALYSIS")
        print("="*70)
        
        # Load field data
        Ex, Ey, Ez, x, y, z = self.load_field_data(data, monitor_name)
        
        # Create permittivity map
        eps = self.create_permittivity_map(x, y, z, field_shape=Ex.shape)
        
        # Compute mode volume
        V_eff = self.compute_mode_volume(Ex, Ey, Ez, x, y, z, eps)
        
        # Compute Purcell factor if Q is provided
        F = None
        if Q is not None:
            F = self.compute_purcell_factor(V_eff, Q)
        
        # Compile results
        results = {
            'effective_mode_volume_m3': V_eff,
            'effective_mode_volume_um3': V_eff * 1e18,
            'purcell_factor': F,
            'quality_factor': Q,
            'wavelength_um': self.wavelength_um,
            'emitter_environment_index': _sellmeier_diamond(self.wavelength_um),
            'core_index': self.n_core,
            'cladding_index': self.n_clad,
            'thickness_um': self.thickness_um,
            'field_shape': Ex.shape,
            'coordinate_ranges': {
                'x_um': [x.min()*1e6, x.max()*1e6],
                'y_um': [y.min()*1e6, y.max()*1e6],
                'z_um': [z.min()*1e6, z.max()*1e6]
            }
        }
        
        return results
    
    def plot_field_distribution(self, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
                               x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               eps: np.ndarray, save_file: str = "figures/mode_volume_field.png") -> None:
        """
        Plot 3D field distribution with structure outlines in linear and log scales.
        
        Args:
            Ex, Ey, Ez: Electric field components
            x, y, z: Coordinate arrays
            eps: 3D permittivity array
            save_file: Output filename for plot
        """
        print(f"\n📊 Creating field distribution plots...")
        apply_theme()

        # Compute field intensity
        E_squared = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        
        # Find center slices
        i_center = len(x) // 2
        j_center = len(y) // 2
        k_center = len(z) // 2
        
        # Create figure with 1 row x 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Calculate colormap limits for better visibility (crop to show 1-99th percentile)
        vmin_linear = np.percentile(E_squared, 0.01)
        vmax_linear = np.percentile(E_squared, 99.99)
        
        # Field intensity plots
        # XY plane (z=0) - Top view
        im1 = axes[0].imshow(E_squared[k_center, :, :], 
                            extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6],
                            origin='lower', cmap=mono_cmap, aspect='auto', alpha=1,
                            vmin=vmin_linear, vmax=vmax_linear)
        axes[0].set_title('Top View (XY plane)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('x (µm)')
        axes[0].set_ylabel('y (µm)')
        axes[0].grid(False)  # Remove grid
        plt.colorbar(im1, ax=axes[0], label='|E|²')
        
        # Add structure outline for XY plane using permittivity map
        self._add_structure_outline_xy(axes[0], eps[k_center, :, :], x, y)
        
        # XZ plane (y=0) - Side view
        im2 = axes[1].imshow(E_squared[:, j_center, :], 
                            extent=[x.min()*1e6, x.max()*1e6, z.min()*1e6, z.max()*1e6],
                            origin='lower', cmap=mono_cmap, aspect='auto', alpha=1,
                            vmin=vmin_linear, vmax=vmax_linear)
        axes[1].set_title('Side View (XZ plane)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('x (µm)')
        axes[1].set_ylabel('z (µm)')
        axes[1].grid(False)  # Remove grid
        plt.colorbar(im2, ax=axes[1], label='|E|²')
        
        # Add structure outline for XZ plane
        self._add_structure_outline_xz(axes[1], eps[:, j_center, :], x, z)
        
        # YZ plane (x=0) - Front view
        im3 = axes[2].imshow(E_squared[:, :, i_center], 
                            extent=[y.min()*1e6, y.max()*1e6, z.min()*1e6, z.max()*1e6],
                            origin='lower', cmap=mono_cmap, aspect='auto', alpha=1,
                            vmin=vmin_linear, vmax=vmax_linear)
        axes[2].set_title('Front View (YZ plane)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('y (µm)')
        axes[2].set_ylabel('z (µm)')
        axes[2].grid(False)  # Remove grid
        plt.colorbar(im3, ax=axes[2], label='|E|²')
        
        # Add structure outline for YZ plane
        self._add_structure_outline_yz(axes[2], eps[:, :, i_center], y, z)
        
        
        plt.tight_layout()
        from pathlib import Path as _Path
        _path = _Path(save_file)
        if not _path.is_absolute():
            _repo_root = _Path(__file__).resolve().parents[1]
            _path = _repo_root / _path
        _path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(_path), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Field distribution plots saved to {_path}")
    
    def _add_structure_outline_xy(self, ax, eps_slice, x, y):
        """Add structure outline to XY plane using permittivity slice."""
        from skimage import measure
        
        # Find contours where permittivity changes significantly
        # Use a threshold between air and silicon permittivity
        eps_threshold = (self.n_clad**2 + self.n_core**2) / 2
        
        try:
            contours = measure.find_contours(eps_slice, eps_threshold)
            
            for contour in contours:
                if len(contour) > 10:  # Only plot substantial contours
                    # Convert contour indices to physical coordinates
                    y_coords = y[contour[:, 0].astype(int)]
                    x_coords = x[contour[:, 1].astype(int)]
                    
                    # Classify as cavity (high permittivity) or hole (low permittivity)
                    avg_eps = np.mean(eps_slice[contour[:, 0].astype(int), contour[:, 1].astype(int)])
                    
                    if avg_eps > eps_threshold * 1.2:  # Cavity
                        ax.plot(x_coords*1e6, y_coords*1e6, 'w-', linewidth=2, alpha=1)
                    else:  # Hole
                        ax.plot(x_coords*1e6, y_coords*1e6, 'w-', linewidth=1, alpha=1)
        except Exception as e:
            print(f"Warning: Could not add structure outline to XY plane: {e}")
    
    def _add_structure_outline_xz(self, ax, eps_slice, x, z):
        """Add structure outline to XZ plane using permittivity slice."""
        from skimage import measure
        
        eps_threshold = (self.n_clad**2 + self.n_core**2) / 2
        
        try:
            contours = measure.find_contours(eps_slice, eps_threshold)
            
            for contour in contours:
                if len(contour) > 10:
                    # Convert contour indices to physical coordinates
                    z_coords = z[contour[:, 0].astype(int)]
                    x_coords = x[contour[:, 1].astype(int)]
                    
                    # For XZ plane, we plot the top and bottom boundaries
                    ax.plot(x_coords*1e6, z_coords*1e6, 'w-', linewidth=2, alpha=1)
        except Exception as e:
            print(f"Warning: Could not add structure outline to XZ plane: {e}")
    
    def _add_structure_outline_yz(self, ax, eps_slice, y, z):
        """Add structure outline to YZ plane using permittivity slice."""
        from skimage import measure
        
        eps_threshold = (self.n_clad**2 + self.n_core**2) / 2
        
        try:
            contours = measure.find_contours(eps_slice, eps_threshold)
            
            for contour in contours:
                if len(contour) > 10:
                    # Convert contour indices to physical coordinates
                    z_coords = z[contour[:, 0].astype(int)]
                    y_coords = y[contour[:, 1].astype(int)]
                    
                    # For YZ plane, we plot the top and bottom boundaries
                    ax.plot(y_coords*1e6, z_coords*1e6, 'w-', linewidth=2, alpha=1)
        except Exception as e:
            print(f"Warning: Could not add structure outline to YZ plane: {e}")
    
    def save_results(self, results: Dict, filename: str = "data/summaries/mode_volume_results.json") -> None:
        """Save analysis results to file."""
        import json
        from pathlib import Path as _Path
        _path = _Path(filename)
        if not _path.is_absolute():
            _repo_root = _Path(__file__).resolve().parents[1]
            _path = _repo_root / _path
        _path.parent.mkdir(parents=True, exist_ok=True)
        with open(_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Mode volume analysis results saved to {_path}")


def analyze_mode_volume(data_path: str,
                       cavity_gds: str = "gds/Cavity.gds",
                       holes_gds: str = "gds/Holes.gds",
                       thickness_um: float = 0.14,
                       wavelength_um: float = 0.62,
                       monitor_name: str = "fld_3d_box",
                       Q: Optional[float] = None,
                       save_results: bool = True,
                       create_plots: bool = True) -> Dict:
    """
    Convenience function to analyze mode volume from simulation data.
    
    Args:
        data_path: Path to simulation data file
        cavity_gds: Path to cavity GDS file
        holes_gds: Path to holes GDS file
        thickness_um: Slab thickness in micrometers
        wavelength_um: Analysis wavelength in micrometers
        monitor_name: Name of 3D field monitor
        Q: Quality factor (optional, for Purcell calculation)
        save_results: Whether to save results to file
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    data = td.SimulationData.from_file(data_path)
    
    # Create analyzer
    analyzer = ModeVolumeAnalyzer(
        cavity_gds=cavity_gds,
        holes_gds=holes_gds,
        thickness_um=thickness_um,
        wavelength_um=wavelength_um
    )
    
    # Perform analysis
    results = analyzer.analyze_mode_volume(data, monitor_name=monitor_name, Q=Q)
    
    # Create plots if requested
    if create_plots:
        Ex, Ey, Ez, x, y, z = analyzer.load_field_data(data, monitor_name)
        eps = analyzer.create_permittivity_map(x, y, z, field_shape=Ex.shape)
        analyzer.plot_field_distribution(Ex, Ey, Ez, x, y, z, eps)
    
    # Save results if requested
    if save_results:
        analyzer.save_results(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mode_volume_analysis.py <data_file.hdf5> [Q_factor]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    Q = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    results = analyze_mode_volume(
        data_path=data_file,
        Q=Q,
        save_results=True,
        create_plots=True
    )
    
    print("\n✅ Mode volume analysis completed!")
    print(f"Effective mode volume: {results['effective_mode_volume_um3']:.3f} µm³")
    if results['purcell_factor'] is not None:
        print(f"Purcell factor: {results['purcell_factor']:.2f}")

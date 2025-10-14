#!/usr/bin/env python3
"""
Simulation Setup Module

This module handles the creation and configuration of Tidy3D simulations
for photonic cavity analysis. It provides functions to set up geometry,
materials, sources, and monitors for comprehensive electromagnetic simulations.

Key Features:
- GDS file processing for geometry extraction
- Diamond material properties at optical wavelengths
- Configurable simulation parameters (thickness, wavelength, etc.)
- Comprehensive monitor setup (basic, far-field, mode volume)
- Idempotent simulation creation (won't overwrite existing files)
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import gdstk
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union

from .plot_style import apply_theme

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Physical constants
C0 = 299_792_458.0  # Speed of light in vacuum (m/s)


class SimulationSetup:
    """
    Handles the setup and configuration of Tidy3D simulations for photonic cavities.
    """
    
    def __init__(self,
                 thickness_um: float = 0.14,
                 wavelength_um: float = 0.62,
                 source_bandwidth_rel: float = 0.12,
                 cavity_gds: Optional[str] = None,
                 holes_gds: Optional[str] = None,
                 sidewall_angle_deg: float = 0.0,
                 trapezoid_slices: int = 1):
        """
        Initialize simulation setup parameters.

        Args:
            thickness_um: Cavity thickness in micrometers
            wavelength_um: Analysis wavelength in micrometers
            source_bandwidth_rel: Relative bandwidth (Δf/f0) of the Gaussian source
        """
        self.thickness_um = thickness_um
        self.wavelength_um = wavelength_um
        self.source_bandwidth_rel = source_bandwidth_rel
        # Store optional GDS filenames/paths (can be names like "Cavity_Design.gds")
        self._cavity_gds_input = cavity_gds
        self._holes_gds_input = holes_gds
        self.params = self._setup_geometry_parameters()
        # Trapezoid sidewall configuration (0 disables, slices>=2 enables stepped taper)
        self.sidewall_angle_deg = float(sidewall_angle_deg or 0.0)
        self.trapezoid_slices = int(trapezoid_slices or 1)
        self.diamond_medium, self.clad_medium, self.f0_center = self._create_diamond_medium()
        
    def _setup_geometry_parameters(self) -> Dict:
        """Setup geometry parameters with improved far-field configuration."""
        print("="*70)
        print("🔧 SETTING UP GEOMETRY PARAMETERS")
        print("="*70)
        
        # File paths (resolve relative to repository root so notebooks/scripts work)
        repo_root = Path(__file__).resolve().parents[1]
        # Determine cavity GDS path
        if self._cavity_gds_input:
            cav_candidate = Path(self._cavity_gds_input)
            gds_path = cav_candidate if cav_candidate.is_absolute() else (repo_root / "gds" / cav_candidate.name)
            gds_path = gds_path.resolve()
        else:
            gds_path = (repo_root / "gds" / "Cavity_Design.gds").resolve()
        # Determine holes GDS path
        if self._holes_gds_input:
            hol_candidate = Path(self._holes_gds_input)
            hole_gds_path = hol_candidate if hol_candidate.is_absolute() else (repo_root / "gds" / hol_candidate.name)
            hole_gds_path = hole_gds_path.resolve()
        else:
            hole_gds_path = (repo_root / "gds" / "Holes_Design.gds").resolve()
        
        # Improved padding - keep negative X padding to avoid spurious scattering
        pad_x_neg = -5.0   # left side (negative to avoid waveguide scattering)
        pad_x_pos = -5.0   # right side (negative to avoid waveguide scattering)
        pad_y_neg = 2.0    # bottom
        pad_y_pos = 2.0    # top
        pad_z_neg = 2.0    # below
        pad_z_pos = 2.0    # above
        
        # Other parameters
        n_clad = 1.0       # Air cladding
        chunk_max = 100    # Max geometries per structure chunk
        top_cell_name = "TOP"
        hole_layer = (0, 0)
        
        print(f"✓ Geometry parameters loaded")
        print(f"  - Cavity thickness: {self.thickness_um} µm")
        print(f"  - Wavelength: {self.wavelength_um} µm")
        print(f"  - Optimized padding: x=[{pad_x_neg}, {pad_x_pos}], y=[{pad_y_neg}, {pad_y_pos}], z=[{pad_z_neg}, {pad_z_pos}]")
        
        return {
            'gds_path': gds_path,
            'hole_gds_path': hole_gds_path,
            'thickness_um': self.thickness_um,
            'wavelength_um': self.wavelength_um,
            'pad_x_neg': pad_x_neg,
            'pad_x_pos': pad_x_pos,
            'pad_y_neg': pad_y_neg,
            'pad_y_pos': pad_y_pos,
            'pad_z_neg': pad_z_neg,
            'pad_z_pos': pad_z_pos,
            'n_clad': n_clad,
            'chunk_max': chunk_max,
            'top_cell_name': top_cell_name,
            'hole_layer': hole_layer
        }
    
    def _create_diamond_medium(self) -> Tuple[td.Medium, td.Medium, float]:
        """Create diamond medium using full Sellmeier dispersion model."""
        print("\n🔬 Creating diamond medium with full Sellmeier dispersion...")

        def n_diamond_sellmeier(lambda_um: float) -> float:
            """Sellmeier model for diamond (λ in µm).

            n^2(λ) = 1 + (B1 λ^2)/(λ^2 - C1) + (B2 λ^2)/(λ^2 - C2)
            Coefficients from literature (valid roughly 0.23–5 µm):
              B1=0.3306, C1=0.175^2; B2=4.3356, C2=0.106^2
            """
            lam2 = float(lambda_um)**2
            B1, C1 = 0.3306, 0.175**2
            B2, C2 = 4.3356, 0.106**2
            n2 = 1.0 + B1 * lam2 / (lam2 - C1) + B2 * lam2 / (lam2 - C2)
            return float(np.sqrt(n2))

        # Calculate refractive index at the analysis wavelength for reference
        n_diamond_ref = n_diamond_sellmeier(self.wavelength_um)

        print(f"  - Using full Sellmeier dispersion model")
        print(f"  - Coefficients: B1=0.3306, C1=0.175²; B2=4.3356, C2=0.106²")
        print(f"  - Reference n(λ={self.wavelength_um:.3f} µm) = {n_diamond_ref:.6f}")
        print(f"  - Full frequency-dependent dispersion enabled for FDTD simulation")

        # Create Sellmeier medium with proper coefficients
        # Note: Tidy3D expects coefficients in the format [(B1, C1), (B2, C2), ...]
        sellmeier_coeffs = [
            (0.3306, 0.175**2),  # (B1, C1)
            (4.3356, 0.106**2)   # (B2, C2)
        ]
        
        diamond_medium = td.Sellmeier(coeffs=sellmeier_coeffs)
        clad_medium = td.Medium(permittivity=1.0)

        # Calculate center frequency
        f0_center = C0 / (self.wavelength_um * 1e-6)

        return diamond_medium, clad_medium, f0_center

    @staticmethod
    def diamond_n_sellmeier(lambda_um: float) -> float:
        """Public helper: diamond refractive index n(λ) via Sellmeier (no prints)."""
        lam2 = float(lambda_um)**2
        B1, C1 = 0.3306, 0.175**2
        B2, C2 = 4.3356, 0.106**2
        n2 = 1.0 + B1 * lam2 / (lam2 - C1) + B2 * lam2 / (lam2 - C2)
        return float(np.sqrt(n2))
    
    def extract_geometry_from_gds(self) -> Dict:
        """Extract geometry from GDS files with improved bounds."""
        print("\n📐 Extracting geometry from GDS files...")
        
        # Load main GDS
        lib = gdstk.read_gds(str(self.params['gds_path']))
        gds_scale = lib.unit / 1e-6  # µm / user-unit
        cells_by_name = {cell.name: cell for cell in lib.cells}
        
        tops = lib.top_level()
        if not tops:
            raise RuntimeError("No top-level cells found in GDS.")
        top_cell = tops[0]
        
        print(f"  - Using cell: {top_cell.name}")
        
        # Get bounding box
        bbox = top_cell.bounding_box()
        if bbox is None:
            raise RuntimeError("Could not determine bounding box.")
        
        # Convert to µm
        xmin_orig, ymin_orig = np.array(bbox[0]) * gds_scale
        xmax_orig, ymax_orig = np.array(bbox[1]) * gds_scale
        
        print(f"  - Original bbox: x[{xmin_orig:.3f}, {xmax_orig:.3f}], y[{ymin_orig:.3f}, {ymax_orig:.3f}]")
        
        # Keep original geometry coordinates
        xmin = xmin_orig
        xmax = xmax_orig  
        ymin = ymin_orig
        ymax = ymax_orig
        
        # Apply negative padding to simulation domain boundaries
        left   = xmin - self.params['pad_x_neg']
        right  = xmax + self.params['pad_x_pos']
        bottom = ymin - self.params['pad_y_neg']
        top    = ymax + self.params['pad_y_pos']
        down   = -self.params['thickness_um']/2 - self.params['pad_z_neg']
        up     =  self.params['thickness_um']/2 + self.params['pad_z_pos']
        
        print(f"  - Geometry bbox: x[{xmin:.3f}, {xmax:.3f}], y[{ymin:.3f}, {ymax:.3f}]")
        print(f"  - Simulation domain: left={left:.3f}, right={right:.3f}, bottom={bottom:.3f}, top={top:.3f}")
        print(f"  - X truncation: {xmax_orig - xmin_orig:.3f} → {right - left:.3f} µm (reduced by {xmax_orig - xmin_orig - (right - left):.3f} µm)")
        
        # Calculate simulation size (truncated domain)
        size_x = right - left
        size_y = top - bottom
        size_z = up - down
        
        # Calculate center
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        cz = 0.0
        
        print(f"  - Simulation size: ({size_x:.3f}, {size_y:.3f}, {size_z:.3f}) µm")
        print(f"  - Simulation center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) µm")
        
        return {
            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
            'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
            'cx': cx, 'cy': cy, 'cz': cz
        }
    
    def create_core_structure(self, geom_params: Dict) -> td.Structure:
        """Create core structure from rectangle or trapezoid (stepped taper)."""
        print("\n🏗️  Creating core structure...")
        
        rect_vertices = [
            (geom_params['xmin'], geom_params['ymin']),
            (geom_params['xmin'], geom_params['ymax']),
            (geom_params['xmax'], geom_params['ymax']),
            (geom_params['xmax'], geom_params['ymin'])
        ]

        # If sidewall taper requested, use native PolySlab sidewall_angle (no slicing)
        if self.sidewall_angle_deg <= 0.0:
            core_geo = td.PolySlab(
                vertices=rect_vertices,
                axis=2,
                slab_bounds=(-self.params['thickness_um']/2, self.params['thickness_um']/2),
                reference_plane="middle",
            )
            core_struct = td.Structure(geometry=core_geo, medium=self.diamond_medium)
            print(f"  - Core rectangle created with {len(rect_vertices)} vertices")
            print(f"  - Thickness: {self.params['thickness_um']} µm")
            return core_struct

        # Trapezoid using PolySlab sidewall_angle
        t = float(self.params['thickness_um'])
        angle_rad = float(np.deg2rad(self.sidewall_angle_deg))
        width_x = geom_params['xmax'] - geom_params['xmin']
        width_y = geom_params['ymax'] - geom_params['ymin']
        inset_top = t * np.tan(angle_rad)
        if 2 * inset_top >= min(width_x, width_y):
            print("⚠️  Sidewall angle too large for given footprint; reducing to fit.")
            # Cap angle so that top width remains positive in both x and y
            max_inset = 0.49 * min(width_x, width_y)
            angle_rad = np.arctan(max_inset / t)
            inset_top = t * np.tan(angle_rad)

        core_geo = td.PolySlab(
            vertices=rect_vertices,
            axis=2,
            slab_bounds=(-t/2, t/2),
            reference_plane="bottom",  # define vertices on bottom plane
            sidewall_angle=angle_rad,
        )
        core_struct = td.Structure(geometry=core_geo, medium=self.diamond_medium)

        top_width_x = max(width_x - 2 * inset_top, 0.0)
        top_width_y = max(width_y - 2 * inset_top, 0.0)
        print(f"  - Trapezoidal core (native sidewall): sidewall_angle={self.sidewall_angle_deg:.2f}°")
        print(f"  - Bottom width (x × y): {width_x:.3f} µm × {width_y:.3f} µm")
        print(f"  - Top width    (x × y): {top_width_x:.3f} µm × {top_width_y:.3f} µm")
        print(f"  - Thickness: {t} µm")
        return core_struct
    
    def create_hole_structures(self, geom_params: Dict) -> List[td.Structure]:
        """Create hole structures from GDS."""
        print("\n🕳️  Creating hole structures...")
        
        # Load holes GDS
        lib_holes = gdstk.read_gds(str(self.params['hole_gds_path']))
        gds_scale_holes = lib_holes.unit / 1e-6  # µm / user-unit
        holes_cells_by_name = {cell.name: cell for cell in lib_holes.cells}
        
        tops_h = lib_holes.top_level()
        if not tops_h:
            raise RuntimeError("No top-level cells found in holes GDS.")
        hole_cell = tops_h[0]
        
        print(f"  - Using holes cell: {hole_cell.name}")
        
        # Create hole geometry
        holes_geo = td.Geometry.from_gds(
            gds_cell=hole_cell,
            gds_layer=self.params['hole_layer'][0],
            gds_dtype=self.params['hole_layer'][1],
            axis=2,
            slab_bounds=(-self.params['thickness_um']/2, self.params['thickness_um']/2),
            reference_plane="middle",
            gds_scale=gds_scale_holes,
        )
        
        # Handle geometry grouping
        holes_geoms = getattr(holes_geo, "geometries", [holes_geo])
        hole_structs = []
        
        for i in range(0, len(holes_geoms), self.params['chunk_max']):
            chunk = holes_geoms[i:i+self.params['chunk_max']]
            geom = td.GeometryGroup(geometries=chunk) if len(chunk) > 1 else chunk[0]
            hole_structs.append(td.Structure(geometry=geom, medium=self.clad_medium))
        
        print(f"  - Imported {len(holes_geoms)} hole polygon(s)")
        print(f"  - Created {len(hole_structs)} hole structure(s)")
        print(f"  - Thickness: {self.params['thickness_um']} µm")
        
        return hole_structs
    
    def create_sources_and_monitors(self, geom_params: Dict) -> Tuple[td.PointDipole, List[td.Monitor]]:
        """Create sources and monitors with improved configuration."""
        print("\n📡 Creating sources and monitors...")
        
        # Create source
        source = td.PointDipole(
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            source_time=td.GaussianPulse(
                freq0=self.f0_center,
                fwidth=self.f0_center * self.source_bandwidth_rel,
            ),
            polarization="Ey",
        )

        print(f"  - Source: Point dipole at ({geom_params['cx']:.3f}, {geom_params['cy']:.3f}, 0.0)")
        print(f"  - Frequency: {self.f0_center/1e12:.3f} THz")
        print(f"  - Bandwidth: {self.f0_center * self.source_bandwidth_rel/1e12:.2f} THz")
        
        # Create basic monitors
        monitors = []
        
        # Probe monitor (time data)
        probe_monitor = td.FieldTimeMonitor(
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            size=(0, 0, 0),
            name="probe",
            interval=5,  # High temporal resolution
        )
        monitors.append(probe_monitor)
        
        # Flux monitor
        flux_monitor = td.FluxMonitor(
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            size=(geom_params['size_x'] * 0.8, geom_params['size_y'] * 0.8, 0),
            freqs=[self.f0_center],
            name="flux",
        )
        monitors.append(flux_monitor)
        
        # Field monitor (near field)
        field_monitor = td.FieldMonitor(
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            size=(geom_params['size_x'] * 0.6, geom_params['size_y'] * 0.6, 0),
            freqs=[self.f0_center],
            name="field_near",
        )
        monitors.append(field_monitor)
        
        print(f"  - Created {len(monitors)} basic monitors")
        
        return source, monitors

    def create_minimal_q_probe(self, geom_params: Dict) -> Tuple[td.PointDipole, List[td.Monitor]]:
        """Create a minimal source and only the time-domain probe needed for Q analysis."""
        print("\n📡 Creating minimal source and Q-probe monitor...")

        source = td.PointDipole(
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            source_time=td.GaussianPulse(
                freq0=self.f0_center,
                fwidth=self.f0_center * self.source_bandwidth_rel,
            ),
            polarization="Ey",
        )

        probe_monitor = td.FieldTimeMonitor(
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            size=(0, 0, 0),
            name="probe",
            interval=5,
        )

        print("  - Created minimal monitor set: ['probe']")
        return source, [probe_monitor]
    
    def create_farfield_monitors(self, geom_params: Dict) -> List[td.Monitor]:
        """Create improved far-field monitors."""
        print("\n🌐 Creating improved far-field monitors...")

        monitor_z = 1.5
        monitor_size = 8.0
        print(f"  - Positioning monitors at z = {monitor_z} µm")
        print(f"  - Monitor size: {monitor_size} µm")

        monitors = []

        # Cartesian sampling
        x_samples = np.linspace(-4, 4, 50)
        y_samples = np.linspace(-4, 4, 50)

        cartesian_monitor = td.FieldProjectionCartesianMonitor(
            center=(geom_params['cx'], geom_params['cy'], monitor_z),
            size=(monitor_size, monitor_size, 0.0),
            freqs=[self.f0_center],
            name="farfield_cartesian",
            x=x_samples,
            y=y_samples,
            proj_axis=2,  # z-normal plane
        )
        monitors.append(cartesian_monitor)

        # K-space sampling
        u_max = 0.95
        ux_samples = np.linspace(-u_max, u_max, 40)
        uy_samples = np.linspace(-u_max, u_max, 40)

        kspace_monitor = td.FieldProjectionKSpaceMonitor(
            center=(geom_params['cx'], geom_params['cy'], monitor_z),
            size=(monitor_size, monitor_size, 0.0),
            freqs=[self.f0_center],
            name="farfield_kspace",
            ux=ux_samples,
            uy=uy_samples,
            proj_axis=2,  # z-normal plane
        )
        monitors.append(kspace_monitor)

        # Angle sampling (radians)
        theta_deg = np.linspace(0.0, 90.0, 100)     # 0..π/2 hemisphere above the plane
        phi_deg   = np.linspace(0.0, 360.0, 200)    # full azimuth
        theta = np.deg2rad(theta_deg)
        phi   = np.deg2rad(phi_deg)

        angle_monitor = td.FieldProjectionAngleMonitor(
            center=(geom_params['cx'], geom_params['cy'], monitor_z),
            size=(monitor_size, monitor_size, 0.0),
            freqs=[self.f0_center],
            name="farfield_angles",
            theta=theta,
            phi=phi,
            normal_dir="+",  # observe into +z hemisphere
        )
        monitors.append(angle_monitor)

        print(f"  - Created {len(monitors)} improved far-field monitors")
        return monitors
    
    def create_mode_volume_monitor(self, geom_params: Dict) -> td.Monitor:
        """Create 3D field monitor for mode volume analysis."""
        return td.FieldMonitor(
            name="fld_3d_box",
            center=(geom_params['cx'], geom_params['cy'], 0.0),
            size=(min(geom_params['size_x'], 6.0),
                  min(geom_params['size_y'], 3.0),
                  min(geom_params['size_z'], 2.0)),
            fields=["Ex", "Ey", "Ez"],
            freqs=[self.f0_center],
            interval_space=(1, 1, 1),
        )
    
    def create_simulation(self, run_time_ps: float = 10.0, min_steps_per_wvl: int = 18) -> td.Simulation:
        """Create the complete simulation with all components."""
        print("\n🚀 Creating complete simulation...")
        
        # Extract geometry
        geom_params = self.extract_geometry_from_gds()
        
        # Create structures
        core_struct = self.create_core_structure(geom_params)
        hole_structs = self.create_hole_structures(geom_params)
        all_structures = [core_struct] + hole_structs
        
        # Create sources and monitors
        source, basic_monitors = self.create_sources_and_monitors(geom_params)
        farfield_monitors = self.create_farfield_monitors(geom_params)
        modevol_monitor = self.create_mode_volume_monitor(geom_params)
        
        # Combine all monitors
        all_monitors = basic_monitors + farfield_monitors + [modevol_monitor]
        
        # Create simulation with specified run time
        run_time = run_time_ps * 1e-12  # Convert ps to seconds
        
        simulation = td.Simulation(
            size=(geom_params['size_x'], geom_params['size_y'], geom_params['size_z']),
            center=(geom_params['cx'], geom_params['cy'], geom_params['cz']),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=self.wavelength_um),
            structures=all_structures,
            sources=[source],
            monitors=all_monitors,
            run_time=run_time,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )
        
        print(f"✓ Complete simulation created successfully")
        print(f"  - Structures: {len(all_structures)}")
        print(f"  - Monitors: {len(all_monitors)}")
        print(f"  - Run time: {run_time_ps:.1f} ps")
        print(f"  - Grid: min_steps_per_wvl={min_steps_per_wvl} with wavelength={self.wavelength_um} µm")
        print(f"  - Boundaries: PML(8 layers)")
        
        return simulation

    def create_q_scout_simulation(self, run_time_ps: float = 10.0, min_steps_per_wvl: int = 18) -> td.Simulation:
        """Create a simplified simulation for the scout stage (Q-only probe)."""
        print("\n🚀 Creating minimal scout simulation (Q-only)...")

        # Extract geometry
        geom_params = self.extract_geometry_from_gds()

        # Create structures
        core_struct = self.create_core_structure(geom_params)
        hole_structs = self.create_hole_structures(geom_params)
        all_structures = [core_struct] + hole_structs

        # Create minimal source + probe
        source, probe_monitors = self.create_minimal_q_probe(geom_params)

        # Create simulation with specified run time
        run_time = run_time_ps * 1e-12  # Convert ps to seconds

        simulation = td.Simulation(
            size=(geom_params['size_x'], geom_params['size_y'], geom_params['size_z']),
            center=(geom_params['cx'], geom_params['cy'], geom_params['cz']),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=self.wavelength_um),
            structures=all_structures,
            sources=[source],
            monitors=probe_monitors,
            run_time=run_time,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )

        print(f"✓ Minimal scout simulation created successfully")
        print(f"  - Structures: {len(all_structures)}")
        print(f"  - Monitors: {len(probe_monitors)} (probe only)")
        print(f"  - Run time: {run_time_ps:.1f} ps")
        return simulation
    
    def save_simulation(self, simulation: td.Simulation, filename: str) -> None:
        """Save the simulation to a file."""
        print(f"\n💾 Saving simulation to {filename}...")
        simulation.to_file(filename)
        print(f"✓ Simulation saved successfully")
        print(f"  - File: {filename}")
        print(f"  - Size: {Path(filename).stat().st_size / 1024:.1f} KB")
    
    def visualize_simulation(self, simulation: td.Simulation) -> None:
        """Visualize the simulation setup with a simplified geometry view."""
        print("\n📊 Visualizing simulation setup...")

        apply_theme()

        # Create single consolidated geometry visualization (nearfield planes)
        print("  - Creating simulation visualization...")
        fig = plt.figure(figsize=(16, 12), constrained_layout=False)

        # Layout: left column (XY and XZ), right column (YZ)
        outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.12)
        left = outer[0].subgridspec(nrows=2, ncols=1, hspace=0.15)

        # XY plane (z=0) - top view
        ax_xy = fig.add_subplot(left[0])
        simulation.plot(z=0, ax=ax_xy)
        ax_xy.set_title('XY Plane (z=0) - Top View', pad=1, fontsize=14, fontweight='bold')
        ax_xy.set_aspect('auto')

        # XZ plane (y=0) - side view
        ax_xz = fig.add_subplot(left[1], sharex=ax_xy)
        simulation.plot(y=0, ax=ax_xz)
        ax_xz.set_title('XZ Plane (y=0) - Side View', pad=1, fontsize=14, fontweight='bold')
        ax_xz.set_aspect('auto')

        # YZ plane (x=0) - front view
        ax_yz = fig.add_subplot(outer[1])
        simulation.plot(x=0, ax=ax_yz)
        ax_yz.set_title('YZ Plane (x=0) - Front View', pad=2, fontsize=14, fontweight='bold')
        ax_yz.set_aspect('auto')

        fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06)
        fig_dir = Path(__file__).resolve().parents[1] / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_dir / 'simulation_setup.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Visualization saved to figures/simulation_setup.png")


def create_simulation_setup(thickness_um: float = 0.14, 
                          wavelength_um: float = 0.62,
                          run_time_ps: float = 10.0,
                          save_file: Optional[str] = None,
                          visualize: bool = True) -> td.Simulation:
    """
    Convenience function to create a complete simulation setup.
    
    Args:
        thickness_um: Cavity thickness in micrometers
        wavelength_um: Analysis wavelength in micrometers  
        run_time_ps: Simulation run time in picoseconds
        save_file: Optional filename to save simulation (if None, auto-generates)
        visualize: Whether to create visualization plots
        
    Returns:
        Configured Tidy3D simulation object
    """
    setup = SimulationSetup(thickness_um=thickness_um, wavelength_um=wavelength_um)
    simulation = setup.create_simulation(run_time_ps=run_time_ps)
    
    if save_file is None:
        thickness_str = f"_{thickness_um}um" if thickness_um != 0.14 else "_default"
        save_file = f"simulation{thickness_str}.json"
    
    setup.save_simulation(simulation, save_file)
    
    if visualize:
        setup.visualize_simulation(simulation)
    
    return simulation


if __name__ == "__main__":
    # Example usage
    simulation = create_simulation_setup(
        thickness_um=0.14,
        wavelength_um=0.62,
        run_time_ps=10.0,
        visualize=True
    )

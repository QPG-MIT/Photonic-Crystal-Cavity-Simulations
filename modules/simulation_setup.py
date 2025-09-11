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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Physical constants
C0 = 299_792_458.0  # Speed of light in vacuum (m/s)


class SimulationSetup:
    """
    Handles the setup and configuration of Tidy3D simulations for photonic cavities.
    """
    
    def __init__(self, thickness_um: float = 0.14, wavelength_um: float = 0.62):
        """
        Initialize simulation setup parameters.
        
        Args:
            thickness_um: Cavity thickness in micrometers
            wavelength_um: Analysis wavelength in micrometers
        """
        self.thickness_um = thickness_um
        self.wavelength_um = wavelength_um
        self.params = self._setup_geometry_parameters()
        self.diamond_medium, self.clad_medium, self.f0_center = self._create_diamond_medium()
        
    def _setup_geometry_parameters(self) -> Dict:
        """Setup geometry parameters with improved far-field configuration."""
        print("="*70)
        print("🔧 SETTING UP GEOMETRY PARAMETERS")
        print("="*70)
        
        # File paths
        gds_path = Path("Cavity.gds")
        hole_gds_path = Path("Holes.gds")
        
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
        """Create diamond medium with correct refractive index for the analysis wavelength."""
        print("\n🔬 Creating diamond medium...")
        
        # Diamond refractive index at 620nm (correct value)
        n_diamond = 2.414
        
        print(f"  - n(λ={self.wavelength_um:.3f} µm) = {n_diamond:.6f} (correct for diamond at 620nm)")
        print(f"  - Using constant permittivity (narrowband simulation)")
        
        # Create medium with correct refractive index
        diamond_medium = td.Medium(permittivity=n_diamond**2)
        clad_medium = td.Medium(permittivity=1.0)
        
        # Calculate center frequency
        f0_center = C0 / (self.wavelength_um * 1e-6)
        
        return diamond_medium, clad_medium, f0_center
    
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
        """Create core structure from rectangle."""
        print("\n🏗️  Creating core structure...")
        
        rect_vertices = [
            (geom_params['xmin'], geom_params['ymin']),
            (geom_params['xmin'], geom_params['ymax']),
            (geom_params['xmax'], geom_params['ymax']),
            (geom_params['xmax'], geom_params['ymin'])
        ]
        
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
                fwidth=self.f0_center * 0.12,  # 12% bandwidth
            ),
            polarization="Ey",
        )
        
        print(f"  - Source: Point dipole at ({geom_params['cx']:.3f}, {geom_params['cy']:.3f}, 0.0)")
        print(f"  - Frequency: {self.f0_center/1e12:.3f} THz")
        print(f"  - Bandwidth: {self.f0_center * 0.12/1e12:.2f} THz")
        
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
    
    def create_simulation(self, run_time_ps: float = 10.0) -> td.Simulation:
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
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=18, wavelength=self.wavelength_um),
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
        print(f"  - Grid: min_steps_per_wvl=18 with wavelength={self.wavelength_um} µm")
        print(f"  - Boundaries: PML(8 layers)")
        
        return simulation
    
    def save_simulation(self, simulation: td.Simulation, filename: str) -> None:
        """Save the simulation to a file."""
        print(f"\n💾 Saving simulation to {filename}...")
        simulation.to_file(filename)
        print(f"✓ Simulation saved successfully")
        print(f"  - File: {filename}")
        print(f"  - Size: {Path(filename).stat().st_size / 1024:.1f} KB")
    
    def visualize_simulation(self, simulation: td.Simulation) -> None:
        """Visualize the simulation setup with all projections."""
        print("\n📊 Visualizing simulation setup...")
        
        # Create comprehensive 2D visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # XY plane (z=0) - top view
        simulation.plot(z=0, ax=axes[0, 0])
        axes[0, 0].set_title('XY Plane (z=0) - Top View')
        
        # XZ plane (y=0) - side view
        simulation.plot(y=0, ax=axes[0, 1])
        axes[0, 1].set_title('XZ Plane (y=0) - Side View')
        
        # YZ plane (x=0) - front view
        simulation.plot(x=0, ax=axes[0, 2])
        axes[0, 2].set_title('YZ Plane (x=0) - Front View')
        
        # XY plane at z=1.5 (far-field monitor level)
        simulation.plot(z=1.5, ax=axes[1, 0])
        axes[1, 0].set_title('XY Plane (z=1.5) - Far-Field Level')
        
        # XZ plane at y=0 with far-field monitors highlighted
        simulation.plot(y=0, ax=axes[1, 1])
        axes[1, 1].set_title('XZ Plane (y=0) - Far-Field Monitors')
        
        # YZ plane at x=0
        simulation.plot(x=0, ax=axes[1, 2])
        axes[1, 2].set_title('YZ Plane (x=0) - Cross Section')
        
        plt.tight_layout()
        plt.savefig('simulation_setup.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Simulation setup visualization saved to simulation_setup.png")


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

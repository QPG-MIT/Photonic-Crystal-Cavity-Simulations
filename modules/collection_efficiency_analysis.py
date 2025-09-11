#!/usr/bin/env python3
"""
Collection Efficiency Analysis Module for Photonic Cavity Simulations

This module provides comprehensive collection efficiency analysis including:
- Collection efficiency calculation
- K-space analysis
- Radiation pattern analysis
- Collection optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from pathlib import Path
import warnings
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Dict, Tuple, Optional
from .plot_style import apply_theme

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CollectionEfficiencyAnalyzer:
    """
    Comprehensive collection efficiency analyzer for Tidy3D simulations
    """
    
    def __init__(self, wavelength_um: float = 0.62, NA: float = 0.9, n_bg: float = 1.0):
        """
        Initialize the collection efficiency analyzer
        
        Args:
            wavelength_um: Analysis wavelength in micrometers
            NA: Numerical aperture for collection
            n_bg: Background refractive index
        """
        self.wavelength_um = wavelength_um
        self.NA = NA
        self.n_bg = n_bg
        self.results = {}
    
    def analyze_collection_efficiency(self, data: td.SimulationData,
                                    monitor_names: Optional[list] = None,
                                    save_results: bool = True,
                                    create_plots: bool = True) -> Dict:
        """
        Perform comprehensive collection efficiency analysis
        
        Args:
            data: Tidy3D simulation data
            monitor_names: List of monitor names to analyze
            save_results: Whether to save results to file
            create_plots: Whether to create visualization plots
            
        Returns:
            Dictionary with analysis results
        """
        print("="*70)
        print("📊 COLLECTION EFFICIENCY ANALYSIS")
        print("="*70)
        
        # Default monitor names to look for
        if monitor_names is None:
            monitor_names = [
                'farfield_cartesian',
                'farfield_kspace', 
                'farfield_angles',
                'farfield_bfp'
            ]
        
        # Find available monitors
        available_monitors = []
        for name in monitor_names:
            if name in data.monitor_data:
                available_monitors.append(name)
                print(f"✓ Found monitor: {name}")
            else:
                print(f"✗ Monitor not found: {name}")
        
        if not available_monitors:
            print("No far-field monitors found in data!")
            return {}
        
        # Analyze collection efficiency for each monitor
        results = {}
        for monitor_name in available_monitors:
            print(f"\n--- Analyzing collection efficiency for {monitor_name} ---")
            monitor_results = self._analyze_monitor_collection(data[monitor_name], monitor_name)
            results[monitor_name] = monitor_results
        
        # Calculate overall collection efficiency
        overall_results = self._calculate_overall_collection_efficiency(results)
        results['overall'] = overall_results
        
        # Note: Collection efficiency plots are now handled by far-field analysis
        # to avoid duplication and provide better visualization
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        self.results = results
        return results
    
    def _analyze_monitor_collection(self, monitor_data, monitor_name: str) -> Dict:
        """
        Analyze collection efficiency for a specific monitor
        """
        results = {}
        
        try:
            # Get field components - handle different monitor types
            Ex, Ey, Ez = None, None, None
            
            if hasattr(monitor_data, 'Ex'):
                # Cartesian field components
                Ex = monitor_data.Ex.isel(f=0).values
                Ey = monitor_data.Ey.isel(f=0).values
                Ez = monitor_data.Ez.isel(f=0).values if hasattr(monitor_data, 'Ez') else None
            elif hasattr(monitor_data, 'Er'):
                # Spherical field components (far-field)
                Er = monitor_data.Er.isel(f=0).values
                Etheta = monitor_data.Etheta.isel(f=0).values
                Ephi = monitor_data.Ephi.isel(f=0).values
                # Convert to Cartesian for analysis
                Ex = Ephi  # Approximate mapping
                Ey = Etheta
                Ez = Er
            else:
                print(f"  - No field data found in {monitor_name}")
                return results
            
            # Calculate intensity
            if Ez is not None:
                I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
            else:
                I = np.abs(Ex)**2 + np.abs(Ey)**2
            
            # Get coordinates
            coords = self._get_monitor_coordinates(monitor_data, monitor_name)
            
            # Calculate collection efficiency based on monitor type
            if 'angles' in monitor_name:
                collection_results = self._calculate_angular_collection(I, coords)
            elif 'kspace' in monitor_name:
                collection_results = self._calculate_kspace_collection(I, coords)
            elif 'cartesian' in monitor_name:
                collection_results = self._calculate_cartesian_collection(I, coords)
            else:
                collection_results = self._calculate_generic_collection(I, coords)
            
            results.update(collection_results)
            
            # Store field data
            results['field_data'] = {
                'Ex': Ex,
                'Ey': Ey,
                'Ez': Ez,
                'intensity': I,
                'coordinates': coords
            }
            
            hemisphere_note = collection_results.get('hemisphere_coverage', 'unknown')
            print(f"  - Collection efficiency ({hemisphere_note}): {collection_results.get('collection_efficiency', 0):.3f}")
            print(f"  - Collection efficiency (full sphere): {collection_results.get('collection_efficiency_full_sphere', 0):.3f}")
            print(f"  - Total power ({hemisphere_note}): {collection_results.get('total_power', 0):.2e}")
            print(f"  - Total power (full sphere): {collection_results.get('total_power_full_sphere', 0):.2e}")
            print(f"  - Collected power: {collection_results.get('collected_power', 0):.2e}")
            
        except Exception as e:
            print(f"  - Error analyzing {monitor_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _get_monitor_coordinates(self, monitor_data, monitor_name: str) -> Dict:
        """
        Extract coordinates from monitor data
        """
        coords = {}
        
        try:
            # Get coordinates from available field components
            field_component = None
            if hasattr(monitor_data, 'Ex'):
                field_component = monitor_data.Ex
            elif hasattr(monitor_data, 'Er'):
                field_component = monitor_data.Er
            elif hasattr(monitor_data, 'Etheta'):
                field_component = monitor_data.Etheta
            elif hasattr(monitor_data, 'Ephi'):
                field_component = monitor_data.Ephi
            
            if field_component is not None:
                # Get all available coordinates
                for coord_name in ['x', 'y', 'z', 'kx', 'ky', 'kz', 'theta', 'phi']:
                    if coord_name in field_component.coords:
                        coord_data = field_component.coords[coord_name]
                        coords[coord_name] = coord_data.values if hasattr(coord_data, 'values') else coord_data
                        
        except Exception as e:
            print(f"  - Warning: Could not extract coordinates: {e}")
        
        return coords
    
    def _calculate_angular_collection(self, I: np.ndarray, coords: Dict) -> Dict:
        """
        Calculate collection efficiency from angular radiation pattern
        """
        results = {}
        
        try:
            theta = coords.get('theta')
            phi = coords.get('phi')
            
            if theta is None:
                print("  - No angular coordinates available")
                return results
            
            # Calculate maximum collection angle
            theta_max = np.arcsin(self.NA / self.n_bg)
            
            # Create coordinate grids and solid angle weighting (same as far-field analysis)
            if len(I.shape) == 3:
                # 3D pattern (freq, theta, phi) - remove frequency dimension
                I_2d = I[0, :, :]  # Remove frequency dimension
                theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
            elif len(I.shape) == 2 and phi is not None:
                # 2D angular pattern
                theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
                I_2d = I
            else:
                # 1D angular pattern
                if theta.ndim > 1:
                    theta_1d = theta.flatten()
                else:
                    theta_1d = theta
                # For 1D, create a simple meshgrid
                theta_2d, phi_2d = np.meshgrid(theta_1d, [0], indexing='ij')
                I_2d = I[:, np.newaxis] if len(I.shape) == 1 else I
            
            # Calculate solid angle weighting factor sin(θ)
            sin_theta = np.sin(theta_2d)
            
            # Calculate total power with solid angle weighting
            total_power = np.sum(I_2d * sin_theta)
            
            # Create collection mask
            collection_mask = theta_2d <= theta_max
            
            # Calculate collected power with solid angle weighting
            collected_power = np.sum((I_2d * sin_theta)[collection_mask])
            collection_efficiency = collected_power / total_power if total_power > 0 else 0
            
            # Calculate additional metrics
            max_intensity = np.max(I)
            directivity = 4 * np.pi * max_intensity / total_power if total_power > 0 else 0
            
            # For symmetric structures, the angular monitor typically only covers upper hemisphere (0° to 90°)
            # If theta range is 0° to 90°, we need to account for the full 4π steradian emission
            theta_range_deg = np.degrees(theta_2d.max() - theta_2d.min())
            if theta_range_deg <= 90:  # Only upper hemisphere
                # For symmetric structure, total emission is 2x the upper hemisphere
                total_power_full_sphere = total_power * 2
                collection_efficiency_full_sphere = collected_power / total_power_full_sphere
                hemisphere_note = "upper hemisphere only"
            else:  # Full sphere or other range
                total_power_full_sphere = total_power
                collection_efficiency_full_sphere = collection_efficiency
                hemisphere_note = "full sphere"
            
            results = {
                'collection_efficiency': float(collection_efficiency),
                'collection_efficiency_full_sphere': float(collection_efficiency_full_sphere),
                'total_power': float(total_power),
                'total_power_full_sphere': float(total_power_full_sphere),
                'collected_power': float(collected_power),
                'hemisphere_coverage': hemisphere_note,
                'theta_max_deg': float(np.degrees(theta_max)),
                'directivity': float(directivity),
                'max_intensity': float(max_intensity)
            }
            
        except Exception as e:
            print(f"  - Error in angular collection calculation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_kspace_collection(self, I: np.ndarray, coords: Dict) -> Dict:
        """
        Calculate collection efficiency from k-space data
        """
        results = {}
        
        try:
            kx = coords.get('kx')
            ky = coords.get('ky')
            
            if kx is None or ky is None:
                print("  - No k-space coordinates available")
                return results
            
            # Calculate total power
            total_power = np.sum(I)
            
            # Calculate k-space collection radius
            k0 = 2 * np.pi / (self.wavelength_um * 1e-6)  # Free space k
            k_collection = k0 * self.NA / self.n_bg
            
            # Create collection mask
            if len(I.shape) == 2:
                # 2D k-space pattern
                kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
                k_magnitude = np.sqrt(kx_2d**2 + ky_2d**2)
                collection_mask = k_magnitude <= k_collection
            else:
                # 1D k-space pattern
                k_magnitude = np.sqrt(kx**2 + ky**2) if ky is not None else np.abs(kx)
                collection_mask = k_magnitude <= k_collection
            
            # Calculate collected power
            collected_power = np.sum(I[collection_mask])
            collection_efficiency = collected_power / total_power if total_power > 0 else 0
            
            # Calculate k-space metrics
            k_max = np.sqrt(kx.max()**2 + ky.max()**2) if ky is not None else kx.max()
            NA_effective = k_max / k0
            
            results = {
                'collection_efficiency': float(collection_efficiency),
                'total_power': float(total_power),
                'collected_power': float(collected_power),
                'k_collection': float(k_collection),
                'k_max': float(k_max),
                'NA_effective': float(NA_effective)
            }
            
        except Exception as e:
            print(f"  - Error in k-space collection calculation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_cartesian_collection(self, I: np.ndarray, coords: Dict) -> Dict:
        """
        Calculate collection efficiency from cartesian data
        """
        results = {}
        
        try:
            x = coords.get('x')
            y = coords.get('y')
            
            if x is None or y is None:
                print("  - No cartesian coordinates available")
                return results
            
            # Calculate total power
            total_power = np.sum(I)
            
            # Calculate collection radius (approximate)
            # This is a simplified calculation - in practice would need more sophisticated analysis
            collection_radius = self.wavelength_um * self.NA / (2 * np.pi * self.n_bg)
            
            # Create collection mask
            if len(I.shape) == 2:
                # 2D cartesian pattern
                x_2d, y_2d = np.meshgrid(x, y, indexing='ij')
                r_magnitude = np.sqrt(x_2d**2 + y_2d**2)
                collection_mask = r_magnitude <= collection_radius
            else:
                # 1D cartesian pattern
                r_magnitude = np.sqrt(x**2 + y**2) if y is not None else np.abs(x)
                collection_mask = r_magnitude <= collection_radius
            
            # Calculate collected power
            collected_power = np.sum(I[collection_mask])
            collection_efficiency = collected_power / total_power if total_power > 0 else 0
            
            results = {
                'collection_efficiency': float(collection_efficiency),
                'total_power': float(total_power),
                'collected_power': float(collected_power),
                'collection_radius_um': float(collection_radius)
            }
            
        except Exception as e:
            print(f"  - Error in cartesian collection calculation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_generic_collection(self, I: np.ndarray, coords: Dict) -> Dict:
        """
        Calculate collection efficiency for generic data
        """
        results = {}
        
        try:
            # Calculate total power
            total_power = np.sum(I)
            
            # Simple collection efficiency based on intensity distribution
            # This is a very simplified approach
            max_intensity = np.max(I)
            high_intensity_mask = I >= 0.1 * max_intensity
            collected_power = np.sum(I[high_intensity_mask])
            collection_efficiency = collected_power / total_power if total_power > 0 else 0
            
            results = {
                'collection_efficiency': float(collection_efficiency),
                'total_power': float(total_power),
                'collected_power': float(collected_power),
                'max_intensity': float(max_intensity)
            }
            
        except Exception as e:
            print(f"  - Error in generic collection calculation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_overall_collection_efficiency(self, results: Dict) -> Dict:
        """
        Calculate overall collection efficiency from all monitors
        """
        print("\n📊 Overall collection efficiency analysis:")
        
        overall_results = {}
        
        try:
            # Collect all collection efficiencies
            efficiencies = []
            total_powers = []
            collected_powers = []
            
            for monitor_name, monitor_results in results.items():
                if 'collection_efficiency' in monitor_results:
                    efficiencies.append(monitor_results['collection_efficiency'])
                    total_powers.append(monitor_results.get('total_power', 0))
                    collected_powers.append(monitor_results.get('collected_power', 0))
            
            if efficiencies:
                # Calculate weighted average
                total_power_sum = sum(total_powers)
                collected_power_sum = sum(collected_powers)
                overall_efficiency = collected_power_sum / total_power_sum if total_power_sum > 0 else 0
                
                # Calculate statistics
                mean_efficiency = np.mean(efficiencies)
                std_efficiency = np.std(efficiencies)
                max_efficiency = np.max(efficiencies)
                min_efficiency = np.min(efficiencies)
                
                overall_results = {
                    'overall_efficiency': float(overall_efficiency),
                    'mean_efficiency': float(mean_efficiency),
                    'std_efficiency': float(std_efficiency),
                    'max_efficiency': float(max_efficiency),
                    'min_efficiency': float(min_efficiency),
                    'total_power': float(total_power_sum),
                    'collected_power': float(collected_power_sum),
                    'num_monitors': len(efficiencies)
                }
                
                print(f"  - Overall collection efficiency: {overall_efficiency:.3f} ({overall_efficiency*100:.1f}%)")
                print(f"  - Mean efficiency: {mean_efficiency:.3f} ± {std_efficiency:.3f}")
                print(f"  - Range: {min_efficiency:.3f} to {max_efficiency:.3f}")
                print(f"  - Total power: {total_power_sum:.2e}")
                print(f"  - Collected power: {collected_power_sum:.2e}")
            else:
                print("  - No collection efficiency data available")
                
        except Exception as e:
            print(f"  - Error calculating overall collection efficiency: {e}")
            overall_results['error'] = str(e)
        
        return overall_results
    
    def _create_collection_plots(self, results: Dict):
        """
        Create comprehensive collection efficiency visualization plots
        """
        print("\n📊 Creating collection efficiency plots...")
        apply_theme()

        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Collection efficiency comparison
        ax1 = axes[0, 0]
        monitor_names = []
        efficiencies = []
        
        for monitor_name, monitor_results in results.items():
            if monitor_name != 'overall' and 'collection_efficiency' in monitor_results:
                monitor_names.append(monitor_name.replace('farfield_', ''))
                efficiencies.append(monitor_results['collection_efficiency'])
        
        if monitor_names:
            bars = ax1.bar(monitor_names, efficiencies, alpha=0.7)
            ax1.set_ylabel('Collection Efficiency')
            ax1.set_title('Collection Efficiency by Monitor')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, efficiency in zip(bars, efficiencies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{efficiency:.3f}', ha='center', va='bottom')
        
        # Plot 2: Power distribution
        ax2 = axes[0, 1]
        total_powers = []
        collected_powers = []
        
        for monitor_name, monitor_results in results.items():
            if monitor_name != 'overall' and 'total_power' in monitor_results:
                total_powers.append(monitor_results['total_power'])
                collected_powers.append(monitor_results.get('collected_power', 0))
        
        if total_powers:
            x = np.arange(len(monitor_names))
            width = 0.35
            
            ax2.bar(x - width/2, total_powers, width, label='Total Power', alpha=0.7)
            ax2.bar(x + width/2, collected_powers, width, label='Collected Power', alpha=0.7)
            ax2.set_xlabel('Monitor')
            ax2.set_ylabel('Power')
            ax2.set_title('Power Distribution')
            ax2.set_xticks(x)
            ax2.set_xticklabels(monitor_names)
            ax2.legend()
            ax2.set_yscale('log')
        
        # Plot 3: Collection efficiency vs NA (if available)
        ax3 = axes[1, 0]
        if 'overall' in results and 'overall_efficiency' in results['overall']:
            # Create a simple plot showing the current NA
            nas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            # This is a simplified model - in practice would need more sophisticated calculation
            theoretical_efficiencies = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            
            ax3.plot(nas, theoretical_efficiencies, label='Theoretical', linewidth=2)
            ax3.axvline(x=self.NA, linestyle='--', label=f'Current NA = {self.NA}')
            ax3.axhline(y=results['overall']['overall_efficiency'], linestyle='--', 
                       label=f'Measured = {results["overall"]["overall_efficiency"]:.3f}')
            ax3.set_xlabel('Numerical Aperture')
            ax3.set_ylabel('Collection Efficiency')
            ax3.set_title('Collection Efficiency vs NA')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        if 'overall' in results:
            overall = results['overall']
            stats = ['Overall', 'Mean', 'Max', 'Min']
            values = [
                overall.get('overall_efficiency', 0),
                overall.get('mean_efficiency', 0),
                overall.get('max_efficiency', 0),
                overall.get('min_efficiency', 0)
            ]
            
            bars = ax4.bar(stats, values, alpha=0.7)
            ax4.set_ylabel('Collection Efficiency')
            ax4.set_title('Collection Efficiency Statistics')
            ax4.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('collection_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Collection efficiency plots saved to 'collection_efficiency_analysis.png'")
    
    def _save_results(self, results: Dict):
        """
        Save analysis results to JSON file
        """
        import json
        
        # Prepare results for JSON serialization
        json_results = {}
        for monitor_name, monitor_results in results.items():
            if monitor_name == 'overall':
                json_results[monitor_name] = monitor_results
            else:
                # Skip large arrays for JSON
                json_results[monitor_name] = {
                    k: v for k, v in monitor_results.items() 
                    if k != 'field_data'
                }
        
        filename = 'collection_efficiency_results.json'
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Collection efficiency results saved to {filename}")


def analyze_collection_efficiency(data_path: str,
                                monitor_names: Optional[list] = None,
                                wavelength_um: float = 0.62,
                                NA: float = 0.9,
                                n_bg: float = 1.0,
                                save_results: bool = True,
                                create_plots: bool = True) -> Dict:
    """
    Convenience function to analyze collection efficiency from simulation data
    
    Args:
        data_path: Path to simulation data file
        monitor_names: List of monitor names to analyze
        wavelength_um: Analysis wavelength in micrometers
        NA: Numerical aperture for collection
        n_bg: Background refractive index
        save_results: Whether to save results to file
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    data = td.SimulationData.from_file(data_path)
    
    # Create analyzer and run analysis
    analyzer = CollectionEfficiencyAnalyzer(wavelength_um=wavelength_um, NA=NA, n_bg=n_bg)
    results = analyzer.analyze_collection_efficiency(
        data=data,
        monitor_names=monitor_names,
        save_results=save_results,
        create_plots=create_plots
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = analyze_collection_efficiency(
        data_path="results_0.14um.hdf5",
        monitor_names=['farfield_cartesian', 'farfield_kspace', 'farfield_angles'],
        wavelength_um=0.62,
        NA=0.9,
        n_bg=1.0,
        save_results=True,
        create_plots=True
    )
    
    print("\nCollection efficiency analysis completed!")
    if 'overall' in results:
        eff = results['overall']['overall_efficiency']
        print(f"Overall collection efficiency: {eff:.3f} ({eff*100:.1f}%)")

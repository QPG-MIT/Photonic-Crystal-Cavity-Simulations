#!/usr/bin/env python3
"""
Far-Field Analysis Module for Photonic Cavity Simulations

This module provides comprehensive far-field analysis including:
- K-space analysis
- Collection efficiency calculation
- Radiation pattern analysis
- Far-field visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from pathlib import Path
import warnings
from scipy.signal import hilbert
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Dict, Tuple, Optional
try:
    from .plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap
except ImportError:
    from plot_style import apply_theme, PALETTE, mono_cmap, bipolar_cmap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FarFieldAnalyzer:
    """
    Comprehensive far-field analyzer for Tidy3D simulations
    """
    
    def __init__(self, wavelength_um: float = 0.62, NA: float = 0.9, n_bg: float = 1.0):
        """
        Initialize the far-field analyzer
        
        Args:
            wavelength_um: Analysis wavelength in micrometers
            NA: Numerical aperture for collection efficiency
            n_bg: Background refractive index
        """
        self.wavelength_um = wavelength_um
        self.NA = NA
        self.n_bg = n_bg
        self.results = {}
    
    def analyze_farfield(self, data: td.SimulationData,
                        monitor_names: Optional[list] = None,
                        save_results: bool = True,
                        create_plots: bool = True) -> Dict:
        """
        Perform comprehensive far-field analysis
        
        Args:
            data: Tidy3D simulation data
            monitor_names: List of far-field monitor names to analyze
            save_results: Whether to save results to file
            create_plots: Whether to create visualization plots
            
        Returns:
            Dictionary with analysis results
        """
        print("="*70)
        print("🌐 FAR-FIELD ANALYSIS")
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
        
        # Analyze each available monitor
        results = {}
        for monitor_name in available_monitors:
            print(f"\n--- Analyzing {monitor_name} ---")
            monitor_results = self._analyze_monitor(data[monitor_name], monitor_name)
            results[monitor_name] = monitor_results
        
        # Calculate collection efficiency if possible
        collection_results = self._calculate_collection_efficiency(results)
        if collection_results:
            results['collection_efficiency'] = collection_results
        
        # Create plots if requested
        if create_plots:
            self._create_farfield_plots(results)
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        self.results = results
        return results
    
    def _analyze_monitor(self, monitor_data, monitor_name: str) -> Dict:
        """
        Analyze a specific far-field monitor
        """
        results = {}
        
        try:
            # Get field components - handle different monitor types
            if hasattr(monitor_data, 'Ex'):
                # Standard field monitor
                Ex = monitor_data.Ex.isel(f=0).values
                Ey = monitor_data.Ey.isel(f=0).values
                Ez = monitor_data.Ez.isel(f=0).values if hasattr(monitor_data, 'Ez') else None
            elif hasattr(monitor_data, 'Etheta'):
                # Far-field projection monitor
                # Handle 4D data (ux, uy, r, f) by removing r and f dimensions
                if 'r' in monitor_data.Etheta.dims:
                    Etheta = monitor_data.Etheta.isel(f=0, r=0).values
                    Ephi = monitor_data.Ephi.isel(f=0, r=0).values
                    Er = monitor_data.Er.isel(f=0, r=0).values if hasattr(monitor_data, 'Er') else None
                else:
                    Etheta = monitor_data.Etheta.isel(f=0).values
                    Ephi = monitor_data.Ephi.isel(f=0).values
                    Er = monitor_data.Er.isel(f=0).values if hasattr(monitor_data, 'Er') else None
                # Convert to Cartesian-like components for analysis
                Ex = Etheta  # Approximate mapping
                Ey = Ephi
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
            
            # Calculate basic statistics
            results['intensity_stats'] = {
                'max': float(np.max(I)),
                'mean': float(np.mean(I)),
                'total': float(np.sum(I)),
                'shape': I.shape
            }
            
            # Calculate radiation pattern metrics
            radiation_metrics = self._calculate_radiation_metrics(I, coords, monitor_name)
            results['radiation_metrics'] = radiation_metrics
            
            # Store field data
            results['field_data'] = {
                'Ex': Ex,
                'Ey': Ey,
                'Ez': Ez,
                'intensity': I,
                'coordinates': coords
            }
            
            print(f"  - Intensity range: {np.min(I):.2e} to {np.max(I):.2e}")
            print(f"  - Total intensity: {np.sum(I):.2e}")
            print(f"  - Field shape: {I.shape}")
            
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
            if 'cartesian' in monitor_name:
                # Cartesian coordinates
                if hasattr(monitor_data, 'Ex'):
                    coords['x'] = monitor_data.Ex.coords.get('x', None)
                    coords['y'] = monitor_data.Ex.coords.get('y', None)
                    coords['z'] = monitor_data.Ex.coords.get('z', None)
                elif hasattr(monitor_data, 'Etheta'):
                    coords['x'] = monitor_data.Etheta.coords.get('x', None)
                    coords['y'] = monitor_data.Etheta.coords.get('y', None)
                    coords['z'] = monitor_data.Etheta.coords.get('z', None)
            elif 'kspace' in monitor_name:
                # K-space coordinates
                if hasattr(monitor_data, 'Ex'):
                    coords['kx'] = monitor_data.Ex.coords.get('kx', None)
                    coords['ky'] = monitor_data.Ex.coords.get('ky', None)
                    coords['kz'] = monitor_data.Ex.coords.get('kz', None)
                elif hasattr(monitor_data, 'Etheta'):
                    coords['ux'] = monitor_data.Etheta.coords.get('ux', None)
                    coords['uy'] = monitor_data.Etheta.coords.get('uy', None)
                    coords['r'] = monitor_data.Etheta.coords.get('r', None)
            elif 'angles' in monitor_name:
                # Angular coordinates
                if hasattr(monitor_data, 'Ex'):
                    coords['theta'] = monitor_data.Ex.coords.get('theta', None)
                    coords['phi'] = monitor_data.Ex.coords.get('phi', None)
                elif hasattr(monitor_data, 'Etheta'):
                    coords['theta'] = monitor_data.Etheta.coords.get('theta', None)
                    coords['phi'] = monitor_data.Etheta.coords.get('phi', None)
                    coords['r'] = monitor_data.Etheta.coords.get('r', None)
            elif 'bfp' in monitor_name:
                # Back focal plane coordinates
                if hasattr(monitor_data, 'Ex'):
                    coords['x'] = monitor_data.Ex.coords.get('x', None)
                    coords['y'] = monitor_data.Ex.coords.get('y', None)
                elif hasattr(monitor_data, 'Etheta'):
                    coords['x'] = monitor_data.Etheta.coords.get('x', None)
                    coords['y'] = monitor_data.Etheta.coords.get('y', None)
            
            # Convert to numpy arrays if they exist
            for key, value in coords.items():
                if value is not None:
                    coords[key] = value.values if hasattr(value, 'values') else value
                    
        except Exception as e:
            print(f"  - Warning: Could not extract coordinates: {e}")
        
        return coords
    
    def _calculate_radiation_metrics(self, I: np.ndarray, coords: Dict, monitor_name: str) -> Dict:
        """
        Calculate radiation pattern metrics
        """
        metrics = {}
        
        try:
            if 'angles' in monitor_name and 'theta' in coords and 'phi' in coords:
                # Angular radiation pattern
                theta = coords['theta']
                phi = coords['phi']
                
                # Calculate directivity
                total_power = np.sum(I)
                max_intensity = np.max(I)
                directivity = 4 * np.pi * max_intensity / total_power if total_power > 0 else 0
                
                # Calculate beam width (FWHM)
                beam_width = self._calculate_beam_width(I, theta, phi)
                
                metrics.update({
                    'directivity': float(directivity),
                    'beam_width_deg': beam_width,
                    'max_intensity': float(max_intensity),
                    'total_power': float(total_power)
                })
                
                print(f"  - Directivity: {directivity:.2f}")
                print(f"  - Beam width: {beam_width:.1f}°")
                
            elif 'kspace' in monitor_name and 'kx' in coords and 'ky' in coords:
                # K-space analysis
                kx = coords['kx']
                ky = coords['ky']
                
                # Calculate k-space extent
                k_max = np.sqrt(kx.max()**2 + ky.max()**2)
                k_min = np.sqrt(kx.min()**2 + ky.min()**2)
                
                # Calculate effective numerical aperture
                k0 = 2 * np.pi / (self.wavelength_um * 1e-6)  # Free space k
                NA_eff = k_max / k0
                
                metrics.update({
                    'k_max': float(k_max),
                    'k_min': float(k_min),
                    'NA_effective': float(NA_eff),
                    'k_space_extent': float(k_max - k_min)
                })
                
                print(f"  - K-space extent: {k_min:.2e} to {k_max:.2e}")
                print(f"  - Effective NA: {NA_eff:.3f}")
                
        except Exception as e:
            print(f"  - Warning: Could not calculate radiation metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_beam_width(self, I: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
        """
        Calculate beam width (FWHM) from angular radiation pattern
        """
        try:
            # Find maximum intensity
            max_idx = np.unravel_index(np.argmax(I), I.shape)
            
            # Extract profile through maximum
            if len(I.shape) == 2:
                # 2D pattern
                profile = I[max_idx[0], :] if max_idx[0] < I.shape[0] else I[:, max_idx[1]]
                angles = theta if max_idx[0] < I.shape[0] else phi
            else:
                # 1D pattern
                profile = I
                angles = theta if len(theta) > len(phi) else phi
            
            # Find FWHM
            max_val = np.max(profile)
            half_max = max_val / 2
            
            # Find indices where intensity is above half maximum
            above_half = profile >= half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                width_deg = angles[indices[-1]] - angles[indices[0]]
                return float(width_deg)
            
        except Exception as e:
            print(f"  - Warning: Could not calculate beam width: {e}")
        
        return 0.0
    
    def _calculate_collection_efficiency(self, results: Dict) -> Dict:
        """
        Calculate collection efficiency based on available data
        """
        print("\n📊 Collection efficiency analysis:")
        
        collection_results = {}
        
        try:
            # Look for angular data to calculate collection efficiency
            angular_data = None
            for monitor_name, monitor_results in results.items():
                if 'angles' in monitor_name and 'field_data' in monitor_results:
                    angular_data = monitor_results['field_data']
                    break
            
            if angular_data is None:
                print("  - No angular data available for collection efficiency")
                return collection_results
            
            # Extract intensity and angles
            I = angular_data['intensity']
            theta = angular_data['coordinates'].get('theta')
            phi = angular_data['coordinates'].get('phi')
            
            if theta is None or phi is None:
                print("  - Angular coordinates not available")
                return collection_results
            
            # Calculate collection efficiency with proper solid angle weighting
            # This uses the same method as the comprehensive analysis
            
            # Define collection cone (within NA)
            theta_max = np.arcsin(self.NA / self.n_bg)  # Maximum collection angle
            
            # Create coordinate grids and solid angle weighting
            if len(I.shape) == 3:
                # 3D pattern (freq, theta, phi) - remove frequency dimension
                I_2d = I[0, :, :]  # Remove frequency dimension
                theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
            elif len(I.shape) == 2:
                # 2D pattern - ensure theta and phi have correct shapes
                if theta.ndim == 1 and phi.ndim == 1:
                    theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
                else:
                    # Use existing 2D arrays
                    theta_2d, phi_2d = theta, phi
                I_2d = I
            else:
                # 1D pattern - ensure theta is 1D
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
            
            collection_results = {
                'total_power': float(total_power),
                'collected_power': float(collected_power),
                'collection_efficiency': float(collection_efficiency),
                'collection_efficiency_full_sphere': float(collection_efficiency_full_sphere),
                'total_power_full_sphere': float(total_power_full_sphere),
                'hemisphere_coverage': hemisphere_note,
                'NA': self.NA,
                'n_bg': self.n_bg,
                'theta_max_deg': float(np.degrees(theta_max))
            }
            
            print(f"  - Total power ({hemisphere_note}): {total_power:.2e}")
            print(f"  - Total power (full sphere): {total_power_full_sphere:.2e}")
            print(f"  - Collected power: {collected_power:.2e}")
            print(f"  - Collection efficiency ({hemisphere_note}): {collection_efficiency:.3f} ({collection_efficiency*100:.1f}%)")
            print(f"  - Collection efficiency (full sphere): {collection_efficiency_full_sphere:.3f} ({collection_efficiency_full_sphere*100:.1f}%)")
            print(f"  - Collection angle: ±{np.degrees(theta_max):.1f}°")
            
        except Exception as e:
            print(f"  - Error calculating collection efficiency: {e}")
            collection_results['error'] = str(e)
        
        return collection_results
    
    def _create_farfield_plots(self, results: Dict):
        """
        Create comprehensive far-field visualization plots
        """
        print("\n📊 Creating far-field analysis plots...")
        apply_theme()

        
        # Find angular monitor for collection efficiency vs NA plot
        angular_monitor = None
        for monitor_name, monitor_results in results.items():
            if 'angles' in monitor_name and 'field_data' in monitor_results:
                angular_monitor = (monitor_name, monitor_results)
                break
        
        # Determine number of monitors to plot (excluding cartesian)
        n_monitors = len([k for k in results.keys() if k != 'collection_efficiency' and 'cartesian' not in k])
        if n_monitors == 0:
            print("  - No monitors to plot")
            return
        
        # Create figure for far-field monitors only (collection efficiency plot is separate)
        n_cols = 2
        n_rows = (n_monitors + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot non-cartesian monitors
        for monitor_name, monitor_results in results.items():
            if monitor_name == 'collection_efficiency' or 'cartesian' in monitor_name or plot_idx >= n_monitors:
                continue
                
            if 'field_data' not in monitor_results:
                continue
            
            ax = axes[plot_idx]
            field_data = monitor_results['field_data']
            I = field_data['intensity']
            coords = field_data['coordinates']
            
            # Plot based on monitor type
            if 'kspace' in monitor_name and ('kx' in coords or 'ux' in coords) and ('ky' in coords or 'uy' in coords):
                # K-space plot with proper orientation
                kx = coords.get('kx', coords.get('ux'))
                ky = coords.get('ky', coords.get('uy'))
                
                # Apply orient_for_imshow logic (transpose if needed)
                I_oriented = self._orient_for_imshow(I, kx, ky)
                
                im = ax.imshow(I_oriented, extent=[kx.min(), kx.max(), ky.min(), ky.max()], 
                              origin='lower', cmap=mono_cmap, aspect='auto')
                ax.set_xlabel('kx (1/µm)')
                ax.set_ylabel('ky (1/µm)')
                ax.set_title('K-Space Far-Field Distribution')
                ax.grid(False)
                
            elif 'angles' in monitor_name and 'theta' in coords and 'phi' in coords:
                # Angular plot
                theta, phi = coords['theta'], coords['phi']
                
                # Handle coordinate shape mismatches
                if len(I.shape) == 2:
                    # 2D pattern - ensure coordinates match intensity shape
                    if theta.ndim == 1 and phi.ndim == 1:
                        # Create 2D coordinate grids
                        theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
                    else:
                        # Use existing coordinates, but ensure they match I shape
                        theta_2d, phi_2d = theta, phi
                    
                    # Ensure coordinate arrays match intensity shape
                    if theta_2d.shape != I.shape:
                        # If shapes don't match, use extent instead
                        theta_min, theta_max = theta_2d.min(), theta_2d.max()
                        phi_min, phi_max = phi_2d.min(), phi_2d.max()
                        im = ax.imshow(I, extent=[phi_min, phi_max, theta_min, theta_max], 
                                      origin='lower', cmap=mono_cmap, aspect='auto')
                    else:
                        im = ax.imshow(I, extent=[phi_2d.min(), phi_2d.max(), theta_2d.min(), theta_2d.max()], 
                                      origin='lower', cmap=mono_cmap, aspect='auto')
                    ax.set_xlabel('φ (deg)')
                    ax.set_ylabel('θ (deg)')
                    ax.grid(False)
                elif len(I.shape) == 3:
                    # 3D pattern - remove frequency dimension for plotting
                    I_2d = I[0, :, :]  # Remove frequency dimension
                    if theta.ndim == 1 and phi.ndim == 1:
                        # Create 2D coordinate grids
                        theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
                    else:
                        # Use existing coordinates
                        theta_2d, phi_2d = theta, phi
                    
                    # Use extent for plotting to avoid coordinate shape issues
                    theta_min, theta_max = theta_2d.min(), theta_2d.max()
                    phi_min, phi_max = phi_2d.min(), phi_2d.max()
                    im = ax.imshow(I_2d, extent=[phi_min, phi_max, theta_min, theta_max], 
                                  origin='lower', cmap=mono_cmap, aspect='auto')
                    ax.set_xlabel('φ (deg)')
                    ax.set_ylabel('θ (deg)')
                    ax.grid(False)
                else:
                    # 1D pattern
                    if theta.ndim > 1:
                        theta_1d = theta.flatten()
                    else:
                        theta_1d = theta
                    # Ensure I is 1D for plotting
                    I_1d = I.flatten() if I.ndim > 1 else I
                    ax.plot(theta_1d, I_1d, linewidth=2, color=PALETTE[0])
                    ax.set_xlabel('θ (deg)')
                    ax.set_ylabel('Intensity')
                    ax.grid(False)
                ax.set_title('Angular Radiation Pattern')
                
            else:
                # Generic plot
                im = ax.imshow(I, cmap=mono_cmap, aspect='auto')
                ax.set_title('Far-Field Intensity Distribution')
                ax.grid(False)
            
            # Add colorbar for image plots
            if 'im' in locals():
                plt.colorbar(im, ax=ax, label='Intensity')
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('farfield_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create separate collection efficiency vs NA plot
        if angular_monitor:
            self._create_collection_efficiency_plot(angular_monitor)
    
    def _create_collection_efficiency_plot(self, angular_monitor):
        """
        Create a separate plot for collection efficiency vs numerical aperture
        """
        print("\n📊 Creating collection efficiency vs NA plot...")
        
        monitor_name, monitor_results = angular_monitor
        field_data = monitor_results['field_data']
        I = field_data['intensity']
        coords = field_data['coordinates']
        
        theta = coords['theta']
        phi = coords['phi']
        
        # Create coordinate grids and solid angle weighting
        if len(I.shape) == 3:
            I_2d = I[0, :, :]  # Remove frequency dimension
        else:
            I_2d = I
            
        theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
        sin_theta = np.sin(theta_2d)
        
        # Calculate total power with solid angle weighting (full sphere)
        # For full sphere, we need to account for both hemispheres
        total_power = np.sum(I_2d * sin_theta)
        
        # Check if we only have upper hemisphere data (theta <= 90°)
        theta_max_available = np.degrees(theta.max())
        if theta_max_available <= 90.0:
            # Double the power to account for full sphere (symmetric structure)
            total_power *= 2.0
        
        # Calculate collection efficiency for different NAs
        na_values = np.linspace(0.1, 1.0, 20)
        collection_efficiencies = []
        
        for na in na_values:
            theta_max = np.arcsin(na)
            collection_mask = theta_2d <= theta_max
            collected_power = np.sum((I_2d * sin_theta)[collection_mask])
            collection_efficiency = collected_power / total_power if total_power > 0 else 0
            collection_efficiencies.append(collection_efficiency)
        
        apply_theme()
        fig, ax = plt.subplots()
        eff_percent = np.array(collection_efficiencies) * 100
        ax.plot(na_values, eff_percent, marker='o', linewidth=2.5, markersize=6, color=PALETTE[0])
        current_na = self.NA
        current_eff = np.interp(current_na, na_values, collection_efficiencies) * 100
        ax.axvline(current_na, linestyle='--', color=PALETTE[1], alpha=1, linewidth=1, zorder=-1)
        ax.axhline(current_eff, linestyle='--', color=PALETTE[2], alpha=1, linewidth=1, zorder=-1)
        ax.set_xlabel('Numerical Aperture (NA)')
        ax.set_ylabel('Collection Efficiency (%)')
        ax.set_title('Collection Efficiency vs Numerical Aperture')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig('collection_efficiency_vs_na.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Collection efficiency vs NA plot saved to 'collection_efficiency_vs_na.png'")
    
    def _plot_collection_efficiency_vs_na(self, angular_monitor, ax):
        """
        Plot collection efficiency vs numerical aperture (legacy method for combined plots)
        """
        # This method is kept for backward compatibility but not used in the new separate plot approach
        pass
    
    def _orient_for_imshow(self, A, x, y):
        """
        Return A oriented for imshow (rows=y, cols=x), using x,y lengths for disambiguation.
        Based on comprehensive_farfield_analysis.py logic.
        """
        A2 = np.squeeze(A)
        if A2.shape == (len(x), len(y)):
            return A2.T  # (x,y) -> (y,x)
        if A2.shape == (len(y), len(x)):
            return A2    # already (y,x)
        raise ValueError(f"Array shape {A2.shape} does not match (len(x),len(y)) or (len(y),len(x)).")
    
    def _save_results(self, results: Dict):
        """
        Save analysis results to JSON file
        """
        import json
        
        # Prepare results for JSON serialization
        json_results = {}
        for monitor_name, monitor_results in results.items():
            if monitor_name == 'collection_efficiency':
                json_results[monitor_name] = monitor_results
            else:
                # Skip large arrays for JSON
                json_results[monitor_name] = {
                    'intensity_stats': monitor_results.get('intensity_stats', {}),
                    'radiation_metrics': monitor_results.get('radiation_metrics', {})
                }
        
        filename = 'farfield_analysis_results.json'
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Far-field analysis results saved to {filename}")


def analyze_farfield(data_path: str,
                    monitor_names: Optional[list] = None,
                    wavelength_um: float = 0.62,
                    NA: float = 0.9,
                    n_bg: float = 1.0,
                    save_results: bool = True,
                    create_plots: bool = True) -> Dict:
    """
    Convenience function to analyze far-field from simulation data
    
    Args:
        data_path: Path to simulation data file
        monitor_names: List of far-field monitor names to analyze
        wavelength_um: Analysis wavelength in micrometers
        NA: Numerical aperture for collection efficiency
        n_bg: Background refractive index
        save_results: Whether to save results to file
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    data = td.SimulationData.from_file(data_path)
    
    # Create analyzer and run analysis
    analyzer = FarFieldAnalyzer(wavelength_um=wavelength_um, NA=NA, n_bg=n_bg)
    results = analyzer.analyze_farfield(
        data=data,
        monitor_names=monitor_names,
        save_results=save_results,
        create_plots=create_plots
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = analyze_farfield(
        data_path="results_0.14um.hdf5",
        monitor_names=['farfield_cartesian', 'farfield_kspace', 'farfield_angles'],
        wavelength_um=0.62,
        NA=0.9,
        n_bg=1.0,
        save_results=True,
        create_plots=True
    )
    
    print("\nFar-field analysis completed!")
    if 'collection_efficiency' in results:
        eff = results['collection_efficiency']['collection_efficiency']
        print(f"Collection efficiency: {eff:.3f} ({eff*100:.1f}%)")

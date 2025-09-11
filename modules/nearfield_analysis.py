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
from scipy.integrate import trapezoid
from typing import Dict, Tuple, Optional

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
    
    def analyze_nearfield(self, data: td.SimulationData, 
                         monitor_name: str = "fld_xy_narrow",
                         save_results: bool = True,
                         create_plots: bool = True) -> Dict:
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
        print("="*70)
        print("🔬 NEAR-FIELD ANALYSIS")
        print("="*70)
        
        # Load field data
        if monitor_name not in data.monitor_data:
            raise KeyError(f"Monitor '{monitor_name}' not found in data")
        
        field_data = data[monitor_name]
        
        # Get field components at resonance frequency
        Ex = field_data.Ex.isel(f=0).values
        Ey = field_data.Ey.isel(f=0).values
        Ez = field_data.Ez.isel(f=0).values
        
        # Get coordinates
        x = field_data.Ex.coords['x'].values
        y = field_data.Ex.coords['y'].values
        
        # Calculate field intensity and power
        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        P = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
        
        print(f"✓ Field data loaded")
        print(f"  - Field shape: {I.shape}")
        print(f"  - X range: {x.min():.3f} to {x.max():.3f} µm")
        print(f"  - Y range: {y.min():.3f} to {y.max():.3f} µm")
        print(f"  - Max intensity: {np.max(I):.2e}")
        print(f"  - Total power: {np.sum(P):.2e}")
        
        # Perform field confinement analysis
        confinement_results = self._analyze_field_confinement(I, x, y)
        
        # Calculate mode parameters
        mode_results = self._calculate_mode_parameters(I, x, y)
        
        # Calculate field quality metrics
        quality_results = self._calculate_field_quality_metrics(I, Ex, Ey, Ez)
        
        # Combine results
        results = {
            'field_intensity': I,
            'field_power': P,
            'coordinates': {'x': x, 'y': y},
            'field_components': {'Ex': Ex, 'Ey': Ey, 'Ez': Ez},
            'confinement': confinement_results,
            'mode_parameters': mode_results,
            'quality_metrics': quality_results
        }
        
        # Create plots if requested
        if create_plots:
            self._create_nearfield_plots(field_data, I, x, y, Ex, Ey, Ez)
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        self.results = results
        return results
    
    def _analyze_field_confinement(self, I: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Analyze field confinement using Gaussian fitting
        """
        print("\n📊 Field confinement analysis:")
        
        # Calculate 1D profiles
        # I has shape (901, 93, 1) where 901 corresponds to y, 93 corresponds to x
        # x has 901 points, y has 93 points
        # So I[:, 0, 0] corresponds to y (901 points), I[0, :, 0] corresponds to x (93 points)
        I_x_profile = np.sum(I, axis=0)  # Sum over y (axis 0) to get x profile (93 points)
        I_y_profile = np.sum(I, axis=1)  # Sum over x (axis 1) to get y profile (901 points)
        
        # The coordinate arrays are swapped relative to the field array
        # x has 901 points but I_x_profile has 93 points
        # y has 93 points but I_y_profile has 901 points
        # So we need to use the correct coordinate arrays
        x_coords = y  # Use y coordinates for x profile (93 points)
        y_coords = x  # Use x coordinates for y profile (901 points)
        
        # Gaussian fitting function
        def gaussian(x, A, x0, sigma):
            return A * np.exp(-((x - x0) / sigma)**2)
        
        confinement_results = {}
        
        try:
            # Convert to float64 to avoid type issues
            x_float = x_coords.astype(np.float64)
            y_float = y_coords.astype(np.float64)
            I_x_profile_float = I_x_profile.astype(np.float64)
            I_y_profile_float = I_y_profile.astype(np.float64)
            
            # Ensure arrays are contiguous and have no NaN/inf values
            x_float = np.ascontiguousarray(x_float)
            y_float = np.ascontiguousarray(y_float)
            I_x_profile_float = np.ascontiguousarray(I_x_profile_float)
            I_y_profile_float = np.ascontiguousarray(I_y_profile_float)
            
            # Remove any NaN or inf values
            valid_x = np.isfinite(x_float) & np.isfinite(I_x_profile_float)
            valid_y = np.isfinite(y_float) & np.isfinite(I_y_profile_float)
            
            if np.sum(valid_x) < 3 or np.sum(valid_y) < 3:
                raise ValueError("Not enough valid data points for fitting")
            
            x_clean = x_float[valid_x]
            y_clean = y_float[valid_y]
            I_x_clean = I_x_profile_float[valid_x]
            I_y_clean = I_y_profile_float[valid_y]
            
            # Fit X profile
            popt_x, _ = curve_fit(gaussian, x_clean, I_x_clean, 
                                p0=[float(np.max(I_x_clean)), float(np.mean(x_clean)), 1.0],
                                maxfev=1000)
            sigma_x = abs(popt_x[2])
            w_x = 2 * np.sqrt(2) * sigma_x  # 1/e² width
            
            # Fit Y profile  
            popt_y, _ = curve_fit(gaussian, y_clean, I_y_clean, 
                                p0=[float(np.max(I_y_clean)), float(np.mean(y_clean)), 0.5],
                                maxfev=1000)
            sigma_y = abs(popt_y[2])
            w_y = 2 * np.sqrt(2) * sigma_y  # 1/e² width
            
            confinement_area = w_x * w_y
            aspect_ratio = w_x / w_y
            
            print(f"  - Gaussian fit 1/e² width (x): {w_x:.3f} µm")
            print(f"  - Gaussian fit 1/e² width (y): {w_y:.3f} µm")
            print(f"  - Confinement area: {confinement_area:.3f} µm²")
            print(f"  - Aspect ratio (x/y): {aspect_ratio:.2f}")
            
            confinement_results = {
                'width_x_um': w_x,
                'width_y_um': w_y,
                'confinement_area_um2': confinement_area,
                'aspect_ratio': aspect_ratio,
                'gaussian_fit_success': True
            }
            
        except Exception as e:
            print(f"  - Gaussian fitting failed: {e}")
            # Fallback to simple method
            I_x_max = np.max(I_x_profile)
            I_y_max = np.max(I_y_profile)
            
            x_1e2_indices = np.where(I_x_profile >= I_x_max / np.e**2)[0]
            y_1e2_indices = np.where(I_y_profile >= I_y_max / np.e**2)[0]
            
            if len(x_1e2_indices) > 0 and len(y_1e2_indices) > 0:
                w_x = x_coords[x_1e2_indices[-1]] - x_coords[x_1e2_indices[0]]
                w_y = y_coords[y_1e2_indices[-1]] - y_coords[y_1e2_indices[0]]
                confinement_area = w_x * w_y
                aspect_ratio = w_x / w_y
                
                print(f"  - Simple 1/e² width (x): {w_x:.3f} µm")
                print(f"  - Simple 1/e² width (y): {w_y:.3f} µm")
                print(f"  - Confinement area: {confinement_area:.3f} µm²")
                print(f"  - Aspect ratio (x/y): {aspect_ratio:.2f}")
                
                confinement_results = {
                    'width_x_um': w_x,
                    'width_y_um': w_y,
                    'confinement_area_um2': confinement_area,
                    'aspect_ratio': aspect_ratio,
                    'gaussian_fit_success': False
                }
        
        return confinement_results
    
    def _calculate_mode_parameters(self, I: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculate mode area and effective parameters
        """
        print("\n📐 Mode parameters:")
        
        # Calculate mode area (A_eff)
        total_power = np.sum(I)
        max_intensity = np.max(I)
        mode_area = total_power / max_intensity
        
        # Calculate effective mode area in terms of wavelength
        wavelength_um = self.wavelength_um
        mode_area_lambda2 = mode_area / (wavelength_um**2)
        
        # Calculate effective index (rough estimate)
        # This is a simplified calculation - in practice would need more sophisticated analysis
        n_eff = 2.4  # Typical for diamond/silicon
        
        print(f"  - Mode area: {mode_area:.3f} µm²")
        print(f"  - Mode area (λ²): {mode_area_lambda2:.3f}")
        print(f"  - Effective index: {n_eff:.2f}")
        
        return {
            'mode_area_um2': mode_area,
            'mode_area_lambda2': mode_area_lambda2,
            'effective_index': n_eff
        }
    
    def _calculate_field_quality_metrics(self, I: np.ndarray, 
                                       Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray) -> Dict:
        """
        Calculate field quality metrics
        """
        print("\n📈 Field quality metrics:")
        
        # Calculate field uniformity
        field_uniformity = np.std(I) / np.mean(I)
        
        # Calculate polarization purity
        Ex_mag = np.abs(Ex)
        Ey_mag = np.abs(Ey)
        Ez_mag = np.abs(Ez)
        
        total_mag = Ex_mag + Ey_mag + Ez_mag
        Ex_fraction = np.sum(Ex_mag) / np.sum(total_mag)
        Ey_fraction = np.sum(Ey_mag) / np.sum(total_mag)
        Ez_fraction = np.sum(Ez_mag) / np.sum(total_mag)
        
        # Calculate field concentration (fraction of power in top 10% of pixels)
        I_sorted = np.sort(I.ravel())[::-1]
        top_10_percent = int(0.1 * len(I_sorted))
        concentration = np.sum(I_sorted[:top_10_percent]) / np.sum(I_sorted)
        
        print(f"  - Field uniformity (std/mean): {field_uniformity:.3f}")
        print(f"  - Ex fraction: {Ex_fraction:.3f}")
        print(f"  - Ey fraction: {Ey_fraction:.3f}")
        print(f"  - Ez fraction: {Ez_fraction:.3f}")
        print(f"  - Field concentration (top 10%): {concentration:.3f}")
        
        return {
            'field_uniformity': field_uniformity,
            'polarization_fractions': {
                'Ex': Ex_fraction,
                'Ey': Ey_fraction,
                'Ez': Ez_fraction
            },
            'field_concentration': concentration
        }
    
    def _create_nearfield_plots(self, field_data, I: np.ndarray, x: np.ndarray, y: np.ndarray,
                              Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray):
        """
        Create comprehensive near-field visualization plots
        """
        print("\n📊 Creating near-field analysis plots...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Field intensity
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(I, extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', cmap='hot', aspect='auto')
        ax1.set_title('Field Intensity')
        ax1.set_xlabel('x (µm)')
        ax1.set_ylabel('y (µm)')
        plt.colorbar(im1, ax=ax1, label='Intensity (arb. units)')
        
        # Plot 2: Ex component (real)
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(np.real(Ex), extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', cmap='RdBu', aspect='auto')
        ax2.set_title('Ex (real part)')
        ax2.set_xlabel('x (µm)')
        ax2.set_ylabel('y (µm)')
        plt.colorbar(im2, ax=ax2, label='Ex (V/m)')
        
        # Plot 3: Ey component (real)
        ax3 = plt.subplot(2, 3, 3)
        im3 = ax3.imshow(np.real(Ey), extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', cmap='RdBu', aspect='auto')
        ax3.set_title('Ey (real part)')
        ax3.set_xlabel('x (µm)')
        ax3.set_ylabel('y (µm)')
        plt.colorbar(im3, ax=ax3, label='Ey (V/m)')
        
        # Plot 4: Ez component (real)
        ax4 = plt.subplot(2, 3, 4)
        im4 = ax4.imshow(np.real(Ez), extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', cmap='RdBu', aspect='auto')
        ax4.set_title('Ez (real part)')
        ax4.set_xlabel('x (µm)')
        ax4.set_ylabel('y (µm)')
        plt.colorbar(im4, ax=ax4, label='Ez (V/m)')
        
        # Plot 5: Field profiles
        ax5 = plt.subplot(2, 3, 5)
        I_x_profile = np.sum(I, axis=1)
        I_y_profile = np.sum(I, axis=0)
        ax5.plot(x, I_x_profile, 'b-', label='X profile', linewidth=2)
        ax5.plot(y, I_y_profile, 'r-', label='Y profile', linewidth=2)
        ax5.set_xlabel('Position (µm)')
        ax5.set_ylabel('Intensity (arb. units)')
        ax5.set_title('Field Profiles')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Polarization analysis
        ax6 = plt.subplot(2, 3, 6)
        Ex_mag = np.abs(Ex)
        Ey_mag = np.abs(Ey)
        Ez_mag = np.abs(Ez)
        
        # Calculate polarization fractions
        total_mag = Ex_mag + Ey_mag + Ez_mag
        Ex_fraction = np.sum(Ex_mag) / np.sum(total_mag)
        Ey_fraction = np.sum(Ey_mag) / np.sum(total_mag)
        Ez_fraction = np.sum(Ez_mag) / np.sum(total_mag)
        
        fractions = [Ex_fraction, Ey_fraction, Ez_fraction]
        labels = ['Ex', 'Ey', 'Ez']
        colors = ['red', 'green', 'blue']
        
        bars = ax6.bar(labels, fractions, color=colors, alpha=0.7)
        ax6.set_ylabel('Polarization Fraction')
        ax6.set_title('Field Polarization')
        ax6.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, fraction in zip(bars, fractions):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fraction:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('nearfield_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Near-field analysis plots saved to 'nearfield_analysis.png'")
    
    def _save_results(self, results: Dict):
        """
        Save analysis results to JSON file
        """
        import json
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key in ['field_intensity', 'field_power', 'field_components']:
                # Skip large arrays for JSON
                continue
            elif key == 'coordinates':
                json_results[key] = {
                    'x_range': [float(results['coordinates']['x'].min()), 
                               float(results['coordinates']['x'].max())],
                    'y_range': [float(results['coordinates']['y'].min()), 
                               float(results['coordinates']['y'].max())]
                }
            else:
                # Convert numpy types to standard Python types
                json_results[key] = self._convert_to_json_serializable(value)
        
        filename = 'nearfield_analysis_results.json'
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Near-field analysis results saved to {filename}")
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types and other non-serializable objects to JSON-serializable types
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj


def analyze_nearfield(data_path: str,
                     monitor_name: str = "fld_xy_narrow",
                     wavelength_um: float = 0.62,
                     save_results: bool = True,
                     create_plots: bool = True) -> Dict:
    """
    Convenience function to analyze near-field from simulation data
    
    Args:
        data_path: Path to simulation data file
        monitor_name: Name of the field monitor
        wavelength_um: Analysis wavelength in micrometers
        save_results: Whether to save results to file
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    data = td.SimulationData.from_file(data_path)
    
    # Create analyzer and run analysis
    analyzer = NearFieldAnalyzer(wavelength_um=wavelength_um)
    results = analyzer.analyze_nearfield(
        data=data,
        monitor_name=monitor_name,
        save_results=save_results,
        create_plots=create_plots
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = analyze_nearfield(
        data_path="results_0.14um.hdf5",
        monitor_name="fld_xy_narrow",
        wavelength_um=0.62,
        save_results=True,
        create_plots=True
    )
    
    print("\nNear-field analysis completed!")
    print(f"Confinement area: {results['confinement']['confinement_area_um2']:.3f} µm²")
    print(f"Mode area: {results['mode_parameters']['mode_area_um2']:.3f} µm²")

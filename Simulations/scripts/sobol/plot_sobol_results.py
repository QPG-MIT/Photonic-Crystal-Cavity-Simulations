#!/usr/bin/env python3
"""
Plot Sobol Analysis Results

Visualization script for Sobol analysis:
- Reads HDF5 simulation results from data/results/sobol_32/
- Creates 3D visualizations of parameter space with color-coded Q-factors and wavelengths
- Generates publication-ready plots showing parameter sensitivity

Can be run at any time after run_sobol_simulations.py completes.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import glob

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.q_factor_analysis import ResonanceConfig, analyze_q_factor

def load_sobol_data():
    """Load all Sobol simulation results and extract parameters, Q-factors, and wavelengths."""
    
    # Set up the improved configuration
    ResonanceConfig.guard_cycles = 12
    ResonanceConfig.fit_window_lo = 0.60
    ResonanceConfig.fit_window_hi = 0.95
    ResonanceConfig.max_modes_for_fit = 2
    ResonanceConfig.do_plot = False  # Disable plotting for batch analysis
    
    # Find all Sobol result files
    sobol_dir = REPO_ROOT / "data" / "results" / "sobol_32"
    if not sobol_dir.exists():
        print(f"❌ Sobol results directory not found: {sobol_dir}")
        return None
    
    result_files = list(sobol_dir.glob("results_sobol_*.hdf5"))
    print(f"📁 Found {len(result_files)} Sobol result files")
    
    if not result_files:
        print("❌ No Sobol result files found")
        return None
    
    # Load and analyze each file
    data = []
    
    for i, result_file in enumerate(sorted(result_files), 1):
        print(f"[{i}/{len(result_files)}] Analyzing {result_file.name}...", end=" ", flush=True)
        
        try:
            # Extract parameters from filename
            filename = result_file.stem
            import re
            w_match = re.search(r'_w([0-9.]+)_', filename)
            r_match = re.search(r'_r([0-9.]+)_', filename)
            t_match = re.search(r'_t([0-9.]+)', filename)
            
            w_val = float(w_match.group(1)) if w_match else None
            r_val = float(r_match.group(1)) if r_match else None
            t_val = float(t_match.group(1)) if t_match else None
            
            # Run Q-factor analysis
            q_results = analyze_q_factor(
                data_path=str(result_file),
                wavelength_um=0.62,
                save_results=False
            )
            
            # Extract key metrics
            selected_index = q_results.get('selected_index', 0)
            q_factors = q_results.get('q_factors', [])
            frequencies_hz = q_results.get('frequencies_hz', [])
            
            # Calculate the actual Q_top and wavelength
            if selected_index < len(q_factors) and selected_index < len(frequencies_hz):
                q_top = q_factors[selected_index]
                freq_hz = frequencies_hz[selected_index]
                # Convert frequency (Hz) to wavelength (nm)
                # c = f * λ, so λ = c / f
                # c = 299792458 m/s = 299792458 * 1e9 nm/s
                # wavelength_nm = c_nm_per_s / freq_hz
                # But we need to be careful: if freq_hz is ~4.8e14 Hz (for 625 nm)
                # Then: wavelength_nm = (299792458 * 1e9) / 4.8e14 = 624.6 nm ✓
                # The issue is freq_hz might be in a different unit or we need to check
                # Let's use: wavelength (nm) = 299792458 / (freq_hz / 1e9) 
                # Or better: wavelength (nm) = 299792458 * 1e9 / freq_hz
                # Actually, let's use the standard formula: λ = c / f
                # c = 2.99792458e8 m/s = 2.99792458e17 nm/s
                # So: λ (nm) = 2.99792458e17 / f (Hz)
                wavelength_nm = 2.99792458e17 / freq_hz if freq_hz > 0 else 0
            else:
                q_top = 0
                wavelength_nm = 0
            
            # Store data
            data.append({
                'filename': result_file.name,
                'w': w_val,
                'r': r_val,
                't': t_val,
                'q_factor': q_top,
                'wavelength_nm': wavelength_nm,
                'frequency_hz': freq_hz if selected_index < len(frequencies_hz) else 0
            })
            
            print(f"✓ Q: {q_top:.0f}, λ: {wavelength_nm:.1f}nm")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return data

def create_3d_plots(data):
    """Create 3D scatter plots with color-coded Q-factors and wavelengths."""
    
    if not data:
        print("❌ No data to plot")
        return
    
    # Extract data arrays
    w_vals = np.array([d['w'] for d in data])
    r_vals = np.array([d['r'] for d in data])
    t_vals = np.array([d['t'] for d in data])
    q_factors = np.array([d['q_factor'] for d in data])
    wavelengths = np.array([d['wavelength_nm'] for d in data])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Plot 1: Q-factors color-coded
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(w_vals, r_vals, t_vals, c=q_factors, cmap='viridis', s=100, alpha=0.8)
    ax1.set_xlabel('Width (μm)')
    ax1.set_ylabel('Radius (μm)')
    ax1.set_zlabel('Thickness (μm)')
    ax1.set_title('Sobol Parameter Space\n(Color: Q-factor)')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, label='Q-factor')
    
    # Plot 2: Wavelengths color-coded
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(w_vals, r_vals, t_vals, c=wavelengths, cmap='plasma', s=100, alpha=0.8)
    ax2.set_xlabel('Width (μm)')
    ax2.set_ylabel('Radius (μm)')
    ax2.set_zlabel('Thickness (μm)')
    ax2.set_title('Sobol Parameter Space\n(Color: Wavelength)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, label='Wavelength (nm)')
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    Q-factors: {q_factors.min():.0f} - {q_factors.max():.0f}
    Wavelengths: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm
    Mean Q: {q_factors.mean():.0f}
    Mean λ: {wavelengths.mean():.1f} nm
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = REPO_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sobol_3d_parameter_space.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 3D plots saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def create_parameter_sensitivity_plots(data):
    """Create 2D projections showing parameter sensitivity."""
    
    if not data:
        return
    
    # Extract data arrays
    w_vals = np.array([d['w'] for d in data])
    r_vals = np.array([d['r'] for d in data])
    t_vals = np.array([d['t'] for d in data])
    q_factors = np.array([d['q_factor'] for d in data])
    wavelengths = np.array([d['wavelength_nm'] for d in data])
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Q-factor vs Width
    scatter1 = axes[0, 0].scatter(w_vals, q_factors, c=r_vals, cmap='viridis', s=100, alpha=0.8)
    axes[0, 0].set_xlabel('Width (μm)')
    axes[0, 0].set_ylabel('Q-factor')
    axes[0, 0].set_title('Q-factor vs Width\n(Color: Radius)')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Radius (μm)')
    
    # Q-factor vs Radius
    scatter2 = axes[0, 1].scatter(r_vals, q_factors, c=t_vals, cmap='plasma', s=100, alpha=0.8)
    axes[0, 1].set_xlabel('Radius (μm)')
    axes[0, 1].set_ylabel('Q-factor')
    axes[0, 1].set_title('Q-factor vs Radius\n(Color: Thickness)')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Thickness (μm)')
    
    # Wavelength vs Width
    scatter3 = axes[1, 0].scatter(w_vals, wavelengths, c=r_vals, cmap='viridis', s=100, alpha=0.8)
    axes[1, 0].set_xlabel('Width (μm)')
    axes[1, 0].set_ylabel('Wavelength (nm)')
    axes[1, 0].set_title('Wavelength vs Width\n(Color: Radius)')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Radius (μm)')
    
    # Wavelength vs Radius
    scatter4 = axes[1, 1].scatter(r_vals, wavelengths, c=t_vals, cmap='plasma', s=100, alpha=0.8)
    axes[1, 1].set_xlabel('Radius (μm)')
    axes[1, 1].set_ylabel('Wavelength (nm)')
    axes[1, 1].set_title('Wavelength vs Radius\n(Color: Thickness)')
    plt.colorbar(scatter4, ax=axes[1, 1], label='Thickness (μm)')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = REPO_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sobol_parameter_sensitivity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Parameter sensitivity plots saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def save_data_summary(data):
    """Save the analyzed data to a JSON file for further analysis."""
    
    output_dir = REPO_ROOT / "data" / "results_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sobol_analysis_summary.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = []
    for item in data:
        serializable_item = {
            'filename': item['filename'],
            'w': float(item['w']) if item['w'] is not None else None,
            'r': float(item['r']) if item['r'] is not None else None,
            't': float(item['t']) if item['t'] is not None else None,
            'q_factor': float(item['q_factor']),
            'wavelength_nm': float(item['wavelength_nm']),
            'frequency_hz': float(item['frequency_hz'])
        }
        serializable_data.append(serializable_item)
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_points': len(data),
                'parameter_ranges': {
                    'width': [min([d['w'] for d in data if d['w'] is not None]), 
                             max([d['w'] for d in data if d['w'] is not None])],
                    'radius': [min([d['r'] for d in data if d['r'] is not None]), 
                              max([d['r'] for d in data if d['r'] is not None])],
                    'thickness': [min([d['t'] for d in data if d['t'] is not None]), 
                                 max([d['t'] for d in data if d['t'] is not None])]
                },
                'q_factor_range': [min([d['q_factor'] for d in data]), 
                                 max([d['q_factor'] for d in data])],
                'wavelength_range': [min([d['wavelength_nm'] for d in data]), 
                                   max([d['wavelength_nm'] for d in data])]
            },
            'data': serializable_data
        }, f, indent=2)
    
    print(f"💾 Data summary saved to: {output_file}")

def main():
    """Main function to run the 3D analysis and visualization."""
    
    print("🚀 Starting Sobol 3D Parameter Space Analysis")
    
    # Load data
    data = load_sobol_data()
    if not data:
        return
    
    print(f"\n📊 Loaded {len(data)} data points")
    
    # Create output directory (already handled in individual functions)
    
    # Create 3D plots
    print("\n🎨 Creating 3D parameter space plots...")
    create_3d_plots(data)
    
    # Create parameter sensitivity plots
    print("\n📈 Creating parameter sensitivity plots...")
    create_parameter_sensitivity_plots(data)
    
    # Save data summary
    print("\n💾 Saving data summary...")
    save_data_summary(data)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()



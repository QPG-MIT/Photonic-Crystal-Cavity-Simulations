#!/usr/bin/env python3
"""
Plotting script for statistical width estimation results.
Shows distribution plots with Gaussian fits for hole radii and waveguide widths.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.optimize import curve_fit

# ============================================================================
# CONFIGURATION - Change these values as needed
# ============================================================================
NUM_BINS = 10  # Number of bins for histograms
# ============================================================================


def load_detailed_results():
    """Load detailed results from JSON file"""
    json_file = "results_detailed.json"
    if not Path(json_file).exists():
        print(f"❌ File not found: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    return results


def extract_hole_data(results):
    """Extract hole radius data from detailed results"""
    hole_radii = []
    for result in results:
        if result['holes'] and len(result['holes']) > 0:
            for hole in result['holes']:
                hole_radii.append(hole['radius_nm'])
    return np.array(hole_radii)


def extract_waveguide_data(results):
    """Extract waveguide width data from detailed results"""
    waveguide_widths = []
    for result in results:
        if result['rail_separation_nm'] is not None:
            waveguide_widths.append(result['rail_separation_nm'])
    return np.array(waveguide_widths)


def gaussian_function(x, amplitude, mean, std):
    """Gaussian function for curve fitting"""
    return amplitude * np.exp(-((x - mean) / std) ** 2 / 2)


def fit_gaussian_to_data(data, bins=NUM_BINS):
    """Fit a Gaussian curve to the data and return parameters"""
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    amplitude_guess = np.max(hist)
    mean_guess = np.mean(data)
    std_guess = np.std(data)
    
    try:
        popt, pcov = curve_fit(gaussian_function, bin_centers, hist, 
                              p0=[amplitude_guess, mean_guess, std_guess],
                              maxfev=10000)
        
        amplitude, mean, std = popt
        
        y_pred = gaussian_function(bin_centers, *popt)
        ss_res = np.sum((hist - y_pred) ** 2)
        ss_tot = np.sum((hist - np.mean(hist)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        param_errors = np.sqrt(np.diag(pcov))
        
        return {
            'amplitude': amplitude,
            'mean': mean,
            'std': std,
            'r_squared': r_squared,
            'amplitude_error': param_errors[0],
            'mean_error': param_errors[1],
            'std_error': param_errors[2],
            'bin_centers': bin_centers,
            'histogram': hist,
            'fitted_curve': y_pred
        }
    except Exception as e:
        print(f"Warning: Gaussian fitting failed: {e}")
        return None


def plot_distributions():
    """Create distribution plots for hole radii and waveguide widths"""
    print("📊 Loading analysis results...")
    
    results = load_detailed_results()
    if results is None:
        return
    
    hole_radii = extract_hole_data(results)
    waveguide_widths = extract_waveguide_data(results)
    
    print(f"  Total holes analyzed: {len(hole_radii)}")
    print(f"  Images with waveguide data: {len(waveguide_widths)}")
    
    # Fit Gaussians
    print("🔬 Fitting Gaussian distributions...")
    hole_gaussian = fit_gaussian_to_data(hole_radii)
    width_gaussian = fit_gaussian_to_data(waveguide_widths)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Statistical Analysis of Waveguide Holes and Widths with Gaussian Fits', 
                 fontsize=16, fontweight='bold')
    
    # 1. Hole radius distribution (histogram)
    axes[0, 0].hist(hole_radii, bins=NUM_BINS, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].set_title('Distribution of Hole Radii with Gaussian Fit')
    axes[0, 0].set_xlabel('Hole Radius (nm)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    if hole_gaussian:
        x_range = np.linspace(np.min(hole_radii), np.max(hole_radii), 100)
        gaussian_curve = gaussian_function(x_range, hole_gaussian['amplitude'], 
                                          hole_gaussian['mean'], hole_gaussian['std'])
        axes[0, 0].plot(x_range, gaussian_curve, 'r-', linewidth=2, 
                       label=f'Gaussian Fit (σ={hole_gaussian["std"]:.1f} nm)')
    
    mean_radius = np.mean(hole_radii)
    median_radius = np.median(hole_radii)
    axes[0, 0].axvline(mean_radius, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_radius:.1f} nm')
    axes[0, 0].axvline(median_radius, color='green', linestyle='--', linewidth=2, label=f'Median: {median_radius:.1f} nm')
    axes[0, 0].legend()
    
    # 2. Hole radius box plot
    axes[0, 1].boxplot(hole_radii, vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0, 1].set_title('Hole Radius Box Plot')
    axes[0, 1].set_ylabel('Hole Radius (nm)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Waveguide width distribution (histogram)
    axes[1, 0].hist(waveguide_widths, bins=NUM_BINS, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    axes[1, 0].set_title('Distribution of Waveguide Widths with Gaussian Fit')
    axes[1, 0].set_xlabel('Waveguide Width (nm)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    if width_gaussian:
        x_range = np.linspace(np.min(waveguide_widths), np.max(waveguide_widths), 100)
        gaussian_curve = gaussian_function(x_range, width_gaussian['amplitude'], 
                                          width_gaussian['mean'], width_gaussian['std'])
        axes[1, 0].plot(x_range, gaussian_curve, 'r-', linewidth=2, 
                       label=f'Gaussian Fit (σ={width_gaussian["std"]:.1f} nm)')
    
    mean_width = np.mean(waveguide_widths)
    median_width = np.median(waveguide_widths)
    axes[1, 0].axvline(mean_width, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_width:.1f} nm')
    axes[1, 0].axvline(median_width, color='green', linestyle='--', linewidth=2, label=f'Median: {median_width:.1f} nm')
    axes[1, 0].legend()
    
    # 4. Waveguide width box plot
    axes[1, 1].boxplot(waveguide_widths, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[1, 1].set_title('Waveguide Width Box Plot')
    axes[1, 1].set_ylabel('Waveguide Width (nm)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and show plot
    output_path = "hole_and_waveguide_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved distribution plot: {output_path}")
    plt.show()
    
    # Save Gaussian parameters
    if hole_gaussian or width_gaussian:
        save_gaussian_parameters(hole_gaussian, width_gaussian)


def save_gaussian_parameters(hole_gaussian, width_gaussian):
    """Save Gaussian parameters to JSON file"""
    save_data = {}
    
    if hole_gaussian:
        save_data['hole_radii'] = {
            'mean': float(hole_gaussian['mean']),
            'std': float(hole_gaussian['std']),
            'amplitude': float(hole_gaussian['amplitude']),
            'mean_error': float(hole_gaussian['mean_error']),
            'std_error': float(hole_gaussian['std_error']),
            'r_squared': float(hole_gaussian['r_squared'])
        }
    
    if width_gaussian:
        save_data['waveguide_widths'] = {
            'mean': float(width_gaussian['mean']),
            'std': float(width_gaussian['std']),
            'amplitude': float(width_gaussian['amplitude']),
            'mean_error': float(width_gaussian['mean_error']),
            'std_error': float(width_gaussian['std_error']),
            'r_squared': float(width_gaussian['r_squared'])
        }
    
    if save_data:
        output_file = "gaussian_distribution_parameters.json"
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"💾 Saved Gaussian parameters to: {output_file}")


def main():
    """Main function - plots distributions by default"""
    plot_distributions()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Cavity Distribution Plotting
Simple plotting functions: distribution with gaussian, heatmaps, and three distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import analysis functions
from analyze import extract_center_wavelengths, load_and_process_grid_data, fit_gaussian_to_distribution


def setup_style():
    """Configure matplotlib style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_distribution_with_gaussian():
    """Plot normal distribution with Gaussian fit."""
    print("Extracting center wavelengths...")
    center_wavelengths = extract_center_wavelengths()
    
    if len(center_wavelengths) == 0:
        print("No data found!")
        return
    
    wavelengths_nm = center_wavelengths * 1e9
    
    print("Fitting Gaussian...")
    fit_results = fit_gaussian_to_distribution(wavelengths_nm, bins=20)
    
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(wavelengths_nm, bins=20, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black', linewidth=0.5, label='Data')
    
    # Plot Gaussian fit
    if fit_results['success']:
        x_fit = np.linspace(wavelengths_nm.min(), wavelengths_nm.max(), 1000)
        y_fit = fit_results['amplitude'] * np.exp(
            -0.5 * ((x_fit - fit_results['mean']) / fit_results['stddev']) ** 2
        )
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Gaussian Fit')
        
        # Print fit parameters
        print(f"Gaussian Fit:")
        print(f"  Mean: {fit_results['mean']:.2f} ± {fit_results['mean_error']:.2f} nm")
        print(f"  Std Dev: {fit_results['stddev']:.2f} ± {fit_results['stddev_error']:.2f} nm")
        print(f"  FWHM: {fit_results['fwhm']:.2f} nm")
        print(f"  R²: {fit_results['r_squared']:.4f}")
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Cavity Distribution with Gaussian Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = Path('outputs/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / 'cavity_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path / 'cavity_distribution.png'}")
    
    plt.show()
    return fig


def plot_heatmaps():
    """Plot heatmaps for wavelength and Q factor."""
    print("Loading and processing data...")
    data = load_and_process_grid_data()
    
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Center wavelength heatmap
    im1 = axes[0].imshow(data['avg_center_map'] * 1e9, cmap='viridis', 
                         origin='lower', aspect='auto')
    axes[0].set_title('Average Center Wavelength (nm)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Column', fontsize=10)
    axes[0].set_ylabel('Row', fontsize=10)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Wavelength (nm)', fontsize=9)
    
    # Q factor heatmap
    im2 = axes[1].imshow(data['avg_q_map'], cmap='plasma', 
                        origin='lower', aspect='auto')
    axes[1].set_title('Average Q Factor', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Column', fontsize=10)
    axes[1].set_ylabel('Row', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Q Factor', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('outputs/figures/heatmaps')
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / 'wavelength_q_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path / 'wavelength_q_heatmaps.png'}")
    
    plt.show()
    return fig


def plot_three_distributions():
    """Plot three distributions: wavelength, Q factor, and fill factor."""
    print("Loading and processing data...")
    data = load_and_process_grid_data()
    
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distribution Statistics', fontsize=14, fontweight='bold')
    
    # Center wavelength distribution
    if len(data['all_centers']) > 0:
        axes[0].hist(data['all_centers'] * 1e9, bins=20, 
                    color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Center Wavelength (nm)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Wavelength Distribution\n(Mean: {np.mean(data["all_centers"])*1e9:.1f} nm)')
        axes[0].grid(True, alpha=0.3)
    
    # Q factor distribution
    if len(data['all_Qs']) > 0:
        axes[1].hist(data['all_Qs'], bins=20, 
                    color='indigo', alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Q Factor')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Q Factor Distribution\n(Mean: {np.mean(data["all_Qs"]):.0f})')
        axes[1].grid(True, alpha=0.3)
    
    # Fill factor distribution
    if len(data['all_fills']) > 0:
        axes[2].hist(data['all_fills'], bins=20, 
                    color='midnightblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[2].set_xlabel('Fill Factor')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'Fill Factor Distribution\n(Mean: {np.mean(data["all_fills"]):.2f})')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('outputs/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / 'three_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path / 'three_distributions.png'}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['heatmaps', 'heatmap']:
            plot_heatmaps()
        elif arg in ['distributions', 'distribution', 'three']:
            plot_three_distributions()
        elif arg in ['gaussian', 'dist']:
            plot_distribution_with_gaussian()
        else:
            print("Usage: python plot.py [option]")
            print("  No argument or 'gaussian': distribution plot with Gaussian fit")
            print("  'heatmaps': wavelength and Q factor heatmaps")
            print("  'distributions': three distribution plots (wavelength, Q factor, fill factor)")
    else:
        plot_distribution_with_gaussian()

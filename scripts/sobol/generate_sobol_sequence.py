#!/usr/bin/env python3
"""
Generate Sobol Sequence for Parameter Sensitivity Analysis

Step 1 of the Sobol analysis workflow:
- Creates a 32-point Sobol sequence for the three parameters:
  * Width (bottom width): 0.314 µm ± 30%
  * Radius (hole radius): 0.043 µm ± 30% 
  * Thickness: 0.136 µm ± 30%
- Saves sequence to data/sobol_sequence_32.npz
- Generates 3D visualization plots

Next step: Run run_sobol_simulations.py to execute simulations for these points.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sobol_seq import i4_sobol_generate
except ImportError:
    print("Error: sobol-seq package not found. Install with: pip install sobol-seq")
    sys.exit(1)


def generate_sobol_points(n_points: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Generate Sobol sequence points for 3D parameter space.
    
    Args:
        n_points: Number of Sobol sequence points to generate
        
    Returns:
        Tuple of (percentages, actual_values) where:
        - percentages: Array of shape (n_points, 3) with percentage variations [-30, +30]
        - actual_values: Array of shape (n_points, 3) with actual parameter values
    """
    # Generate Sobol sequence in [0,1]^3
    sobol_points = i4_sobol_generate(3, n_points)
    
    # Center point values
    center_width = 0.314  # µm
    center_radius = 0.043  # µm  
    center_thickness = 0.136  # µm
    
    # Convert to percentage variations [-30, +30]%
    percentages = (sobol_points - 0.5) * 2 * 30  # Scale to [-30, +30]%
    
    # Convert to actual parameter values
    actual_values = np.zeros_like(percentages)
    actual_values[:, 0] = center_width * (1 + percentages[:, 0] / 100)  # Width
    actual_values[:, 1] = center_radius * (1 + percentages[:, 1] / 100)  # Radius
    actual_values[:, 2] = center_thickness * (1 + percentages[:, 2] / 100)  # Thickness
    
    return percentages, actual_values


def create_3d_plots(percentages: np.ndarray, actual_values: np.ndarray, 
                   output_dir: Path) -> None:
    """Create 3D visualization plots for Sobol sequence points.
    
    Args:
        percentages: Percentage variations array
        actual_values: Actual parameter values array
        output_dir: Directory to save plots
    """
    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Plot 1: Percentage variations
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(percentages[:, 0], percentages[:, 1], percentages[:, 2], 
                          c=range(len(percentages)), cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('Width Variation (%)')
    ax1.set_ylabel('Radius Variation (%)')
    ax1.set_zlabel('Thickness Variation (%)')
    ax1.set_title('Sobol Sequence: Percentage Variations')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for sequence order
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    cbar1.set_label('Sequence Order')
    
    # Plot 2: Actual parameter values
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(actual_values[:, 0], actual_values[:, 1], actual_values[:, 2], 
                          c=range(len(actual_values)), cmap='plasma', s=50, alpha=0.8)
    ax2.set_xlabel('Width (µm)')
    ax2.set_ylabel('Radius (µm)')
    ax2.set_zlabel('Thickness (µm)')
    ax2.set_title('Sobol Sequence: Actual Parameter Values')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for sequence order
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    cbar2.set_label('Sequence Order')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'sobol_3d_sequence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"3D Sobol sequence plot saved as {output_path}")


def print_parameter_table(percentages: np.ndarray, actual_values: np.ndarray) -> None:
    """Print a formatted table of the Sobol sequence points.
    
    Args:
        percentages: Percentage variations array
        actual_values: Actual parameter values array
    """
    print("\n" + "="*80)
    print("SOBOL SEQUENCE PARAMETER POINTS")
    print("="*80)
    print(f"{'Point':<6} {'Width %':<10} {'Radius %':<10} {'Thickness %':<12} {'Width (µm)':<12} {'Radius (µm)':<12} {'Thickness (µm)':<15}")
    print("-"*80)
    
    for i in range(len(percentages)):
        print(f"{i+1:<6} {percentages[i,0]:<10.1f} {percentages[i,1]:<10.1f} {percentages[i,2]:<12.1f} "
              f"{actual_values[i,0]:<12.3f} {actual_values[i,1]:<12.3f} {actual_values[i,2]:<15.3f}")
    
    print("-"*80)
    print(f"Total points: {len(percentages)}")
    print(f"Parameter ranges:")
    print(f"  Width: {actual_values[:,0].min():.3f} - {actual_values[:,0].max():.3f} µm")
    print(f"  Radius: {actual_values[:,1].min():.3f} - {actual_values[:,1].max():.3f} µm") 
    print(f"  Thickness: {actual_values[:,2].min():.3f} - {actual_values[:,2].max():.3f} µm")


def main() -> None:
    """Main function to generate and visualize Sobol sequence."""
    print("Generating 32-point Sobol sequence for parameter sensitivity analysis...")
    
    # Generate Sobol sequence points
    percentages, actual_values = generate_sobol_points(n_points=32)
    
    # Create output directory
    output_dir = REPO_ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Create 3D plots
    print("Creating 3D visualization plots...")
    create_3d_plots(percentages, actual_values, output_dir)
    
    # Print parameter table
    print_parameter_table(percentages, actual_values)
    
    # Save data for later use
    data_path = REPO_ROOT / "data" / "sobol_sequence_32.npz"
    data_path.parent.mkdir(exist_ok=True)
    np.savez(data_path, percentages=percentages, actual_values=actual_values)
    print(f"\nSobol sequence data saved to {data_path}")
    
    print("\nNext steps:")
    print("1. Review the 3D plots to verify good coverage of parameter space")
    print("2. Run: python scripts/sobol/run_sobol_simulations.py")
    print("3. Then: python scripts/sobol/generate_surrogate_data.py")
    print("4. Finally: python scripts/sobol/inverse_monte_carlo_analysis.py")


if __name__ == "__main__":
    main()



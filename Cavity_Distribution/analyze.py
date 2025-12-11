#!/usr/bin/env python3
"""
Cavity Distribution Analysis
All data processing, Gaussian fitting, and statistical analysis functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Loading Functions
# ============================================================================

def extract_center_wavelengths(data_dir='data/raw/parent_chip3/Gen'):
    """Extract all center wavelengths from CSV files."""
    all_centers = []
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found!")
    for csv_file in data_dir_path.glob('*.csv'):
        if csv_file.name.endswith('.py'):
            continue
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            if data.shape[1] >= 2:
                all_centers.extend(data[:, 0])
        except Exception:
            continue
    return np.array(all_centers)


def load_and_process_grid_data(data_dir='data/raw/parent_chip3/Gen', grid_shape=(7, 16)):
    """Load and process all CSV files from the data directory into grid maps."""
    avg_center_map = np.full(grid_shape, np.nan)
    avg_fwhm_map = np.full(grid_shape, np.nan)
    avg_q_map = np.full(grid_shape, np.nan)
    dev_centers = np.full(grid_shape, np.nan)
    dev_fwhms = np.full(grid_shape, np.nan)
    spread_map = np.full(grid_shape, np.nan)
    fill_map = np.full(grid_shape, np.nan)
    
    all_centers = []
    all_Qs = []
    all_fills = []
    
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found!")
    
    csv_files = list(data_dir_path.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files to process...")
    
    for csv_file in csv_files:
        fname = csv_file.name
        if fname.endswith('.py'):
            continue
            
        try:
            parts = fname.split('_')
            if len(parts) >= 2:
                row = int(parts[0])
                col = int(parts[1])
            else:
                continue
        except (ValueError, IndexError):
            continue

        if not (0 <= row < grid_shape[0]) or not (0 <= col < grid_shape[1]):
            continue

        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            if data.size == 0:
                continue
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
        except Exception:
            continue

        if data.shape[1] < 2:
            continue

        center_wavelengths = data[:, 0]
        q_factors = data[:, 0] / data[:, 1]
        fwhms = data[:, 1]
        
        all_centers.extend(center_wavelengths)
        all_Qs.extend(q_factors)
        
        avg_center = np.mean(center_wavelengths)
        avg_fwhm = np.mean(fwhms)
        avg_q = np.mean(q_factors)
        
        dev_centers[row, col] = np.std(center_wavelengths)
        dev_fwhms[row, col] = np.std(fwhms)
        spread = np.ptp(center_wavelengths)
        fill_factor = len(center_wavelengths) / 15
        all_fills.append(fill_factor)
        
        avg_center_map[row, col] = avg_center
        avg_fwhm_map[row, col] = avg_fwhm
        avg_q_map[row, col] = avg_q
        spread_map[row, col] = spread
        fill_map[row, col] = fill_factor
        
        print(f"Processed {fname}: {len(center_wavelengths)} resonances, avg center: {avg_center*1e9:.1f} nm")

    return {
        'avg_center_map': avg_center_map,
        'avg_fwhm_map': avg_fwhm_map,
        'avg_q_map': avg_q_map,
        'dev_centers': dev_centers,
        'dev_fwhms': dev_fwhms,
        'spread_map': spread_map,
        'fill_map': fill_map,
        'all_centers': np.array(all_centers),
        'all_Qs': np.array(all_Qs),
        'all_fills': np.array(all_fills),
        'grid_shape': grid_shape
    }


# ============================================================================
# Gaussian Fitting Functions
# ============================================================================

def gaussian_function(x, amplitude, mean, stddev):
    """Gaussian function for fitting."""
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)


def fit_gaussian_to_distribution(data, bins=50):
    """Fit a Gaussian curve to a data distribution."""
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    amplitude_guess = np.max(hist)
    mean_guess = np.mean(data)
    stddev_guess = np.std(data)
    
    try:
        popt, pcov = curve_fit(
            gaussian_function, bin_centers, hist, 
            p0=[amplitude_guess, mean_guess, stddev_guess],
            maxfev=10000
        )
        
        y_pred = gaussian_function(bin_centers, *popt)
        ss_res = np.sum((hist - y_pred) ** 2)
        ss_tot = np.sum((hist - np.mean(hist)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        param_errors = np.sqrt(np.diag(pcov))
        fit_amplitude, fit_mean, fit_stddev = popt
        fit_amplitude_err, fit_mean_err, fit_stddev_err = param_errors
        fwhm = 2.355 * fit_stddev
        
        return {
            'success': True,
            'parameters': popt,
            'parameter_errors': param_errors,
            'covariance_matrix': pcov,
            'r_squared': r_squared,
            'amplitude': fit_amplitude,
            'amplitude_error': fit_amplitude_err,
            'mean': fit_mean,
            'mean_error': fit_mean_err,
            'stddev': fit_stddev,
            'stddev_error': fit_stddev_err,
            'fwhm': fwhm,
            'bin_centers': bin_centers,
            'histogram': hist,
            'bin_edges': bin_edges,
            'fit_curve': y_pred
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# Statistical Tests
# ============================================================================

def perform_statistical_tests(data, fit_results):
    """Perform statistical tests to assess the quality of the Gaussian fit."""
    if not fit_results.get('success', False):
        print("Cannot perform statistical tests - Gaussian fit failed")
        return
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS OF GAUSSIAN FIT")
    print("="*60)
    
    fitted_samples = np.random.normal(
        fit_results['mean'], 
        fit_results['stddev'], 
        len(data)
    )
    ks_statistic, ks_pvalue = stats.ks_2samp(data, fitted_samples)
    
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_statistic:.4f}")
    print(f"  P-value: {ks_pvalue:.4f}")
    print(f"  Interpretation: {'Data appears Gaussian' if ks_pvalue > 0.05 else 'Data may not be Gaussian'}")
    
    ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(data, dist='norm')
    print(f"\nAnderson-Darling Test for Normality:")
    print(f"  Statistic: {ad_statistic:.4f}")
    print(f"  Critical values: {ad_critical_values}")
    print(f"  Significance levels: {ad_significance_levels}")
    
    if len(data) <= 5000:
        shapiro_statistic, shapiro_pvalue = stats.shapiro(data)
        print(f"\nShapiro-Wilk Test:")
        print(f"  Statistic: {shapiro_statistic:.4f}")
        print(f"  P-value: {shapiro_pvalue:.4f}")
        print(f"  Interpretation: {'Data appears normal' if shapiro_pvalue > 0.05 else 'Data may not be normal'}")
    
    print(f"\nAdditional Statistics:")
    print(f"  Sample size: {len(data)}")
    print(f"  Mean (data): {np.mean(data):.2f}")
    print(f"  Mean (fit): {fit_results['mean']:.2f}")
    print(f"  Std Dev (data): {np.std(data):.2f}")
    print(f"  Std Dev (fit): {fit_results['stddev']:.2f}")
    print(f"  Skewness: {stats.skew(data):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.4f}")
    print("="*60)


# ============================================================================
# Export Functions
# ============================================================================

def export_cavity_data(data_dir='data/raw/parent_chip3/Gen', output_file='data/processed/cavity_distribution_data.npz', bins=25):
    """Export cavity distribution data to .npz file."""
    print("Extracting center wavelengths...")
    center_wavelengths = extract_center_wavelengths(data_dir)
    
    if len(center_wavelengths) == 0:
        print("No data found!")
        return

    wavelengths_nm = center_wavelengths * 1e9
    hist, bin_edges = np.histogram(wavelengths_nm, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    popt, pcov = curve_fit(
        gaussian_function, bin_centers, hist,
        p0=[np.max(hist), np.mean(wavelengths_nm), np.std(wavelengths_nm)],
        maxfev=10000
    )
    fit_amp, fit_mu, fit_sigma = popt
    fit_errors = np.sqrt(np.diag(pcov))
    
    y_pred = gaussian_function(bin_centers, *popt)
    ss_res = np.sum((hist - y_pred)**2)
    ss_tot = np.sum((hist - np.mean(hist))**2)
    r2 = 1 - ss_res/ss_tot
    fwhm = 2.355 * fit_sigma

    print(f"μ = {fit_mu:.2f} nm, σ = {fit_sigma:.2f} nm, FWHM = {fwhm:.2f} nm, R² = {r2:.4f}")

    range_sigma = 3.5
    x_min = fit_mu - range_sigma * fit_sigma
    x_max = fit_mu + range_sigma * fit_sigma
    x_fit = np.linspace(x_min, x_max, 300)
    y_fit = gaussian_function(x_fit, *popt)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_file,
             center_wavelengths_m=center_wavelengths,
             wavelengths_nm=wavelengths_nm,
             histogram_counts=hist,
             bin_edges=bin_edges,
             bin_centers=bin_centers,
             fit_amplitude=fit_amp,
             fit_mean=fit_mu,
             fit_stddev=fit_sigma,
             fit_amplitude_error=fit_errors[0],
             fit_mean_error=fit_errors[1],
             fit_stddev_error=fit_errors[2],
             r_squared=r2,
             fwhm=fwhm,
             x_fit=x_fit,
             y_fit=y_fit,
             n_data_points=len(center_wavelengths),
             n_bins=bins,
             fit_range_sigma=range_sigma)

    print(f"Data exported to '{output_file}'")
    print(f"Contains {len(center_wavelengths)} data points")
    print(f"Histogram with {bins} bins")
    print(f"Gaussian fit: μ={fit_mu:.2f}±{fit_errors[1]:.2f} nm, σ={fit_sigma:.2f}±{fit_errors[2]:.2f} nm")


# ============================================================================
# Main Analysis Function
# ============================================================================

def main_analysis():
    """Main function to run the complete analysis."""
    print("Gaussian Fit Analysis for Silicon Hard Mask Cavity Distribution")
    print("=" * 70)
    
    try:
        print("Extracting center wavelengths from CSV files...")
        center_wavelengths = extract_center_wavelengths()
        
        if len(center_wavelengths) == 0:
            print("No center wavelength data found!")
            return
        
        print(f"\nExtracted {len(center_wavelengths)} center wavelength measurements")
        wavelengths_nm = center_wavelengths * 1e9
        print(f"Wavelength range: {wavelengths_nm.min():.2f} - {wavelengths_nm.max():.2f} nm")
        print(f"Mean wavelength: {np.mean(wavelengths_nm):.2f} nm")
        print(f"Standard deviation: {np.std(wavelengths_nm):.2f} nm")
        
        print("\nFitting Gaussian curve to distribution...")
        fit_results = fit_gaussian_to_distribution(wavelengths_nm)
        
        if fit_results['success']:
            print("Gaussian fit successful!")
            print(f"Fit parameters:")
            print(f"  Amplitude: {fit_results['amplitude']:.4f} ± {fit_results['amplitude_error']:.4f}")
            print(f"  Mean: {fit_results['mean']:.2f} ± {fit_results['mean_error']:.2f} nm")
            print(f"  Standard deviation: {fit_results['stddev']:.2f} ± {fit_results['stddev_error']:.2f} nm")
            print(f"  FWHM: {fit_results['fwhm']:.2f} nm")
            print(f"  R-squared: {fit_results['r_squared']:.4f}")
            
            perform_statistical_tests(wavelengths_nm, fit_results)
        else:
            print(f"Gaussian fit failed: {fit_results['error']}")
        
        print("\nGaussian fit analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export_cavity_data()
    else:
        main_analysis()


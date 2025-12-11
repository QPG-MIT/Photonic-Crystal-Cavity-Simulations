# Sensitivity Analysis Scripts

This folder contains scripts for sensitivity analysis, visualization, and inverse uncertainty analysis.

## Scripts

### Visualization Scripts

#### `plot_wavelength_sensitivity_complete_good.py` ⭐
- **Purpose**: Complete wavelength sensitivity analysis and plotting
- **Function**: Analyzes all modes, generates publication-ready compact plots
- **Usage**: Main visualization for wavelength sensitivity analysis

#### `plot_oned_sensitivity.py`
- **Purpose**: Plot 1D sensitivity analysis results
- **Function**: Visualizes parameter variations, simulation metrics, parameter space coverage
- **Input**: `data/oned_sensitivity_analysis/oned_sensitivity_results.csv`
- **Usage**: Visualization for 1D cutline analysis

#### `plot_cutlines_q.py`
- **Purpose**: Plot cutline results (Q vs variation, wavelength vs variation)
- **Function**: Creates plots from q_summary JSON files
- **Input**: `data/summaries/q_summary_*.json`
- **Usage**: Main visualization for 1D cutline results

#### `plot_lambda_slices.py`
- **Purpose**: Plot λ(t) slices for different (w, r) combinations
- **Function**: Visualizes inversion landscape for thickness prediction
- **Usage**: Specialized visualization for inverse analysis

#### `plot_resonance_details.py`
- **Purpose**: Run Q-factor analysis for each cutline result and save per-case plots
- **Function**: Analyzes resonance details, saves diagnostic plots
- **Usage**: Debugging problematic cases

### Analysis Scripts

#### `inverse_monte_carlo_uncertainty_analysis_v2.py` ⭐
- **Purpose**: Inverse Monte Carlo analysis to predict thickness distributions
- **Function**: Inverts surrogate model given manufacturing uncertainties in width/radius/wavelength
- **Input**: `data/results_summaries/surrogate_model_data.json`
- **Usage**: Core for manufacturing uncertainty analysis

### Reporting Scripts

#### `report_cutlines_q.py`
- **Purpose**: Build report for each cutline simulation from Q summaries
- **Function**: Reads q_summary JSON files, generates CSV report
- **Input**: `data/summaries/q_summary_*.json`
- **Output**: `data/summaries/cutlines_q_report.csv`
- **Usage**: Generate summary reports from cutline analysis

## Workflow

### 1D Cutline Sensitivity Analysis:
1. **Run simulations**: `../run_simulation/run_oned_cutlines_quick.py`
2. **Visualize**: `plot_oned_sensitivity.py` or `plot_cutlines_q.py`
3. **Report**: `report_cutlines_q.py`

### Wavelength Sensitivity:
1. **Analyze and plot**: `plot_wavelength_sensitivity_complete_good.py`

### Inverse Uncertainty Analysis:
1. **Run analysis**: `inverse_monte_carlo_uncertainty_analysis_v2.py`
2. **Visualize slices**: `plot_lambda_slices.py` (optional)

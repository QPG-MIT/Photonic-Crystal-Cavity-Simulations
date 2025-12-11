# Sobol Sequence Analysis Scripts

This folder contains scripts for Sobol sequence generation, simulation data processing, visualization, and utilities.

## Scripts

### `sobol_3d_analysis.py` ⭐ **Start Here**
- **Purpose**: Generate and visualize 32-point Sobol sequence for 3D parameter space (width, radius, thickness)
- **Function**: Creates Sobol sequence points, generates 3D plots, saves data for simulations
- **Output**: `data/sobol_sequence_32.npz` with parameter points
- **Usage**: First step in Sobol parameter sensitivity analysis

### `generate_surrogate_model_data.py`
- **Purpose**: Generate surrogate model training data from Sobol HDF5 results
- **Function**: Extracts Q-factors and wavelengths from simulation results, creates JSON for surrogate modeling
- **Input**: `data/results/sobol_32/*.hdf5` (from `../run_simulation/run_sobol_32_simulations.py`)
- **Output**: `data/results_summaries/surrogate_model_data.json`
- **Usage**: Run after Sobol simulations complete

### `plot_sobol_3d_analysis_good.py`
- **Purpose**: Create 3D visualizations of Sobol parameter space with Q-factors and wavelengths
- **Function**: Color-codes points by Q-factor/wavelength, creates publication-ready plots
- **Input**: `data/results/sobol_32/*.hdf5`
- **Usage**: Visualize Sobol simulation results

### `plot_3d_with_surrogate_predictions.py`
- **Purpose**: Visualize original simulated points + surrogate model predictions
- **Function**: Shows denser sampling of parameter space using trained surrogate models
- **Input**: Requires trained surrogate model data
- **Usage**: Validate surrogate models and explore parameter space

### `regenerate_failed_gds.py`
- **Purpose**: Regenerate failed GDS files from Sobol sequence setup
- **Function**: Identifies failed GDS generations and retries them
- **Usage**: Utility for fixing GDS generation issues

## Workflow

1. **Generate Sobol sequence**: `sobol_3d_analysis.py`
2. **Run simulations**: `../run_simulation/run_sobol_32_simulations.py`
3. **Extract data**: `generate_surrogate_model_data.py`
4. **Visualize results**: `plot_sobol_3d_analysis_good.py` or `plot_3d_with_surrogate_predictions.py`

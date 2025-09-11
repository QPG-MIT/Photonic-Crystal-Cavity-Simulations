# Photonic Cavity Analysis

This repository contains a complete photonic cavity analysis workflow for silicon hard mask simulations.

## Project Structure

- `modules/` - Analysis modules for different aspects of photonic cavity characterization
- `*.gds` - GDSII files containing the cavity and hole geometries
- `*.ipynb` - Jupyter notebooks for interactive analysis
- `results_*.hdf5` - Simulation results in HDF5 format
- `run_complete_analysis.py` - Script to run the complete analysis workflow

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Analysis
Open the Jupyter notebooks for interactive analysis:
- `Photonic_Cavity_Analysis.ipynb` - Basic analysis
- `Photonic_Cavity_Analysis_Complete.ipynb` - Complete analysis workflow

### Command Line Analysis
Run the complete analysis from the command line:
```bash
python run_complete_analysis.py
```

## Analysis Modules

The `modules/` directory contains specialized analysis modules:

- `q_factor_analysis.py` - Quality factor analysis
- `mode_volume_analysis.py` - Mode volume calculations
- `polarization_analysis.py` - Polarization analysis
- `nearfield_analysis.py` - Near-field analysis
- `farfield_analysis.py` - Far-field analysis
- `collection_efficiency_analysis.py` - Collection efficiency calculations
- `simulation_runner.py` - Simulation execution
- `simulation_setup.py` - Simulation configuration

## Dependencies

Key dependencies include:
- `tidy3d` - FDTD simulation engine
- `gdsfactory` - GDSII file handling
- `numpy`, `scipy` - Numerical computing
- `matplotlib` - Plotting
- `h5py` - HDF5 file handling
- `jupyter` - Interactive notebooks

See `requirements.txt` for the complete list of dependencies.

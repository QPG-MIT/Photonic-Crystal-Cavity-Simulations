# Run Simulation Scripts

This folder contains scripts that execute Tidy3D simulations and run analysis workflows.

## Scripts

### `run_analysis.py` ⭐ **Main Entry Point**
- **Purpose**: Main orchestrator for two-stage photonic cavity analysis workflow
- **Function**: 
  - Scout stage: Broadband simulation to locate resonance frequency/Q-factor
  - Lock-in stage: Narrowband simulation with detailed analyses (mode volume, polarization, near/far-field, collection efficiency)
- **Usage**: Primary script for complete cavity analysis

### `run_sobol_32_simulations.py`
- **Purpose**: Run all 32 Sobol sequence simulations in parallel
- **Function**: Executes Tidy3D simulations for each Sobol point, analyzes Q-factors
- **Usage**: Run after generating Sobol sequence with `../sobol/sobol_3d_analysis.py`

### `run_remaining_sobol.py`
- **Purpose**: Run only the remaining Sobol simulations that haven't completed
- **Function**: Checks which simulations are done, runs missing ones
- **Usage**: Resume interrupted Sobol simulation runs

### `run_oned_cutlines_quick.py`
- **Purpose**: Run 1D cutline sensitivity simulations (cheap preset)
- **Function**: Generates scaled GDS, runs minimal scout simulations for parameter variations
- **Usage**: Quick 1D parameter sensitivity analysis

### `preview_simulation_setup_good.py`
- **Purpose**: Preview Tidy3D simulation setup without running
- **Function**: Builds simulation, renders geometry/monitor layout figure
- **Usage**: Check geometry and monitors before running expensive simulations

## Workflow

1. **Preview setup**: `preview_simulation_setup_good.py` (optional, to check geometry)
2. **Run simulations**: 
   - For Sobol analysis: `run_sobol_32_simulations.py` or `run_remaining_sobol.py`
   - For 1D cutlines: `run_oned_cutlines_quick.py`
   - For complete analysis: `run_analysis.py`

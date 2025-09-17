# Photonic Crystal Cavity Simulations

A complete photonic cavity analysis workflow for silicon hard mask simulations using Tidy3D FDTD solver.

## Repository Structure

This repository contains only the essential files needed to run the analysis:

```
├── modules/                          # Analysis modules
│   ├── collection_efficiency_analysis.py
│   ├── farfield_analysis.py
│   ├── mode_volume_analysis.py
│   ├── nearfield_analysis.py
│   ├── plot_style.py
│   ├── polarization_analysis.py
│   ├── q_factor_analysis.py
│   ├── simulation_runner.py
│   └── simulation_setup.py
├── scripts/
│   └── run_complete_analysis.py      # Main analysis script
├── notebooks/
│   └── Photonic_Cavity_Workflow.ipynb # Interactive analysis notebook
├── gds/                              # Device geometry files
│   ├── Cavity.gds
│   └── Holes.gds
├── data/
│   └── results/                      # Sample results (0.19µm only)
│       ├── results_lockin_full_0.19um.hdf5
│       └── results_scout_q_only_0.19um.hdf5
├── requirements.txt                  # Python dependencies
└── README.md
```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abuzzimit/Photonic-Crystal-Cavity-Simulations.git
   cd Photonic-Crystal-Cavity-Simulations
   ```

2. **Set up environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run analysis:**
   ```bash
   # Command line (recommended for first run)
   python scripts/run_complete_analysis.py
   
   # Or interactive notebook
   jupyter notebook notebooks/Photonic_Cavity_Workflow.ipynb
   ```

## Analysis Workflow

The analysis consists of two main stages:

### 1. Scout Stage
- **Purpose**: Find cavity resonance wavelength
- **Output**: `data/results/results_scout_q_only_*.hdf5`
- **Analysis**: Q-factor calculation and resonance identification

### 2. Lock-in Stage  
- **Purpose**: Detailed analysis at resonance
- **Output**: `data/results/results_lockin_full_*.hdf5`
- **Analysis**: Mode volume, polarization, near-field, far-field, collection efficiency

## Analysis Modules

| Module | Purpose |
|--------|---------|
| `simulation_setup.py` | Tidy3D simulation configuration and geometry loading |
| `simulation_runner.py` | Simulation execution and result management |
| `q_factor_analysis.py` | Quality factor calculation from frequency domain data |
| `mode_volume_analysis.py` | Mode volume and Purcell factor calculations |
| `polarization_analysis.py` | Far-field polarization analysis |
| `nearfield_analysis.py` | Near-field intensity and field analysis |
| `farfield_analysis.py` | Far-field radiation pattern analysis |
| `collection_efficiency_analysis.py` | Collection efficiency vs numerical aperture |

## Output Structure

When you run the analysis, the following directories will be created:

- `data/simulations/` - Simulation configuration files
- `data/results/` - HDF5 simulation results
- `data/summaries/` - JSON analysis summaries
- `figures/` - Generated plots and visualizations
- `logs/` - Analysis logs

## Key Dependencies

- **`tidy3d`** - FDTD simulation engine
- **`gdstk`** - GDSII file handling
- **`numpy`, `scipy`** - Numerical computing
- **`matplotlib`** - Plotting and visualization
- **`h5py`** - HDF5 file I/O
- **`jupyter`** - Interactive notebooks

See `requirements.txt` for the complete dependency list.

## Notes

- The repository contains sample results for 0.19µm cavity size
- All file paths are resolved relative to the repository root
- The analysis can be run from any directory within the repository
- Large result files (>100MB) are excluded from the repository but generated locally during analysis

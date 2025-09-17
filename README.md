# Diamond Nanobeam Photonic Crystal Cavity Simulations

A complete photonic cavity analysis workflow for diamond nanobeam 1D photonic crystal cavities designed for tin vacancy (SnV) defects, fabricated using microtransfer printing of silicon hard masks. Simulations performed using Tidy3D FDTD solver.

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
│   └── run_analysis.py               # Main analysis script
├── notebooks/
│   └── analysis_notebook.ipynb       # Interactive analysis notebook
├── gds/                              # Diamond nanobeam cavity geometry files
│   ├── Cavity.gds                    # Main nanobeam cavity structure
│   └── Holes.gds                     # 1D photonic crystal holes
├── data/
│   ├── results/                      # Simulation results (HDF5 files)
│   ├── simulations/                  # Simulation configuration files (JSON)
│   └── summaries/                    # Analysis summaries
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
   # Create virtual environment in a parallel directory (outside the project)
   cd ..
   python -m venv photonic-cavity-env
   source photonic-cavity-env/bin/activate  # On Windows: photonic-cavity-env\Scripts\activate
   cd Photonic-Crystal-Cavity-Simulations
   pip install -r requirements.txt
   ```

   **For future sessions**, activate the environment from the project directory:
   ```bash
   # From the project root directory
   source ../photonic-cavity-env/bin/activate  # On Windows: ..\photonic-cavity-env\Scripts\activate
   ```

3. **Run analysis:**
   ```bash
   # Command line (recommended for first run)
   python scripts/run_analysis.py
   
   # Or interactive notebook
   jupyter notebook notebooks/analysis_notebook.ipynb
   ```

## Analysis Workflow

The analysis consists of two main stages for characterizing diamond nanobeam 1D photonic crystal cavities:

### 1. Scout Stage
- **Purpose**: Find cavity resonance wavelength for SnV defect coupling
- **Output**: `data/results/results_scout_q_only_*.hdf5`
- **Analysis**: Q-factor calculation and resonance identification in diamond

### 2. Lock-in Stage  
- **Purpose**: Detailed analysis at resonance for quantum applications
- **Output**: `data/results/results_lockin_full_*.hdf5`
- **Analysis**: Mode volume, polarization, near-field, far-field, collection efficiency for SnV integration

## Analysis Modules

| Module | Purpose |
|--------|---------|
| `simulation_setup.py` | Tidy3D simulation configuration and diamond nanobeam cavity geometry loading |
| `simulation_runner.py` | Simulation execution and result management |
| `q_factor_analysis.py` | Quality factor calculation for diamond cavity resonances |
| `mode_volume_analysis.py` | Mode volume and Purcell factor calculations for SnV coupling |
| `polarization_analysis.py` | Far-field polarization analysis for quantum applications |
| `nearfield_analysis.py` | Near-field intensity and field analysis in diamond |
| `farfield_analysis.py` | Far-field radiation pattern analysis |
| `collection_efficiency_analysis.py` | Collection efficiency vs numerical aperture for SnV emission |

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

- The repository contains sample results for 0.19 µm diamond nanobeam cavity size
- All file paths are resolved relative to the repository root
- The analysis can be run from any directory within the repository
- Large result files (>100 MB) are excluded from the repository but generated locally during analysis
- Nanobeam 1D photonic crystal cavities are designed for tin vacancy (SnV) defect integration in diamond
- Fabrication uses microtransfer printing of silicon hard masks for precise patterning
- Virtual environment is created in a parallel directory to keep the project directory clean

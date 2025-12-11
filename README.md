# Photonic Crystal Cavity Simulations

This repository collects simulation scripts, analysis utilities, and data processing
pipelines for photonic crystal cavity work. It combines several self-contained
analyses (SEM image processing, FDTD simulations, thickness estimation, and
statistical fitting) that previously lived in separate folders.

## Getting started

1. **Create an environment** with Python 3.11+ and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > The requirements file aggregates tools from all sub-projects and pins a few
   > bleeding-edge versions; if you run into resolver errors, loosen the versions
   > for the affected packages or install only the subsets you need.
2. **Configure Tidy3D** if you plan to run simulations: export `TIDY3D_API_KEY`
   (and optionally `TIDY3D_REGION`) so `tidy3d.web` can authenticate.
3. **Clone the repo with data**: most scripts assume the existing directory
   layout (e.g., `Simulations/data`, `Cavity_Distribution/data`).

## Repository structure

- `Simulations/`: Tidy3D simulation assets
  - `modules/`: reusable analysis helpers (Q-factor, mode volume, far/near-field
    plots) and the `simulation_runner.py` CLI for managing Tidy3D jobs and
    caching results.
  - `gds/`: GDS layouts and reconstruction utilities for cavity geometries.
  - `data/` and `notebooks/`: example simulation inputs/outputs and analysis
    notebooks.
- `Cavity_Distribution/`: Gaussian fitting and statistics for cavity center
  wavelengths (`analyze.py`, `plot.py`), expecting CSV grids under
  `data/raw/...`.
- `Statistical_width_estimate/`: SEM image processing to extract rail spacing
  and hole radii; main scripts are `analyze.py` (batch processing) and
  `plot_results.py` (visualization of saved `.npz` results).
- `Thickness estimate/`: Scripts for extracting mask thickness from SEM imagery
  (`thickness_extraction.py`) and visualizing prism overlays.

## Common workflows

### Run a Tidy3D simulation
```bash
python Simulations/modules/simulation_runner.py <simulation.json> [output.hdf5]
```
The runner will estimate cost (if possible), reuse cached results, and validate
monitors before returning a `SimulationData` file.

### Analyze cavity wavelength distributions
```bash
python Cavity_Distribution/analyze.py
python Cavity_Distribution/plot.py
```
These scripts load wavelength CSV grids from `data/raw/...`, fit Gaussians, and
produce summary figures in `figures/`.

### Estimate feature widths from SEMs
```bash
python Statistical_width_estimate/analyze.py  # processes batch of SEM images
python Statistical_width_estimate/plot_results.py
```
Outputs (centerlines, masks, measurements) are saved in `analysis_data.npz` and
visualized via Matplotlib.

### Extract mask thickness
```bash
python "Thickness estimate/thickness_extraction.py"
```
Uses SEM overlay geometry to infer thickness; preview images live alongside the
scripts for reference.

## Notes and caveats
- Many scripts suppress warnings for cleaner output; when debugging, consider
  removing those filters to surface upstream warnings.
- The repository mixes data-heavy workflows with simulation runners; if you only
  need one component, install the minimum dependencies for that sub-folder to
  keep environments lightweight.

#!/usr/bin/env python3
"""
Simulation Runner Module

This module handles the execution of Tidy3D simulations with cost estimation,
idempotent execution (won't re-run if results exist), and comprehensive error handling.

Key Features:
- Cost estimation before running simulations
- Idempotent execution (checks for existing results)
- Progress monitoring and error handling
- Integration with Tidy3D web interface
- Automatic result validation
"""

import numpy as np
import tidy3d as td
from tidy3d import web
import os
from pathlib import Path
import warnings
from typing import Optional, Dict, Any, Tuple
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class SimulationRunner:
    """
    Handles the execution of Tidy3D simulations with comprehensive error handling.
    """
    
    def __init__(self, task_name: str = "photonic_cavity_simulation"):
        """
        Initialize the simulation runner.
        
        Args:
            task_name: Base name for Tidy3D tasks
        """
        self.task_name = task_name
        self.results_cache = {}
        # Attempt to configure web API from environment for clearer diagnostics
        api_key = os.environ.get("TIDY3D_API_KEY")
        region = os.environ.get("TIDY3D_REGION")
        try:
            if api_key:
                # Try the newer API first, then fall back to older API
                try:
                    web.configure(api_key=api_key, region=region)
                except TypeError:
                    # Newer Tidy3D versions might not accept api_key parameter
                    try:
                        web.configure(region=region)
                        # Set API key via environment or other method
                        os.environ["TIDY3D_API_KEY"] = api_key
                    except Exception:
                        pass
            else:
                # If already authenticated, this is a no-op; otherwise prompts in interactive sessions
                try:
                    web.login()
                except Exception:
                    pass
        except Exception as _cfg_exc:
            # Non-fatal; configuration can still occur elsewhere
            print(f"⚠️  Tidy3D web.configure/login failed: {_cfg_exc}")
        
    def load_simulation(self, sim_file: str) -> td.Simulation:
        """
        Load a simulation from file.
        
        Args:
            sim_file: Path to simulation JSON file
            
        Returns:
            Loaded Tidy3D simulation object
        """
        print("="*60)
        print("🚀 LOADING SIMULATION")
        print("="*60)
        
        if not Path(sim_file).exists():
            raise FileNotFoundError(f"Simulation file not found: {sim_file}")
        
        print(f"Loading simulation from {sim_file}...")
        simulation = td.Simulation.from_file(sim_file)
        
        print(f"✓ Simulation loaded successfully")
        print(f"  - Structures: {len(simulation.structures)}")
        print(f"  - Monitors: {len(simulation.monitors)}")
        print(f"  - Run time: {simulation.run_time*1e12:.1f} ps")
        print(f"  - Size: {simulation.size} µm")
        
        # List monitors
        print(f"\nMonitors:")
        for i, monitor in enumerate(simulation.monitors):
            print(f"  {i+1}. {monitor.name}: {type(monitor).__name__}")
        
        return simulation
    
    def estimate_cost(self, simulation: td.Simulation, task_name: Optional[str] = None) -> Optional[float]:
        """
        Estimate the cost of running the simulation.
        
        Args:
            simulation: Tidy3D simulation object
            task_name: Optional custom task name
            
        Returns:
            Estimated cost in FlexCredits, or None if estimation failed
        """
        if task_name is None:
            task_name = self.task_name
            
        print(f"\n💰 ESTIMATING SIMULATION COST")
        print(f"  - Task name: {task_name}")
        print(f"  - Run time: {simulation.run_time*1e12:.1f} ps")
        print(f"  - Size: {simulation.size} µm")
        print(f"  - Monitors: {len(simulation.monitors)}")
        
        try:
            # Create a job to estimate cost
            job = web.Job(simulation=simulation, task_name=task_name, verbose=False)
            
            # Estimate the cost
            print("  - Calculating cost estimate...")
            estimated_cost = job.estimate_cost(verbose=False)
            
            print(f"\n💳 ESTIMATED COST: {estimated_cost:.3f} FlexCredits")
            print(f"  - This is the maximum cost (if simulation runs full time)")
            print(f"  - Actual cost may be lower if early shut-off is triggered")
            
            # Clean up the job
            job.delete()
            
            return estimated_cost
            
        except Exception as e:
            # Provide clearer diagnostics for common causes
            has_token = bool(os.environ.get("TIDY3D_API_KEY"))
            region = os.environ.get("TIDY3D_REGION") or "default"
            print("❌ Could not estimate cost.")
            print(f"  - Reason: {repr(e)}")
            print(f"  - TIDY3D_API_KEY present in env: {has_token}")
            print(f"  - TIDY3D_REGION: {region}")
            print("  - Tip: On Tidy3D >=2.9, use tidy3d.web.configure(api_key=...) or tidy3d.web.login().")
            return None
    
    def check_existing_results(self, results_path: str, expected_monitors: Optional[list] = None) -> bool:
        """
        Check if simulation results already exist.
        
        Args:
            results_path: Path to expected results file
            
        Returns:
            True if results exist and are valid, False otherwise
        """
        if not Path(results_path).exists():
            return False
        
        try:
            # Try to load the results to validate they're complete
            data = td.SimulationData.from_file(results_path)
            
            # Check if we have the expected monitors
            if expected_monitors is None:
                expected_monitors = ['probe', 'flux', 'field_near']
            available_monitors = list(data.monitor_data.keys())
            
            missing_monitors = [m for m in expected_monitors if m not in available_monitors]
            if missing_monitors:
                print(f"⚠️  Results file exists but missing monitors: {missing_monitors}")
                return False
            
            print(f"✓ Found existing results: {results_path}")
            print(f"  - Available monitors: {available_monitors}")
            return True
            
        except Exception as e:
            print(f"⚠️  Results file exists but is corrupted: {e}")
            return False
    
    def run_simulation(self, 
                     simulation: td.Simulation, 
                     results_path: str = "simulation_results.hdf5",
                     task_name: Optional[str] = None,
                     force_rerun: bool = False,
                     estimate_cost_first: bool = True,
                     expected_monitors: Optional[list] = None,
                     auto_confirm: bool = False) -> Optional[td.SimulationData]:
        """
        Run the simulation with comprehensive error handling.
        
        Args:
            simulation: Tidy3D simulation object
            results_path: Path to save results
            task_name: Optional custom task name
            force_rerun: Force re-run even if results exist
            estimate_cost_first: Whether to estimate cost before running
            
        Returns:
            Simulation data if successful, None if failed or skipped
        """
        if task_name is None:
            task_name = self.task_name
            
        print(f"\n🎯 RUNNING SIMULATION")
        print(f"  - Task name: {task_name}")
        print(f"  - Output path: {results_path}")
        print(f"  - Run time: {simulation.run_time*1e12:.1f} ps")
        
        # Check for existing results
        if not force_rerun and self.check_existing_results(results_path, expected_monitors=expected_monitors):
            print("✓ Using existing results (use force_rerun=True to override)")
            try:
                return td.SimulationData.from_file(results_path)
            except Exception as e:
                print(f"❌ Failed to load existing results: {e}")
                print("  - Will re-run simulation")
        
        # Estimate cost if requested
        estimated_cost = None
        if estimate_cost_first:
            estimated_cost = self.estimate_cost(simulation, task_name)
        
        # Get user confirmation
        if estimated_cost is not None:
            print(f"\n⚠️  This simulation will cost approximately {estimated_cost:.3f} FlexCredits!")
            if auto_confirm:
                print("Auto-confirm enabled; proceeding with simulation.")
                confirm = 'y'
            else:
                confirm = input("Do you want to proceed with the simulation? (y/N): ").strip().lower()
        else:
            print("\n⚠️  Could not estimate cost. Proceeding without cost information.")
            if auto_confirm:
                print("Auto-confirm enabled; proceeding with simulation.")
                confirm = 'y'
            else:
                confirm = input("Are you sure you want to run the simulation? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("Simulation cancelled.")
            return None
        
        # Run the simulation
        print("\n🚀 Running simulation...")
        start_time = time.time()
        
        try:
            data = web.run(
                simulation,
                task_name=task_name,
                folder_name="default",
                path=results_path,
                verbose=True
            )
            
            elapsed_time = time.time() - start_time
            print(f"✓ Simulation completed successfully in {elapsed_time:.1f} seconds!")
            
            # Validate results
            self._validate_results(data)
            
            return data
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_results(self, data: td.SimulationData) -> None:
        """
        Validate simulation results for completeness.
        
        Args:
            data: Simulation data to validate
        """
        print("\n🔍 Validating simulation results...")
        
        # Check for required monitors
        required_monitors = ['probe', 'flux', 'field_near']
        available_monitors = list(data.monitor_data.keys())
        
        missing_monitors = [m for m in required_monitors if m not in available_monitors]
        if missing_monitors:
            print(f"⚠️  Missing required monitors: {missing_monitors}")
        else:
            print("✓ All required monitors present")
        
        # Check probe data
        if 'probe' in available_monitors:
            probe_data = data['probe']
            if hasattr(probe_data, 'Ey'):
                t = probe_data.Ey.coords['t'].values
                print(f"✓ Probe data: {len(t)} time points, {t[-1]*1e12:.2f} ps duration")
            else:
                print("⚠️  Probe data missing Ey field")
        
        # Check flux data
        if 'flux' in available_monitors:
            flux_data = data['flux']
            if hasattr(flux_data, 'flux'):
                flux_value = flux_data.flux.values[0]
                print(f"✓ Flux data: {flux_value:.2e}")
            else:
                print("⚠️  Flux data missing flux values")
        
        # Check field data
        if 'field_near' in available_monitors:
            field_data = data['field_near']
            if hasattr(field_data, 'Ex'):
                print(f"✓ Field data: {field_data.Ex.shape}")
            else:
                print("⚠️  Field data missing Ex field")
        
        # Check far-field monitors
        farfield_monitors = [m for m in available_monitors if 'farfield' in m]
        if farfield_monitors:
            print(f"✓ Far-field monitors: {farfield_monitors}")
        else:
            print("⚠️  No far-field monitors found")
        
        print("✓ Results validation complete")
    
    def load_existing_data(self, results_path: str) -> Optional[td.SimulationData]:
        """
        Load existing simulation data.
        
        Args:
            results_path: Path to results file
            
        Returns:
            Simulation data if successful, None if failed
        """
        print(f"\n📁 LOADING EXISTING DATA")
        
        if not Path(results_path).exists():
            print(f"❌ Data file not found: {results_path}")
            return None
        
        print(f"Loading data from {results_path}...")
        try:
            data = td.SimulationData.from_file(results_path)
            print("✓ Data loaded successfully!")
            
            # Validate results
            self._validate_results(data)
            
            return data
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return None
    
    def get_simulation_info(self, simulation: td.Simulation) -> Dict[str, Any]:
        """
        Get comprehensive information about a simulation.
        
        Args:
            simulation: Tidy3D simulation object
            
        Returns:
            Dictionary with simulation information
        """
        info = {
            'frequency_thz': simulation.sources[0].source_time.freq0 / 1e12,
            'wavelength_nm': 3e8 / simulation.sources[0].source_time.freq0 * 1e9,
            'run_time_ps': simulation.run_time * 1e12,
            'size_um': simulation.size,
            'center_um': simulation.center,
            'num_structures': len(simulation.structures),
            'num_monitors': len(simulation.monitors),
            'monitor_names': [m.name for m in simulation.monitors],
            'grid_cells': simulation.grid.num_cells,
            'boundary_conditions': str(simulation.boundary_spec)
        }
        
        return info


def run_simulation_from_file(sim_file: str,
                           results_path: str = "data/results/simulation_results.hdf5",
                           task_name: str = "photonic_cavity_simulation",
                           force_rerun: bool = False,
                           estimate_cost: bool = True) -> Optional[td.SimulationData]:
    """
    Convenience function to run a simulation from a file.
    
    Args:
        sim_file: Path to simulation JSON file
        results_path: Path to save results
        task_name: Name for Tidy3D task
        force_rerun: Force re-run even if results exist
        estimate_cost: Whether to estimate cost before running
        
    Returns:
        Simulation data if successful, None if failed
    """
    runner = SimulationRunner(task_name=task_name)
    simulation = runner.load_simulation(sim_file)
    return runner.run_simulation(
        simulation=simulation,
        results_path=results_path,
        force_rerun=force_rerun,
        estimate_cost_first=estimate_cost
    )


def load_simulation_data(results_path: str) -> Optional[td.SimulationData]:
    """
    Convenience function to load existing simulation data.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Simulation data if successful, None if failed
    """
    runner = SimulationRunner()
    return runner.load_existing_data(results_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simulation_runner.py <simulation_file.json> [results_file.hdf5]")
        sys.exit(1)
    
    sim_file = sys.argv[1]
    results_file = sys.argv[2] if len(sys.argv) > 2 else "data/results/simulation_results.hdf5"
    
    data = run_simulation_from_file(
        sim_file=sim_file,
        results_path=results_file,
        force_rerun=False,
        estimate_cost=True
    )
    
    if data is not None:
        print("\n✅ Simulation completed successfully!")
        print("You can now run analysis on the results.")
    else:
        print("\n❌ Simulation failed or was cancelled.")

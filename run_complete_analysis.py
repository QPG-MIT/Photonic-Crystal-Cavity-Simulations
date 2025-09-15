#!/usr/bin/env python3
"""
Complete Photonic Cavity Analysis Runner

This script runs all the analysis cells from the Jupyter notebook sequentially
to test the complete workflow and identify any issues.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append('modules')

# Import all analysis modules
from modules.q_factor_analysis import analyze_q_factor
from modules.mode_volume_analysis import analyze_mode_volume
from modules.polarization_analysis import analyze_polarization
from modules.nearfield_analysis import analyze_nearfield
from modules.farfield_analysis import analyze_farfield
from modules.collection_efficiency_analysis import analyze_collection_efficiency

# Physical constants
C0 = 299792458.0  # Speed of light in vacuum (m/s)
thickness_um = 0.14

def run_complete_analysis():
    """
    Run the complete photonic cavity analysis workflow
    """
    print("=" * 80)
    print("🔬 COMPLETE PHOTONIC CAVITY ANALYSIS")
    print("=" * 80)
    
    # Configuration (matching the notebook)
    CONFIG = {
        'results_file': f'results_{thickness_um}um.hdf5',
        'wavelength_um': 0.650,
        'thickness_um': thickness_um,
        'NA': 0.9,
        'n_bg': 1.0,
        'n_emitter': 2.414
    }
    
    print(f"Configuration:")
    print(f"  Results file: {CONFIG['results_file']}")
    print(f"  Wavelength: {CONFIG['wavelength_um']} µm")
    print(f"  Thickness: {CONFIG['thickness_um']} µm")
    print(f"  NA: {CONFIG['NA']}")
    print(f"  Background index: {CONFIG['n_bg']}")
    print()
    
    results = {}
    
    # ========================================================================
    # 1. Q-FACTOR ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("📊 Q-FACTOR ANALYSIS")
    print("=" * 80)
    
    try:
        q_results = analyze_q_factor(
            data_path=CONFIG['results_file'],
            save_results=True
        )
        results['q_factors'] = q_results
        print("✅ Q-factor analysis completed successfully!")
        print(f"  Q-factors: {q_results['q_factors']}")
        if 'decay_times_s' in q_results:
            decay_times_ps = np.array(q_results['decay_times_s']) * 1e12  # Convert to ps
            print(f"  Decay times: {decay_times_ps} ps")
        
        # Calculate resonance wavelength from Q-factor analysis
        resonance_frequency_hz = q_results['frequencies_hz'][0]  # Use first mode
        resonance_wavelength_um = C0 / resonance_frequency_hz * 1e6  # Convert to micrometers
        print(f"  Resonance frequency: {resonance_frequency_hz/1e12:.6f} THz")
        print(f"  Resonance wavelength: {resonance_wavelength_um:.6f} µm")
        print(f"  Original wavelength: {CONFIG['wavelength_um']:.6f} µm")
        print(f"  Wavelength difference: {abs(resonance_wavelength_um - CONFIG['wavelength_um']):.6f} µm")
        
        # Update CONFIG to use resonance wavelength for subsequent analyses
        CONFIG['resonance_wavelength_um'] = resonance_wavelength_um
        print(f"  ✓ Updated CONFIG to use resonance wavelength: {resonance_wavelength_um:.6f} µm")
        
    except Exception as e:
        print(f"❌ Q-factor analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # ========================================================================
    # 2. MODE VOLUME ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📐 MODE VOLUME ANALYSIS")
    print("=" * 80)
    
    try:
        Q = q_results['q_factors'][0]  # Use first mode's Q-factor
        mv_results = analyze_mode_volume(
            data_path=CONFIG['results_file'],
            thickness_um=CONFIG['thickness_um'],
            wavelength_um=CONFIG['resonance_wavelength_um'],  # Use resonance wavelength
            Q=Q,
            n_emitter=CONFIG['n_emitter'],
            save_results=True,
            create_plots=True
        )
        results['mode_volume'] = mv_results
        print("✅ Mode volume analysis completed successfully!")
        print(f"  Effective mode volume: {mv_results['effective_mode_volume_um3']:.3f} µm³")
        if mv_results['purcell_factor'] is not None:
            print(f"  Purcell factor: {mv_results['purcell_factor']:.2f}")
        
    except Exception as e:
        print(f"❌ Mode volume analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 3. POLARIZATION ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔄 POLARIZATION ANALYSIS")
    print("=" * 80)
    
    try:
        pol_results = analyze_polarization(
            data_path=CONFIG['results_file'],
            wavelength_um=CONFIG['resonance_wavelength_um'],  # Use resonance wavelength
            save_results=True,
            create_plots=True
        )
        results['polarization'] = pol_results
        print("✅ Polarization analysis completed successfully!")
        print(f"  DoLP: {pol_results.dolp:.3f}")
        print(f"  DoCP: {pol_results.docp:.3f}")
        print(f"  DoP: {pol_results.dop:.3f}")
        print(f"  Psi: {pol_results.psi:.1f}°")
        print(f"  Chi: {pol_results.chi:.1f}°")
        
    except Exception as e:
        print(f"❌ Polarization analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 4. NEAR-FIELD ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔬 NEAR-FIELD ANALYSIS")
    print("=" * 80)
    
    try:
        nf_results = analyze_nearfield(
            data_path=CONFIG['results_file'],
            monitor_name="field_near",
            wavelength_um=CONFIG['resonance_wavelength_um'],  # Use resonance wavelength
            save_results=True,
            create_plots=True
        )
        results['nearfield'] = nf_results
        print("✅ Near-field analysis completed successfully!")
        print(f"  Confinement area: {nf_results['confinement']['confinement_area_um2']:.3f} µm²")
        print(f"  Mode area: {nf_results['mode_parameters']['mode_area_um2']:.3f} µm²")
        print(f"  Field uniformity: {nf_results['quality_metrics']['field_uniformity']:.3f}")
        print(f"  Field concentration: {nf_results['quality_metrics']['field_concentration']:.3f}")
        
    except Exception as e:
        print(f"❌ Near-field analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 5. FAR-FIELD ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("🌐 FAR-FIELD ANALYSIS")
    print("=" * 80)
    
    try:
        ff_results = analyze_farfield(
            data_path=CONFIG['results_file'],
            monitor_names=['farfield_kspace', 'farfield_angles'],  # Exclude cartesian
            wavelength_um=CONFIG['resonance_wavelength_um'],  # Use resonance wavelength
            NA=CONFIG['NA'],
            n_bg=CONFIG['n_bg'],
            save_results=True,
            create_plots=True
        )
        results['farfield'] = ff_results
        print("✅ Far-field analysis completed successfully!")
        
        # Print far-field results
        for monitor_name, monitor_results in ff_results.items():
            if monitor_name != 'collection_efficiency' and 'radiation_metrics' in monitor_results:
                metrics = monitor_results['radiation_metrics']
                print(f"  {monitor_name}:")
                if 'directivity' in metrics:
                    print(f"    Directivity: {metrics['directivity']:.2f}")
                if 'beam_width_deg' in metrics:
                    print(f"    Beam width: {metrics['beam_width_deg']:.1f}°")
                if 'NA_effective' in metrics:
                    print(f"    Effective NA: {metrics['NA_effective']:.3f}")
        
        # Print collection efficiency
        if 'collection_efficiency' in ff_results:
            ce = ff_results['collection_efficiency']
            print(f"  Collection efficiency: {ce['collection_efficiency']:.3f} ({ce['collection_efficiency']*100:.1f}%)")
            print(f"  Collection angle: ±{ce['theta_max_deg']:.1f}°")
        
    except Exception as e:
        print(f"❌ Far-field analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 6. COLLECTION EFFICIENCY ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 COLLECTION EFFICIENCY ANALYSIS")
    print("=" * 80)
    
    try:
        ce_results = analyze_collection_efficiency(
            data_path=CONFIG['results_file'],
            monitor_names=['farfield_kspace', 'farfield_angles'],  # Exclude cartesian
            wavelength_um=CONFIG['resonance_wavelength_um'],  # Use resonance wavelength
            NA=CONFIG['NA'],
            n_bg=CONFIG['n_bg'],
            save_results=True,
            create_plots=False  # Plots now handled by far-field analysis
        )
        results['collection_efficiency'] = ce_results
        print("✅ Collection efficiency analysis completed successfully!")
        
        # Print collection efficiency results
        for monitor_name, monitor_results in ce_results.items():
            if monitor_name != 'overall' and 'collection_efficiency' in monitor_results:
                eff = monitor_results['collection_efficiency']
                print(f"  {monitor_name}: {eff:.3f} ({eff*100:.1f}%)")
        
        if 'overall' in ce_results:
            overall = ce_results['overall']
            print(f"\n  Overall collection efficiency: {overall['overall_efficiency']:.3f} ({overall['overall_efficiency']*100:.1f}%)")
            print(f"  Mean efficiency: {overall['mean_efficiency']:.3f} ± {overall['std_efficiency']:.3f}")
            print(f"  Range: {overall['min_efficiency']:.3f} to {overall['max_efficiency']:.3f}")
        
    except Exception as e:
        print(f"❌ Collection efficiency analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("Completed analyses:")
    for analysis_name in results.keys():
        print(f"  ✅ {analysis_name}")
    
    print(f"\nTotal analyses completed: {len(results)}/6")
    
    if len(results) == 6:
        print("🎉 All analyses completed successfully!")
    else:
        print(f"⚠️  {6 - len(results)} analysis(es) failed")
    
    return results

if __name__ == "__main__":
    # Run the complete analysis
    results = run_complete_analysis()
    
    print("\n" + "=" * 80)
    print("🏁 ANALYSIS COMPLETE")
    print("=" * 80)

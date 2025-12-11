#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 17:25:15 2025

@author: gclark
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# === USER INPUT ===
data_dir = '.'
grid_shape = (7, 16)  # (rows, cols)
filename_prefix_length = 4  # Length of the part like '4_10'
# ===================

# Output maps
avg_center_map = np.full(grid_shape, np.nan)
avg_fwhm_map = np.full(grid_shape, np.nan)
dev_centers= np.full(grid_shape, np.nan)
dev_fwhms= np.full(grid_shape, np.nan)
spread_map = np.full(grid_shape, np.nan)
fill_map = np.full(grid_shape, np.nan)
centers=[0] 
Qs=[0]
fills=[0]

for fname in os.listdir(data_dir):
    if not fname.endswith('.csv') or '_' not in fname[:filename_prefix_length]:
        continue

    # Extract row and col from filename like "4_10_peaks.csv"
    try:
        row_str, col_str = fname[:filename_prefix_length].split('_')
        row = int(row_str)
        col = int(col_str)
    except Exception:
        print(f"Skipping {fname}: couldn't parse position.")
        continue

    if not (0 <= row < grid_shape[0]) or not (0 <= col < grid_shape[1]):
        print(f"Skipping {fname}: position out of grid bounds.")
        continue

    # Load data
    full_path = os.path.join(data_dir, fname)
    try:
        data = np.loadtxt(full_path, delimiter=',', skiprows=1)
        if data.size == 0:
            raise ValueError("Empty file")
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)  # for single-entry files
    except Exception:
        print(f"Warning: {fname} is empty or unreadable. Setting values to zero.")
        avg_center_map[row, col] = 630e-9 
        avg_fwhm_map[row, col] = 0
        spread_map[row, col] = 0
        continue


    except Exception:
        print(f"Skipping {fname}: error reading file.")
        continue

    if data.shape[1] < 2:
        print(f"Skipping {fname}: expected 2 columns.")
        continue

    center_wavelengths = data[:, 0]
    fwhms = data[:, 0]/data[:,1]
    centers=np.hstack((centers,center_wavelengths))
    Qs=np.hstack((Qs,fwhms))
    
    # Compute statistics
    avg_center = np.mean(center_wavelengths)
    avg_fwhm = np.mean(fwhms)
    
    #get standard deviations....
    devs=(center_wavelengths-avg_center)
    devs=devs**2
    var=np.sum(devs)/len(center_wavelengths)
    dev_centers[row,col]=np.sqrt(var)
    
    devs_fwhm=(fwhms-avg_fwhm)**2
    var_fwhm=np.sum(devs_fwhm)/len(fwhms)
    spread = np.ptp(center_wavelengths)  # peak-to-peak = max - min
    fill_factor = len(center_wavelengths)/15
    fills.append(fill_factor)
    print("dev center is...", fill_factor)
    dev_fwhms[row,col]=np.sqrt(var_fwhm)

    avg_center_map[row, col] = avg_center
    avg_fwhm_map[row, col] = avg_fwhm
    spread_map[row, col] = spread
    fill_map[row,col]=fill_factor

# === Plot heatmaps ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(dev_centers, cmap='GnBu', origin='lower', aspect='auto', alpha=0.95)
plt.colorbar(label='Center Wavelength (nm)')
plt.title('Center Wavelength per Scan')
# ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
# ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
# ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

plt.subplot(1, 3, 2)
plt.imshow(dev_fwhms, cmap='plasma', origin='lower', aspect='auto', alpha=0.95)
plt.colorbar(label='Dev FWHM (nm)')
plt.title('Dev Q per Scan')
# ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
# ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
# ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

plt.subplot(1, 3, 3)
plt.imshow(fill_map, cmap='cividis', origin='lower', aspect='auto')
plt.colorbar(label='Fill fraction')
plt.title('Fraction of nanobeams with a cavity')

plt.tight_layout()
plt.show()

#plot histograms
bins = np.arange(6.05e-7, 6.55e-7 ,0.02e-7)
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.hist(centers, bins=bins,color='skyblue', alpha=0.95, edgecolor='black', linewidth=1)
plt.xlabel('center wavelength (nm)')
plt.ylabel('count')

plt.subplot(1,3,2)
plt.hist(Qs, bins=20, color='indigo', alpha=0.7, edgecolor='black', linewidth=1)  
plt.xlabel('Q factor')
plt.ylabel('count')

plt.subplot(1,3,3)
plt.hist(fills, bins=15, color='midnightblue', alpha=0.7, edgecolor='black', linewidth=1)  
plt.xlabel('Fill factor')
plt.ylabel('count')

plt.tight_layout()
plt.show()




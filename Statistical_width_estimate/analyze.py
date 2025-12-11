#!/usr/bin/env python3
"""
Complete analysis script for SEM images - processes all images and saves results.
Combines all analysis functions from hole_analysis.py with batch processing.
"""

import numpy as np
import cv2
import json
import csv
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DEVICE PARAMETERS
# ============================================================================
SCALE_PX_PER_NM = 0.2951
RAIL_SEPARATION_MIN_PX = 74
RAIL_SEPARATION_MAX_PX = 118
HOLE_RADIUS_MIN_PX = 6
HOLE_RADIUS_MAX_PX = 18

# Analysis parameters
RANSAC_ITERATIONS = 1000
INLIER_THRESHOLD = 4.0  # Increased for thick rails
STRIP_MASK_FACTOR = 0.45
HOLE_MIN_SIZE = 50  # Increased minimum size
HOLE_MAX_SIZE = 2000  # Increased maximum size
CIRCLE_RESIDUAL_THRESHOLD = 2.0  # More lenient
CENTER_STRIP_FACTOR = 0.25  # Very restrictive - holes must be very close to midline (from legacy)
RAIL_DISTANCE_FACTOR = 0.3  # More restrictive - holes must be further from rails (from legacy)

# Hole filtering parameters - IMPROVED to reduce spurious holes (from legacy_code_1)
MIDLINE_TIGHT_FRAC = 0.08  # More restrictive to reduce spurious holes (from legacy)
MIN_HOLE_SPACING_NM = 110  # More restrictive to remove spurious holes (from legacy)
RADIUS_Z_MAX = 8.0  # More permissive for radius consistency (from legacy)

# ============================================================================
# CORE ANALYSIS FUNCTIONS (from hole_analysis.py)
# ============================================================================

def estimate_theta_structure_tensor(img, sigma=1.2, rho=3.0):
    """
    Global tangent angle from the structure tensor.
    sigma: pre-smoothing for gradients, rho: smoothing for tensor fields
    """
    # Scharr grads (better rotational accuracy than Sobel)
    Gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    Gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)

    if sigma > 0:
        Gx = cv2.GaussianBlur(Gx, (0,0), sigma)
        Gy = cv2.GaussianBlur(Gy, (0,0), sigma)

    Jxx = Gx*Gx
    Jxy = Gx*Gy
    Jyy = Gy*Gy

    if rho > 0:
        Jxx = cv2.GaussianBlur(Jxx, (0,0), rho)
        Jxy = cv2.GaussianBlur(Jxy, (0,0), rho)
        Jyy = cv2.GaussianBlur(Jyy, (0,0), rho)

    # dominant tangent direction t = eigenvector of largest eigenvalue
    # angle of t:
    theta = 0.5 * np.arctan2(2*Jxy.mean(), (Jxx.mean() - Jyy.mean()))
    return float(theta)

def load_and_preprocess_image(image_path):
    """Load image and create edge map with simple, reliable Canny."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    # Simple preprocessing
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    denoised = cv2.fastNlMeansDenoising(bilateral, None, h=8, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised, (0,0), 0.8)
    
    # Very permissive Canny thresholds
    low = 10   # Very low threshold to detect more edges
    high = 60  # Very low high threshold to be very permissive
    edges = cv2.Canny(blurred, low, high, L2gradient=True)
    
    # Light closing to connect rail edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    return img, edges

def get_edge_points(edges):
    """Extract edge points from binary edge image."""
    ys, xs = np.where(edges > 0)
    return np.column_stack((xs, ys))

def keep_elongated_edges(edges):
    """Filter edges to keep only elongated components (rails)."""
    num, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros_like(edges)
    
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 8:  # Balanced approach - not too strict
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect >= 1.5:  # Balanced approach - keep elongated pieces
            keep[lab == i] = 255
    
    return keep

def sample_along_normal_all_pixels(img, theta, frac=0.10):
    """
    Return (s, g_n) for the top frac of pixels by |g_n| whose gradient
    direction is within ±20° of the normal. Useful when Canny is unreliable.
    """
    m = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0); gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    g_n = gx*m[0] + gy*m[1]
    ang = np.arctan2(gy, gx)
    normal = theta + np.pi/2
    def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
    mask_dir = (np.abs(wrap(ang - normal)) < np.deg2rad(20)).astype(np.uint8)

    h, w = img.shape
    yy, xx = np.mgrid[0:h,0:w].astype(np.float32)
    s = xx*m[0] + yy*m[1]

    g = g_n[mask_dir>0].ravel()
    s = s[mask_dir>0].ravel()

    # keep strongest |g| fraction
    k = max(5000, int(len(g)*frac))
    idx = np.argpartition(np.abs(g), -k)[-k:]
    return s[idx], g[idx]

def fit_parallel_rails_gradient(img, edges_for_rails):
    """
    Rail detection by PCA orientation + signed-normal-gradient pairing + refinement.
    - img: grayscale image (uint8)
    - edges_for_rails: binary edge map filtered to elongated components (use keep_elongated_edges)
    Returns: theta, d1, d2, inlier_mask, score
    """
    pts = get_edge_points(edges_for_rails)
    if len(pts) < 20:
        return None, None, None, None, 0
    
    # --- Stable orientation via PCA (t = tangent, m = normal)
    C = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(C)
    t = vecs[:, np.argmax(vals)]
    theta = np.arctan2(t[1], t[0])
    m = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)  # unit normal

    # --- Gradients and their projection along the normal (signed)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    g_n = gx * m[0] + gy * m[1]  # signed gradient along normal

    # Use only elongated-edge pixels as samples (suppresses holes)
    ys, xs = np.where(edges_for_rails > 0)
    s = xs * m[0] + ys * m[1]  # normal coordinates of sample pixels
    g_samples = g_n[ys, xs]

    if len(s) < 20:
        # Fallback: use gradient-only sampling when Canny is unreliable
        print(f"  Using gradient fallback (only {len(s)} edge points)")
        s, g_samples = sample_along_normal_all_pixels(img, theta, frac=0.10)
        if len(s) < 20:
            return None, None, None, None, 0
    
    # --- Build two histograms: positive and negative responses along the normal
    s_min, s_max = float(np.min(s)), float(np.max(s))
    nb = 2048
    bins = np.linspace(s_min, s_max, nb + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    pos_w = np.maximum(g_samples, 0.0)
    neg_w = np.maximum(-g_samples, 0.0)

    Hpos, _ = np.histogram(s, bins=bins, weights=pos_w)
    Hneg, _ = np.histogram(s, bins=bins, weights=neg_w)

    # Smooth
    Hpos = cv2.GaussianBlur(Hpos.astype(np.float32)[None, :], (0, 0), 3.5).ravel()
    Hneg = cv2.GaussianBlur(Hneg.astype(np.float32)[None, :], (0, 0), 3.5).ravel()

    # --- Peak picking with polarity constraint (inner rails have opposite sign)
    def find_local_maxima(h):
        return np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1

    pks_pos = find_local_maxima(Hpos)
    pks_neg = find_local_maxima(Hneg)

    # Option A: Choose outermost opposite-polarity peaks (outer rails)
    # pick significant peaks (avoid tiny ripples)
    tau_pos = 0.35 * float(Hpos.max())
    tau_neg = 0.35 * float(Hneg.max())
    cand_pos = [(centers[i], Hpos[i]) for i in pks_pos if Hpos[i] >= tau_pos]
    cand_neg = [(centers[i], Hneg[i]) for i in pks_neg if Hneg[i] >= tau_neg]

    d1 = d2 = None
    if cand_pos and cand_neg:
        # outermost pair by coordinate (max sep) -> outer rails
        s_pos_right, _ = max(cand_pos, key=lambda p: p[0])  # largest s among + peaks
        s_neg_left,  _ = min(cand_neg, key=lambda p: p[0])  # smallest s among - peaks
        if s_pos_right > s_neg_left:
            sep = s_pos_right - s_neg_left
            if RAIL_SEPARATION_MIN_PX <= sep <= RAIL_SEPARATION_MAX_PX:
                d1, d2 = s_neg_left, s_pos_right

    # fallback to previous candidate logic if that failed to set d1,d2
    if d1 is None or d2 is None:
        candidates = []
        for i in pks_pos:
            for j in pks_neg:
                # try both orders; distance is absolute
                sep = abs(centers[j] - centers[i])
                if RAIL_SEPARATION_MIN_PX <= sep <= RAIL_SEPARATION_MAX_PX:
                    score = Hpos[i] + Hneg[j]
                    # Prefer peaks that are both strong and narrow (sharper rails)
                    # Add a small curvature term by looking at neighbors
                    sharp = (Hpos[i] - 0.5*(Hpos[i-1] + Hpos[i+1])) + (Hneg[j] - 0.5*(Hneg[j-1] + Hneg[j+1]))
                    candidates.append((score + 0.25*sharp, centers[i], centers[j]))

        if not candidates:
            # Fallback: use two strongest overall peaks from (Hpos+Hneg), no polarity
            H = Hpos + Hneg
            peaks = find_local_maxima(H)
            if len(peaks) < 2:
                return None, None, None, None, 0
            # choose pair with valid separation and max sum
            pairs = []
            for a in peaks:
                for b in peaks:
                    if b <= a: continue
                    sep = abs(centers[b] - centers[a])
                    if RAIL_SEPARATION_MIN_PX <= sep <= RAIL_SEPARATION_MAX_PX:
                        pairs.append((H[a] + H[b], centers[a], centers[b]))
            if not pairs:
                return None, None, None, None, 0
            _, d1, d2 = max(pairs, key=lambda x: x[0])
        else:
            _, d1, d2 = max(candidates, key=lambda x: x[0])

    # --- Sub-pixel refinement: maximize signed response in a small window
    def refine_d(d_init, sign=+1):
        # sample a fine grid around d_init and pick argmax of signed histogram
        win = 3.5  # pixels in normal coords
        grid = np.linspace(d_init - win, d_init + win, 41)  # ~0.175 px step
        # accumulate signed responses for points close to each candidate d
        best_d, best_val = d_init, -1e9
        for dv in grid:
            mask = np.abs(s - dv) < 2.0  # band width
            val = np.sum((g_samples if sign > 0 else -g_samples)[mask])
            if val > best_val:
                best_val, best_d = val, dv
        return best_d

    # Determine which is positive/negative by local sign dominance
    # average sign near each d
    sign1 = np.sign(np.sum(g_samples[np.abs(s - d1) < 2.0]) + 1e-6)
    sign2 = np.sign(np.sum(g_samples[np.abs(s - d2) < 2.0]) + 1e-6)

    d1 = refine_d(d1, +1 if sign1 >= 0 else -1)
    d2 = refine_d(d2, +1 if sign2 >= 0 else -1)

    # --- Inliers and score
    s_all = xs * m[0] + ys * m[1]  # normal coords of the elongated-edge set
    inliers = (np.abs(s_all - d1) < 3.0) | (np.abs(s_all - d2) < 3.0)
    score = int(inliers.sum())

    # Ensure d1 < d2 (ordering)
    if d1 > d2:
        d1, d2 = d2, d1

    return float(theta), float(d1), float(d2), inliers, score

def _fit_d_from_points(m, pts, w):
    """Fit d from points using weighted 1D mean of s = m·x."""
    s = pts @ m
    w = np.clip(w, 1e-6, None)
    return float(np.sum(w * s) / np.sum(w))

def _fit_theta_from_points(pts, w):
    """Fit theta from points using weighted PCA to get tangent."""
    mu = np.sum(pts * w[:, None], axis=0) / np.sum(w)
    X = pts - mu
    # weighted covariance
    C = (X.T * w) @ X / np.sum(w)
    vals, vecs = np.linalg.eigh(C)
    t = vecs[:, np.argmax(vals)]  # tangent
    return float(np.arctan2(t[1], t[0]))

def refine_rails_iterative(img, theta_init, d1_init, d2_init, iters=3, band_px=4.0):
    """
    Snap each rail to its sub-pixel edge using weighted TLS on gradient-weighted points.
    Returns refined (theta, d1, d2).
    """
    theta = float(theta_init)
    d1, d2 = float(d1_init), float(d2_init)

    # precompute gradients
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pts_all = np.column_stack((xx.ravel(), yy.ravel()))

    for _ in range(iters):
        m = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)  # normal
        t = np.array([ np.cos(theta), np.sin(theta)], dtype=np.float32)  # tangent

        # signed normal gradient as edge strength (weight)
        g_n = (gx * m[0] + gy * m[1]).ravel()

        # normal coordinates of all pixels
        s = pts_all @ m

        # narrow bands for each rail
        band1 = np.abs(s - d1) < band_px
        band2 = np.abs(s - d2) < band_px

        # collect points + weights; keep only strong gradients of the expected sign
        # determine local sign near each rail
        sign1 = np.sign(np.sum(g_n[np.abs(s - d1) < 2.0]) + 1e-6)
        sign2 = np.sign(np.sum(g_n[np.abs(s - d2) < 2.0]) + 1e-6)

        pts1 = pts_all[band1]
        w1   = (sign1 * g_n[band1]).clip(min=0.0)
        pts2 = pts_all[band2]
        w2   = (sign2 * g_n[band2]).clip(min=0.0)

        # if weights are too sparse, fall back to un-signed magnitude
        if np.sum(w1 > 0) < 200: w1 = np.abs(g_n[band1])
        if np.sum(w2 > 0) < 200: w2 = np.abs(g_n[band2])

        # robust downweight: IRLS (Huber-like)
        def irls_weights(w):
            if len(w) == 0: return w
            mval = np.median(w)
            mad  = np.median(np.abs(w - mval)) + 1e-6
            r = np.abs(w - mval) / (6.0*mad)     # 6*MAD ~ outlier cutoff
            return w / (1.0 + r*r)
        w1 = irls_weights(w1); w2 = irls_weights(w2)

        # Fit theta for each rail from its points (weighted PCA), then average → enforce parallelism
        theta1 = _fit_theta_from_points(pts1, w1) if len(pts1) > 50 else theta
        theta2 = _fit_theta_from_points(pts2, w2) if len(pts2) > 50 else theta
        # bring angles into same branch
        if np.abs(theta1 - theta2) > np.pi/2:
            if theta1 > theta2: theta1 -= np.pi
            else:               theta2 -= np.pi
        theta = (theta1 + theta2) / 2.0

        # Recompute normal with the averaged theta, then refit d1,d2 as weighted means of m·x
        m = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        d1 = _fit_d_from_points(m, pts1, w1) if len(pts1) > 50 else d1
        d2 = _fit_d_from_points(m, pts2, w2) if len(pts2) > 50 else d2

        # order
        if d1 > d2: d1, d2 = d2, d1

    return theta, d1, d2

def estimate_theta_from_holes(holes):
    """Get stable angle from hole centers using PCA."""
    if not holes or len(holes) < 3:
        return None
    P = np.array([h['center'] for h in holes], dtype=np.float32)
    P -= P.mean(axis=0, keepdims=True)
    # PCA tangent = first eigenvector
    C = (P.T @ P) / len(P)
    vals, vecs = np.linalg.eigh(C)
    t = vecs[:, np.argmax(vals)]
    return float(np.arctan2(t[1], t[0]))  # tangent angle

def _local_maxima_1d(h):
    """Find local maxima in 1D array."""
    return np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1

def find_outer_and_inner_rails(img, edges_for_rails, theta,
                               inner_min=RAIL_SEPARATION_MIN_PX,
                               inner_max=RAIL_SEPARATION_MAX_PX,
                               peak_frac=0.30, outer_margin_px=8):
    """
    Returns:
        (d_out_lo, d_out_hi), (d_in_lo, d_in_hi)  # inner can be None if absent
    Coordinates are along the normal m for the provided theta.
    """
    # normal
    m = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    # sample only elongated-edge pixels (suppresses dots/holes)
    ys, xs = np.where(edges_for_rails > 0)
    if len(xs) < 20:
        return None, None

    # gradients projected on normal (signed)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    g_samples = gx[ys, xs] * m[0] + gy[ys, xs] * m[1]

    # normal coordinates of samples
    s = xs.astype(np.float32) * m[0] + ys.astype(np.float32) * m[1]

    # histograms of positive/negative signed responses
    nb = 2048
    smin, smax = float(s.min()), float(s.max())
    bins = np.linspace(smin, smax, nb+1); centers = 0.5*(bins[:-1]+bins[1:])
    Hpos, _ = np.histogram(s, bins=bins, weights=np.maximum(g_samples, 0.0))
    Hneg, _ = np.histogram(s, bins=bins, weights=np.maximum(-g_samples, 0.0))
    Hpos = cv2.GaussianBlur(Hpos.astype(np.float32)[None,:], (0,0), 3.5).ravel()
    Hneg = cv2.GaussianBlur(Hneg.astype(np.float32)[None,:], (0,0), 3.5).ravel()

    # candidate peaks (reject tiny ripples)
    pks_pos = [i for i in _local_maxima_1d(Hpos) if Hpos[i] >= peak_frac*float(Hpos.max()+1e-6)]
    pks_neg = [i for i in _local_maxima_1d(Hneg) if Hneg[i] >= peak_frac*float(Hneg.max()+1e-6)]
    if not pks_pos or not pks_neg:
        return None, None

    # make all (+, −) pairs with their geometry and strength
    pairs = []
    for ip in pks_pos:
        for ineg in pks_neg:
            d_lo, d_hi = sorted([centers[ip], centers[ineg]])
            sep = d_hi - d_lo
            strength = Hpos[ip] + Hneg[ineg]
            pairs.append((sep, strength, d_lo, d_hi, ip, ineg))
    if not pairs:
        return None, None

    # OUTER = farthest-apart opposite-sign pair (allow any sep >= inner_min)
    outer = max([p for p in pairs if p[0] >= inner_min], key=lambda x: (x[0], x[1]), default=None)
    if outer is None:
        return None, None
    _, _, d_out_lo, d_out_hi, _, _ = outer

    # INNER = strongest pair inside the outer interval with sep within inner band
    inner_candidates = [p for p in pairs
                        if inner_min <= p[0] <= inner_max
                        and (d_out_lo + outer_margin_px) <= p[2]   # inner low inside outer by a margin
                        and p[3] <= (d_out_hi - outer_margin_px)]  # inner high inside outer by a margin
    inner = max(inner_candidates, key=lambda x: (x[1], -abs(0.5*(x[2]+x[3]) - 0.5*(d_out_lo+d_out_hi))),
                default=None)

    d_in_lo = d_in_hi = None
    if inner is not None:
        _, _, d_in_lo, d_in_hi, _, _ = inner

    return (float(d_out_lo), float(d_out_hi)), (None if inner is None else (float(d_in_lo), float(d_in_hi)))

def create_strip_mask(edge_points, theta, d1, d2):
    """Create mask for points between the rails."""
    m = np.array([-np.sin(theta), np.cos(theta)])  # unit NORMAL (correct)
    dm = (d1 + d2) / 2
    rail_separation = abs(d2 - d1)
    
    distances_to_midline = np.abs(np.dot(edge_points, m) - dm)
    strip_mask = distances_to_midline < STRIP_MASK_FACTOR * rail_separation
    
    return strip_mask

def _tm(theta):
    t = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    m = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    return t, m

def filter_holes_custom_restrictive(holes, theta, d1, d2):
    """Custom filtering with very restrictive parameters to remove off-axis holes (from legacy_code_1)."""
    if not holes:
        return holes
    
    # Calculate midline
    m = np.array([-np.sin(theta), np.cos(theta)])
    dm = (d1 + d2) / 2
    rail_separation = abs(d2 - d1)
    tight_tol = MIDLINE_TIGHT_FRAC * rail_separation
    
    # 1. Filter by midline distance (very restrictive)
    filtered_holes = []
    for hole in holes:
        center = hole['center']
        center_off = abs(np.dot(center, m) - dm)
        if center_off <= tight_tol:
            filtered_holes.append(hole)
    
    if len(filtered_holes) < 3:
        return filtered_holes
    
    # 2. Filter by radius range (30-70 nm) - more restrictive
    radius_filtered = []
    for hole in filtered_holes:
        radius_nm = hole['radius_px'] / SCALE_PX_PER_NM
        if 30.0 <= radius_nm <= 70.0:
            radius_filtered.append(hole)
    
    if len(radius_filtered) < 3:
        return radius_filtered
    
    # 3. Filter by radius consistency
    radii = np.array([h['radius_px'] for h in radius_filtered], dtype=np.float32)
    med = float(np.median(radii))
    mad = float(np.median(np.abs(radii - med)) + 1e-6)
    
    # Make radius filtering very permissive to include smaller holes
    # If MAD is very small, use a percentage-based range instead
    if mad < 0.5:  # If all radii are very similar
        lo, hi = med * 0.7, med * 1.5  # Allow 30% variation in both directions
    else:
        lo, hi = med - 12.0*mad, med + RADIUS_Z_MAX*mad
    
    consistency_filtered = []
    for hole in radius_filtered:
        radius_px = hole['radius_px']
        if lo <= radius_px <= hi:
            consistency_filtered.append(hole)
    
    if len(consistency_filtered) < 3:
        return consistency_filtered
    
    # 4. Filter by spacing - smarter approach (only keep holes not close to ANY others)
    t, _ = _tm(theta)
    centers = np.array([h['center'] for h in consistency_filtered])
    us = centers @ t
    order = np.argsort(us)
    
    # Calculate all pairwise distances
    distances = np.zeros((len(order), len(order)))
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                distances[i, j] = np.linalg.norm(centers[order[i]] - centers[order[j]]) / SCALE_PX_PER_NM
    
    # Find holes that are too close to multiple other holes (likely spurious)
    keep_indices = []
    
    for i in range(len(order)):
        current_idx = order[i]
        too_close_count = 0
        
        # Count how many other holes are too close to this one
        for j in range(len(order)):
            if i != j and distances[i, j] < MIN_HOLE_SPACING_NM:
                too_close_count += 1
        
        # Keep holes that are not too close to multiple others - more restrictive
        if too_close_count == 0:  # Only keep holes that are not close to any others
            keep_indices.append(current_idx)
    
    final_holes = [consistency_filtered[i] for i in sorted(keep_indices)]
    
    return final_holes

def detect_circular_holes_improved(edge_points, strip_mask, theta, d1, d2, img=None, img_shape=None):
    """Improved hole detection using HoughCircles and contour analysis."""
    if not np.any(strip_mask):
        return []
    
    # Get strip edge points
    strip_points = edge_points[strip_mask]
    
    if len(strip_points) == 0:
        return []
    
    # Create binary image for analysis
    min_x, min_y = np.min(strip_points, axis=0)
    max_x, max_y = np.max(strip_points, axis=0)
    
    h = int(max_y - min_y) + 20
    w = int(max_x - min_x) + 20
    binary_img = np.zeros((h, w), dtype=np.uint8)
    
    # Mark edge points
    for x, y in strip_points:
        img_x = int(x - min_x + 10)
        img_y = int(y - min_y + 10)
        if 0 <= img_x < w and 0 <= img_y < h:
            binary_img[img_y, img_x] = 255
    
    holes = []
    m = np.array([-np.sin(theta), np.cos(theta)])  # unit NORMAL (correct)
    dm = (d1 + d2) / 2
    rail_separation = abs(d2 - d1)
    
    # Method 1: HoughCircles
    try:
        # Blur the sparse edge image a bit and lower the vote threshold
        work = cv2.GaussianBlur(binary_img, (0,0), 1.2)
        circles = cv2.HoughCircles(
            work,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=12,
            param1=60,
            param2=18,  # lower param2 finds more candidates
            minRadius=HOLE_RADIUS_MIN_PX,
            maxRadius=HOLE_RADIUS_MAX_PX
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Convert back to original coordinates
                center_x = x + min_x - 10
                center_y = y + min_y - 10
                center = np.array([center_x, center_y])
                
                # Check constraints
                center_dist_to_midline = abs(np.dot(center, m) - dm)
                if center_dist_to_midline > CENTER_STRIP_FACTOR * rail_separation:
                    continue
                
                dist_to_rail1 = abs(np.dot(center, m) - d1)
                dist_to_rail2 = abs(np.dot(center, m) - d2)
                min_dist_to_rail = min(dist_to_rail1, dist_to_rail2)
                if min_dist_to_rail <= RAIL_DISTANCE_FACTOR * rail_separation:
                    continue
                
                radius_nm = r / SCALE_PX_PER_NM
                
                holes.append({
                    'center': center,
                    'radius_px': r,
                    'radius_nm': radius_nm,
                    'residual': 0,  # HoughCircles doesn't provide residual
                    'num_points': 0,
                    'method': 'HoughCircles'
                })
    except Exception as e:
        print(f"HoughCircles failed: {e}")
    
    # Fallback: Try grayscale strip if HoughCircles found too few
    if len(holes) < 3 and img is not None:
        try:
            # Create grayscale strip from original image
            h, w = img.shape[:2]
            grayscale_strip = np.zeros_like(img)
            
            # Create strip region
            m = np.array([-np.sin(theta), np.cos(theta)])
            dm = (d1 + d2) / 2
            rail_separation = abs(d2 - d1)
            strip_width = CENTER_STRIP_FACTOR * rail_separation
            
            for y in range(h):
                for x in range(w):
                    center_dist_to_midline = abs(np.dot([x, y], m) - dm)
                    if center_dist_to_midline <= strip_width:
                        grayscale_strip[y, x] = img[y, x]
            
            # Apply CLAHE to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if len(grayscale_strip.shape) == 3:
                grayscale_strip = cv2.cvtColor(grayscale_strip, cv2.COLOR_BGR2GRAY)
            grayscale_strip = clahe.apply(grayscale_strip)
            
            # Try HoughCircles on grayscale strip with relaxed parameters
            circles_gray = cv2.HoughCircles(
                grayscale_strip, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=8,
                param1=50, param2=12,  # More relaxed
                minRadius=HOLE_RADIUS_MIN_PX, maxRadius=HOLE_RADIUS_MAX_PX
            )
            
            if circles_gray is not None:
                circles_gray = np.round(circles_gray[0, :]).astype("int")
                for (x, y, r) in circles_gray:
                    center = np.array([x, y])
                    
                    # Check if center is within strip
                    center_dist_to_midline = abs(np.dot(center, m) - dm)
                    
                    if center_dist_to_midline <= CENTER_STRIP_FACTOR * rail_separation:
                        radius_nm = r / SCALE_PX_PER_NM
                        holes.append({
                            'center': center,
                            'radius_px': r,
                            'radius_nm': radius_nm,
                            'residual': 0,
                            'num_points': 0,
                            'method': 'HoughCircles_Gray'
                        })
        except Exception as e:
            print(f"Grayscale fallback failed: {e}")
    
    # Method 2: Contour analysis
    try:
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < HOLE_MIN_SIZE or area > HOLE_MAX_SIZE:
                continue
            
            # Fit circle to contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if not (HOLE_RADIUS_MIN_PX <= radius <= HOLE_RADIUS_MAX_PX):
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.3:  # Not circular enough
                continue
            
            # Convert back to original coordinates
            center_x = x + min_x - 10
            center_y = y + min_y - 10
            center = np.array([center_x, center_y])
            
            # Check constraints
            center_dist_to_midline = abs(np.dot(center, m) - dm)
            if center_dist_to_midline > CENTER_STRIP_FACTOR * rail_separation:
                continue
            
            dist_to_rail1 = abs(np.dot(center, m) - d1)
            dist_to_rail2 = abs(np.dot(center, m) - d2)
            min_dist_to_rail = min(dist_to_rail1, dist_to_rail2)
            if min_dist_to_rail <= RAIL_DISTANCE_FACTOR * rail_separation:
                continue
            
            radius_nm = radius / SCALE_PX_PER_NM
            
            holes.append({
                'center': center,
                'radius_px': radius,
                'radius_nm': radius_nm,
                'residual': 0,
                'num_points': len(contour),
                'circularity': circularity,
                'method': 'Contour'
            })
    except Exception as e:
        print(f"Contour analysis failed: {e}")
    
    # Tangent-aware NMS to handle duplicates properly
    if len(holes) > 1:
        # Project centers onto tangent for NMS
        t = np.array([np.cos(theta), np.sin(theta)])
        centers_projected = np.array([h['center'] @ t for h in holes])
        
        # Sort by tangent coordinate
        sorted_indices = np.argsort(centers_projected)
        filtered_holes = []
        
        for i, idx in enumerate(sorted_indices):
            current_hole = holes[idx]
            is_duplicate = False
            
            # Check against already accepted holes
            for accepted_hole in filtered_holes:
                # Distance in tangent direction
                t_dist = abs(centers_projected[idx] - (accepted_hole['center'] @ t))
                # Distance in normal direction  
                m = np.array([-np.sin(theta), np.cos(theta)])
                n_dist = abs(current_hole['center'] @ m - accepted_hole['center'] @ m)
                
                # Keep if far enough in either direction
                if t_dist < 15 and n_dist < 8:  # Tangent-aware threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_holes.append(current_hole)
        
        holes = filtered_holes
    
    # In-bounds/border guard
    if img_shape is not None:
        h, w = img_shape
    elif holes:
        # Estimate image bounds from hole centers
        all_x = [h['center'][0] for h in holes]
        all_y = [h['center'][1] for h in holes]
        w = int(max(all_x) + 100)  # Add margin
        h = int(max(all_y) + 100)  # Add margin
    else:
        h, w = 1000, 1000  # Default fallback
    
    margin = 20  # Keep holes away from borders
    
    final_holes = []
    for hole in holes:
        x, y = hole['center']
        if margin <= x < w - margin and margin <= y < h - margin:
            final_holes.append(hole)
    
    return final_holes

def analyze_single_image_improved(image_path):
    """Analyze a single image with improved hole detection."""
    print(f"Processing: {Path(image_path).name}")
    
    # Load and preprocess with robust angle-aware Canny
    img, edges = load_and_preprocess_image(image_path)
    if img is None:
        return None
    
    # Get all edge points for hole detection
    all_edge_points = get_edge_points(edges)
    if len(all_edge_points) < 10:
        print(f"  Too few edge points: {len(all_edge_points)}")
        return None
    
    # Filter edges to keep only elongated components for rail detection
    edges_long = keep_elongated_edges(edges)  # for rails
    rail_edge_points = get_edge_points(edges_long)
    
    # Use filtered edges for rail detection, all edges for hole detection
    if len(rail_edge_points) < 10:
        print(f"  Too few rail edge points: {len(rail_edge_points)}")
        return None
    
    # Two-pass approach: rough rails → holes → outer/inner rails
    # 1st pass – rough rails (existing fit_parallel_rails_gradient)
    theta0, d1_0, d2_0, _, _ = fit_parallel_rails_gradient(img, edges_long)
    if theta0 is None:
        print(f"  Failed to detect rails in first pass")
        return None
    theta0, d1_0, d2_0 = refine_rails_iterative(img, theta0, d1_0, d2_0, iters=2, band_px=5.0)

    # detect holes using that strip (as you already do)
    strip_mask0 = create_strip_mask(all_edge_points, theta0, d1_0, d2_0)
    holes0 = detect_circular_holes_improved(all_edge_points, strip_mask0, theta0, d1_0, d2_0, img, img.shape)

    # θ from holes (or fallback to θ0)
    theta_h = estimate_theta_from_holes(holes0) or theta0

    # find outer and inner rails simultaneously
    outer_pair, inner_pair = find_outer_and_inner_rails(
        img, edges_long, theta_h,
        inner_min=RAIL_SEPARATION_MIN_PX,
        inner_max=RAIL_SEPARATION_MAX_PX,
        peak_frac=0.30,              # 0.25–0.35 works well
        outer_margin_px=8            # push inner away from the outer walls
    )

    # fallbacks if needed
    if outer_pair is None:
        print(f"  Failed to detect outer rails, using fallback")
        outer_pair = (d1_0, d2_0)
    if inner_pair is None:
        # it may be missing in some SEMs; that's fine
        pass

    # final sub-pixel snap for outer rails
    (d1o, d2o) = outer_pair
    theta_o, d1o, d2o = refine_rails_iterative(img, theta_h, d1o, d2o, iters=3, band_px=5.0)

    # final sub-pixel snap for inner rails (if present)
    if inner_pair is not None:
        (d1i, d2i) = inner_pair
        theta_i, d1i, d2i = refine_rails_iterative(img, theta_o, d1i, d2i, iters=3, band_px=4.5)  # use θ from outer
    else:
        d1i = d2i = None

    # Use outer rails for analysis
    theta, d1, d2 = theta_o, d1o, d2o
    if theta is None:
        print(f"  Failed to detect rails")
        return None
    
    print(f"  Rails detected: θ={np.degrees(theta):.1f}°, separation={abs(d2-d1):.1f}px")
    
    # Create strip mask using all edge points (not filtered)
    strip_mask = create_strip_mask(all_edge_points, theta, d1, d2)
    
    # Detect holes with improved method using all edge points (not filtered)
    holes = detect_circular_holes_improved(all_edge_points, strip_mask, theta, d1, d2, img, img.shape)
    print(f"  Initial detection: {len(holes)} holes")
    
    # Apply custom restrictive filtering to remove spurious holes (from legacy)
    holes = filter_holes_custom_restrictive(holes, theta, d1, d2)
    print(f"  After filtering: {len(holes)} holes")
    
    # Convert to nanometers
    rail_separation_nm = abs(d2 - d1) / SCALE_PX_PER_NM
    
    # Calculate score from final inliers
    m = np.array([-np.sin(theta), np.cos(theta)])
    s_all = all_edge_points @ m
    inliers = (np.abs(s_all - d1) < 3.0) | (np.abs(s_all - d2) < 3.0)
    score = int(inliers.sum())
    
    results = {
        'image_name': Path(image_path).name,
        'theta_deg': np.degrees(theta),
        'd1_px': d1,
        'd2_px': d2,
        'rail_separation_px': abs(d2 - d1),
        'rail_separation_nm': rail_separation_nm,
        'num_inliers': score,
        'num_holes': len(holes),
        'holes': holes
    }
    
    return results

def create_overlay_improved(img, edges, results, output_path):
    """Create overlay visualization with 3 subplots: original SEM, Canny edges, rails and circles only (from legacy)."""
    if not results:
        return None
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()
        if img_rgb.shape[2] == 1:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    height, width = img_rgb.shape[:2]
    
    # Create figure with 3 subplots: original SEM, Canny edges, rails and circles only
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Analysis: {Path(output_path).stem}', fontsize=14, fontweight='bold')
    
    # 1. Original SEM image
    axes[0].imshow(img_rgb, cmap='gray')
    axes[0].set_title('Original SEM')
    axes[0].axis('off')
    axes[0].set_aspect('equal')
    
    # 2. Canny edges
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Canny Edges')
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    
    # 3. Rails and circles only (same color - red)
    axes[2].set_xlim(0, width)
    axes[2].set_ylim(height, 0)  # Flip y-axis to match image coordinates
    axes[2].set_title('Rails & Holes')
    axes[2].axis('off')
    axes[2].set_aspect('equal')
    
    # Draw rails and holes on the third subplot
    if 'theta_deg' in results and 'd1_px' in results and 'd2_px' in results:
        theta = np.radians(results['theta_deg'])
        d1, d2 = results['d1_px'], results['d2_px']
        
        # Create line endpoints using correct coordinate system
        t = np.array([np.cos(theta), np.sin(theta)])
        m = np.array([-np.sin(theta), np.cos(theta)])
        
        # Get points along the rails
        w, h = width, height
        t_coords = np.linspace(-w, w, 1000)
        
        # Rail 1
        rail_center = np.array([w/2, h/2])
        rail_proj1 = rail_center - (np.dot(rail_center, m) - d1) * m
        rail_points1 = np.array([rail_proj1 + t_coord * t for t_coord in t_coords])
        valid_mask1 = (rail_points1[:, 0] >= 0) & (rail_points1[:, 0] < w) & \
                     (rail_points1[:, 1] >= 0) & (rail_points1[:, 1] < h)
        if np.any(valid_mask1):
            valid_points1 = rail_points1[valid_mask1]
            axes[2].plot(valid_points1[:, 0], valid_points1[:, 1], 'red', linewidth=2, label='Rails')
        
        # Rail 2
        rail_proj2 = rail_center - (np.dot(rail_center, m) - d2) * m
        rail_points2 = np.array([rail_proj2 + t_coord * t for t_coord in t_coords])
        valid_mask2 = (rail_points2[:, 0] >= 0) & (rail_points2[:, 0] < w) & \
                     (rail_points2[:, 1] >= 0) & (rail_points2[:, 1] < h)
        if np.any(valid_mask2):
            valid_points2 = rail_points2[valid_mask2]
            axes[2].plot(valid_points2[:, 0], valid_points2[:, 1], 'red', linewidth=2)
        
        # Draw holes (red circles, same as rails)
        if 'holes' in results:
            for hole in results['holes']:
                center = hole['center']
                radius_px = hole['radius_px']
                
                # Draw circle
                circle = plt.Circle(center, radius_px, color='red', fill=False, linewidth=2)
                axes[2].add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_fallback_overlay(img, edges, image_name, output_path):
    """Create fallback overlay when rail detection fails - just show edge detection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Edge detection results
    ax2.imshow(edges, cmap='gray')
    ax2.set_title(f'Edge Detection Results\n{image_name}\n(Rail detection failed)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# BATCH PROCESSING (from analyze_all_images.py)
# ============================================================================

def analyze_all_images(png_dir="PNG", save_overlays=True, overlay_dir="analysis_overlays"):
    """Process all PNG images in the directory and save results."""
    print("🔍 PROCESSING ALL IMAGES")
    print("=" * 50)
    
    png_files = sorted(list(Path(png_dir).glob("*.png")))
    if not png_files:
        print(f"❌ No PNG files found in {png_dir} directory")
        return None, None
    
    print(f"📁 Found {len(png_files)} PNG files")
    print("=" * 50)
    
    # Create overlay directory if saving overlays
    if save_overlays:
        overlay_path = Path(overlay_dir)
        overlay_path.mkdir(exist_ok=True)
        print(f"📁 Overlay images will be saved to: {overlay_dir}/")
    
    all_results = []
    successful_analyses = 0
    failed_analyses = 0
    
    for i, png_file in enumerate(png_files):
        print(f"\n📊 Processing {i+1}/{len(png_files)}: {png_file.name}")
        
        try:
            # Analyze image (this loads the image internally)
            results = analyze_single_image_improved(str(png_file))
            
            # Save overlay image (need to reload image/edges for visualization)
            if save_overlays:
                img, edges = load_and_preprocess_image(str(png_file))
                overlay_filename = overlay_path / f"{png_file.stem}_analysis.png"
                if results and results.get('holes'):
                    create_overlay_improved(img, edges, results, str(overlay_filename))
                else:
                    create_fallback_overlay(img, edges, png_file.name, str(overlay_filename))
            
            if results and results.get('holes'):
                successful_analyses += 1
                num_holes = len(results['holes'])
                print(f"✅ Analysis successful: {num_holes} holes detected")
                print(f"   Rails: θ={results['theta_deg']:.1f}°, sep={results['rail_separation_nm']:.1f}nm")
                
                # Convert holes to serializable format
                holes_serializable = []
                for hole in results['holes']:
                    holes_serializable.append({
                        'center': hole['center'].tolist() if hasattr(hole['center'], 'tolist') else hole['center'],
                        'radius_px': float(hole['radius_px']),
                        'radius_nm': float(hole.get('radius_nm', hole['radius_px'] / SCALE_PX_PER_NM))
                    })
                
                all_results.append({
                    'image_name': png_file.name,
                    'theta_deg': results['theta_deg'],
                    'rail_separation_nm': results['rail_separation_nm'],
                    'num_holes': num_holes,
                    'holes': holes_serializable
                })
            else:
                failed_analyses += 1
                print(f"❌ Analysis failed for {png_file.name}")
                all_results.append({
                    'image_name': png_file.name,
                    'theta_deg': None,
                    'rail_separation_nm': None,
                    'num_holes': 0,
                    'holes': []
                })
        except Exception as e:
            failed_analyses += 1
            print(f"❌ Error processing {png_file.name}: {e}")
            all_results.append({
                'image_name': png_file.name,
                'theta_deg': None,
                'rail_separation_nm': None,
                'num_holes': 0,
                'holes': []
            })
    
    # Save results
    csv_filename = "results_summary.csv"
    json_filename = "results_detailed.json"
    
    print(f"\n💾 Saving results to {csv_filename}")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'theta_deg', 'rail_separation_nm', 'num_holes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                'image_name': result['image_name'],
                'theta_deg': result['theta_deg'],
                'rail_separation_nm': result['rail_separation_nm'],
                'num_holes': result['num_holes']
            })
    
    print(f"💾 Saving detailed results to {json_filename}")
    with open(json_filename, 'w') as jsonfile:
        json.dump(all_results, jsonfile, indent=2)
    
    print("\n🎉 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"✅ Successful analyses: {successful_analyses}")
    print(f"❌ Failed analyses: {failed_analyses}")
    print(f"📊 Total images: {len(png_files)}")
    print(f"📈 Success rate: {successful_analyses / len(png_files) * 100:.1f}%")
    print(f"💾 Results saved to: {csv_filename} and {json_filename}")
    if save_overlays:
        print(f"🖼️  Overlay images saved to: {overlay_dir}/")
    
    return csv_filename, json_filename

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    analyze_all_images(png_dir="PNG")


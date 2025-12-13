#!/usr/bin/env python3
"""
Automatic thickness extraction with uncertainty propagation from width inputs.

This version clones v3 and adds Monte Carlo propagation of uncertainties in
top/bottom widths (Wt, Wb) to estimate the uncertainty of the thickness.
"""

import argparse
import json
import math
from typing import Tuple, Dict, List

import numpy as np
import matplotlib
# Use MacOSX backend for interactive plots (native on macOS, no extra dependencies)
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

# Global small-text defaults
matplotlib.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 9,
})
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import io, color, util, exposure, feature, transform, morphology


# --------------------- utilities ---------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return util.img_as_float(img)
    if img.shape[-1] == 4:
        img = color.rgba2rgb(img)
        
    return color.rgb2gray(img)


def azimuth_from_image_angle(psi_deg: float, theta_deg: float) -> float:
    """Recover true azimuth φ from measured image angle ψ given tilt θ."""
    th = math.radians(theta_deg)
    ps = math.radians(psi_deg)
    return math.degrees(math.atan(math.tan(ps) / max(math.cos(th), 1e-12)))


# --------------------- angle estimation ---------------------

def estimate_azimuth(gray: np.ndarray) -> float:
    """
    Estimate *image* angle ψ (in deg) from dominant line orientation via Hough.
    Returns ψ (angle of the lines in the image).
    """
    img_eq = exposure.equalize_adapthist(gray, clip_limit=0.01)
    edges = feature.canny(img_eq, sigma=2.0, low_threshold=0.1, high_threshold=0.3)
    # skimage>=0.25: use footprint_rectangle with keyword arg
    edges = morphology.binary_closing(edges, morphology.footprint_rectangle(shape=(3, 3)))

    h, theta, d = transform.hough_line(edges)
    acc, ang, dist = transform.hough_line_peaks(h, theta, d, num_peaks=20)
    if len(ang) == 0:
        return 0.0

    # Convert to line angles (relative to x-axis); horizontal ≈ 0°
    line_angles = ang - np.pi/2
    line_angles_deg = [math.degrees(a) for a in line_angles]

    print(f"Debug: Line angles found: {line_angles_deg}")
    print(f"Debug: Accumulator values: {acc}")

    # Prefer lines near 0° (horizontal)
    horizontal_lines = [i for i, angle in enumerate(line_angles_deg) if abs(angle) < 10]
    if horizontal_lines:
        weights = acc[horizontal_lines] / np.sum(acc[horizontal_lines])
        psi = np.average([line_angles_deg[i] for i in horizontal_lines], weights=weights)
        print(f"Debug: Method 1 (weighted near 0°): {psi:.3f}°")
        return float(psi)

    # Fallback near -180° (also horizontal)
    horizontal_lines_180 = [i for i, angle in enumerate(line_angles_deg) if abs(angle + 180) < 10]
    if horizontal_lines_180:
        angles_180 = [line_angles_deg[i] + 180 if line_angles_deg[i] < -90 else line_angles_deg[i]
                      for i in horizontal_lines_180]
        weights = acc[horizontal_lines_180] / np.sum(acc[horizontal_lines_180])
        psi = np.average(angles_180, weights=weights)
        print(f"Debug: Method 2 (near -180°): {psi:.3f}°")
        return float(psi)

    # Strongest line
    strongest_idx = np.argmax(acc)
    psi = line_angles_deg[strongest_idx]
    print(f"Debug: Method 3 (strongest line): {psi:.3f}°")
    return float(psi)


# --------------------- line detection ---------------------

def detect_horizontal_lines(rot_img: np.ndarray, angle_tol_deg: float = 15) -> Tuple[float, np.ndarray]:
    """Detect (now nearly horizontal) lines in the rotated image; return common slope and intercepts (unordered)."""
    img_eq = exposure.equalize_adapthist(rot_img, clip_limit=0.01)
    edges = feature.canny(img_eq, sigma=1.5, low_threshold=0.05, high_threshold=0.2)

    h, theta, d = transform.hough_line(edges)
    # ask for more peaks to help clustering
    acc, ang, dist = transform.hough_line_peaks(h, theta, d, num_peaks=40)
    if acc is None or len(acc) == 0:
        raise RuntimeError("No Hough lines found.")

    line_angles = ang - np.pi/2

    # accept near 0° OR near ±π as horizontal
    tol = math.radians(angle_tol_deg)
    def is_horizontal(a: float) -> bool:
        a_wrapped = ((a + math.pi) % (2 * math.pi)) - math.pi  # [-π, π]
        return (abs(a_wrapped) < tol) or (abs(abs(a_wrapped) - math.pi) < tol)

    sel = [i for i in range(len(acc)) if is_horizontal(line_angles[i])]
    if not sel:
        # wider tolerance fallback
        tol2 = math.radians(max(20, angle_tol_deg))
        def is_horizontal_wide(a: float) -> bool:
            a_wrapped = ((a + math.pi) % (2 * math.pi)) - math.pi
            return (abs(a_wrapped) < tol2) or (abs(abs(a_wrapped) - math.pi) < tol2)
        sel = [i for i in range(len(acc)) if is_horizontal_wide(line_angles[i])]
    if not sel:
        raise RuntimeError("No near-horizontal lines found.")

    cand = [(float(acc[i]), float(line_angles[i]), float(dist[i])) for i in sel]
    cand.sort(key=lambda t: t[0], reverse=True)

    def build_clusters(candidates, rho_thresh):
        clusters = []
        for a, angle, rho in candidates:
            placed = False
            for cl in clusters:
                if abs(rho - cl["rho_mean"]) < rho_thresh:
                    cl["members"].append((a, angle, rho))
                    cl["rho_mean"] = float(np.mean([m[2] for m in cl["members"]]))
                    placed = True
                    break
            if not placed:
                clusters.append({"members":[(a,angle,rho)], "rho_mean": float(rho)})
        return clusters

    # try standard clustering
    clusters = build_clusters(cand, rho_thresh=4.0)
    if len(clusters) < 3:
        # relax clustering threshold
        clusters = build_clusters(cand, rho_thresh=8.0)

    if len(clusters) < 3:
        # last-resort: pick top 3 peaks with distinct rhos (>= 8 px apart)
        distinct = []
        used_rhos = []
        for a, angle, rho in cand:
            if all(abs(rho - r) >= 8.0 for r in used_rhos):
                distinct.append({"members":[(a, angle, rho)], "rho_mean": rho})
                used_rhos.append(rho)
            if len(distinct) == 3:
                break
        clusters = distinct

    if len(clusters) < 3:
        raise RuntimeError("Fewer than 3 horizontal clusters found.")

    # keep strongest 3 clusters
    clusters.sort(key=lambda cl: sum(m[0] for m in cl["members"]), reverse=True)
    clusters = clusters[:3]

    # Convert clusters to y=mx+b via weighted average
    lines_mb = []
    for cl in clusters:
        w = np.array([m[0] for m in cl["members"]])
        angs = np.array([m[1] for m in cl["members"]])
        rhos = np.array([m[2] for m in cl["members"]])
        ang = float(np.average(angs, weights=w))
        rho = float(np.average(rhos, weights=w))
        theta0 = ang + np.pi/2
        s = math.sin(theta0)
        c = math.cos(theta0)
        if abs(s) < 1e-3:
            continue
        m = -c / s
        b = rho / s
        lines_mb.append((float(m), float(b)))

    if len(lines_mb) < 3:
        raise RuntimeError("Fewer than 3 valid lines.")

    # Enforce common slope and consistent intercepts
    m_common = float(np.median([mb[0] for mb in lines_mb]))
    xs = np.linspace(0, rot_img.shape[1]-1, 60)
    bs = []
    for m, b in lines_mb:
        ys = m*xs + b
        b_fix = float(np.median(ys - m_common*xs))
        bs.append(b_fix)
    bs = np.array(bs, float)
    return m_common, bs


# --------------------- relabeling & geometry ---------------------

def relabel_lines_by_geometry(
    m: float,
    bs: np.ndarray,
    Wt_nm: float,
    Wb_nm: float,
    theta_deg: float,
    phi_deg: float,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Given three intercepts 'bs' (unordered), choose the two top lines as the pair whose
    spacing is the median of the three, and compute scales with φ (true azimuth).
    Returns bs_ordered = [b_top_far, b_top_near, b_bottom].
    """
    assert len(bs) == 3, "Expected exactly 3 lines."
    denom = math.sqrt(1.0 + m*m)

    pairs = [(0,1), (0,2), (1,2)]
    dists_px = {p: abs((bs[p[1]] - bs[p[0]])/denom) for p in pairs}

    proj = math.cos(math.radians(theta_deg)) * math.cos(math.radians(phi_deg))  # uses φ (true)
    target_top_nm = Wt_nm * proj

    # robustly choose top pair as median distance
    vals = {p: dists_px[p] for p in pairs}
    median_val = float(np.median(list(vals.values())))
    top_pair = min(pairs, key=lambda p: abs(vals[p] - median_val))

    px_per_nm_v = dists_px[top_pair] / max(target_top_nm, 1e-9)
    i_far_top, i_near_top = top_pair
    i_bottom = ({0,1,2} - set(top_pair)).pop()

    dT_px = dists_px[top_pair]
    dFB_px = abs((bs[i_bottom] - bs[i_near_top]) / denom)

    min_bottom_nm = 0.5*(Wt_nm + Wb_nm)*proj
    min_bottom_px = min_bottom_nm * px_per_nm_v

    swapped_top_roles = False
    if dFB_px < 0.9 * min_bottom_px:
        alt_dFB_px = abs((bs[i_bottom] - bs[i_far_top]) / denom)
        if alt_dFB_px > dFB_px:
            i_far_top, i_near_top = i_near_top, i_far_top
            dFB_px = alt_dFB_px
            swapped_top_roles = True

    # enforce order so that dT_px > 0
    b_far = float(bs[i_far_top])
    b_near = float(bs[i_near_top])
    if b_near <= b_far:
        i_far_top, i_near_top = i_near_top, i_far_top
        b_far, b_near = b_near, b_far
        swapped_top_roles = True

    dFB_px = abs((bs[i_bottom] - bs[i_near_top]) / denom)
    dT_px = abs((bs[i_near_top] - bs[i_far_top]) / denom)

    bs_ordered = np.array([bs[i_far_top], bs[i_near_top], bs[i_bottom]], dtype=float)

    info = {
        "proj": float(proj),
        "target_top_nm": float(target_top_nm),
        "pair01_px": float(dists_px[(0,1)]),
        "pair02_px": float(dists_px[(0,2)]),
        "pair12_px": float(dists_px[(1,2)]),
        "chosen_top_pair": int( (0 if top_pair==(0,1) else 1 if top_pair==(0,2) else 2) ),
        "px_per_nm_v_est": float(px_per_nm_v),
        "dT_px": float(dT_px),
        "dFB_px": float(dFB_px),
        "min_bottom_nm_width_only": float(min_bottom_nm),
        "min_bottom_px_width_only": float(min_bottom_px),
        "swapped_top_roles": bool(swapped_top_roles),
        "i_far_top": int(i_far_top),
        "i_near_top": int(i_near_top),
        "i_bottom": int(i_bottom),
    }
    if verbose:
        print("\n=== Geometry Debug ===")
        print(f"phi_true_deg (used)       : {phi_deg:.3f}°")
        print(f"theta_deg                 : {theta_deg:.3f}°")
        print(f"proj = cosθ·cosφ          : {info['proj']:.6f}")
        print(f"Wt*proj (nm)              : {info['target_top_nm']:.3f}")
        print(f"Pair distances (px)       : (0,1)={info['pair01_px']:.3f}, (0,2)={info['pair02_px']:.3f}, (1,2)={info['pair12_px']:.3f}")
        print(f"Chosen top pair idx code  : {info['chosen_top_pair']}  (0:(0,1), 1:(0,2), 2:(1,2))")
        print(f"px_per_nm_v (from top)    : {info['px_per_nm_v_est']:.6f}")
        print(f"dT_px (top pair)          : {info['dT_px']:.3f}")
        print(f"dFB_px (nearTop→bottom)   : {info['dFB_px']:.3f}")
        print(f"min bottom (nm,width-only): {info['min_bottom_nm_width_only']:.3f} nm  -> {info['min_bottom_px_width_only']:.3f} px")
        print(f"Swapped top roles?        : {info['swapped_top_roles']}")
        print(f"Indices (farTop, nearTop, bottom): ({info['i_far_top']}, {info['i_near_top']}, {info['i_bottom']})")
        print("=======================\n")

    return bs_ordered, info


# --------------------- compute & plots ---------------------

def compute_thickness(m: float, bs: np.ndarray, Wt_nm: float, Wb_nm: float, 
                      theta_deg: float, phi_deg: float) -> dict:
    """Compute thickness using projection formulas (sign-robust), using φ (true)."""
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    denom = math.sqrt(1.0 + m * m)

    b_top, b_mid, b_bot = float(bs[0]), float(bs[1]), float(bs[2])
    dT_px = (b_mid - b_top) / denom
    dFB_px = (b_bot - b_mid) / denom

    proj = math.cos(theta) * math.cos(phi)
    s_px_per_nm = dT_px / (Wt_nm * proj)  # vertical px per nm

    min_bottom_px = 0.5 * (Wt_nm + Wb_nm) * proj * s_px_per_nm

    if dFB_px < min_bottom_px:
        t_mag_nm = (min_bottom_px - dFB_px) / (s_px_per_nm * max(math.sin(theta), 1e-12))
        sign = +1  # bottom projects toward the top pair
    else:
        t_mag_nm = (dFB_px - min_bottom_px) / (s_px_per_nm * max(math.sin(theta), 1e-12))
        sign = -1  # bottom projects away

    t_nm = sign * t_mag_nm

    return {
        "dT_px": float(dT_px),
        "dFB_px": float(dFB_px),
        "px_per_nm_v": float(s_px_per_nm),
        "thickness_nm": float(t_nm),
    }


def plot_overlay(rot_img: np.ndarray, m: float, bs: np.ndarray, 
                 Wt: float, Wb: float, theta_deg: float, phi_true_deg: float, psi_deg: float) -> dict:
    """Plot overlay with detected lines and model using φ (true) and showing ψ (image)."""
    res = compute_thickness(m, bs, Wt, Wb, theta_deg, phi_true_deg)

    denom = math.sqrt(1.0 + m * m)
    proj = math.cos(math.radians(theta_deg)) * math.cos(math.radians(phi_true_deg))
    dT_model_px = res["px_per_nm_v"] * (Wt * proj)
    dFB_model_px = res["px_per_nm_v"] * (0.5 * (Wt + Wb) * proj + res["thickness_nm"] * math.sin(math.radians(theta_deg)))
    dTN_model_px = dFB_model_px - dT_model_px

    b_mid = float(bs[1])
    b_top = b_mid - dT_model_px * denom
    b_bot = b_mid + dTN_model_px * denom

    X = np.linspace(0, rot_img.shape[1] - 1, 400)

    # Calculate figure size: maintain image aspect ratio, add fixed padding amounts
    rot_h, rot_w = rot_img.shape[:2]
    aspect_ratio = rot_w / rot_h
    
    # Base image size (maintains aspect ratio)
    base_width = 10
    base_height = base_width / aspect_ratio
    
    # Add fixed padding amounts (same absolute amount on all sides)
    padding = 1.0  # inches of padding on each side
    fig_width = base_width + 2 * padding
    fig_height = base_height + 2 * padding
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    # Position axes to center the image with fixed padding
    pad_frac = padding / fig_width  # horizontal padding fraction
    pad_frac_v = padding / fig_height  # vertical padding fraction
    fig.subplots_adjust(left=pad_frac, right=1-pad_frac, top=1-pad_frac_v, bottom=pad_frac_v)
    
    ax.imshow(rot_img, cmap="gray")
    # Detected lines (red) only
    for j, b in enumerate(bs):
        ax.plot(X, m * X + b, linewidth=1, color="red", label="Detected" if j == 0 else "")

    t_abs = abs(res['thickness_nm'])
    ax.set_title(f"Thickness ≈ {t_abs:.0f} nm | θ={theta_deg:.1f}°, φ(true)={phi_true_deg:.2f}°, ψ(img)={psi_deg:.2f}°")
    ax.legend()
    ax.set_axis_off()
    ax.set_xlim(0, rot_w)
    ax.set_ylim(rot_h, 0)  # Image coordinates: y increases downward
    plt.show()
    return res


def build_trapezoid_prism_vertices(Wt: float, Wb: float, t: float, L: float) -> np.ndarray:
    """Build trapezoid prism vertices aligned along y-axis."""
    top_left  = np.array([-Wt/2, 0,  t/2])
    top_right = np.array([ Wt/2, 0,  t/2])
    bot_right = np.array([ Wb/2, 0, -t/2])
    bot_left  = np.array([-Wb/2, 0, -t/2])
    verts = np.array([
        top_left, top_right, bot_right, bot_left,
        top_left + [0, L, 0], top_right + [0, L, 0], bot_right + [0, L, 0], bot_left + [0, L, 0]
    ], dtype=float)
    return verts


def plot_prism(ax, V: np.ndarray):
    """Plot 3D trapezoid prism."""
    faces = [
        [V[0], V[1], V[2], V[3]],  # front
        [V[4], V[5], V[6], V[7]],  # back
        [V[0], V[1], V[5], V[4]],  # top
        [V[3], V[2], V[6], V[7]],  # bottom
        [V[1], V[2], V[6], V[5]],  # right
        [V[0], V[3], V[7], V[4]],  # left
    ]
    pc = Poly3DCollection(faces, facecolors=[(0.7,0.8,1,0.6)], edgecolors='k', linewidths=0.8)
    ax.add_collection3d(pc)


def visualize_3d(Wt: float, Wb: float, t: float, L: float, theta_deg: float, phi_true_deg: float, psi_deg: float):
    """Show 3D trapezoid prism; set camera using θ and φ(true)."""
    V = build_trapezoid_prism_vertices(Wt, Wb, t, L)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_prism(ax, V)

    ax.set_xlabel('x (width, nm)')
    ax.set_ylabel('y (length, nm)')
    ax.set_zlabel('z (thickness, nm)')
    ax.set_title(f'Trapezoid Prism (θ={theta_deg:.1f}°, φ(true)={phi_true_deg:.2f}°, ψ(img)={psi_deg:.2f}°, t={t:.0f} nm)')

    # Equal aspect
    ranges = V.max(axis=0) - V.min(axis=0)
    max_range = ranges.max()
    mins = V.min(axis=0)
    centers = mins + ranges / 2
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(centers[0]-max_range/2, centers[0]+max_range/2)
    ax.set_ylim(centers[1]-max_range/2, centers[1]+max_range/2)
    ax.set_zlim(centers[2]-max_range/2, centers[2]+max_range/2)

    # Camera: elevation tied to θ, azimuth tied to φ(true)
    camera_elev = 90 - theta_deg
    camera_azim = phi_true_deg
    ax.view_init(elev=camera_elev, azim=camera_azim)

    plt.tight_layout()
    plt.show()


# --------------------- PNG overlay saving ---------------------

def save_png_overlay(rot_img: np.ndarray, m: float, bs: np.ndarray,
                     Wt: float, Wb: float, theta_deg: float, phi_true_deg: float,
                     thickness_nm: float, out_path: str = "overlay.png", dpi: int = 300):
    """
    Save PNG overlay on ROTATED image with detected lines only.
    """
    import matplotlib as mpl
    mpl.use('Agg')  # Use non-GUI backend for PNG generation
    import matplotlib.pyplot as plt

    rot_h, rot_w = rot_img.shape[:2]
    fig_w_in = 10
    fig_h_in = fig_w_in * (rot_h / rot_w)
    
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    
    # Background: rotated image
    ax.imshow(rot_img, cmap="gray", interpolation="nearest")
    
    X = np.linspace(0, rot_w - 1, 400)
    
    # Detected lines (red) only
    for j, b in enumerate(bs):
        ax.plot(X, m * X + b, linewidth=1.5, color="red", label="Detected" if j == 0 else "")
    
    t_abs = abs(thickness_nm)
    ax.set_title(f"Thickness ≈ {t_abs:.0f} nm | θ={theta_deg:.1f}°, φ(true)={phi_true_deg:.2f}°")
    ax.legend()
    ax.set_axis_off()
    ax.set_xlim(0, rot_w)
    ax.set_ylim(rot_h, 0)
    
    fig.savefig(out_path, dpi=dpi, transparent=False, bbox_inches='tight')
    plt.close(fig)


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Automatic thickness extraction with uncertainty propagation")
    ap.add_argument("--image", default="cavity_sem_45degree.png", help="SEM image path")
    ap.add_argument("--wt", type=float, default=238.0, help="Top width Wt (nm)")
    ap.add_argument("--wb", type=float, default=314.0, help="Bottom width Wb (nm)")
    ap.add_argument("--wt-std", type=float, default=10.0, help="Uncertainty (std) of Wt (nm)")
    ap.add_argument("--wb-std", type=float, default=10.0, help="Uncertainty (std) of Wb (nm)")
    ap.add_argument("--mc-samples", type=int, default=2000, help="Number of Monte Carlo samples")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for Monte Carlo")
    ap.add_argument("--theta", type=float, default=45.0, help="Tilt elevation (deg)")
    ap.add_argument("--length", type=float, default=4000.0, help="Prism length (nm)")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots for faster batch runs")
    args = ap.parse_args()

    # Load image with better error handling
    import os
    if not os.path.exists(args.image):
        print(f"ERROR: Image file not found: {args.image}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please check the image path or run from the correct directory.")
        return
    
    try:
        img = io.imread(args.image)
        print(f"Loaded image: {args.image} (shape: {img.shape})")
    except Exception as e:
        print(f"ERROR: Failed to load image '{args.image}': {e}")
        return
    
    try:
        gray = to_gray(img)
        gray = exposure.equalize_adapthist(gray, clip_limit=0.01)
    except Exception as e:
        print(f"ERROR: Failed to process image: {e}")
        return

    # 1) Measure *image* angle ψ
    psi_deg = estimate_azimuth(gray)
    print(f"Detected image angle ψ: {psi_deg:.2f}°")

    # 2) Rotate image: use proven working angle when ψ is reasonable (kept consistent with v3)
    if abs(psi_deg) < 10:
        rotation_angle = -2.8
        print(f"Rotating image by {rotation_angle:.2f}° (ψ was {psi_deg:.2f}°)")
    else:
        rotation_angle = -2.8
        print(f"Rotating image by known working angle {rotation_angle:.2f}° (ψ was {psi_deg:.2f}°)")
    rot = transform.rotate(gray, rotation_angle, resize=True, preserve_range=True, mode="constant", cval=0)

    # 3) Recover true azimuth φ from ψ and θ; use φ in projection math
    phi_true_deg = azimuth_from_image_angle(psi_deg, args.theta)
    print(f"Recovered true azimuth φ: {phi_true_deg:.2f}° (from ψ and θ)")

    # Detect lines on rotated image with error handling
    try:
        m_common, bs_raw = detect_horizontal_lines(rot)
        print(f"Detected (unordered) line intercepts b: {np.array2string(bs_raw, precision=3)}")
    except RuntimeError as e:
        print(f"ERROR: Line detection failed: {e}")
        print(f"This usually means the image doesn't have clear horizontal lines.")
        print(f"Try:")
        print(f"  1. Check that the image has visible horizontal edges")
        print(f"  2. Try a different image or adjust the rotation angle")
        print(f"  3. Check image quality and contrast")
        return
    except Exception as e:
        print(f"ERROR: Unexpected error during line detection: {e}")
        import traceback
        traceback.print_exc()
        return

    # Relabel based on geometry using φ(true) with nominal widths
    bs, dbg = relabel_lines_by_geometry(
        m=m_common,
        bs=bs_raw,
        Wt_nm=args.wt,
        Wb_nm=args.wb,
        theta_deg=args.theta,
        phi_deg=phi_true_deg,
        verbose=True
    )
    print(f"Ordered lines [farTop, nearTop, bottom] b: {np.array2string(bs, precision=3)}")

    # Compute thickness with nominal widths
    res_nominal = compute_thickness(m_common, bs, args.wt, args.wb, args.theta, phi_true_deg)
    t_nm_nominal = abs(float(res_nominal["thickness_nm"])) if np.isfinite(res_nominal["thickness_nm"]) else 0.0
    
    # Show plots if not disabled
    if not args.no_plots:
        print("Generating 2D overlay plot...")
        _ = plot_overlay(rot, m_common, bs, args.wt, args.wb, args.theta, phi_true_deg, psi_deg)
        print("Generating 3D visualization...")
        visualize_3d(args.wt, args.wb, t_nm_nominal, args.length, args.theta, phi_true_deg, psi_deg)
        print("Plots should be displayed now!")

    # Save overlay matching the plot
    print("\n=== Saving Overlay ===")
    
    # PNG overlay on rotated image (matching plot_overlay)
    png_out = "overlay.png"
    save_png_overlay(
        rot_img=rot,
        m=m_common,
        bs=bs,
        Wt=args.wt,
        Wb=args.wb,
        theta_deg=args.theta,
        phi_true_deg=phi_true_deg,
        thickness_nm=t_nm_nominal,
        out_path=png_out,
        dpi=300,
    )
    print(f"PNG overlay saved to: {png_out}")
    print("==========================================\n")

    # Monte Carlo propagation for Wt/Wb uncertainties (geometry fixed: m, bs, θ, φ)
    rng = np.random.default_rng(args.seed)
    if args.mc_samples > 0 and (args.wt_std > 0 or args.wb_std > 0):
        Wt_samples = rng.normal(loc=args.wt, scale=max(args.wt_std, 0.0), size=args.mc_samples)
        Wb_samples = rng.normal(loc=args.wb, scale=max(args.wb_std, 0.0), size=args.mc_samples)
        # enforce positive widths
        Wt_samples = np.clip(Wt_samples, 1e-9, None)
        Wb_samples = np.clip(Wb_samples, 1e-9, None)

        t_samples = np.empty(args.mc_samples, dtype=float)
        for i in range(args.mc_samples):
            res_i = compute_thickness(m_common, bs, float(Wt_samples[i]), float(Wb_samples[i]), args.theta, phi_true_deg)
            t_samples[i] = float(res_i["thickness_nm"])

        t_mean = float(np.mean(t_samples))
        t_std = float(np.std(t_samples, ddof=1)) if args.mc_samples > 1 else 0.0
        t_p05, t_p50, t_p95 = [float(q) for q in np.percentile(t_samples, [5, 50, 95])]
    else:
        t_samples = np.array([res_nominal["thickness_nm"]], dtype=float)
        t_mean = float(t_samples[0])
        t_std = 0.0
        t_p05 = t_p50 = t_p95 = float(t_samples[0])

    # Results payload
    out = {
        "input_file": args.image,
        "detected_image_angle_deg": float(psi_deg),
        "true_azimuth_deg": float(phi_true_deg),
        "rotation_angle_deg": float(rotation_angle),
        "m_common": float(m_common),
        "b_top_mid_bot": [float(bs[0]), float(bs[1]), float(bs[2])],
        "wt_nm": float(args.wt),
        "wb_nm": float(args.wb),
        "wt_std_nm": float(args.wt_std),
        "wb_std_nm": float(args.wb_std),
        "mc_samples": int(args.mc_samples),
        **{k: float(v) for k, v in res_nominal.items()},
        "thickness_mean_nm": t_mean,
        "thickness_std_nm": t_std,
        "thickness_p05_nm": t_p05,
        "thickness_p50_nm": t_p50,
        "thickness_p95_nm": t_p95,
        "debug": dbg
    }

    print("\nResults:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



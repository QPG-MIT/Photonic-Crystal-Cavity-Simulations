#!/usr/bin/env python3
"""
Automatic thickness extraction from tilted SEM (scale-free) - Version 2

Uses the robust line detection approach from debug_rotation.py:
1. CLAHE -> Bilateral -> MORPH_CLOSE (to fill holes) -> Canny -> HoughLinesP
2. Group segments by y-intercept
3. Fit lines with robust loss (DIST_WELSCH)

Key features:
- NO pixel scale needed: Uses scale-free formula
- NO hard-coded rotation: Uses measured image angle
- Robust line detection that handles surface holes/pits
- Advanced prism visualization with face-based hidden-line removal (from visualize_structure.py)
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

import cv2
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Editable text in vector exports
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42


# =============================================================================
# DATA CLASSES (from visualize_structure.py)
# =============================================================================

@dataclass
class PrismDims:
    Wt: float  # top width (x extent at z=+t/2)
    Wb: float  # bottom width (x extent at z=-t/2)
    t: float   # thickness (z extent)
    L: float   # length (y extent)

@dataclass
class View:
    azim: float   # Matplotlib azimuth (deg) about world +z
    elev: float   # Matplotlib elevation (deg)
    mode: str = "ortho"  # 'ortho' or 'persp'
    f: float = 2000.0    # focal length for perspective


# =============================================================================
# IMAGE I/O
# =============================================================================

def read_gray(path: str) -> np.ndarray:
    """Read image and convert to grayscale uint8."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    # Handle RGBA
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


# =============================================================================
# LINE DETECTION (from debug_rotation.py - the working approach)
# =============================================================================

def preprocess_for_edges(img_u8: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for edge detection using the approach that works:
    CLAHE -> Bilateral -> MORPH_CLOSE (fill holes) -> Canny
    
    Returns: (preprocessed_image, edges)
    """
    # 1) CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(img_u8)
    
    # 2) Bilateral filter (edge-preserving denoising)
    g_denoised = cv2.bilateralFilter(g, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3) MORPH_CLOSE to fill small dark holes BEFORE Canny
    # This is the key fix - prevents holes from creating spurious edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    g_closed = cv2.morphologyEx(g_denoised, cv2.MORPH_CLOSE, k)
    
    # 4) Canny edge detection
    edges = cv2.Canny(g_closed, threshold1=140, threshold2=150, apertureSize=3, L2gradient=True)
    
    if debug:
        print(f"  Preprocessing: CLAHE -> bilateral -> MORPH_CLOSE(9x9) -> Canny(140,150)")
        print(f"  Edge pixels: {np.sum(edges > 0)} / {edges.size} ({100*np.sum(edges > 0)/edges.size:.2f}%)")
    
    return g_closed, edges


def group_by_intercept(segments: np.ndarray, tol_px: float = 4.0) -> List[List]:
    """
    Group line segments by their y-intercept b in y = mx + b.
    Works well for nearly horizontal lines.
    
    Returns list of groups, each group is a list of (b, length, angle_deg, (x1,y1,x2,y2)).
    """
    items = []
    for (x1, y1, x2, y2) in segments:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 5:  # skip near-vertical
            continue
        m = dy / dx
        b = y1 - m * x1
        length = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        items.append((b, length, angle_deg, (x1, y1, x2, y2)))

    if not items:
        return []

    # Sort by intercept and cluster
    items.sort(key=lambda t: t[0])
    groups = []
    cur = [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) <= tol_px:
            cur.append(it)
        else:
            groups.append(cur)
            cur = [it]
    groups.append(cur)
    return groups


def fit_line_from_segments(group: List) -> Tuple[float, float, float, float, float]:
    """
    Fit one line to all endpoints in a group using robust loss (DIST_WELSCH).
    Returns (vx, vy, x0, y0, angle_deg).
    """
    pts = []
    for (_, _, _, (x1, y1, x2, y2)) in group:
        pts.append([x1, y1])
        pts.append([x2, y2])
    pts = np.asarray(pts, dtype=np.float32)

    # DIST_WELSCH is robust to outliers - prevents stray segments from pulling the fit
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_WELSCH, 0, 0.01, 0.01).flatten()
    angle_deg = math.degrees(math.atan2(vy, vx))
    return (float(vx), float(vy), float(x0), float(y0), float(angle_deg))


def detect_lines(img_u8: np.ndarray, n_lines: int = 3,
                 hough_threshold: int = 40,
                 min_line_length: int = 200,
                 max_line_gap: int = 20,
                 intercept_tol: float = 4.0,
                 debug: bool = False) -> Tuple[float, np.ndarray, dict]:
    """
    Detect n_lines horizontal lines using the robust approach:
    Preprocessing -> HoughLinesP -> Group by intercept -> Fit with robust loss.
    
    Returns: (mean_angle_deg, y_intercepts, debug_info)
    """
    h, w = img_u8.shape[:2]
    
    if debug:
        print("\n=== LINE DETECTION (HoughLinesP + Robust Fitting) ===")
    
    # Step 1: Preprocess
    preprocessed, edges = preprocess_for_edges(img_u8, debug=debug)
    
    # Step 2: HoughLinesP for line segments
    linesP = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    
    if linesP is None:
        raise RuntimeError("No line segments found by HoughLinesP")
    
    segments = linesP[:, 0, :]
    if debug:
        print(f"  HoughLinesP: found {len(segments)} segments")
    
    # Step 3: Group by y-intercept
    groups = group_by_intercept(segments, tol_px=intercept_tol)
    if debug:
        print(f"  Grouping: {len(groups)} groups (tol={intercept_tol}px)")
    
    if len(groups) < n_lines:
        raise RuntimeError(f"Only found {len(groups)} line groups, need {n_lines}")
    
    # Step 4: Keep top n_lines groups by total segment length
    groups = sorted(groups, key=lambda grp: sum(it[1] for it in grp), reverse=True)[:n_lines]
    
    # Step 5: Fit lines to groups
    fits = []
    y_intercepts = []
    for i, grp in enumerate(groups):
        fit = fit_line_from_segments(grp)
        fits.append(fit)
        
        # Compute y-intercept at x=0 from the fitted line
        vx, vy, x0, y0, angle_deg = fit
        if abs(vx) > 1e-6:
            m = vy / vx
            b = y0 - m * x0
        else:
            b = y0  # vertical line
        y_intercepts.append(b)
        
        if debug:
            median_intercept = np.median([it[0] for it in grp])
            print(f"  Line {i+1}: angle={angle_deg:.3f}°, y_intercept={b:.2f}px, n_segs={len(grp)}")
    
    # Sort by y-intercept (top to bottom)
    sorted_indices = np.argsort(y_intercepts)
    y_intercepts = np.array([y_intercepts[i] for i in sorted_indices])
    fits = [fits[i] for i in sorted_indices]
    
    # Compute mean angle
    angles = np.array([f[4] for f in fits])
    ang_rad = np.deg2rad(angles)
    mean_angle = math.degrees(math.atan2(np.mean(np.sin(ang_rad)), np.mean(np.cos(ang_rad))))
    
    if debug:
        print(f"  Final lines (sorted by y): y_intercepts={np.array2string(y_intercepts, precision=2)}")
        print(f"  Mean angle: {mean_angle:.3f}°")
        print("=== END LINE DETECTION ===\n")
    
    debug_info = {
        "preprocessed": preprocessed,
        "edges": edges,
        "segments": segments,
        "groups": groups,
        "fits": fits,
        "mean_angle_deg": mean_angle,
    }
    
    return mean_angle, y_intercepts, debug_info


# =============================================================================
# GEOMETRY / THICKNESS CALCULATION
# =============================================================================

def azimuth_from_image_angle(psi_deg: float, theta_deg: float) -> float:
    """
    Recover true azimuth φ from measured image angle ψ given tilt θ:
    tan(ψ) = tan(φ)*cos(θ)  ->  φ = arctan(tan(ψ)/cos(θ))
    """
    th = math.radians(theta_deg)
    ps = math.radians(psi_deg)
    return math.degrees(math.atan(math.tan(ps) / max(math.cos(th), 1e-12)))


def thickness_scale_free(
    dT_px: float,
    dNB_px: float,
    Wt_nm: float,
    Wb_nm: float,
    theta_deg: float,
    phi_deg: float
) -> float:
    """
    Scale-free formula:
      t = cot(θ)*cos(φ) * ( Wt*(dNB/dT) - (Wb-Wt)/2 )

    dT, dNB can be in pixels; only their ratio is used.
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    if dT_px <= 1e-12:
        return float("nan")

    cot_theta = math.cos(theta) / max(math.sin(theta), 1e-12)
    pref = cot_theta * math.cos(phi)

    bracket = Wt_nm * (dNB_px / dT_px) - (Wb_nm - Wt_nm) / 2.0
    return pref * bracket


# =============================================================================
# LABELING (enumerate all labelings and pick best)
# =============================================================================

def relabel_lines_scale_free(
    m: float,
    bs: np.ndarray,
    Wt_nm: float,
    Wb_nm: float,
    theta_deg: float,
    phi_deg: float,
    t_min_nm: float,
    t_max_nm: float,
    prefer_positive: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Enumerate all labelings [farTop, nearTop, bottom] among 3 lines,
    compute scale-free thickness, and pick the best by plausibility score.
    """
    assert len(bs) == 3
    denom = math.sqrt(1.0 + m * m)

    candidates = []
    # Since lines are sorted by y-intercept (top to bottom), enforce physical ordering:
    # farTop (index 0, topmost) < nearTop (index 1, middle) < bottom (index 2, bottommost)
    # Only consider assignments where: far < near < bottom in terms of indices
    # Verify y-intercepts are in correct order (should be true after sorting in detect_lines)
    assert bs[0] < bs[1] < bs[2], f"Lines not properly sorted: bs={bs}"
    for far, near, bottom in [(0, 1, 2)]:  # Only one valid ordering after sorting
        # distances along normal
        dT = abs((bs[near] - bs[far]) / denom)   # top-top
        dNB = abs((bs[near] - bs[bottom]) / denom)    # nearTop-bottom

        t_nm = thickness_scale_free(dT, dNB, Wt_nm, Wb_nm, theta_deg, phi_deg)

        # score
        score = 0.0
        if not np.isfinite(t_nm):
            score += 1e9
        else:
            # enforce range
            if not (t_min_nm <= abs(t_nm) <= t_max_nm):
                score += 1e6 + 1e3 * min(abs(abs(t_nm) - t_min_nm), abs(abs(t_nm) - t_max_nm))

            # prefer positive thickness
            if prefer_positive and t_nm < 0:
                score += 1e4

            # penalize extreme ratios
            r = dNB / max(dT, 1e-12)
            if r <= 0:
                score += 1e5
            if r < 0.1 or r > 5.0:
                score += 1e4

            # small regularizer
            score += 0.01 * abs(t_nm)

        candidates.append({
            "far": far, "near": near, "bottom": bottom,
            "dT_px": float(dT), "dNB_px": float(dNB),
            "ratio": float(dNB / max(dT, 1e-12)),
            "t_nm": float(t_nm),
            "score": float(score),
        })

    candidates.sort(key=lambda c: c["score"])
    best = candidates[0]

    bs_ordered = np.array([bs[best["far"]], bs[best["near"]], bs[best["bottom"]]], dtype=float)

    info = {
        "chosen_indices": {"far": best["far"], "near": best["near"], "bottom": best["bottom"]},
        "dT_px": best["dT_px"],
        "dNB_px": best["dNB_px"],
        "ratio_dNB_over_dT": best["ratio"],
        "thickness_nm": best["t_nm"],
        "score": best["score"],
        "all_candidates": candidates if verbose else None,
    }

    if verbose:
        print("\n=== Labeling (scale-free) ===")
        for n, c in enumerate(candidates[:6]):
            print(f"[{n}] far={c['far']} near={c['near']} bot={c['bottom']} | "
                  f"dT={c['dT_px']:.3f}px dNB={c['dNB_px']:.3f}px r={c['ratio']:.3f} | "
                  f"t={c['t_nm']:.2f} nm | score={c['score']:.2f}")
        print(f"--> CHOSEN: far={best['far']} near={best['near']} bottom={best['bottom']} | t={best['t_nm']:.2f} nm")
        print("=============================\n")

    return bs_ordered, info


# =============================================================================
# PRISM GEOMETRY (from visualize_structure.py)
# =============================================================================

def prism_vertices(d: PrismDims) -> np.ndarray:
    """Return 8x3 array of prism vertices."""
    z_top = +d.t/2
    z_bot = -d.t/2
    xt = d.Wt/2
    xb = d.Wb/2
    y  = d.L/2

    # Top (z=+t/2), CCW from (+x,+y)
    v0 = np.array([+xt, +y, z_top])
    v1 = np.array([-xt, +y, z_top])
    v2 = np.array([-xt, -y, z_top])
    v3 = np.array([+xt, -y, z_top])
    # Bottom (z=-t/2), CCW from (+x,+y) seen from -z
    v4 = np.array([+xb, +y, z_bot])
    v5 = np.array([-xb, +y, z_bot])
    v6 = np.array([-xb, -y, z_bot])
    v7 = np.array([+xb, -y, z_bot])

    return np.stack([v0,v1,v2,v3,v4,v5,v6,v7], axis=0)

# Faces as index lists (quads) - CCW winding for outward normals
FACES = [
    [0,1,2,3],  # top - CCW from above (normal points up +z)
    [4,7,6,5],  # bottom - CW from above = CCW from below (normal points down -z)
    [0,4,5,1],  # side +y - CCW from outside +y (normal points +y)
    [3,2,6,7],  # side -y - CCW from outside -y (normal points -y)
    [1,5,6,2],  # side -x - CCW from outside -x (normal points -x)
    [0,3,7,4],  # side +x - CCW from outside +x (normal points +x)
]

FACE_NAMES = [
    "top",
    "bottom",
    "side +y",
    "side -y",
    "side -x",
    "side +x"
]

def unique_edges_from_faces(faces: List[List[int]]) -> List[Tuple[int,int]]:
    """Unique undirected edges from quads."""
    edges = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            i, j = face[k], face[(k+1) % n]
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return list(edges)

def edges_to_faces_map(faces: List[List[int]]) -> dict:
    """
    Create a mapping from edge (tuple of sorted vertex indices) to list of face indices.
    
    Returns:
        Dictionary mapping (v1, v2) edge tuple -> list of face indices that contain this edge
    """
    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            i, j = face[k], face[(k+1) % n]
            edge = (i, j) if i < j else (j, i)
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
    return edge_to_faces

def calculate_face_normal(vertices: np.ndarray, face_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the outward normal vector for a face.
    
    Args:
        vertices: 8x3 array of all vertices
        face_indices: List of vertex indices forming the face (CCW order)
        
    Returns:
        (normal, center): normal vector (unit vector) and face center point
    """
    # Get face vertices
    face_verts = vertices[face_indices]
    
    # Calculate center
    center = np.mean(face_verts, axis=0)
    
    # Calculate normal using cross product of two edges
    # Since vertices are CCW, (v1-v0) x (v2-v0) gives outward normal
    v0, v1, v2 = face_verts[0], face_verts[1], face_verts[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    
    # Normalize
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0])  # fallback
    
    return normal, center


# =============================================================================
# CAMERA & PROJECTION (from visualize_structure.py)
# =============================================================================

def camera_axes_from_mpl(elev_deg: float, azim_deg: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Right-handed camera basis like mplot3d: right, up, forward (scene->camera)."""
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    vdir = np.array([math.cos(a)*math.cos(e), math.sin(a)*math.cos(e), math.sin(e)], dtype=float)
    world_up = np.array([0.0, 0.0, 1.0], float)
    right = np.cross(vdir, world_up)
    if np.linalg.norm(right) < 1e-12:
        right = np.array([1.0, 0.0, 0.0], float)
    right /= np.linalg.norm(right)
    up = np.cross(vdir, right); up /= np.linalg.norm(up)
    forward = vdir / np.linalg.norm(vdir)
    return right, up, forward

def world_to_camera(P_world: np.ndarray, view: View) -> np.ndarray:
    """World -> camera coords (u,v,w). Note azimuth sign flip to match earlier 2D convention."""
    right, up, fwd = camera_axes_from_mpl(view.elev, -view.azim)
    R_cam = np.stack([right, up, fwd], axis=0)
    return (R_cam @ P_world.T).T

def project_from_camera(P_cam: np.ndarray, view: View) -> Tuple[np.ndarray, np.ndarray]:
    """Camera -> 2D projection (UV) and keep w."""
    u, v, w = P_cam[:,0], P_cam[:,1], P_cam[:,2]
    if view.mode == 'ortho':
        UV = np.stack([u, -v], axis=1)  # flip v to match mplot3d
    elif view.mode == 'persp':
        eps = 1e-9
        zc = np.maximum(w, eps)
        UV = np.stack([view.f * u / zc, view.f * -v / zc], axis=1)
    else:
        raise ValueError("view.mode must be 'ortho' or 'persp'")
    return UV, w


# =============================================================================
# PLOTTING
# =============================================================================

def draw_fitted_line_matplotlib(ax, fit, img_shape, color='r', linewidth=2):
    """Draw a fitted line on matplotlib axis."""
    vx, vy, x0, y0, _ = fit
    h, w = img_shape[:2]

    if abs(vx) < 1e-6:
        ax.plot([x0, x0], [0, h - 1], color=color, linewidth=linewidth, alpha=0.8)
        return

    t0 = (0 - x0) / vx
    t1 = ((w - 1) - x0) / vx
    y_at_0 = y0 + t0 * vy
    y_at_w = y0 + t1 * vy
    ax.plot([0, w - 1], [y_at_0, y_at_w], color=color, linewidth=linewidth, alpha=0.8)


def plot_overlay(img: np.ndarray, fits: List,
                 theta_deg: float, phi_deg: float, psi_deg: float,
                 t_nm: float, labeling: Dict, debug_info: dict = None):
    """
    Plot image with detected lines overlay.
    Uses the actual fitted lines (vx, vy, x0, y0) to draw correctly.
    """
    h, w = img.shape[:2]
    
    n_panels = 4 if debug_info else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]
    
    if debug_info:
        # Panel 1: Original
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('1. Original Image')
        axes[0].axis('off')
        
        # Panel 2: Preprocessed
        axes[1].imshow(debug_info.get('preprocessed', img), cmap='gray')
        axes[1].set_title('2. Preprocessed (CLAHE + bilateral + MORPH_CLOSE)')
        axes[1].axis('off')
        
        # Panel 3: Edges
        axes[2].imshow(debug_info.get('edges', img), cmap='gray')
        axes[2].set_title('3. Canny Edges')
        axes[2].axis('off')
        
        ax_final = axes[3]
    else:
        ax_final = axes[0]
    
    # Final panel: Image with detected lines using the actual fits
    ax_final.imshow(img, cmap='gray')
    
    # Get ordered indices from labeling
    idx_order = [labeling["far"], labeling["near"], labeling["bottom"]]
    colors = ['r', 'b', 'g']  # Red=farTop, Blue=nearTop, Green=bottom
    labels = ['farTop', 'nearTop', 'bottom']
    
    for j, idx in enumerate(idx_order):
        fit = fits[idx]
        color = colors[j % len(colors)]
        label = labels[j % len(labels)]
        draw_fitted_line_matplotlib(ax_final, fit, img.shape, color=color, linewidth=2)
        # Add label near the line
        vx, vy, x0, y0, angle = fit
        if abs(vx) > 1e-6:
            m = vy / vx
            b = y0 - m * x0
            ax_final.text(w - 50, m * (w - 50) + b, label, color=color, fontsize=10, 
                         fontweight='bold', va='center')
    
    ax_final.set_title(f"t ≈ {t_nm:.1f} nm | θ_tilt={theta_deg:.1f}°, φ={phi_deg:.2f}°, ψ_line={psi_deg:.2f}°")
    ax_final.set_axis_off()
    ax_final.set_xlim(0, w)
    ax_final.set_ylim(h, 0)
    
    plt.tight_layout()
    plt.savefig('overlay.png', dpi=150, bbox_inches='tight')
    print("Saved overlay.png")
    plt.show()


def plot_prism_2d(ax: plt.Axes, dims: PrismDims, view: View,
                  lw_visible: float = 1.6,
                  lw_hidden: float  = 1.2,
                  dash_pattern: Tuple[float, float] = (5.0, 3.0),
                  debug: bool = False,
                  clean_export: bool = False) -> None:
    """
    Draw edges using face-based classification (from visualize_structure.py):
      - visible (solid): edges between blue-blue or blue-red faces
      - hidden (dashed): edges between two red faces
    """
    V = prism_vertices(dims)
    P_cam = world_to_camera(V, view)
    UV, w = project_from_camera(P_cam, view)

    # Calculate camera direction vector - MUST match the one used in world_to_camera
    right, up, forward = camera_axes_from_mpl(view.elev, -view.azim)
    
    # Calculate dot product for each face to determine if blue or red
    face_colors_map = []  # True for blue (positive dot), False for red (negative dot)
    
    if debug:
        print("\n" + "="*60)
        print("FACE-BASED EDGE CLASSIFICATION")
        print("="*60)
        print("Face classification:")
    
    for i, face_indices in enumerate(FACES):
        normal, _ = calculate_face_normal(V, face_indices)
        dot_product = np.dot(normal, forward)
        is_blue = dot_product > 0
        face_colors_map.append(is_blue)
        if debug:
            color_name = "BLUE" if is_blue else "RED"
            print(f"  Face {i}: dot={dot_product:+.4f} -> {color_name}")
    
    # Create edge-to-faces mapping
    edge_to_faces = edges_to_faces_map(FACES)
    edges = unique_edges_from_faces(FACES)
    
    if debug:
        print(f"\nFound {len(edges)} unique edges")
        print("\nEdge styling:")
    
    # Classify edges
    solid_edges = []
    dashed_edges = []
    
    for edge in edges:
        v1, v2 = edge
        # Find which faces this edge belongs to
        face_indices = edge_to_faces.get(edge, [])
        
        if len(face_indices) == 2:
            # Edge between two faces
            face1_blue = face_colors_map[face_indices[0]]
            face2_blue = face_colors_map[face_indices[1]]
            
            # Dashed if: (red, red) - both faces facing away from camera
            # Solid if: (blue, blue) or (blue, red) or (red, blue)
            if not face1_blue and not face2_blue:
                # Both red -> dashed (hidden)
                dashed_edges.append(edge)
                if debug:
                    print(f"  Edge {v1}-{v2}: RED-RED -> DASHED")
            else:
                # At least one blue -> solid (visible)
                solid_edges.append(edge)
                if debug:
                    f1_type = "BLUE" if face1_blue else "RED"
                    f2_type = "BLUE" if face2_blue else "RED"
                    print(f"  Edge {v1}-{v2}: {f1_type}-{f2_type} -> SOLID")
        else:
            # Edge belongs to only one face (shouldn't happen for closed solid)
            if debug:
                print(f"  Edge {v1}-{v2}: WARNING - belongs to {len(face_indices)} faces")
            solid_edges.append(edge)  # Default to solid
    
    if debug:
        print(f"\nSummary: {len(solid_edges)} solid edges, {len(dashed_edges)} dashed edges")
        print("="*60)
    
    # Draw dashed edges first (hidden)
    for edge in dashed_edges:
        v1, v2 = edge
        ax.plot([UV[v1, 0], UV[v2, 0]], [UV[v1, 1], UV[v2, 1]],
                color='k', lw=lw_hidden,
                dashes=dash_pattern,
                solid_capstyle='round', dash_capstyle='round', zorder=2)
    
    # Draw solid edges on top (visible)
    for edge in solid_edges:
        v1, v2 = edge
        ax.plot([UV[v1, 0], UV[v2, 0]], [UV[v1, 1], UV[v2, 1]],
                linestyle='-', color='k', lw=lw_visible,
                solid_capstyle='round', zorder=3)

    # Axes housekeeping
    ax.set_aspect('equal', adjustable='datalim')
    allx, ally = UV[:,0], UV[:,1]
    cx, cy = np.mean(allx), np.mean(ally)
    rng = max(allx.max()-allx.min(), ally.max()-ally.min()) * 0.55 + 1e-6
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cy - rng, cy + rng)
    
    if clean_export:
        # Remove all decorations for clean export
        ax.axis('off')  # Remove axes, labels, ticks, and frame
        ax.set_facecolor('white')  # White background (or 'none' for transparent)
    else:
        ax.set_xlabel('u (camera x)')
        ax.set_ylabel('v (camera -y)')
        ax.grid(False)


def plot_prism_3d_styled_edges(vertices: np.ndarray, view: View):
    """
    Plot the prism in 3D with edges styled based on adjacent faces:
    - Solid edges: between two blue faces OR between blue and red faces
    - Dashed edges: between two red faces
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate camera direction vector
    right, up, forward = camera_axes_from_mpl(view.elev, -view.azim)
    
    # Calculate dot product for each face to determine if blue or red
    face_colors_map = []  # True for blue (positive dot), False for red (negative dot)
    
    for i, face_indices in enumerate(FACES):
        normal, _ = calculate_face_normal(vertices, face_indices)
        dot_product = np.dot(normal, forward)
        is_blue = dot_product > 0
        face_colors_map.append(is_blue)
    
    # Create edge-to-faces mapping
    edge_to_faces = edges_to_faces_map(FACES)
    edges = unique_edges_from_faces(FACES)
    
    # Prepare edges with styling
    solid_edges = []
    dashed_edges = []
    
    for edge in edges:
        v1, v2 = edge
        # Find which faces this edge belongs to
        face_indices = edge_to_faces.get(edge, [])
        
        if len(face_indices) == 2:
            # Edge between two faces
            face1_blue = face_colors_map[face_indices[0]]
            face2_blue = face_colors_map[face_indices[1]]
            
            # Solid if: (blue, blue) or (blue, red) or (red, blue)
            # Dashed if: (red, red)
            if not face1_blue and not face2_blue:
                # Both red -> dashed
                dashed_edges.append(edge)
            else:
                # At least one blue -> solid
                solid_edges.append(edge)
        else:
            solid_edges.append(edge)  # Default to solid
    
    # Draw faces (semi-transparent)
    face_polygons = []
    for face_indices in FACES:
        face_verts = vertices[face_indices]
        face_polygons.append(face_verts)
    
    face_collection = Poly3DCollection(
        face_polygons,
        facecolors=(0.8, 0.8, 0.8, 0.3),  # Light gray, very transparent
        edgecolors='none',  # No face edges, we'll draw them separately
        alpha=0.3
    )
    ax.add_collection3d(face_collection)
    
    # Draw solid edges
    for edge in solid_edges:
        v1, v2 = edge
        ax.plot3D(
            [vertices[v1, 0], vertices[v2, 0]],
            [vertices[v1, 1], vertices[v2, 1]],
            [vertices[v1, 2], vertices[v2, 2]],
            color='black',
            linewidth=2.0,
            solid_capstyle='round'
        )
    
    # Draw dashed edges
    for edge in dashed_edges:
        v1, v2 = edge
        ax.plot3D(
            [vertices[v1, 0], vertices[v2, 0]],
            [vertices[v1, 1], vertices[v2, 1]],
            [vertices[v1, 2], vertices[v2, 2]],
            color='black',
            linewidth=1.5,
            linestyle='--',
            dashes=(5, 3)
        )
    
    # Set equal aspect ratio and limits
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges))
    centers = mins + ranges / 2.0
    
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
    ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
    ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)
    
    ax.set_xlabel('x (nm)', fontsize=12)
    ax.set_ylabel('y (nm)', fontsize=12)
    ax.set_zlabel('z (nm)', fontsize=12)
    ax.set_title(f'Prism with Styled Edges\n(Solid=blue-blue or blue-red, Dashed=red-red)\n(azim={view.azim:.1f}°, elev={view.elev:.1f}°)', 
                 fontsize=14, fontweight='bold')
    
    ax.view_init(elev=view.elev, azim=view.azim)
    plt.tight_layout()
    return fig, ax


def visualize_prism(Wt: float, Wb: float, t: float, L: float, 
                    theta_deg: float, phi_deg: float, psi_deg: float,
                    show_2d: bool = True, show_3d: bool = True, debug: bool = False):
    """
    Visualize the prism using the sophisticated visualization from visualize_structure.py.
    
    Args:
        Wt, Wb, t, L: Prism dimensions
        theta_deg: Tilt angle (elevation)
        phi_deg: True azimuth (azimuth)
        psi_deg: Image line angle (for title only)
        show_2d: Whether to show 2D projection
        show_3d: Whether to show 3D view
        debug: Whether to print debug info
    """
    dims = PrismDims(Wt=Wt, Wb=Wb, t=t, L=L)
    # Convert theta (tilt) to elevation, phi (azimuth) to azimuth
    # In matplotlib: elev is angle from horizontal plane, azim is rotation about z-axis
    view = View(azim=phi_deg, elev=theta_deg, mode='ortho')
    
    if show_2d:
        # Regular plot with title and axes
        fig2d, ax2d = plt.subplots(figsize=(8, 8))
        plot_prism_2d(ax2d, dims, view, debug=debug, clean_export=False)
        ax2d.set_title(f"2D projection — ortho (azim={phi_deg:.1f}°, elev={theta_deg:.1f}°)\n"
                      f"Face-based hidden-line removal | t={t:.1f} nm")
        fig2d.tight_layout()
        plt.savefig('prism_2d_projection.png', dpi=150, bbox_inches='tight')
        print("Saved prism_2d_projection.png")
        plt.show()
        
        # Clean SVG export - only the projection lines, no axes/labels/title
        fig2d_clean, ax2d_clean = plt.subplots(figsize=(8, 8))
        plot_prism_2d(ax2d_clean, dims, view, debug=debug, clean_export=True)
        fig2d_clean.tight_layout()
        # For clean export, use transparent background and tight padding
        fig2d_clean.patch.set_facecolor('none')  # Transparent figure background
        plt.savefig('prism_2d_projection.svg', bbox_inches='tight', 
                   pad_inches=0, facecolor='none', transparent=True)
        print("Saved prism_2d_projection.svg (clean, lines only)")
        plt.close(fig2d_clean)  # Close the clean figure without showing
    
    if show_3d:
        V = prism_vertices(dims)
        fig3d, ax3d = plot_prism_3d_styled_edges(V, view)
        ax3d.set_title(f"3D Prism Visualization\n"
                      f"θ={theta_deg:.1f}°, φ={phi_deg:.2f}°, ψ={psi_deg:.2f}°, t={t:.1f} nm",
                      fontsize=14, fontweight='bold')
        plt.savefig('prism_3d.png', dpi=150, bbox_inches='tight')
        print("Saved prism_3d.png")
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Scale-free thickness extraction from tilted SEM (v2 with advanced visualization)")
    ap.add_argument("--image", default="cavity_sem_45degree.png", help="SEM image path")
    ap.add_argument("--wt", type=float, default=238.0, help="Top width Wt (nm)")
    ap.add_argument("--wb", type=float, default=314.0, help="Bottom width Wb (nm)")
    ap.add_argument("--wt-std", type=float, default=10.0, help="Std of Wt for MC (nm)")
    ap.add_argument("--wb-std", type=float, default=10.0, help="Std of Wb for MC (nm)")
    ap.add_argument("--mc-samples", type=int, default=2000, help="MC samples for uncertainty")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--theta", type=float, default=45.0, help="Tilt angle θ (deg)")
    ap.add_argument("--length", type=float, default=4000.0, help="3D prism length (nm)")

    ap.add_argument("--t-min", type=float, default=20.0, help="Min plausible |t| (nm)")
    ap.add_argument("--t-max", type=float, default=400.0, help="Max plausible |t| (nm)")
    ap.add_argument("--allow-negative", action="store_true", help="Allow negative t")

    ap.add_argument("--no-plots", action="store_true", help="Disable plots")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    
    # Visualization options
    ap.add_argument("--no-2d", action="store_true", help="Disable 2D projection plot")
    ap.add_argument("--no-3d", action="store_true", help="Disable 3D visualization")
    
    # Line detection parameters
    ap.add_argument("--hough-threshold", type=int, default=40, help="HoughLinesP threshold")
    ap.add_argument("--min-line-length", type=int, default=200, help="HoughLinesP minLineLength")
    ap.add_argument("--max-line-gap", type=int, default=20, help="HoughLinesP maxLineGap")
    ap.add_argument("--intercept-tol", type=float, default=4.0, help="Tolerance for grouping by intercept (px)")
    
    args = ap.parse_args()

    import os
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Load image
    print(f"Loading image: {args.image}")
    gray = read_gray(args.image)
    print(f"Image shape: {gray.shape}, dtype: {gray.dtype}")

    # Detect lines using the robust approach
    psi_deg, bs_raw, debug_info = detect_lines(
        gray,
        n_lines=3,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        intercept_tol=args.intercept_tol,
        debug=True
    )
    
    # Compute slope from angle
    m_common = math.tan(math.radians(psi_deg))
    
    print(f"Detected line angle ψ: {psi_deg:.3f}°")
    print(f"Detected line intercepts (sorted by y): {np.array2string(bs_raw, precision=2)}")
    print(f"Common slope m: {m_common:.3e}")

    # Compute true azimuth phi from image angle psi
    phi_deg = azimuth_from_image_angle(psi_deg, args.theta)
    print(f"Recovered true azimuth φ: {phi_deg:.3f}°")

    # Relabel lines using scale-free scoring
    bs_ordered, dbg = relabel_lines_scale_free(
        m=m_common,
        bs=bs_raw,
        Wt_nm=args.wt,
        Wb_nm=args.wb,
        theta_deg=args.theta,
        phi_deg=phi_deg,
        t_min_nm=args.t_min,
        t_max_nm=args.t_max,
        prefer_positive=(not args.allow_negative),
        verbose=True,
    )
    t_nominal = float(dbg["thickness_nm"])
    print(f"Chosen thickness (nominal): {t_nominal:.3f} nm")

    # Monte Carlo uncertainty from Wt/Wb
    rng = np.random.default_rng(args.seed)
    if args.mc_samples > 0 and (args.wt_std > 0 or args.wb_std > 0):
        Wt_samples = rng.normal(args.wt, max(args.wt_std, 0.0), size=args.mc_samples)
        Wb_samples = rng.normal(args.wb, max(args.wb_std, 0.0), size=args.mc_samples)
        Wt_samples = np.clip(Wt_samples, 1e-6, None)
        Wb_samples = np.clip(Wb_samples, 1e-6, None)

        dT_px = dbg["dT_px"]
        dNB_px = dbg["dNB_px"]

        t_samples = np.empty(args.mc_samples, dtype=float)
        for i in range(args.mc_samples):
            t_samples[i] = thickness_scale_free(
                dT_px, dNB_px,
                float(Wt_samples[i]), float(Wb_samples[i]),
                args.theta, phi_deg
            )

        t_mean = float(np.mean(t_samples))
        t_std = float(np.std(t_samples, ddof=1)) if args.mc_samples > 1 else 0.0
        t_p05, t_p50, t_p95 = [float(q) for q in np.percentile(t_samples, [5, 50, 95])]
    else:
        t_samples = np.array([t_nominal], dtype=float)
        t_mean = float(t_nominal)
        t_std = 0.0
        t_p05 = t_p50 = t_p95 = float(t_nominal)

    # Output
    out = {
        "input_file": args.image,
        "wt_nm": args.wt,
        "wb_nm": args.wb,
        "theta_deg": args.theta,
        "psi_deg": float(psi_deg),
        "phi_deg": float(phi_deg),
        "m_common": float(m_common),
        "b_raw": [float(x) for x in bs_raw],
        "b_ordered_far_near_bottom": [float(x) for x in bs_ordered],
        "dT_px": float(dbg["dT_px"]),
        "dNB_px": float(dbg["dNB_px"]),
        "ratio_dNB_over_dT": float(dbg["ratio_dNB_over_dT"]),
        "thickness_nm": float(t_nominal),
        "mc_samples": int(args.mc_samples),
        "thickness_mean_nm": float(t_mean),
        "thickness_std_nm": float(t_std),
        "thickness_p05_nm": float(t_p05),
        "thickness_p50_nm": float(t_p50),
        "thickness_p95_nm": float(t_p95),
    }

    print("\nResults:")
    print(json.dumps(out, indent=2))

    # Plots
    if not args.no_plots:
        # Pass the actual fits (from debug_info) and labeling info for correct drawing
        fits = debug_info["fits"]
        labeling = dbg["chosen_indices"]
        plot_overlay(gray, fits, args.theta, phi_deg, psi_deg, t_nominal, labeling, debug_info)
        
        # Use the sophisticated prism visualization
        visualize_prism(
            args.wt, args.wb, abs(t_nominal), args.length,
            args.theta, phi_deg, psi_deg,
            show_2d=not args.no_2d,
            show_3d=not args.no_3d,
            debug=args.debug
        )


if __name__ == "__main__":
    main()

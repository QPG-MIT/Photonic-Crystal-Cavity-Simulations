#!/usr/bin/env python3
"""
Trapezoidal Prism — 2D Projection with face-based hidden-line removal

Uses face-based edge classification (from prova_v8):
  - Dashed edges: between two red faces (both facing away from camera)
  - Solid edges: between blue-blue or blue-red faces

Conventions:
  - azim: Matplotlib azimuth (deg), rotation about world +z
  - elev: Matplotlib elevation (deg), tilt toward world +z

Camera model:
  1) Build a camera basis (right, up, forward) from (elev, azim) like mplot3d.
  2) Transform world points into camera coords: P_cam = R_cam @ P_world
  3) Project:
     - ortho: u = x_cam, v = -y_cam  (flip v to match mplot3d screen)
     - persp: u = f * x_cam / z_cam, v = f * -y_cam / z_cam (z_cam > 0 in front)
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Editable text in vector exports
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# ---------- Data classes ----------

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

# ---------- Geometry ----------

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

# ---------- Camera & projection ----------

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

# ---------- Plotting ----------

def plot_prism_2d(ax: plt.Axes, dims: PrismDims, view: View,
                  lw_visible: float = 1.6,
                  lw_hidden: float  = 1.2,
                  dash_pattern: Tuple[float, float] = (5.0, 3.0),
                  debug: bool = False,
                  clean_export: bool = False) -> None:
    """
    Draw edges using face-based classification (from prova_v8):
      - visible (solid): edges between blue-blue or blue-red faces
      - hidden (dashed): edges between two red faces
    """
    V = prism_vertices(dims)
    P_cam = world_to_camera(V, view)
    UV, w = project_from_camera(P_cam, view)

    # Calculate camera direction vector - MUST match the one used in world_to_camera
    # world_to_camera uses camera_axes_from_mpl(view.elev, -view.azim), so we do the same here
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

def plot_prism_3d_with_normals(vertices: np.ndarray, view: View):
    """
    Plot the prism in 3D with faces and outward normal vectors colored by dot product.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare face polygons for Poly3DCollection
    face_polygons = []
    for face_indices in FACES:
        face_verts = vertices[face_indices]
        face_polygons.append(face_verts)
    
    # Create Poly3DCollection for faces
    face_collection = Poly3DCollection(
        face_polygons,
        facecolors=(0.7, 0.8, 1.0, 0.6),  # Light blue, semi-transparent
        edgecolors='black',
        linewidths=1.5,
        alpha=0.6
    )
    ax.add_collection3d(face_collection)
    
    # Calculate and draw normal vectors
    ranges = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    max_range = float(np.max(ranges))
    normal_scale = max_range * 0.1
    
    # Calculate camera direction vector - MUST match world_to_camera (uses -view.azim)
    right, up, forward = camera_axes_from_mpl(view.elev, -view.azim)
    
    print("\n" + "="*60)
    print("FACE NORMALS")
    print("="*60)
    
    for i, (face_indices, face_name) in enumerate(zip(FACES, FACE_NAMES)):
        normal, center = calculate_face_normal(vertices, face_indices)
        
        # Calculate dot product with camera direction
        dot_product = np.dot(normal, forward)
        
        # Color: blue if positive dot product, red if negative
        if dot_product > 0:
            normal_color = 'blue'
            dot_sign = '+'
        else:
            normal_color = 'red'
            dot_sign = '-'
        
        # Draw normal vector as arrow
        ax.quiver(
            center[0], center[1], center[2],
            normal[0], normal[1], normal[2],
            length=normal_scale,
            color=normal_color,
            arrow_length_ratio=0.3,
            linewidth=2,
            label=f'{face_name}: dot={dot_product:+.3f}'
        )
        
        print(f"{face_name:12s}: center=({center[0]:7.2f}, {center[1]:7.2f}, {center[2]:7.2f}), "
              f"normal=({normal[0]:6.3f}, {normal[1]:6.3f}, {normal[2]:6.3f}), "
              f"dot(cam)={dot_product:+.4f} {dot_sign}")
    
    print("="*60)
    
    # Draw camera direction vector
    center_prism = np.mean(vertices, axis=0)
    camera_scale = max_range * 0.3
    ax.quiver(
        center_prism[0], center_prism[1], center_prism[2],
        forward[0], forward[1], forward[2],
        length=camera_scale,
        color='red',
        arrow_length_ratio=0.3,
        linewidth=3,
        label=f'Camera dir: ({forward[0]:.2f}, {forward[1]:.2f}, {forward[2]:.2f})'
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
    ax.set_title(f'Trapezoidal Prism with Outward Normal Vectors\n(azim={view.azim:.1f}°, elev={view.elev:.1f}°)', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(1.05, 1))
    ax.view_init(elev=view.elev, azim=view.azim)
    
    plt.tight_layout()
    return fig, ax


def plot_prism_3d_colored_faces(vertices: np.ndarray, view: View):
    """
    Plot the prism in 3D with faces colored based on normal vector dot product with camera.
    Blue faces: positive dot product (facing toward camera)
    Red faces: negative dot product (facing away from camera)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate camera direction vector
    right, up, forward = camera_axes_from_mpl(view.elev, view.azim)
    
    # Prepare face polygons and calculate colors based on dot product
    face_polygons = []
    face_colors = []
    
    print("\n" + "="*60)
    print("FACE COLORING BASED ON CAMERA DOT PRODUCT")
    print("="*60)
    
    for i, (face_indices, face_name) in enumerate(zip(FACES, FACE_NAMES)):
        face_verts = vertices[face_indices]
        face_polygons.append(face_verts)
        
        # Calculate normal and dot product
        normal, center = calculate_face_normal(vertices, face_indices)
        dot_product = np.dot(normal, forward)
        
        # Color faces: blue if positive dot product, red if negative
        if dot_product > 0:
            face_color = (0.3, 0.5, 1.0, 0.7)  # Blue, semi-transparent
            color_name = "BLUE"
        else:
            face_color = (1.0, 0.3, 0.3, 0.7)  # Red, semi-transparent
            color_name = "RED"
        
        face_colors.append(face_color)
        
        print(f"{face_name:12s}: dot(cam)={dot_product:+.4f} -> {color_name}")
    
    print("="*60)
    
    # Create Poly3DCollection with individual face colors
    face_collection = Poly3DCollection(
        face_polygons,
        facecolors=face_colors,
        edgecolors='black',
        linewidths=1.5,
        alpha=0.7
    )
    ax.add_collection3d(face_collection)
    
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
    ax.set_title(f'Prism with Colored Faces\n(Blue=toward camera, Red=away from camera)\n(azim={view.azim:.1f}°, elev={view.elev:.1f}°)', 
                 fontsize=14, fontweight='bold')
    
    ax.view_init(elev=view.elev, azim=view.azim)
    plt.tight_layout()
    return fig, ax


def plot_prism_3d_styled_edges(vertices: np.ndarray, view: View):
    """
    Plot the prism in 3D with edges styled based on adjacent faces:
    - Solid edges: between two blue faces OR between blue and red faces
    - Dashed edges: between two red faces
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate camera direction vector
    right, up, forward = camera_axes_from_mpl(view.elev, view.azim)
    
    # Calculate dot product for each face to determine if blue or red
    face_colors_map = []  # True for blue (positive dot), False for red (negative dot)
    
    print("\n" + "="*60)
    print("EDGE STYLING BASED ON ADJACENT FACES")
    print("="*60)
    print("Face classification:")
    
    for i, face_indices in enumerate(FACES):
        normal, _ = calculate_face_normal(vertices, face_indices)
        dot_product = np.dot(normal, forward)
        is_blue = dot_product > 0
        face_colors_map.append(is_blue)
        color_name = "BLUE" if is_blue else "RED"
        print(f"  Face {i} ({FACE_NAMES[i]}): dot={dot_product:+.4f} -> {color_name}")
    
    # Create edge-to-faces mapping
    edge_to_faces = edges_to_faces_map(FACES)
    edges = unique_edges_from_faces(FACES)
    
    print(f"\nFound {len(edges)} unique edges")
    
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
    
    print(f"Summary: {len(solid_edges)} solid edges, {len(dashed_edges)} dashed edges")
    print("="*60)
    
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


def plot_prism_3d(ax, dims: PrismDims, view: View,
                  mode: str = 'camera',
                  face_color=(0.7, 0.8, 1.0, 0.6), edge_color='k', linewidth: float = 0.8) -> None:
    V = prism_vertices(dims)
    if mode == 'camera':
        V_disp = V
        faces = [
            [V_disp[0], V_disp[1], V_disp[2], V_disp[3]],  # top
            [V_disp[4], V_disp[7], V_disp[6], V_disp[5]],  # bottom - corrected winding
            [V_disp[0], V_disp[4], V_disp[5], V_disp[1]],  # side +y - corrected winding
            [V_disp[3], V_disp[2], V_disp[6], V_disp[7]],  # side -y
            [V_disp[1], V_disp[5], V_disp[6], V_disp[2]],  # side -x - corrected winding
            [V_disp[0], V_disp[3], V_disp[7], V_disp[4]],  # side +x
        ]
        pc = Poly3DCollection(faces, facecolors=[face_color], edgecolors=edge_color, linewidths=linewidth)
        ax.add_collection3d(pc)

        mins = V_disp.min(axis=0); maxs = V_disp.max(axis=0)
        ranges = maxs - mins; max_range = float(np.max(ranges))
        centers = mins + ranges/2.0
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(centers[0]-max_range/2, centers[0]+max_range/2)
        ax.set_ylim(centers[1]-max_range/2, centers[1]+max_range/2)
        ax.set_zlim(centers[2]-max_range/2, centers[2]+max_range/2)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(elev=view.elev, azim=view.azim)

    elif mode == 'topdown':
        V_cam = world_to_camera(V, view)
        faces = [
            [V_cam[0], V_cam[1], V_cam[2], V_cam[3]],  # top
            [V_cam[4], V_cam[7], V_cam[6], V_cam[5]],  # bottom - corrected winding
            [V_cam[0], V_cam[4], V_cam[5], V_cam[1]],  # side +y - corrected winding
            [V_cam[3], V_cam[2], V_cam[6], V_cam[7]],  # side -y
            [V_cam[1], V_cam[5], V_cam[6], V_cam[2]],  # side -x - corrected winding
            [V_cam[0], V_cam[3], V_cam[7], V_cam[4]],  # side +x
        ]
        pc = Poly3DCollection(faces, facecolors=[face_color], edgecolors=edge_color, linewidths=linewidth)
        ax.add_collection3d(pc)

        mins = V_cam.min(axis=0); maxs = V_cam.max(axis=0)
        ranges = maxs - mins; max_range = float(np.max(ranges))
        centers = mins + ranges/2.0
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(centers[0]-max_range/2, centers[0]+max_range/2)
        ax.set_ylim(centers[1]-max_range/2, centers[1]+max_range/2)
        ax.set_zlim(centers[2]-max_range/2, centers[2]+max_range/2)
        ax.set_xlabel('u (camera x)'); ax.set_ylabel('v (camera y)'); ax.set_zlabel('w (camera z)')
        ax.view_init(elev=90.0, azim=0.0)
        ax.invert_yaxis()

    else:
        raise ValueError("plot_prism_3d mode must be 'camera' or 'topdown'")

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project a trapezoidal prism with face-based hidden-line removal")
    p.add_argument('--json', type=str, default='results.json', help='JSON file with parameters (default: results.json)')
    p.add_argument('--Wt', type=float, default=238.0, help='Top width')
    p.add_argument('--Wb', type=float, default=314.0, help='Bottom width')
    p.add_argument('--t',  type=float, default=136.0, help='Thickness')
    p.add_argument('--L',  type=float, default=2000.0, help='Length (extrusion along y)')

    p.add_argument('--azim', type=float, default=30, help='Matplotlib azimuth (deg, about world +z)')
    p.add_argument('--elev', type=float, default=45.0, help='Matplotlib elevation (deg)')

    p.add_argument('--mode', choices=['ortho','persp'], default='ortho', help='Projection mode')
    p.add_argument('--f', type=float, default=2000.0, help='Focal length for perspective')

    p.add_argument('--outfile', type=str, default='', help='Path to save (SVG/PDF/PNG). If empty, auto-saves SVG if JSON loaded.')
    p.add_argument('--dpi', type=int, default=300, help='DPI for raster outputs')
    p.add_argument('--show-2d', action='store_true', help='Show 2D projection (default if no flags)')
    p.add_argument('--show-3d', action='store_true', help='Show 3D view')
    p.add_argument('--both', action='store_true', help='Show both 2D and 3D views')
    p.add_argument('--no-show', action='store_true', help='Do not display; useful for headless saves')
    p.add_argument('--save-2d', type=str, default='', help='Optional path to save 2D figure')
    p.add_argument('--save-3d', type=str, default='', help='Optional path to save 3D figure')
    p.add_argument('--no-auto-save', action='store_true', help='Disable automatic SVG saving when JSON is loaded')
    p.add_argument('--clean-export', action='store_true', help='Export 2D projection without axes, labels, or background')

    p.add_argument('--view3d', choices=['camera','topdown'], default='camera',
                   help="3D mode: 'camera' uses ax.view_init(elev,azim) on unrotated world. 'topdown' draws in camera coords and looks down +w.")
    p.add_argument('--debug', action='store_true', help='Print debug info about face and edge classification')
    p.add_argument('--show-all-3d', action='store_true', help='Show all 3D visualization plots (normals, colored faces, styled edges)')
    
    # Line appearance
    p.add_argument('--lw-visible', type=float, default=1.6, help='Line width for visible edges')
    p.add_argument('--lw-hidden', type=float, default=1.2, help='Line width for hidden edges')
    p.add_argument('--dash-on', type=float, default=5.0, help='Dash pattern: length of dash (points)')
    p.add_argument('--dash-off', type=float, default=3.0, help='Dash pattern: length of gap (points)')
    
    return p.parse_args()

# ---------- Main ----------

def main():
    args = parse_args()

    # Load JSON file (default: results.json)
    data = {}
    if args.json:
        try:
            with open(args.json, 'r') as jf:
                content = jf.read()
                # Try to extract JSON from text (handle files with debug output before JSON)
                json_start = content.find('{')
                if json_start >= 0:
                    data = json.loads(content[json_start:])
                else:
                    data = json.loads(content)
        except FileNotFoundError:
            if args.json == 'results.json':
                print(f"Note: results.json not found, using default parameters")
            else:
                print(f"Failed to read JSON '{args.json}': file not found")
        except Exception as e:
            print(f"Failed to read JSON '{args.json}': {e}")
            if args.json == 'results.json':
                print(f"Note: Falling back to default parameters")

    # Precedence: JSON -> CLI -> default
    def pick(keys, fallback):
        for k in keys:
            if isinstance(data, dict) and data.get(k) is not None:
                return data.get(k)
        return fallback

    Wt   = float(pick(['wt_nm','Wt','wt'], args.Wt))
    Wb   = float(pick(['wb_nm','Wb','wb'], args.Wb))
    t    = float(pick(['thickness_nm','t','thickness'], args.t))
    L    = float(pick(['length_nm','L','length'], args.L))
    azim = float(pick(['true_azimuth_deg','phi','phi_deg','azim'], args.azim))
    elev = float(pick(['theta_deg','theta','elev'], args.elev))
    
    # Print loaded parameters
    if args.json:
        print(f"\nLoaded from JSON:")
        print(f"  Wt={Wt:.2f} nm, Wb={Wb:.2f} nm, t={t:.2f} nm, L={L:.2f} nm")
        print(f"  azim={azim:.2f}°, elev={elev:.2f}°")
    mode = str(pick(['mode'], args.mode))
    f    = float(pick(['focal_length','f'], args.f))

    dims = PrismDims(Wt=Wt, Wb=Wb, t=t, L=L)
    view = View(azim=azim, elev=elev, mode=mode, f=f)

    any_flag = args.show_2d or args.show_3d or args.both
    show_2d = True if not any_flag else (args.show_2d or args.both)
    show_3d = True if not any_flag else (args.show_3d or args.both)

    if show_2d:
        figsize = (8, 8) if args.debug else (5, 5)
        fig2d, ax2d = plt.subplots(figsize=figsize)
        plot_prism_2d(ax2d, dims, view, 
                      lw_visible=args.lw_visible,
                      lw_hidden=args.lw_hidden,
                      dash_pattern=(args.dash_on, args.dash_off),
                      debug=args.debug,
                      clean_export=args.clean_export)
        if not args.clean_export:
            ax2d.set_title(f"2D projection — {view.mode} (azim={azim}°, elev={elev}°)\nFace-based hidden-line removal")
        fig2d.tight_layout()
        
        # Determine output file name
        outfile_2d = args.outfile
        if not outfile_2d and not args.no_auto_save and data:
            # Auto-generate SVG filename from JSON if available
            json_base = args.json if args.json else 'results.json'
            if isinstance(json_base, str) and json_base.endswith('.json'):
                json_base = json_base[:-5]
            outfile_2d = f"{json_base}_projection.svg"
        
        if outfile_2d and not show_3d:
            # For clean export, use transparent background and tight padding
            if args.clean_export:
                fig2d.patch.set_facecolor('none')  # Transparent figure background
                fig2d.savefig(outfile_2d, dpi=args.dpi, bbox_inches='tight', 
                             pad_inches=0, facecolor='none', transparent=True)
            else:
                fig2d.savefig(outfile_2d, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved 2D projection to {outfile_2d}")
        if args.save_2d:
            if args.clean_export:
                fig2d.patch.set_facecolor('none')
                fig2d.savefig(args.save_2d, dpi=args.dpi, bbox_inches='tight',
                             pad_inches=0, facecolor='none', transparent=True)
            else:
                fig2d.savefig(args.save_2d, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved 2D to {args.save_2d}")

    if show_3d:
        fig3d = plt.figure(figsize=(6,6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        plot_prism_3d(ax3d, dims, view, mode=args.view3d)
        title3d = "3D view — camera" if args.view3d == 'camera' else "3D view — topdown (camera coords)"
        ax3d.set_title(f"{title3d} (azim={azim}°, elev={elev}°)")
        fig3d.tight_layout()
        if args.save_3d:
            fig3d.savefig(args.save_3d, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved 3D to {args.save_3d}")
    
    # Show all 3D visualization plots for verification
    if args.show_all_3d:
        V = prism_vertices(dims)
        
        print("\n" + "="*60)
        print("CREATING ALL 3D VERIFICATION PLOTS")
        print("="*60)
        
        print("\n1. Plot with normal vectors...")
        fig1, ax1 = plot_prism_3d_with_normals(V, view)
        
        print("\n2. Plot with colored faces...")
        fig2, ax2 = plot_prism_3d_colored_faces(V, view)
        
        print("\n3. Plot with styled edges...")
        fig3, ax3 = plot_prism_3d_styled_edges(V, view)

    if not args.no_show:
        if (not args.outfile) or show_3d:
            plt.show()

if __name__ == '__main__':
    main()


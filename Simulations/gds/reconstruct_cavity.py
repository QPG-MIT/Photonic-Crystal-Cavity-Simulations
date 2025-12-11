#!/usr/bin/env python3
"""
Fabrication helper from design GDS files.

Starting from `Cavity_Design.gds` and `Holes_Design.gds`:
- Scale cavity in Y to target height (default 0.274 µm)
- Scale holes so the rightmost hole radius equals target (default 0.043 µm)
- Write scaled holes to `Holes_Fab.gds`
- Create `Cavity_reconstruct.gds` by making a rectangle of the scaled cavity
  bounding box and subtracting the scaled holes geometry.

Requires: gdsfactory, gdstk (provided via requirements.txt)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import click


def _to_polygons_flatten(input_gds: Path):
    """Reads a GDS and returns a flattened gdstk.Cell with only polygons.

    - Merges all top-level cells into a single temporary cell
    - Brings referenced geometry into the cell (flatten)
    - Converts FlexPath/RobustPath objects to polygons
    - Clears paths, leaving only polygonal geometry in `cell.polygons`
    """
    import gdstk  # type: ignore

    lib = gdstk.read_gds(str(input_gds))
    tops = lib.top_level()
    if not tops:
        raise ValueError(f"No top-level cells found in GDS: {input_gds}")

    if len(tops) == 1:
        # Force a consistent top-level cell name for downstream writing
        top = tops[0].copy(name="TOP")
    else:
        # Merge all top-level cells into one cell via references
        merged = gdstk.Cell("TOP")
        for i, c in enumerate(tops):
            merged.add(gdstk.Reference(c))
        merged = merged.copy(name="TOP")
        top = merged

    # Bring all referenced geometry into this cell
    top.flatten(True)

    # Convert paths to polygons and remove paths when present
    new_polys = []
    for fp in list(getattr(top, "flexpaths", []) or []):
        new_polys.extend(fp.to_polygons())
    for rp in list(getattr(top, "robustpaths", []) or []):
        new_polys.extend(rp.to_polygons())
    for poly in new_polys:
        top.add(poly)
    if hasattr(top, "flexpaths"):
        try:
            top.flexpaths = []  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(top, "robustpaths"):
        try:
            top.robustpaths = []  # type: ignore[attr-defined]
        except Exception:
            pass

    return top


def _get_bbox(cell) -> Tuple[float, float, float, float]:
    """Returns (xmin, ymin, xmax, ymax) for a gdstk.Cell's geometry."""
    import numpy as np  # local import to keep module deps minimal at import-time

    bb = cell.bounding_box()
    if bb is None:
        raise ValueError("No geometry found to compute bounding box.")
    xmin, ymin = float(bb[0][0]), float(bb[0][1])
    xmax, ymax = float(bb[1][0]), float(bb[1][1])
    # Defensive: if bbox is degenerate
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("Degenerate bounding box computed from cavity file.")
    return xmin, ymin, xmax, ymax


def _boolean_subtract_rect_minus_polys(
    rect_poly, holes_polys: List
):
    """Returns polygons for (rect) NOT (holes_polys) using gdstk.boolean.

    The resulting polygons inherit the layer/datatype from `rect_poly`.
    """
    import gdstk  # type: ignore

    result = gdstk.boolean(
        [rect_poly],
        holes_polys if holes_polys else [],
        "not",
    )
    return result or []


def _scale_cell_y(cell, scale_y: float):
    """Scale all polygons in a gdstk.Cell by (1.0, scale_y) in-place."""
    for poly in list(cell.polygons):
        poly.scale(1.0, scale_y)
    return cell


def _estimate_center_radius(poly) -> Tuple[Tuple[float, float], float]:
    """Estimate circle center/radius for a hole polygon using algebraic circle fit.

    Uses Kasa's least-squares fit on polygon vertices for improved center
    stability vs simple centroid averaging, which helps preserve alignment.
    Falls back to bbox-based estimate when necessary.
    """
    import numpy as np

    pts = getattr(poly, "points", None)
    if pts is None or len(pts) < 3:
        # fallback via bbox
        bb = poly.bounding_box()
        (x0, y0), (x1, y1) = bb
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        r = max(abs(x1 - x0), abs(y1 - y0)) / 2.0
        return (cx, cy), float(r)

    P = np.asarray(pts, dtype=float)
    x = P[:, 0]
    y = P[:, 1]
    # Kasa fit: solve A*[a,b,c]^T ≈ bvec where x^2+y^2 + a x + b y + c = 0
    A = np.column_stack([x, y, np.ones_like(x)])
    bvec = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        a, b, c = sol
        cx = float(-a / 2.0)
        cy = float(-b / 2.0)
        r = float(np.sqrt(max(cx * cx + cy * cy - c, 0.0)))
    except Exception:
        # Fallback to mean-based estimate
        cxy = P.mean(axis=0)
        r = float(np.linalg.norm(P - cxy, axis=1).mean())
        cx, cy = float(cxy[0]), float(cxy[1])
    return (cx, cy), r


def _find_rightmost_hole(polys: List) -> Tuple[int, Tuple[float, float], float]:
    best = (-1, -1e30, (0.0, 0.0), 0.0)
    for i, p in enumerate(polys):
        (cx, cy), r = _estimate_center_radius(p)
        if cx > best[1]:
            best = (i, cx, (cx, cy), r)
    if best[0] < 0:
        raise ValueError("No hole polygons found.")
    return best[0], best[2], float(best[3])


def _scale_holes_to_radius(holes_cell, target_radius: float):
    """Scale all hole polygons about their own centers so rightmost has target radius."""
    if not holes_cell.polygons:
        raise ValueError("No polygons found in holes cell.")
    idx, _, r = _find_rightmost_hole(list(holes_cell.polygons))
    if r <= 0:
        raise ValueError("Found non-positive measured hole radius.")
    s = float(target_radius) / float(r)
    # scale each polygon about its own center
    for p in list(holes_cell.polygons):
        (cx, cy), _ = _estimate_center_radius(p)
        p.scale(s, s, center=(cx, cy))
    return holes_cell, s


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--cavity-design",
    "cavity_design_gds",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("gds") / "Cavity_Design.gds",
    show_default=True,
    help="Input cavity design GDS file.",
)
@click.option(
    "--holes-design",
    "holes_design_gds",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("gds") / "Holes_Design.gds",
    show_default=True,
    help="Input holes design GDS file.",
)
@click.option(
    "--output",
    "cavity_reconstruct_gds",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("gds") / "Cavity_Fab.gds",
    show_default=True,
    help="Output GDS path for the reconstructed cavity (rectangle minus holes).",
)
@click.option(
    "--holes-fab",
    "holes_fab_gds",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("gds") / "Holes_Fab.gds",
    show_default=True,
    help="Output GDS path for the scaled holes (fabrication).",
)
@click.option(
    "--target-height",
    type=float,
    default=0.314, # used to be 0.274
    show_default=True,
    help="Target cavity total height in microns (Y-extent).",
)
@click.option(
    "--target-hole-radius",
    type=float,
    default=0.043,
    show_default=True,
    help="Target hole radius in microns for the rightmost hole.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Write the reconstructed layout to --output.",
)
@click.option(
    "--show/--no-show",
    default=True,
    show_default=True,
    help="Open viewer to preview the reconstructed layout.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    show_default=True,
    help="Whether to overwrite output if it exists (only with --save).",
)
def main(
    cavity_design_gds: Path,
    holes_design_gds: Path,
    cavity_reconstruct_gds: Path,
    holes_fab_gds: Path,
    target_height: float,
    target_hole_radius: float,
    save: bool,
    show: bool,
    overwrite: bool,
):
    """From design GDS, generate Holes_Fab and Cavity_reconstruct."""
    import gdstk  # type: ignore
    import gdsfactory as gf  # type: ignore

    # Load design cells (merged + flattened + polygonized)
    cavity_cell = _to_polygons_flatten(cavity_design_gds)
    holes_cell = _to_polygons_flatten(holes_design_gds)

    if not cavity_cell.polygons:
        raise ValueError("No polygons found in the cavity GDS.")

    # 1) Scale cavity in Y to achieve target height
    cx0, cy0, cx1, cy1 = _get_bbox(cavity_cell)
    current_height = cy1 - cy0
    if current_height <= 0:
        raise ValueError("Detected non-positive cavity height from the design GDS.")
    sy = float(target_height) / float(current_height)
    _scale_cell_y(cavity_cell, sy)

    # bbox after scaling
    xmin, ymin, xmax, ymax = _get_bbox(cavity_cell)

    # Derive layer/datatype from first cavity polygon when available
    base_layer = getattr(cavity_cell.polygons[0], "layer", 0)
    base_datatype = getattr(cavity_cell.polygons[0], "datatype", 0)

    # 2) Build rectangle covering the scaled cavity extents
    rect = gdstk.rectangle((xmin, ymin), (xmax, ymax), layer=base_layer, datatype=base_datatype)

    # 3) Scale holes so rightmost hole radius matches target
    #    (and filter out degenerate polygons)
    # Clean degenerate first
    cleaned_holes = []
    for p in list(holes_cell.polygons):
        try:
            a = abs(p.area())
        except Exception:
            bb = p.bounding_box()
            if bb is None:
                continue
            (x0, y0), (x1, y1) = bb
            a = max(0.0, (x1 - x0) * (y1 - y0))
        if a > 1e-9:
            cleaned_holes.append(p)
    # Recreate holes_cell with only cleaned polygons
    import gdstk as _gdstk
    tmp_holes_cell = _gdstk.Cell("TOP")
    for p in cleaned_holes:
        tmp_holes_cell.add(p)

    if not tmp_holes_cell.polygons:
        raise ValueError("No valid hole polygons found in the design GDS.")

    holes_cell, scale_s = _scale_holes_to_radius(tmp_holes_cell, target_hole_radius)

    # 4) Save Holes_Fab.gds
    if save:
        out_lib_h = gdstk.Library()
        # Union the holes to merge touching parts without affecting outer subtraction
        try:
            unioned_holes = gdstk.boolean(list(holes_cell.polygons), [], "or")
        except Exception:
            unioned_holes = None
        out_cell_h = gdstk.Cell("TOP")
        for p in (unioned_holes or list(holes_cell.polygons)):
            out_cell_h.add(p)
        out_lib_h.add(out_cell_h)
        if not overwrite and holes_fab_gds.exists():
            raise FileExistsError(f"Output file already exists: {holes_fab_gds}")
        holes_fab_gds.parent.mkdir(parents=True, exist_ok=True)

        # Import into gdsfactory Component for preview and writing
        import os
        import tempfile
        tmp_holes = Path(tempfile.mktemp(suffix="_holes_fab.gds"))
        try:
            out_lib_h.write_gds(str(tmp_holes))
            comp_holes = gf.import_gds(str(tmp_holes))
            try:
                comp_holes.name = "Holes_Fab_preview"
            except Exception:
                pass
        finally:
            try:
                tmp_holes.unlink(missing_ok=True)
            except Exception:
                pass

        if show:
            comp_holes.show()
        # Save using gdstk to preserve top cell name TOP
        out_lib_h.write_gds(str(holes_fab_gds))

    # 5) Perform boolean subtraction: rectangle minus scaled holes
    result_polys = _boolean_subtract_rect_minus_polys(rect, list(holes_cell.polygons))

    # Ensure resulting polygons inherit base layer/datatype
    for p in result_polys:
        try:
            p.layer = base_layer
        except Exception:
            pass
        try:
            p.datatype = base_datatype
        except Exception:
            pass

    # Assemble output cell
    out_cell = gdstk.Cell("TOP")
    for p in result_polys:
        out_cell.add(p)

    # Write to temp GDS and import in gdsfactory for viewing/writing
    out_lib = gdstk.Library()
    out_lib.add(out_cell)

    import tempfile
    tmp_path = Path(tempfile.mktemp(suffix="_cavity_reconstructed.gds"))
    try:
        out_lib.write_gds(str(tmp_path))
        comp = gf.import_gds(str(tmp_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Keep as gdsfactory Component for preview only; do not modify topology
    cavity_fab_preview = gf.boolean(
        comp,
        comp,
        operation="or",
        layer=(base_layer, base_datatype),
    )

    try:
        cavity_fab_preview.name = "Cavity_Fab_preview"
    except Exception:
        pass

    # Preview
    if show:
        cavity_fab_preview.show()

    # Save reconstructed cavity using the preview geometry, but ensure TOP topcell name
    if save:
        if not overwrite and cavity_reconstruct_gds.exists():
            raise FileExistsError(f"Output file already exists: {cavity_reconstruct_gds}")
        cavity_reconstruct_gds.parent.mkdir(parents=True, exist_ok=True)

        # Export preview to temp via gdsfactory, then reload in gdstk to force TOP name
        import tempfile
        tmp_preview = Path(tempfile.mktemp(suffix="_cavity_preview.gds"))
        try:
            import gdsfactory as _gf  # type: ignore
            cavity_fab_preview.write_gds(str(tmp_preview))

            import gdstk as _gdstk2  # type: ignore
            lib_prev = _gdstk2.read_gds(str(tmp_preview))
            tops_prev = lib_prev.top_level()
            if not tops_prev:
                raise RuntimeError("No top-level cell found in preview export.")

            # Create new library with a single top cell named TOP containing the preview
            out_lib_final = _gdstk2.Library()
            if len(tops_prev) == 1:
                top_prev = tops_prev[0]
                top_final = top_prev.copy(name="TOP")
            else:
                # Merge multiple tops under one TOP
                top_final = _gdstk2.Cell("TOP")
                for c in tops_prev:
                    top_final.add(_gdstk2.Reference(c))
                top_final = top_final.copy(name="TOP")
            # Bring all referenced geometry into TOP to avoid external refs in output
            try:
                top_final.flatten(True)
            except Exception:
                pass
            out_lib_final.add(top_final)
            out_lib_final.write_gds(str(cavity_reconstruct_gds))
        finally:
            try:
                tmp_preview.unlink(missing_ok=True)
            except Exception:
                pass
        click.echo(f"Saved: {cavity_reconstruct_gds}")


if __name__ == "__main__":
    main()



"""Plot styling utilities for consistent visuals."""
from matplotlib import pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


blue  = "#002642"
red = "#840032"
yellow = "#e59500"
white = "#FFFAF2"
black  = "#02040f"

# Discrete color palette used across all plots
PALETTE = [red, blue, yellow, black]

# For line plots (discrete)
line_cmap = ListedColormap(PALETTE, name="line_palette")

# ---- Monopolar colormap (low → high) ----
mono_cmap = LinearSegmentedColormap.from_list(
    "mono", [black, blue, red, yellow, white], N=256
)

# ---- Bipolar colormap (negative ↔ positive) ----
bipolar_cmap = LinearSegmentedColormap.from_list(
    "bipolar", [black, blue, white, yellow, red], N=256
)

def apply_theme() -> None:
    """Apply a simple, elegant plotting style using the global palette."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
    plt.rcParams["figure.autolayout"] = True

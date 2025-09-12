"""Plot styling utilities for consistent visuals."""
from matplotlib import pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ---- Color definitions from prova_colori.py ----
dark_green = "#283618"
light_green = "#606C38"
cream = "#FEFAE0"
light_brown = "#DDA15E"
dark_brown = "#BC6C25"

# Discrete color palette used across all plots
PALETTE = [
    dark_green, 
    light_brown, 
    light_green,
    dark_brown, 
    cream
]

# For line plots
line_cmap = ListedColormap(PALETTE)

# ---- Build a monopolar colormap (low → high) ----
# Interpolates smoothly across the palette
mono_cmap = LinearSegmentedColormap.from_list(
    "mono", [dark_green, cream], N=256
)

# ---- Build a bipolar colormap (negative ↔ positive) ----
# Centered at zero, so you may want symmetric coloring
bipolar_cmap = LinearSegmentedColormap.from_list("bipolar", [dark_brown, cream, dark_green], N=256)

def apply_theme() -> None:
    """Apply a simple, elegant plotting style using the global palette."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
    plt.rcParams["figure.autolayout"] = True

"""Plot styling utilities for consistent visuals."""
from matplotlib import pyplot as plt
from cycler import cycler

# Discrete color palette used across all plots
PALETTE = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#EECA3B",  # yellow
    "#B279A2",  # purple
]

def apply_theme() -> None:
    """Apply a simple, elegant plotting style using the global palette."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
    plt.rcParams["figure.autolayout"] = True

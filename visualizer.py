# visualizer.py
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for server/Gradio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# WHO/AEQ PM2.5 breakpoints (μg/m³) for color bands annotation
PM25_LEVELS = [
    (0,   15,  "Good",      "#00e400"),
    (15,  35,  "Moderate",  "#ffff00"),
    (35,  55,  "Sensitive", "#ff7e00"),
    (55,  150, "Unhealthy", "#ff0000"),
    (150, 300, "Hazardous", "#7e0023"),
]


def make_heatmap(
    data_2d:  np.ndarray,
    title:    str,
    vmin:     float = 0.0,
    vmax:     float = 200.0,
    lat:      np.ndarray = None,
    lon:      np.ndarray = None,
    figsize:  tuple = (6, 5),
    dpi:      int   = 110,
) -> Image.Image:
    """
    Render a PM2.5 spatial heatmap.

    Parameters
    ----------
    data_2d : np.ndarray (H, W)
        PM2.5 values in μg/m³.
    title : str
    vmin, vmax : float
        Color scale range.
    lat, lon : np.ndarray (H, W) or None
        Geographic coordinates. If None, pixel indices are used.
    figsize, dpi : figure size and resolution.

    Returns
    -------
    PIL.Image.Image — PNG image ready for Gradio.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    # Extent for geographic axes
    if lat is not None and lon is not None:
        extent = [
            float(lon.min()), float(lon.max()),
            float(lat.min()), float(lat.max()),
        ]
        xlabel, ylabel = "Longitude (°E)", "Latitude (°N)"
    else:
        extent = None
        xlabel, ylabel = "Grid X", "Grid Y"

    cmap = plt.get_cmap("RdYlGn_r")
    im   = ax.imshow(
        data_2d,
        cmap    = cmap,
        vmin    = vmin,
        vmax    = vmax,
        aspect  = "auto",
        extent  = extent,
        origin  = "lower",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PM2.5 (μg/m³)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Axes labels and ticks
    ax.set_xlabel(xlabel, color="white", fontsize=8)
    ax.set_ylabel(ylabel, color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    ax.set_title(title, color="white", fontsize=9, pad=8, fontweight="bold")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()   # .copy() detaches from the BytesIO buffer


def compute_stats(pred_2d: np.ndarray, input_2d: np.ndarray) -> dict:
    """
    Compute summary statistics for a single (H, W) prediction frame.
    Returns a plain dict of human-readable strings.
    """
    return {
        "Predicted Mean PM2.5":          f"{float(pred_2d.mean()):.1f} μg/m³",
        "Predicted Max PM2.5":           f"{float(pred_2d.max()):.1f} μg/m³",
        "Input Mean PM2.5":              f"{float(input_2d.mean()):.1f} μg/m³",
        "High-Risk Pixels  (>75 μg/m³)": str(int((pred_2d > 75).sum())),
        "Unhealthy Pixels (>150 μg/m³)": str(int((pred_2d > 150).sum())),
        "Change vs Input":               f"{float(pred_2d.mean() - input_2d.mean()):+.1f} μg/m³",
    }

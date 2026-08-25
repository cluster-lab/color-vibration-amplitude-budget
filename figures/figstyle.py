"""Shared style for every result figure.

optica-article is article at 10pt, \textwidth = 379.4pt = 5.25 in, captions \small = 9pt.
Figures are drawn at the printed width so 9 pt here is 9 pt on the page.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from _paths import OUT
W = 5.25
CAP = 9.0        # the caption size on the page; nothing in a figure is drawn this large
FS = 8.0         # figure default, one point under the caption
FS_S = 7.0       # secondary text: legends, in-plot labels, tick marks
FS_XS = 7.2      # the numbers inside the discs of the grid figures; figures are now
                 # placed at native size, so this prints as a true 7.2 pt

plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS_S, "ytick.labelsize": FS_S, "legend.fontsize": FS_S,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    # inward ticks, the optics convention for the target venue
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.top": False, "ytick.right": False,
    "axes.grid": False,
    "grid.alpha": 0.30, "grid.linewidth": 0.5, "grid.color": "#DDDDDD",
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_LUM, C_RG, C_S = "#000000", "#D55E00", "#0072B2"
OI = {"orange": "#E69F00", "green": "#009E73", "pink": "#CC79A7",
      "yellow": "#B8960B", "grey": "#6E6E6E", "blue": "#0072B2"}
CMAP = "viridis"          # perceptually uniform and monotonic in lightness

# The five ellipses. The hue each one names is kept, which reads better on the
# chromaticity plane than a maximally separated set does. Under simulated
# dichromacy these colors are not all separable on their own, so marker, dash
# and a direct label are carried alongside: no reading of these figures rests
# on color, and any one of the four channels is enough.
# E14 is the shortest trajectory of the five and the easiest to lose, so it
# takes the solid line and E8, which runs clear of everything, takes the dots.
ELL = {
    8:  {"c": "#009E73", "m": "o", "ls": ":"},      # green
    12: {"c": "#6E6E6E", "m": "s", "ls": "--"},     # near-achromatic, the control
    13: {"c": "#B8960B", "m": "^", "ls": "-."},     # pale yellow
    14: {"c": "#E69F00", "m": "D", "ls": "-"},      # orange
    19: {"c": "#CC79A7", "m": "v", "ls": "-"},      # pink
}
ENAME = {8: "E8", 12: "E12", 13: "E13", 14: "E14", 19: "E19"}


def surface(ax, xs, ys, ps, method="cubic", n=320, label_at=None):
    """Interpolated detection surface with contours; returns the mappable.

    The surface is drawn only over the box the data actually spans, so nothing
    on the page is extrapolated."""
    xlim = (float(np.min(xs)), float(np.max(xs)))
    ylim = (float(np.min(ys)), float(np.max(ys)))
    gx = np.linspace(xlim[0], xlim[1], n)
    gy = np.linspace(ylim[0], ylim[1], n)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([xs, ys])
    Z = griddata(pts, ps, (GX, GY), method=method)
    Zn = griddata(pts, ps, (GX, GY), method="nearest")
    Z = np.where(np.isnan(Z), Zn, Z)          # fill outside the convex hull
    Z = np.clip(Z, 0.0, 1.0)
    im = ax.imshow(Z, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="bilinear",
                   zorder=1)
    minor = [l for l in np.arange(0.1, 1.0, 0.1) if abs(l - 0.5) > 1e-9]
    ax.contour(GX, GY, Z, levels=minor, colors="white", linewidths=0.6,
               alpha=0.45, zorder=2)
    c5 = ax.contour(GX, GY, Z, levels=[0.5], colors="white", linewidths=2.0, zorder=3)
    if label_at is None:
        ax.clabel(c5, fmt=lambda v: "$P(A)=0.5$", fontsize=FS_S, colors="white")
    else:
        ax.text(label_at[0], label_at[1], "$P(A)=0.5$", color="white",
                fontsize=FS_S, ha="center", va="center", zorder=8)
    return im


def levels(vals, gap=0.02):
    """Group the measured values into the levels the design asked for.

    8-bit quantization scatters each nominal level over a small range, so one
    line is drawn at the centre of each group; how far a cell sits from its
    line is the quantization error itself."""
    out, cur = [], [sorted(vals)[0]]
    for v in sorted(vals)[1:]:
        if v - cur[-1] > gap:
            out.append(cur); cur = []
        cur.append(v)
    out.append(cur)
    return [float(np.mean(g)) for g in out]


def gridlines(ax, xs, ys, fine=0.05):
    """A faint dotted lattice, with the sampled levels drawn solid on top of it.

    Lines are clipped to the surface so none of them runs out over white."""
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    for v in np.arange(0, x1 + 1e-9, fine):
        if x0 <= v <= x1:
            ax.plot([v, v], [y0, y1], color="white", lw=0.4, ls=":", alpha=0.22, zorder=4)
    for v in np.arange(0, y1 + 1e-9, fine):
        if y0 <= v <= y1:
            ax.plot([x0, x1], [v, v], color="white", lw=0.4, ls=":", alpha=0.22, zorder=4)
    for v in levels(xs):
        ax.plot([v, v], [y0, y1], color="white", lw=0.9, alpha=0.38, zorder=4)
    for v in levels(ys):
        ax.plot([x0, x1], [v, v], color="white", lw=0.9, alpha=0.38, zorder=4)


def cells(ax, xs, ys, ps, r=None):
    """The measured cells, labelled with their detection rate."""
    ax.scatter(xs, ys, c=ps, cmap=CMAP, vmin=0, vmax=1, s=300,
               edgecolors="white", linewidths=1.0, zorder=6)
    for x, y, p in zip(xs, ys, ps):
        ax.text(x, y, "%.2f" % p, ha="center", va="center", fontsize=FS_XS,
                color="white" if p < 0.62 else "black", zorder=7)


def bar(fig, ax, im):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("$P(A)$", fontsize=FS)
    cb.ax.tick_params(labelsize=FS_S, direction="out", size=2.5)
    cb.outline.set_linewidth(0.6)
    return cb


def panel(ax, tag):
    """A panel label, set clear of the plotting box rather than against it."""
    ax.set_title(tag, fontsize=FS, loc="left", pad=8)


def one_zero(ax, keep="x"):
    """When both axes start at zero, only one of them says so.

    Two zeros meeting in the corner read as one number belonging to neither
    axis, so the label is kept on `keep` and blanked on the other."""
    other = ax.get_yaxis() if keep == "x" else ax.get_xaxis()
    ticks = ax.get_yticks() if keep == "x" else ax.get_xticks()
    dec = max([len(("%g" % round(t, 10)).split(".")[-1]) if "." in ("%g" % round(t, 10))
               else 0 for t in ticks] + [0])
    fmt = lambda v, _pos: "" if abs(v) < 1e-12 else ("%.*f" % (dec, v))
    other.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(fmt))
    return ticks


def finish(fig, name):
    """Both go to out/: the PDF is what the manuscript includes, the PNG is
    there to be looked at."""
    fig.savefig("%s/%s.pdf" % (OUT, name))
    fig.savefig("%s/%s.png" % (OUT, name), dpi=170)
    plt.close(fig)
    print("  %s" % name)

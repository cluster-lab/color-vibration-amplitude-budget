"""The two orthogonal-grid maps, drawn the same way: System C and System B.

The surface is a tensor-product spline on the design lattice rather than a
triangulation, so the contours curve instead of breaking into straight segments;
System B has only three a_RG levels, so it is quadratic in that direction and
cubic in the other. A faint grey lattice runs under everything, which is what
shows in the white margin outside the measured region."""
import json, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from figstyle import W, FS_S, CMAP, levels, cells, bar, finish, one_zero
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from _paths import load_json
d = load_json("results_data.json")


def lattice(xs, ys, ps):
    """Fold the measured cells onto the levels the design asked for."""
    xl, yl = levels(xs), levels(ys)
    Z = np.full((len(yl), len(xl)), np.nan)
    for x, y, p in zip(xs, ys, ps):
        Z[int(np.argmin(np.abs(np.array(yl) - y))),
          int(np.argmin(np.abs(np.array(xl) - x)))] = p
    assert not np.isnan(Z).any(), "a design cell has no data"
    return np.array(xl), np.array(yl), Z


def draw(ax, xs, ys, ps, deg_y, label_at=None, n=400):
    """deg_y is the spline degree along the vertical axis, which System B can
    only take to second order because it has three levels there."""
    xl, yl, Z = lattice(xs, ys, ps)
    # RectBivariateSpline takes the row coordinate first, so kx belongs to yl
    sp = RectBivariateSpline(yl, xl, Z, kx=deg_y, ky=min(3, len(xl) - 1), s=0)
    gx = np.linspace(xl[0], xl[-1], n)
    gy = np.linspace(yl[0], yl[-1], n)
    S = np.clip(sp(gy, gx), 0.0, 1.0)
    GX, GY = np.meshgrid(gx, gy)
    im = ax.imshow(S, origin="lower", extent=[xl[0], xl[-1], yl[0], yl[-1]],
                   cmap=CMAP, vmin=0, vmax=1, aspect="auto",
                   interpolation="bilinear", zorder=2)
    minor = [l for l in np.arange(0.1, 1.0, 0.1) if abs(l - 0.5) > 1e-9]
    ax.contour(GX, GY, S, levels=minor, colors="white", linewidths=0.6, alpha=0.45, zorder=3)
    c5 = ax.contour(GX, GY, S, levels=[0.5], colors="white", linewidths=2.0, zorder=4)
    if label_at is None:
        ax.clabel(c5, fmt=lambda v: "$P(A)=0.5$", fontsize=FS_S, colors="white")
    else:
        ax.text(label_at[0], label_at[1], "$P(A)=0.5$", color="white",
                fontsize=FS_S, ha="center", va="center", zorder=8)
    # the sampled levels, drawn on the surface
    for v in xl:
        ax.plot([v, v], [yl[0], yl[-1]], color="white", lw=0.9, alpha=0.38, zorder=5)
    for v in yl:
        ax.plot([xl[0], xl[-1]], [v, v], color="white", lw=0.9, alpha=0.38, zorder=5)
    cells(ax, xs, ys, ps)
    return im, xl, yl


# ---------------------------------------------------------------- System C
C = d["C"]
xs = np.array([v["aLum"] for v in C.values()])
ys = np.array([v["aRG"] for v in C.values()])
ps = np.array([v["p"] for v in C.values()])

fig, ax = plt.subplots(figsize=(W, 3.28))
ax.set_axisbelow(True)
ax.grid(alpha=0.30, lw=0.5, color="#DDDDDD", zorder=0)
im, xl, yl = draw(ax, xs, ys, ps, deg_y=3)
ax.set_xlabel(r"$a_{\mathrm{Lum}}$"); ax.set_ylabel(r"$a_{\mathrm{RG}}$")
ax.set_xlim(-0.016, 0.262); ax.set_ylim(-0.012, 0.185)
ax.set_xticks([0, 0.1, 0.2]); ax.set_yticks([0, 0.05, 0.10, 0.15])
one_zero(ax, keep="x")
ax.set_aspect("equal")
bar(fig, ax, im)
fig.tight_layout()
finish(fig, "fig_rq2")
print("   System C  a_Lum %s" % np.round(xl, 3))
print("             a_RG  %s" % np.round(yl, 3))

# ---------------------------------------------------------------- System B
B = d["B"]
xs = np.array([v["aS"] for v in B.values()])
ys = np.array([v["aRG"] for v in B.values()])
ps = np.array([v["p"] for v in B.values()])

fig, ax = plt.subplots(figsize=(W, 2.41))
ax.set_axisbelow(True)
ax.grid(alpha=0.30, lw=0.5, color="#DDDDDD", zorder=0)
im, xl, yl = draw(ax, xs, ys, ps, deg_y=2, label_at=(0.33, 0.0715))
ax.set_xlabel(r"$a_{\mathrm{S}}$"); ax.set_ylabel(r"$a_{\mathrm{RG}}$")
ax.set_xlim(-0.02, 0.735); ax.set_ylim(0.036, 0.144)
ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_yticks([0.05, 0.09, 0.13])
bar(fig, ax, im)
fig.tight_layout()
finish(fig, "fig_rq4")
print("   System B  a_S   %s" % np.round(xl, 3))
print("             a_RG  %s" % np.round(yl, 3))

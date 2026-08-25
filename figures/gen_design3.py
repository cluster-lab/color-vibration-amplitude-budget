"""fig_design: what a common MacAdam radius actually delivers to the pathways.

One panel. The two orthogonal grids that used to sit beside it are the same
sampling the contour maps already show, so they are gone. Marker shapes match
the other figures, and the curve interpolates in unit steps of r."""
import json, glob, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from figstyle import W, FS_S, ELL, ENAME, finish, one_zero
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline

from _paths import SHARE, conditions   # noqa: F401  (SHARE goes on sys.path)
import cfvi as C

# labels hang off the large-r end, where the trajectories are well separated
EOFF = {8: (0.026, 0.006), 12: (-0.004, 0.017), 13: (0.030, 0.004),
        14: (-0.008, 0.016), 19: (0.028, 0.005)}


def coord(c1, c2):
    la, ls = C.rgb_to_lms(np.array(c1)), C.rgb_to_lms(np.array(c2))
    m = (la + ls) / 2.0
    cL, cM, cS = (la - ls) / np.where(m > 1e-9, m, 1e-9)
    return abs((cL + cM) / 2), abs(cL - cM) / 2, abs(cS - (cL + cM) / 2) / 2


cond = {(r["ellipse"], r["radius"]): coord(r["c1"], r["c2"])
        for r in conditions("A")}
print("System A conditions: %d" % len(cond))

fig, ax = plt.subplots(figsize=(W, 2.80))
ax.set_axisbelow(True)
ax.grid(alpha=0.30, lw=0.5, color="#DDDDDD")
# E12 is the trajectory that turns back, so it is drawn over the others
ORDER = sorted(ELL, key=lambda e: e == 12)
for i, e in enumerate(ORDER):
    z = 3 + 4 * i
    pts = sorted([(k[1], v) for k, v in cond.items() if k[0] == e])
    rr = np.array([p[0] for p in pts], float)
    rg = np.array([p[1][1] for p in pts])
    s = np.array([p[1][2] for p in pts])
    k = min(3, len(rr) - 1)
    u = np.arange(rr[0], rr[-1] + 1e-9, 1.0)          # every unit of radius
    sx = make_interp_spline(rr, s, k=k)(u)
    sy = make_interp_spline(rr, rg, k=k)(u)
    ax.plot(sx, sy, color=ELL[e]["c"], ls=ELL[e]["ls"], lw=1.5, zorder=z)
    ax.plot(s, rg, ls="none", marker=ELL[e]["m"], ms=5, color=ELL[e]["c"],
            mfc="white", mew=1.3, zorder=z + 1)
    ax.add_patch(FancyArrowPatch((sx[-2], sy[-2]), (sx[-1], sy[-1]), arrowstyle="-|>",
                                 mutation_scale=10, color=ELL[e]["c"], lw=1.5, zorder=z + 2))
    dx, dy = EOFF[e]
    ax.text(sx[-1] + dx, sy[-1] + dy, ENAME[e], color=ELL[e]["c"], fontsize=FS_S,
            ha="center", va="center", zorder=z + 3)
ax.set_xlabel(r"$a_{\mathrm{S}}$")
ax.set_ylabel(r"$a_{\mathrm{RG}}$")
ax.set_xlim(-0.02, 0.72); ax.set_ylim(0, 0.178)
ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_yticks([0, 0.05, 0.10, 0.15])
ax.set_xticks(np.arange(0, 0.71, 0.1), minor=True)
one_zero(ax, keep="x")
fig.tight_layout()
finish(fig, "fig_design")
print("max a_Lum over all System A conditions: %.4f" % max(v[0] for v in cond.values()))

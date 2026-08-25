"""fig_xy: the System A loci on the CIE 1931 xy plane, from the recorded stimuli.

The sRGB gamut is filled with the colors it contains. (a) places the region shown
on the whole diagram and names the five ellipses at their centres, so that (b)
carries no legend and no labels over the data. Each ellipse is drawn at r=10, as
a scale rather than as an envelope, in the same thin black line as fig_budget.
The two panels are given the same box, so (b)'s window is padded in y to the
aspect ratio of (a)."""
import json, glob, os, sys
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from figstyle import W, FS, FS_S, ELL, ENAME, finish, panel, one_zero
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.interpolate import make_interp_spline

from _paths import SHARE, MN, conditions, load_json   # noqa: F401
import cfvi as C
from macadam_newton import ELLIPSE_XY, transform_ellipse, ellipse_radius, uv_to_xy

M = C.M_RGB_TO_XYZ
M_INV = np.linalg.inv(M)
SRGB = np.array([[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]])
# each name is set off its centre in (a); the offsets keep the five apart
LOFF = {8: (-0.036, 0.016), 12: (-0.040, -0.014), 13: (-0.006, 0.030),
        14: (0.040, 0.012), 19: (0.008, -0.036)}


def xy_of(rgb):
    X, Y, Z = M @ np.array(rgb, float)
    s = X + Y + Z
    return (X / s, Y / s) if s > 0 else (0.0, 0.0)


def encode(lin):
    a = np.clip(lin, 0, 1)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def gamut_image(xlim, ylim, n=700):
    """Every displayable chromaticity, painted at its own colour and lightened
    so that black overlays and a legend read cleanly on top of it."""
    gx = np.linspace(xlim[0], xlim[1], n)
    gy = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        XYZ = np.stack([X / Y, np.ones_like(X), (1 - X - Y) / Y], axis=-1)
    rgb = XYZ @ M_INV.T
    inside = (rgb > -1e-6).all(axis=-1) & np.isfinite(rgb).all(axis=-1) & (Y > 1e-6)
    rgb = np.clip(rgb, 0, None)
    peak = rgb.max(axis=-1, keepdims=True)
    rgb = np.divide(rgb, np.where(peak > 1e-9, peak, 1.0))
    rgb = 1 - 0.20 * (1 - rgb)      # a faint wash: the loci and the text on top
                                    # of it have to stay the strongest marks
    img = np.concatenate([encode(rgb), inside[..., None].astype(float)], axis=-1)
    return np.clip(img, 0, 1)


def ellipse_xy(e, r, n=361):
    """The ellipse of centre e scaled by r, built in uv where the pairs were
    placed and then carried to xy, so the outermost pair lands exactly on it."""
    x0, y0, a, b, th = ELLIPSE_XY[e]
    u0, v0, a_uv, b_uv, ang = transform_ellipse(x0, y0, a, b, th)
    t = np.linspace(0, 360, n)
    d = r * np.array([ellipse_radius(tt, a_uv, b_uv, ang) for tt in t])
    u = u0 + d * np.cos(np.deg2rad(t))
    v = v0 + d * np.sin(np.deg2rad(t))
    return np.array([uv_to_xy(uu, vv) for uu, vv in zip(u, v)])


loci, base = defaultdict(list), {}
for row in conditions("A"):
    e, r = row["ellipse"], row["radius"]
    c1, c2 = row["c1"], row["c2"]
    loci[e].append((r, xy_of(c1), xy_of(c2)))
    base.setdefault(e, tuple(encode((np.array(c1) + np.array(c2)) / 2)))
for e in loci:
    loci[e].sort()

R_REF = 10
RING = {e: ellipse_xy(e, R_REF) for e in loci}

FX, FY = (0.0, 0.75), (0.0, 0.85)

# (b) is trimmed to the pairs in x and then padded in y to the aspect of (a),
# so that the two panels come out the same size on the page
pts = [p for e in loci for row in loci[e] for p in row[1:]]
pts += [(ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]) for e in ELL]
P = np.array(pts)
pad = 0.012
ZX = (P[:, 0].min() - pad, P[:, 0].max() + pad)
span_y = (ZX[1] - ZX[0]) * (FY[1] - FY[0]) / (FX[1] - FX[0])
mid_y = 0.5 * (P[:, 1].min() + P[:, 1].max())
ZY = (mid_y - span_y / 2, mid_y + span_y / 2)
print("   region shown  x %.3f..%.3f   y %.3f..%.3f" % (ZX + ZY))
import colour
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
XYZ = cmfs[np.arange(380, 701, 1)]
locus = XYZ[:, :2] / XYZ.sum(axis=1)[:, None]
locus = np.vstack([locus, locus[0]])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(W, W * 0.502),
                               gridspec_kw={"wspace": 0.28})

for ax, xlim, ylim, step in ((axa, FX, FY, 0.2), (axb, ZX, ZY, 0.05)):
    ax.imshow(gamut_image(xlim, ylim), origin="lower",
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=0,
              interpolation="bilinear")
    for v in np.arange(0, 1.0 + 1e-9, step):          # grid, over the fill
        if xlim[0] < v < xlim[1]:
            ax.plot([v, v], ylim, color="0.35", lw=0.4, ls=":", alpha=0.55, zorder=1)
        if ylim[0] < v < ylim[1]:
            ax.plot(xlim, [v, v], color="0.35", lw=0.4, ls=":", alpha=0.55, zorder=1)
    ax.add_patch(Polygon(SRGB, closed=True, fill=False, ec="0.35", lw=0.8, zorder=3))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("CIE 1931 $x$"); ax.set_ylabel("CIE 1931 $y$")

axa.plot(locus[:, 0], locus[:, 1], color="0.30", lw=0.8, zorder=4)
axa.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0],
                        fill=False, ec="black", lw=1.2, zorder=5))
# each centre is named in place with an L-shaped leader. A leader may cross
# the zoom frame, but no label may sit on it: E12 is therefore set right-
# aligned, ending well left of the frame and of the sRGB edge beside it
LEAD = {8:  (0.395, 0.760, "right"),
        12: (0.178, 0.252, "left"),
        13: (0.560, 0.640, "right"),
        14: (0.680, 0.520, "right"),
        19: (0.640, 0.135, "right")}
for e in ELL:
    cx, cy = ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]
    axa.plot(RING[e][:, 0], RING[e][:, 1], color="black", lw=0.8, zorder=6)
    axa.plot([cx], [cy], marker=ELL[e]["m"], ms=3.4, color=ELL[e]["c"],
             mec="black", mew=0.5, zorder=6)
    tx, ty, side = LEAD[e]
    s = 1 if side == "right" else -1
    axa.plot([cx, tx - s * 0.030, tx - s * 0.008], [cy, ty, ty],
             color="0.25", lw=0.7, zorder=5, solid_capstyle="butt")
    axa.text(tx, ty, ENAME[e], color="black", fontsize=FS, zorder=7,
             ha="left" if side == "right" else "right", va="center")
axa.set_xticks([0, 0.2, 0.4, 0.6]); axa.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
one_zero(axa, keep="x")
panel(axa, "(a)")

for e in ELL:
    axb.plot(RING[e][:, 0], RING[e][:, 1], color="black", lw=0.85, zorder=8)
    cen = np.array([ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]])
    # below the smallest measured radius the locus is not guessed: it is the
    # actual construction solved at r = 1, 2, ... (fine_loci.py, validated
    # against the recorded stimuli to |dxy| < 0.0015)
    fine = load_json("fine_loci.json")[str(e)]
    lead = {1: np.array(fine["side1"]), 2: np.array(fine["side2"])}
    first = {side: np.array(loci[e][0][side]) for side in (1, 2)}
    # the solver's +theta side may be either recorded side; match by proximity
    if (np.hypot(*(lead[1][-1] - first[1])) >
            np.hypot(*(lead[1][-1] - first[2]))):
        lead = {1: lead[2], 2: lead[1]}
    for side in (1, 2):
        q = np.array([p[side] for p in loci[e]])
        rr = np.array([p[0] for p in loci[e]], float)
        k = min(3, len(rr) - 1)
        u = np.linspace(rr[0], rr[-1], 240)
        path = np.vstack([cen, lead[side], q[0]])
        axb.plot(path[:, 0], path[:, 1], color=ELL[e]["c"], lw=0.9, zorder=5,
                 solid_capstyle="round")
        axb.plot(make_interp_spline(rr, q[:, 0], k=k)(u),
                 make_interp_spline(rr, q[:, 1], k=k)(u), color=ELL[e]["c"],
                 lw=1.15, zorder=5, solid_capstyle="round")
        axb.plot(q[:, 0], q[:, 1], ls="none", marker=ELL[e]["m"], ms=4.0,
                 color=ELL[e]["c"], mfc="white", mew=1.2, zorder=6)
    cx, cy = ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]
    axb.plot([cx], [cy], marker="s", ms=7.0, color=base[e], mec="black", mew=0.7, zorder=7)
axb.plot([0.3127], [0.3290], marker="*", ms=10, color="black", mfc="none",
         mew=0.8, zorder=8)
# D65 sits almost on top of the E12 base, so its name is carried well clear
axb.plot([0.3127, 0.3127 - 0.030, 0.3127 - 0.042],
         [0.3290, 0.3290 - 0.036, 0.3290 - 0.036],
         color="0.25", lw=0.7, zorder=8, solid_capstyle="butt")
axb.text(0.3127 - 0.046, 0.3290 - 0.036, "D65", fontsize=FS, ha="right",
         va="center", zorder=9)
axb.set_xticks([0.25, 0.35, 0.45]); axb.set_yticks([0.25, 0.35, 0.45])
panel(axb, "(b)")

# one legend for both panels: the marker and colour identify an ellipse in (a)
# as its centre and in (b) as its pair locus
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=ELL[e]["c"], lw=1.15, marker=ELL[e]["m"],
                  ms=4.0, mfc="white", mew=1.2, label=ENAME[e])
           for e in sorted(ELL)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.105),
           ncol=5, fontsize=FS_S, frameon=False, handlelength=2.0,
           columnspacing=1.5, handletextpad=0.5)

finish(fig, "fig_xy")
for e in sorted(loci):
    print("   E%-3d radii %s" % (e, [p[0] for p in loci[e]]))

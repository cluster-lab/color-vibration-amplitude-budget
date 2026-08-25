# -*- coding: utf-8 -*-
"""Fig. 1.

(a) the premise, running downward: A,A,B,B at 90 Hz fuses into one color.
(b) the existing construction at two bases: the pair taken equidistant from the
    center in the chromaticity plane, along the major and along the minor axis
    of the r=25 ellipse. Predicted P(A) spans 3.7-fold.
(c) the chromaticities the model holds equally visible, P(A)=0.2, computed on a
    full two-dimensional grid rather than along the axes, so the shape is the
    real one. Pairs are symmetric in cone contrast, as Fig. 10 builds them.

Predictions use the System-A calibration of Sec. 5.1, the one validated against
the measured System-A rates (r = 0.891); System A is the block in which the
MacAdam-ellipse stimuli were actually run.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from figstyle import FS, FS_S, ELL, ENAME

from _paths import SHARE, MN   # noqa: F401  (puts them on sys.path)
import cfvi as C
from macadam_newton import ELLIPSE_XY

from _paths import OUT as FIGS
M = C.M_RGB_TO_XYZ
M_INV = np.linalg.inv(M)
M_XL = C.M_XYZ_TO_LMS
Y0 = 0.4
B0, GAM, W_RG, W_S, K = -2.693, 45.67, 0.8255, 0.06389, 2.07
BASES = (8, 19)
R_CROSS = 25
LEVEL = 0.20
SRGB = np.array([[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]])


def encode(a):
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def xy_to_rgb(x, y, Y=Y0):
    return M_INV @ np.array([x * Y / y, Y, (1 - x - y) * Y / y])


def wash(xlim, ylim, n=520):
    """The same faint chromaticity wash Fig. 2 uses."""
    gx = np.linspace(*xlim, n)
    gy = np.linspace(*ylim, n)
    X, Y = np.meshgrid(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        XYZ = np.stack([X / Y, np.ones_like(X), (1 - X - Y) / Y], axis=-1)
    rgb = XYZ @ M_INV.T
    inside = (rgb > -1e-6).all(axis=-1) & np.isfinite(rgb).all(axis=-1) & (Y > 1e-6)
    rgb = np.clip(rgb, 0, None)
    peak = rgb.max(axis=-1, keepdims=True)
    rgb = np.divide(rgb, np.where(peak > 1e-9, peak, 1.0))
    rgb = 1 - 0.20 * (1 - rgb)
    return np.clip(np.concatenate(
        [encode(rgb), inside[..., None].astype(float)], axis=-1), 0, 1)


def amps_of(c):
    return (np.abs(c[..., 0] + c[..., 1]),
            np.abs(c[..., 0] - c[..., 1]),
            np.abs(c[..., 2] - (c[..., 0] + c[..., 1]) / 2))


def P_of(c):
    aL, aRG, aS = amps_of(c)
    V = (aL ** K + (W_RG * aRG) ** K + (W_S * aS) ** K) ** (1 / K)
    return 1 / (1 + np.exp(-(B0 + GAM * V))), (aL, aRG, aS)


# ------------------------------------------------------------------ (b) data
cross = {}
for e in BASES:
    x0, y0, a, b, th = ELLIPSE_XY[e]
    arms = []
    for lab, ang, rad in (("major", th, a), ("minor", th + 90, b)):
        h = R_CROSS * rad
        dx, dy = np.cos(np.deg2rad(ang)) * h, np.sin(np.deg2rad(ang)) * h
        p1, p2 = (x0 + dx, y0 + dy), (x0 - dx, y0 - dy)
        r1, r2 = xy_to_rgb(*p1), xy_to_rgb(*p2)
        assert min(r1.min(), r2.min()) > -1e-9 and max(r1.max(), r2.max()) < 1 + 1e-9
        l1, l2 = M_XL @ (M @ r1), M_XL @ (M @ r2)
        c = (l1 - l2) / (l1 + l2)
        P, A = P_of(c)
        arms.append(dict(axis=lab, p1=p1, p2=p2, P=float(P),
                         c1=list(encode(np.clip(r1, 0, 1))),
                         c2=list(encode(np.clip(r2, 0, 1)))))
        print("  E%-3d %-6s P=%.3f   a_Lum %.4f a_RG %.4f a_S %.4f"
              % (e, lab, P, A[0], A[1], A[2]))
    t = np.linspace(0, 360, 241)
    tt = np.deg2rad(t - th)
    rr = a * b / np.hypot(b * np.cos(tt), a * np.sin(tt)) * 10
    cross[e] = dict(center=(x0, y0), arms=arms,
                    ell10=np.column_stack([x0 + rr * np.cos(np.deg2rad(t)),
                                           y0 + rr * np.sin(np.deg2rad(t))]))
vals = [q["P"] for e in BASES for q in cross[e]["arms"]]
print("  spread %.1fx" % (max(vals) / min(vals)))

# ------------------------------------------------------------------ (c) data
CX, CY = (0.13, 0.60), (0.10, 0.60)
n = 460
gx, gy = np.linspace(*CX, n), np.linspace(*CY, n)
GX, GY = np.meshgrid(gx, gy)
with np.errstate(divide="ignore", invalid="ignore"):
    XYZp = np.stack([GX * Y0 / GY, np.full_like(GX, Y0),
                     (1 - GX - GY) * Y0 / GY], axis=-1)
RGBp = XYZp @ M_INV.T
field = {}
for e in BASES:
    x0, y0 = ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]
    rgb0 = xy_to_rgb(x0, y0)
    RGBm = 2 * rgb0 - RGBp                       # the cone-symmetric partner
    good = (np.isfinite(RGBp).all(-1) & (RGBp > -1e-9).all(-1)
            & (RGBp < 1 + 1e-9).all(-1) & (RGBm > -1e-9).all(-1)
            & (RGBm < 1 + 1e-9).all(-1))
    lms0 = M_XL @ (M @ rgb0)
    LMSp = (RGBp @ M.T) @ M_XL.T
    c = LMSp / lms0 - 1.0
    P, A = P_of(c)
    field[e] = np.where(good, P, np.nan)
    # how much does the luminance term actually bend the contour?
    V_no = ((W_RG * A[1]) ** K + (W_S * A[2]) ** K) ** (1 / K)
    P_no = np.where(good, 1 / (1 + np.exp(-(B0 + GAM * V_no))), np.nan)
    band = np.abs(field[e] - LEVEL) < 0.01
    print("  E%-3d field: max P inside gamut %.3f | on the P=%.1f contour, "
          "a_Lum %.4f-%.4f, dropping it moves P by up to %.3f"
          % (e, np.nanmax(field[e]), LEVEL,
             np.nanmin(np.where(band, A[0], np.nan)),
             np.nanmax(np.where(band, A[0], np.nan)),
             np.nanmax(np.where(band, np.abs(P_no - field[e]), np.nan))))

# ------------------------------------------------------------------ drawing
matplotlib.rcParams["savefig.bbox"] = None
fig = plt.figure(figsize=(5.25, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[0.55, 1.0, 1.0],
                      left=0.012, right=0.988, top=0.905, bottom=0.135,
                      wspace=0.10)

# (a)
ax0 = fig.add_subplot(gs[0])
ax0.set_axis_off()
ax0.set_aspect("equal")
SIDE, PITCH = 1.0, 1.06
_arm = cross[BASES[0]]["arms"][1]
cA = _arm["c1"]
cB = _arm["c2"]
# the fused patch is the time average, so the mean is taken in linear light
_lin = 0.5 * (np.clip(xy_to_rgb(*_arm["p1"]), 0, 1)
              + np.clip(xy_to_rgb(*_arm["p2"]), 0, 1))
fused = list(encode(_lin))
top = 8.0
for i in range(8):
    col = cA if (i // 2) % 2 == 0 else cB
    ax0.add_patch(Rectangle((1.20, top - i * PITCH - SIDE), SIDE, SIDE,
                            facecolor=col, edgecolor="0.35", lw=0.5))
    ax0.text(1.08, top - i * PITCH - SIDE / 2, "AB"[(i // 2) % 2], ha="right",
             va="center", fontsize=FS_S, color="0.3")
y_last = top - 7 * PITCH - SIDE

# time runs down the frame sequence, two colors alternating at 90 Hz
ax0.annotate("", xy=(2.62, y_last), xytext=(2.62, top),
             arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.1))
ax0.text(2.86, (top + y_last) / 2, "two\ncolors\n90 Hz", fontsize=FS_S,
         color="0.3", ha="left", va="center")

# and the sequence fuses into one color at the 22.5 Hz cycle
ax0.add_patch(FancyArrowPatch((1.70, y_last - 0.30), (1.70, y_last - 1.30),
                              arrowstyle="simple", mutation_scale=21,
                              color="0.3", lw=0))
ax0.add_patch(Rectangle((1.20, y_last - 2.26), SIDE, SIDE, facecolor=fused,
                        edgecolor="0.35", lw=0.5))
ax0.text(2.32, y_last - 1.76, "single\ncolor\n22.5 Hz", fontsize=FS_S,
         color="0.3", ha="left", va="center")
ax0.set_xlim(0.62, 5.42)
ax0.set_ylim(y_last - 2.60, top + 0.45)


def chroma(ax, xlim, ylim, step):
    ax.imshow(wash(xlim, ylim), origin="lower", zorder=0,
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              interpolation="bilinear")
    for v in np.arange(0, 1.0 + 1e-9, step):
        if xlim[0] < v < xlim[1]:
            ax.plot([v, v], ylim, color="0.35", lw=0.4, ls=":", alpha=0.55,
                    zorder=1)
        if ylim[0] < v < ylim[1]:
            ax.plot(xlim, [v, v], color="0.35", lw=0.4, ls=":", alpha=0.55,
                    zorder=1)
    ax.add_patch(Polygon(SRGB, closed=True, fill=False, ec="0.35", lw=0.8,
                         zorder=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FS_S, length=2.5)
    ax.set_xlabel("CIE 1931 $x$", fontsize=FS_S, labelpad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def leader(ax, tgt, tx, ty, text, color, side):
    s = 1 if side == "right" else -1
    ax.plot([tgt[0], tx - s * 0.016, tx - s * 0.004], [tgt[1], ty, ty],
            color=color, lw=0.55, zorder=8, solid_capstyle="butt")
    ax.text(tx, ty, text, fontsize=FS_S, color=color, zorder=9,
            ha="left" if side == "right" else "right", va="center")


# (b)
BX, BY = (0.135, 0.475), (0.125, 0.515)
axb = fig.add_subplot(gs[1])
chroma(axb, BX, BY, 0.1)
LAB = {(8, "major"): (0.352, 0.497, "right"),
       (8, "minor"): (0.200, 0.452, "left"),
       (19, "major"): (0.408, 0.318, "right"),
       (19, "minor"): (0.412, 0.212, "right")}
NAME_AT = {8: (0.170, 0.315, "left"), 19: (0.300, 0.166, "left")}
for e in BASES:
    col = ELL[e]["c"]
    rec = cross[e]
    axb.plot(rec["ell10"][:, 0], rec["ell10"][:, 1], color="black", lw=0.7,
             zorder=6)
    for arm in rec["arms"]:
        p1, p2 = np.array(arm["p1"]), np.array(arm["p2"])
        axb.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, lw=1.0, zorder=5)
        for p, cc in ((p1, arm["c1"]), (p2, arm["c2"])):
            axb.plot(p[0], p[1], "o", ms=4.4, mfc=cc, mec=col, mew=0.9,
                     zorder=7)
        tx, ty, side = LAB[(e, arm["axis"])]
        leader(axb, p1, tx, ty, "$P(A)=%.2f$" % arm["P"], col, side)
    nx, ny, nside = NAME_AT[e]
    leader(axb, rec["center"], nx, ny, ENAME[e], col, nside)
axb.set_xticks([0.2, 0.3, 0.4])
axb.set_yticks([0.2, 0.3, 0.4, 0.5])
axb.set_ylabel("CIE 1931 $y$", fontsize=FS_S, labelpad=1.5)
axb.set_title("a cross at $r=%d$" % R_CROSS, fontsize=FS_S, color="0.15", pad=3)

# (c)
axc = fig.add_subplot(gs[2])
chroma(axc, BX, BY, 0.1)      # the same window as (b), for direct comparison
for e in BASES:
    col = ELL[e]["c"]
    axc.contourf(GX, GY, field[e], levels=[0.0, LEVEL], colors=[col],
                 alpha=0.22, zorder=4)
    axc.contour(GX, GY, field[e], levels=[LEVEL], colors=[col],
                linewidths=1.2, zorder=5)
    axc.plot(cross[e]["ell10"][:, 0], cross[e]["ell10"][:, 1], color="black",
             lw=0.7, zorder=6)
    cx, cy = cross[e]["center"]
    axc.plot(cx, cy, marker=ELL[e]["m"], ms=4.2, color=col, mec="black",
             mew=0.5, zorder=7)
axc.set_xticks([0.2, 0.3, 0.4])
axc.set_yticks([0.2, 0.3, 0.4, 0.5])
axc.set_yticklabels([])
axc.set_title("equal visibility, $P(A)=%.1f$" % LEVEL, fontsize=FS_S,
              color="0.15", pad=3)
for e, tx, ty, side in ((8, 0.200, 0.470, "left"), (19, 0.428, 0.198, "right")):
    cx, cy = cross[e]["center"]
    leader(axc, (cx, cy), tx, ty, ENAME[e], ELL[e]["c"], side)

# the three axes hold their aspect, so their drawn boxes are narrower than the
# cells they sit in; take the letters from the boxes after the layout settles
fig.canvas.draw()
for ax, tag in ((ax0, "(a)"), (axb, "(b)"), (axc, "(c)")):
    fig.text(max(0.004, ax.get_position().x0 - 0.030), 0.945, tag, fontsize=FS)

out = os.path.join(FIGS, "fig_concept.pdf")
fig.savefig(out)
fig.savefig(os.path.join(FIGS, "fig_concept.png"), dpi=150)
print("wrote", out)

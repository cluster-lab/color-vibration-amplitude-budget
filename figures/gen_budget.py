"""fig_budget: the design rule drawn on the chromaticity diagram.

For three base colors, the red-green and the tritan axis are straight lines
through the base in xy. On each, the heavy segment is the excursion that is both
inside the gamut and below half detection; the dotted line carries the same
amplitude on out to the half-detection threshold, whether or not the gamut
allows it. The r=10 MacAdam ellipse of each base is drawn for scale.

The pair is symmetric in cone contrast about the base, so the two colors always
sit at equal amplitude. Chromaticity is a projection of that space, however, so
the two halves of a segment come out unequal in length, and the gamut is reached
on one side before the other. Amplitudes follow the convention the fit uses,
a_i as in vfit.coord.
"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from figstyle import W, FS, FS_S, finish, panel
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# the budget segment is outlined, so that E12's grey reads as a band rather than
# as a piece of the grey line the axis is extended along
BAND = [pe.Stroke(linewidth=4.6, foreground="black"), pe.Normal()]

from _paths import SHARE, MN   # noqa: F401  (puts them on sys.path)
import cfvi as C
from macadam_newton import ELLIPSE_XY, transform_ellipse, ellipse_radius, uv_to_xy

M = C.M_RGB_TO_XYZ
M_INV = np.linalg.inv(M)
M_XL = C.M_XYZ_TO_LMS
M_LX = np.linalg.inv(M_XL)
SRGB = np.array([[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]])
Y0 = 0.4

# the definitive fit, System C block
B0, GAM, W_RG, W_S = -5.0908, 50.737, 0.8255, 0.06389
T_LUM = -B0 / GAM
T_RG, T_S = T_LUM / W_RG, T_LUM / W_S
print("thresholds  T_Lum %.3f  T_RG %.3f  T_S %.3f" % (T_LUM, T_RG, T_S))

# three bases spanning the region the ellipses cover: near-grey, green, orange
BASES = [(12, "#6E6E6E", "s"), (8, "#009E73", "o"), (14, "#E69F00", "D")]
BNAME = {12: "E12", 8: "E8", 14: "E14"}


def xy_to_rgb(x, y, Y=Y0):
    XYZ = np.array([x * Y / y, Y, (1 - x - y) * Y / y])
    return M_INV @ XYZ


def rgb_to_xy(rgb):
    X, Y, Z = M @ np.asarray(rgb, float)
    s = X + Y + Z
    return X / s, Y / s


def pair(rgb0, axis, a):
    """The two colors of a symmetric pair driving one axis alone by amplitude a.

    Eq. (4) reads the amplitude off Michelson contrasts: a_RG = |c_L - c_M| and
    a_S = |c_S - (c_L + c_M)/2|, so a red-green amplitude a means c_L = -c_M =
    a/2 and a tritan amplitude a means c_S = a with the other two at zero."""
    lms0 = M_XL @ (M @ np.asarray(rgb0, float))
    c = {"RG": np.array([a / 2, -a / 2, 0.0]), "S": np.array([0.0, 0.0, a])}[axis]
    return [M_INV @ (M_LX @ (lms0 * (1 + s * c))) for s in (1, -1)]


def in_gamut(rgb, tol=1e-9):
    return bool(np.all(np.asarray(rgb) >= -tol) and np.all(np.asarray(rgb) <= 1 + tol))


def reach(rgb0, axis, amax, n=4000):
    """How far the amplitude can run before the pair leaves the sRGB gamut."""
    for a in np.linspace(0, amax, n)[::-1]:
        if all(in_gamut(p) for p in pair(rgb0, axis, a)):
            return float(a)
    return 0.0


def span(b, d, xlim, ylim):
    """The straight line through b along d, cut at the edges of the panel."""
    ts = []
    for i, lim in enumerate((xlim, ylim)):
        if abs(d[i]) > 1e-12:
            ts += [(v - b[i]) / d[i] for v in lim]
    lo, hi = -1e9, 1e9
    for i, lim in enumerate((xlim, ylim)):          # keep only what stays inside
        if abs(d[i]) < 1e-12:
            continue
        t0, t1 = sorted([(lim[0] - b[i]) / d[i], (lim[1] - b[i]) / d[i]])
        lo, hi = max(lo, t0), min(hi, t1)
    p, q = b + lo * d, b + hi * d
    return [p[0], q[0]], [p[1], q[1]]


def macadam(e, r, n=361):
    x0, y0, a, b, th = ELLIPSE_XY[e]
    u0, v0, a_uv, b_uv, ang = transform_ellipse(x0, y0, a, b, th)
    t = np.linspace(0, 360, n)
    d = r * np.array([ellipse_radius(tt, a_uv, b_uv, ang) for tt in t])
    return np.array([uv_to_xy(u0 + dd * np.cos(np.deg2rad(tt)),
                              v0 + dd * np.sin(np.deg2rad(tt)))
                     for tt, dd in zip(t, d)])


def encode(lin):
    a = np.clip(lin, 0, 1)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def gamut_image(xlim, ylim, n=600):
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
    rgb = 1 - 0.78 * (1 - rgb)
    img = np.concatenate([encode(rgb), inside[..., None].astype(float)], axis=-1)
    return np.clip(img, 0, 1)


# ---- what each base admits ----------------------------------------------
rows, seg = [], {}
for e, col, mk in BASES:
    x0, y0 = ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]
    rgb0 = xy_to_rgb(x0, y0)
    assert in_gamut(rgb0), "base %d outside the gamut at Y0" % e
    for axis, T in (("RG", T_RG), ("S", T_S)):
        gam = reach(rgb0, axis, 2.0)
        a = min(T, gam)
        # A pair traces a straight line in xy: the modulation is a straight line
        # in LMS through the base, and chromaticity is a projective map, which
        # carries lines to lines. So the direction is fixed by two points and can
        # be extended right across the diagram.
        b = np.array([x0, y0])
        d = rgb_to_xy(pair(rgb0, axis, 0.01)[0]) - b
        d = d / np.hypot(*d)
        # quarter marks of the usable amplitude: the pair of dots facing each
        # other across the base carries the same amplitude, which the unequal
        # lengths of the two halves would otherwise hide
        ticks = np.array([[rgb_to_xy(p) for p in pair(rgb0, axis, f * a)]
                          for f in (0.25, 0.50, 0.75)])
        seg[(e, axis)] = (np.array([rgb_to_xy(p) for p in pair(rgb0, axis, a)]),
                          (b, d), a, gam, gam < T, ticks)
        rows.append((BNAME[e], axis, T, gam, a))
print("\n base  axis   threshold   gamut limit   usable    xy half-length   /(r=10 ellipse)")
for n, ax_, T, gam, a in rows:
    e = [k for k, v in BNAME.items() if v == n][0]
    use = seg[(e, ax_)][0]
    half = 0.5 * float(np.hypot(*(use[0] - use[1])))
    # the r=10 ellipse radius in the same direction, for scale
    d = (use[0] - use[1]) / np.hypot(*(use[0] - use[1]))
    ell = macadam(e, 10) - np.array([ELLIPSE_XY[e][0], ELLIPSE_XY[e][1]])
    rad = float(np.max(ell @ d))
    print("  %-4s %-4s  %8.3f  %11.3f  %7.3f%-15s %8.4f  %10.1fx"
          % (n, ax_, T, gam, a, "  (gamut-bound)" if gam < T else "", half, half / rad))

# the window follows the solid segments; the grey line carries the direction on
# across the whole panel
pts = np.array([q for v in seg.values() for q in v[0]])
pad = 0.03
XL = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
YL = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)
print("\n window x %.3f..%.3f  y %.3f..%.3f" % (XL + YL))

fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.482), gridspec_kw={"wspace": 0.26})
TITLE = {"RG": "red–green axis", "S": "tritan ($S$) axis"}
SYM = {"RG": "a_{\\mathrm{RG}}", "S": "a_{\\mathrm{S}}"}
for ax, axis, tag in ((axes[0], "RG", "(a)"), (axes[1], "S", "(b)")):
    ax.imshow(gamut_image(XL, YL), origin="lower",
              extent=[XL[0], XL[1], YL[0], YL[1]], zorder=0, interpolation="bilinear")
    for v in np.arange(0, 1.0 + 1e-9, 0.1):
        if XL[0] < v < XL[1]:
            ax.plot([v, v], YL, color="0.35", lw=0.4, ls=":", alpha=0.5, zorder=1)
        if YL[0] < v < YL[1]:
            ax.plot(XL, [v, v], color="0.35", lw=0.4, ls=":", alpha=0.5, zorder=1)
    ax.add_patch(Polygon(SRGB, closed=True, fill=False, ec="0.35", lw=0.8, zorder=2))
    for e, col, mk in BASES:
        use, (b, d), a, agam, bound, ticks = seg[(e, axis)]
        ax.plot(*span(b, d, XL, YL), color="0.62", lw=0.8, zorder=3)
        ax.plot(use[:, 0], use[:, 1], color=col, lw=3.2, solid_capstyle="butt",
                zorder=5, path_effects=BAND)
        # open rings, so the colored segment still reads underneath them
        ax.plot(ticks[:, :, 0].ravel(), ticks[:, :, 1].ravel(), ls="none",
                marker="o", ms=3.4, mfc="none", mec="white", mew=1.0, zorder=6)
        ell = macadam(e, 10)
        ax.plot(ell[:, 0], ell[:, 1], color="black", lw=0.7, zorder=6)
        ax.plot([ELLIPSE_XY[e][0]], [ELLIPSE_XY[e][1]], marker=mk, ms=4.2,
                color=col, mec="black", mew=0.5, zorder=7)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_aspect("equal")
    ax.set_xlabel("CIE 1931 $x$"); ax.set_ylabel("CIE 1931 $y$")
    ax.set_title(TITLE[axis], fontsize=FS, loc="right", pad=8)
    panel(ax, tag)

handles = [Line2D([], [], color=c, lw=2.0, marker=m, ms=4.2, mec="black",
                  mew=0.5, label=BNAME[e]) for e, c, m in BASES]
handles += [Line2D([], [], color="0.62", lw=0.8, label="the same axis, extended"),
            Line2D([], [], color="0.45", marker="o", ms=3.0, mfc="none",
                   mec="0.45", mew=0.9, lw=0, label="quarters of the amplitude"),
            Line2D([], [], color="black", lw=0.7, label="MacAdam ellipse, $r=10$")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.17),
           ncol=3, fontsize=FS_S, frameon=False, handlelength=2.2,
           columnspacing=1.6, handletextpad=0.6)
finish(fig, "fig_budget")

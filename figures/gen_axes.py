"""fig_axes: the sensitivity ordering, with the fitted curves the thresholds come from.

Each threshold sits on the psychometric curve that produced it, with a
profile-likelihood interval drawn on the half-detection line, and the measured
points carry cluster-bootstrap intervals over observers.

The fit and the bootstrap run on the per-trial responses, which are not part of
this repository (see README). Their result is in data/axes_fit.json: the
coefficients of the definitive fit, and the three measured series with their
intervals. This script reads that and draws the figure.
"""
import sys

import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from figstyle import W, FS_S, C_LUM, C_RG, C_S, finish
from _paths import load_json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Z, WG = np.polynomial.hermite.hermgauss(61)

# ---- the definitive fit: all three blocks, gamma free per block (Table 1) --
d = load_json("axes_fit.json")
b0, gam, wr, k, sig = d["b0"], d["gam"], d["wr"], d["k"], d["sig"]
ws, b0B, gamB = d["wsB"], d["b0B"], d["gamB"]
TL, TR = d["TL"], d["TR"]
CI_L, CI_R = tuple(d["ciL"]), tuple(d["ciR"])     # profile likelihood
FLOOR = d["floor"]
print("definitive fit: T_Lum %.4f  T_RG %.4f  ratio %.3f" % (TL, TR, 1 / wr))


def marg(eta):
    u = np.sqrt(2) * sig * Z
    return np.array([np.sum(WG / np.sqrt(np.pi) / (1 + np.exp(-(e + u))))
                     for e in np.atleast_1d(eta)])


# ---- the measured series: levels, group means, 95% bootstrap bounds -------
sL, sR, sS = (tuple(np.array(a) for a in d[s]) for s in ("sL", "sR", "sS"))
print("levels  Lum %s\n        RG  %s\n        S   %s" % (sL[0], sR[0], sS[0]))

# a_S was varied only in System B, and never with a_RG at zero. The curve for
# that series therefore comes from the definitive all-block fit, evaluated on
# the a_RG pedestal the series actually sat on.
wsB, wrB, kB = ws, wr, k
RG_PED = 0.050
print("System B series: pedestal a_RG = %.3f, weights from the all-block fit" % RG_PED)

# ---- the figure ---------------------------------------------------------
# The S series is carried past the levels tested, as far as the amplitude at
# which it reaches half detection; that stretch is drawn thin, since it is an
# extrapolation of the fit rather than anything measured.
XHI = 1.10
fig, ax = plt.subplots(figsize=(W, 2.90))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.30, lw=0.5, color="#DDDDDD")
grid = np.logspace(np.log10(0.03), np.log10(XHI), 600)
handles = []
for xs, col, mk, lab, eta in (
        (sL, C_LUM, "o", r"$a_{\mathrm{Lum}}$ alone", b0 + gam * grid),
        (sR, C_RG, "s", r"$a_{\mathrm{RG}}$ alone", b0 + gam * wr * grid),
        (sS, C_S, "^", r"$a_{\mathrm{S}}$, at $a_{\mathrm{RG}}=0.05$",
         b0B + gamB * ((wrB * RG_PED) ** kB + (wsB * grid) ** kB) ** (1 / kB))):
    q = marg(eta)
    inside = (grid >= xs[0].min() * 0.98) & (grid <= xs[0].max() * 1.02)
    ax.plot(grid, q, color=col, lw=0.8, ls=(0, (4, 2)), alpha=0.75, zorder=3)
    ax.plot(grid[inside], q[inside], color=col, lw=1.5, zorder=3)
    ax.errorbar(xs[0], xs[1], yerr=[xs[1] - xs[2], xs[3] - xs[1]], fmt=mk, ms=5,
                color=col, mfc="white", mew=1.3, lw=0, elinewidth=0.9, capsize=2,
                ecolor=col, zorder=4)
    handles.append(Line2D([], [], color=col, lw=1.5, marker=mk, ms=5,
                          mfc="white", mew=1.3, label=lab))
    # the extrapolated crossings stay off the figure: the dashed stretch is a
    # model extrapolation, and marking a point on it would read as a datum
    if q[-1] >= 0.5:
        print("   %-22s extrapolated P(A)=0.50 at %.3f (not drawn)"
              % (lab, float(np.interp(0.5, q, grid))))
ax.axhline(0.5, color="0.55", lw=0.8, ls="-.", zorder=2)
ax.axhline(FLOOR, color="0.55", lw=0.8, ls=":", zorder=2)
for T, ci, col in ((TL, CI_L, C_LUM), (TR, CI_R, C_RG)):
    ax.plot([ci[0], ci[1]], [0.5, 0.5], color=col, lw=3.0, solid_capstyle="butt",
            alpha=0.45, zorder=5)
    ax.plot([T], [0.5], marker="|", ms=13, color=col, mew=2.0, zorder=6)
ax.annotate(r"$T_{\mathrm{Lum}}=%.3f$" % TL, xy=(TL, 0.5), xytext=(0.037, 0.60),
            color=C_LUM, fontsize=FS_S, ha="left", zorder=6,
            arrowprops=dict(arrowstyle="-", color=C_LUM, lw=0.6))
ax.annotate(r"$T_{\mathrm{RG}}=%.3f$" % TR, xy=(TR, 0.5), xytext=(0.20, 0.36),
            color=C_RG, fontsize=FS_S, ha="left", zorder=6,
            arrowprops=dict(arrowstyle="-", color=C_RG, lw=0.6))
# the two reference lines are named in the caption, not on the plot
ax.set_xscale("log")
ax.set_xlabel("cone-opponent contrast on one axis")
ax.set_ylabel("detection rate $P(A)$")
ax.set_xlim(0.03, XHI); ax.set_ylim(0, 1.0)
ax.set_xticks([0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=2.4,
          labelspacing=0.3, fontsize=FS_S, borderpad=0.1)
fig.tight_layout()
finish(fig, "fig_axes")

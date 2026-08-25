"""fig_rq1: detection against MacAdam radius, one panel per the five ellipses.
fig_pred: the same measurements against what Eq. (5) predicts, with the weights
          taken from the other two blocks and only intercept and scale free."""
import json, glob, os, sys
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
sys.stdout.reconfigure(encoding="utf-8")
from figstyle import W, FS_S, ELL, ENAME, finish, panel, one_zero
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _paths import SHARE, conditions, load_json   # noqa: F401
import cfvi as C
FLOOR = load_json("results_data.json")["floor"]

K, W_LUM, W_RG = 2.16, 1.0, 1.0 / 1.32
W_S = W_RG / 30.9


def coord(c1, c2):
    la, ls = C.rgb_to_lms(np.array(c1)), C.rgb_to_lms(np.array(c2))
    m = (la + ls) / 2.0
    cL, cM, cS = (la - ls) / np.where(m > 1e-9, m, 1e-9)
    return abs((cL + cM) / 2), abs(cL - cM) / 2, abs(cS - (cL + cM) / 2) / 2


acc, CI = {}, {}
for row in conditions("A"):
    k = (row["ellipse"], row["radius"])
    acc[k] = {"n": row["n_trials"], "y": row["n_detected"],
              "co": coord(row["c1"], row["c2"])}
    CI[k] = row["ci95"]          # cluster bootstrap over observers, 2000 draws


def boot_ci(k):
    """The interval carried in the condition table; see README."""
    return CI[k]

keys = sorted(acc)
A = np.array([acc[k]["co"] for k in keys])
n = np.array([acc[k]["n"] for k in keys], float)
y = np.array([acc[k]["y"] for k in keys], float)
p = y / n


def vis(a):
    a = np.clip(np.atleast_2d(a), 0, None)   # the spline can dip below zero between knots
    return ((W_LUM * a[:, 0]) ** K + (W_RG * a[:, 1]) ** K + (W_S * a[:, 2]) ** K) ** (1 / K)


def nll(th, x):
    q = np.clip(1 / (1 + np.exp(-(th[0] + th[1] * x))), 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(q) + (n - y) * np.log(1 - q))


V = vis(A)
res = minimize(nll, [-3.0, 40.0], args=(V,), method="Nelder-Mead",
               options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
b0, b1 = res.x
pred = 1 / (1 + np.exp(-(b0 + b1 * V)))
print("V model on System A:  b0 %.3f  b1 %.2f   RMSE %.3f   r %.3f"
      % (b0, b1, np.sqrt(np.mean((pred - p) ** 2)), np.corrcoef(pred, p)[0, 1]))

# ---------------------------------------------------------------- fig_rq1
fig, ax = plt.subplots(figsize=(W, 2.27))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.30, lw=0.5, color="#DDDDDD")
ax.axhline(FLOOR, color="0.55", lw=0.8, ls=":", zorder=2)
for e in ELL:
    idx = [i for i, k in enumerate(keys) if k[0] == e]
    rr = [keys[i][1] for i in idx]
    ci = np.array([boot_ci(keys[i]) for i in idx])
    ax.errorbar(rr, p[idx], yerr=[p[idx] - ci[:, 0], ci[:, 1] - p[idx]],
                color=ELL[e]["c"], ls=ELL[e]["ls"], marker=ELL[e]["m"],
                ms=5, mfc="white", mew=1.3, lw=1.5, elinewidth=0.8, capsize=2,
                zorder=3, clip_on=False)
    ax.text(rr[-1] + 1.5, p[idx][-1], ENAME[e], color=ELL[e]["c"], fontsize=FS_S,
            va="center", zorder=4)
# the floor is named where it runs, in the clear space at the right
ax.text(57, FLOOR + 0.012, "catch false alarms", fontsize=FS_S, color="0.45",
        ha="right", va="bottom")
ax.set_xlabel("MacAdam radius $r$ [JND]")
ax.set_ylabel("detection rate $P(A)$")
ax.set_xlim(0, 58); ax.set_ylim(0, 1.0)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks(np.arange(0, 56, 5), minor=True)
one_zero(ax, keep="x")
fig.tight_layout()
finish(fig, "fig_rq1")

# ---------------------------------------------------------------- fig_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.65), gridspec_kw={"wspace": 0.30})
for a in (ax1, ax2):
    a.set_axisbelow(True)
    a.grid(alpha=0.30, lw=0.5, color="#DDDDDD")
for e in ELL:
    idx = [i for i, k in enumerate(keys) if k[0] == e]
    rr = np.array([keys[i][1] for i in idx], float)
    kk = min(3, len(rr) - 1)
    u = np.arange(rr[0], rr[-1] + 1e-9, 1.0)
    aa = np.column_stack([make_interp_spline(rr, A[idx, j], k=kk)(u) for j in range(3)])
    qq = 1 / (1 + np.exp(-(b0 + b1 * vis(aa))))
    ax1.plot(u, qq, color=ELL[e]["c"], ls=ELL[e]["ls"], lw=1.3)
    ax1.plot(rr, p[idx], ls="none", marker=ELL[e]["m"], ms=4.5, color=ELL[e]["c"],
             mfc="white", mew=1.2)
    ax2.plot(pred[idx], p[idx], ls="none", marker=ELL[e]["m"], ms=4.5,
             color=ELL[e]["c"], mfc="white", mew=1.2)
ax1.axhline(FLOOR, color="0.55", lw=0.8, ls=":", zorder=2)
ax1.set_xlabel("MacAdam radius $r$ [JND]")
ax1.set_ylabel("detection rate $P(A)$")
ax1.set_xlim(0, 58); ax1.set_ylim(0, 1.0)
ax1.set_xticks([0, 20, 40]); ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xticks(np.arange(0, 56, 10), minor=True)
one_zero(ax1, keep="x")
panel(ax1, "(a)")
ax2.plot([0, 1], [0, 1], color="0.55", lw=0.8, ls="--", zorder=2)
ax2.set_xlabel("predicted $P(A)$"); ax2.set_ylabel("measured $P(A)$")
ax2.set_xlim(0, 1.0); ax2.set_ylim(0, 1.0)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
one_zero(ax2, keep="x")
panel(ax2, "(b)")
ax2.text(0.96, 0.06, "$r=%.3f$" % np.corrcoef(pred, p)[0, 1], fontsize=FS_S,
         ha="right", color="0.25")
# the ellipses are named under the figure rather than beside their own curves,
# which at this size ran into the data
handles = [Line2D([], [], color=ELL[e]["c"], ls=ELL[e]["ls"], lw=1.3,
                  marker=ELL[e]["m"], ms=4.5, mfc="white", mew=1.2, label=ENAME[e])
           for e in ELL]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.11),
           ncol=5, fontsize=FS_S, frameon=False,
           handlelength=2.2, columnspacing=1.6, handletextpad=0.6)
# both panels get the same drawing area, so neither reads as the more important
for a in (ax1, ax2):
    a.set_box_aspect(1.0)
finish(fig, "fig_pred")

print("\nE12, measured vs predicted")
for i, k in enumerate(keys):
    if k[0] == 12:
        print("   r=%2d  a_RG %.4f  a_S %.4f   measured %.3f  predicted %.3f"
              % (k[1], A[i, 1], A[i, 2], p[i], pred[i]))

# -*- coding: utf-8 -*-
"""The true pair loci below the smallest measured radius.

For every System-A ellipse, solve the actual stimulus construction of the
paper (steps 1-4: uv placement, CAM16+HK equal-chroma direction, cone
isoluminance) at r = 1, 2, ... up to the smallest measured radius, and cache
the pair chromaticities. gen_xy3.py uses the cache to draw the lead-in from
the base color to the first measured point, instead of inventing it with a
spline.

Validation: the same construction is solved at every measured radius and
compared against the chromaticities recorded in the experiment logs; the
difference must stay within the 8-bit quantization of the framebuffer.
"""
import importlib.util, json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.stdout.reconfigure(encoding="utf-8")
from _paths import CODES, SHARE, DATA, conditions
sys.path.insert(0, CODES)
import macadam_newton as mn
import macadam_newton_cam16hk as hk


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfvi_c = load_as("cfvi_codes", os.path.join(CODES, "cfvi.py"))   # decompose, xyz2lin
cfvi_s = load_as("cfvi_share", os.path.join(SHARE, "cfvi.py"))   # M_RGB_TO_XYZ

Y0 = 0.4


def uvY2lin(u, v, Y):
    x, y = mn.uv_to_xy(u, v)
    return cfvi_c.xyz2lin(mn.xy_to_XYZ(x, y, Y))


def rgb_to_xy(rgb):
    v = cfvi_s.M_RGB_TO_XYZ @ np.clip(np.asarray(rgb, float), 0, None)
    return v[:2] / v.sum()


_theta_cache = {}


def theta_of(e, r):
    """Steps 1-3: the equal-chroma direction and uv half-distance at (e, r)."""
    if (e, r) not in _theta_cache:
        u0, v0, a, b, ang = mn.transform_ellipse(*mn.ELLIPSE_XY[e])
        ell = (a, b, ang)
        g = hk.compute_jacobian_JHK(u0, v0)
        rr = hk.solve_twostage(u0, v0, ell, g, [r])[0]
        d = mn._compute_d(r, rr["theta"], ell)
        _theta_cache[(e, r)] = (u0, v0, np.deg2rad(rr["theta"]), d)
    return _theta_cache[(e, r)]


def pair(e, r, dY):
    u0, v0, th, d = theta_of(e, r)
    ua, va = u0 + d * np.cos(th), v0 + d * np.sin(th)
    us, vs = u0 - d * np.cos(th), v0 - d * np.sin(th)
    return (uvY2lin(ua, va, Y0 + dY), uvY2lin(us, vs, Y0 - dY),
            mn.uv_to_xy(ua, va), mn.uv_to_xy(us, vs))


def solve_pair(e, r):
    """Step 4 on top: the cone-isoluminant pair at (e, r)."""
    f = lambda dY: cfvi_c.decompose(*pair(e, r, dY)[:2])[0]
    dY = brentq(f, -0.2, 0.2, xtol=1e-7)
    return pair(e, r, dY)


# ---- the pairs as the experiment recorded them ---------------------------
rec = {(row["ellipse"], row["radius"]): (row["c1"], row["c2"])
       for row in conditions("A")}

# ---- solve, validate, cache ----------------------------------------------
out = {}
print("validation against the recorded stimuli:")
for e in (8, 12, 13, 14, 19):
    radii = sorted(r for (ee, r) in rec if ee == e)
    worst = 0.0
    for r in radii:
        _, _, xy1, xy2 = solve_pair(e, r)
        q1 = rgb_to_xy(rec[(e, r)][0])
        q2 = rgb_to_xy(rec[(e, r)][1])
        d_direct = max(np.hypot(*(np.array(xy1) - q1)),
                       np.hypot(*(np.array(xy2) - q2)))
        d_swap = max(np.hypot(*(np.array(xy1) - q2)),
                     np.hypot(*(np.array(xy2) - q1)))
        worst = max(worst, float(min(d_direct, d_swap)))
    fine_r = list(range(1, radii[0]))
    lead1, lead2 = [], []
    for r in fine_r:
        _, _, xy1, xy2 = solve_pair(e, r)
        lead1.append([round(float(xy1[0]), 5), round(float(xy1[1]), 5)])
        lead2.append([round(float(xy2[0]), 5), round(float(xy2[1]), 5)])
    x0, y0 = mn.ELLIPSE_XY[e][0], mn.ELLIPSE_XY[e][1]
    out[str(e)] = dict(center=[x0, y0], r=fine_r, side1=lead1, side2=lead2,
                       worst_dxy=round(worst, 5))
    print("  E%-3d measured r %s  worst |dxy| %.5f   lead-in r=%s"
          % (e, radii, worst, fine_r))

json.dump(out, open(os.path.join(DATA, "fine_loci.json"), "w"))
print("saved fine_loci.json")

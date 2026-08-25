"""
CAM16 + Hellwig et al. (2022) H-K補正版 MacAdam楕円最適化。
"""

import numpy as np
import colour

from macadam_newton import (
    ELLIPSE_XY, Y0, xy_to_uv, uv_to_xy, xy_to_XYZ, transform_ellipse,
    ellipse_radius, _compute_d,
)

_SURROUND_CAM16 = colour.appearance.InductionFactors_CAM16(1, 0.69, 1)
XYZ_W = np.array([0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290])
L_A = 64.0
Y_B = 20.0


def hellwig_fh(h_deg):
    h = np.deg2rad(h_deg)
    return (0.160 * np.cos(h) + 0.132 * np.cos(2*h)
            - 0.405 * np.sin(h) + 0.080 * np.sin(2*h) + 0.792)


def J_HK(J, C, h):
    return J + hellwig_fh(h) * (C ** 0.587)


def cam16_JCh(u, v, Y):
    x, y = uv_to_xy(u, v)
    XYZ = xy_to_XYZ(x, y, Y)
    s = colour.appearance.XYZ_to_CAM16(XYZ * 100, XYZ_W * 100, L_A, Y_B, _SURROUND_CAM16)
    return float(s.J), float(s.C), float(s.h)


def cam16_JCh_HK(u, v, Y):
    J, C, h = cam16_JCh(u, v, Y)
    return J_HK(J, C, h), C, h


def eval_delta(u0, v0, ell, r, theta, dY):
    th = np.deg2rad(theta)
    d = _compute_d(r, theta, ell)
    ua, va = u0 + d*np.cos(th), v0 + d*np.sin(th)
    us, vs = u0 - d*np.cos(th), v0 - d*np.sin(th)
    J_hk_a, Ca, ha = cam16_JCh_HK(ua, va, Y0 + dY)
    J_hk_s, Cs, hs = cam16_JCh_HK(us, vs, Y0 - dY)
    dh = abs(ha - hs)
    if dh > 180:
        dh = 360 - dh
    return J_hk_a - J_hk_s, Ca - Cs, dh


def cam16_JC_HK(u, v, Y):
    J_hk, C, h = cam16_JCh_HK(u, v, Y)
    return J_hk, C


def compute_jacobian_JHK(u0, v0, eps_uv=1e-5, eps_Y=1e-4):
    grad = np.zeros(3)
    for i, (du, dv, dYi) in enumerate([(eps_uv,0,0), (0,eps_uv,0), (0,0,eps_Y)]):
        Jp, _ = cam16_JC_HK(u0+du, v0+dv, Y0+dYi)
        Jm, _ = cam16_JC_HK(u0-du, v0-dv, Y0-dYi)
        eps = [eps_uv, eps_uv, eps_Y][i]
        grad[i] = (Jp - Jm) / (2 * eps)
    return grad


def first_order_dY_for_theta(r, ell, theta, grad):
    g_u, g_v, g_Y = grad
    th = np.deg2rad(theta)
    d = _compute_d(r, theta, ell)
    return -d * (g_u * np.cos(th) + g_v * np.sin(th)) / g_Y


def newton_1d_dY(u0, v0, ell, r, theta, dY0, max_iter=10, eps_dY=1e-5):
    dY = dY0
    for _ in range(max_iter):
        dJ_hk, _, _ = eval_delta(u0, v0, ell, r, theta, dY)
        if abs(dJ_hk) < 0.001:
            break
        dJ_p, _, _ = eval_delta(u0, v0, ell, r, theta, dY + eps_dY)
        dJ_m, _, _ = eval_delta(u0, v0, ell, r, theta, dY - eps_dY)
        deriv = (dJ_p - dJ_m) / (2 * eps_dY)
        if abs(deriv) < 1e-12:
            break
        dY -= dJ_hk / deriv
    return dY


def _dC_at_theta(u0, v0, ell, r, theta, grad):
    dY0 = first_order_dY_for_theta(r, ell, theta, grad)
    dY = newton_1d_dY(u0, v0, ell, r, theta, dY0)
    _, dC, _ = eval_delta(u0, v0, ell, r, theta, dY)
    return dC, dY


def nested_newton(u0, v0, ell, r, theta0, grad, max_iter=15, eps_th=0.05):
    theta = theta0
    for _ in range(max_iter):
        dC, dY = _dC_at_theta(u0, v0, ell, r, theta, grad)
        if abs(dC) < 0.01:
            break
        dC_p, _ = _dC_at_theta(u0, v0, ell, r, theta + eps_th, grad)
        dC_m, _ = _dC_at_theta(u0, v0, ell, r, theta - eps_th, grad)
        deriv = (dC_p - dC_m) / (2 * eps_th)
        if abs(deriv) < 1e-12:
            break
        theta = (theta - dC / deriv) % 180
    dY0 = first_order_dY_for_theta(r, ell, theta, grad)
    dY = newton_1d_dY(u0, v0, ell, r, theta, dY0)
    return theta, dY


def solve_twostage(u0, v0, ell, grad, r_values):
    results = []
    prev_th = None
    for r in r_values:
        if prev_th is None:
            best_th, best_absdc = 0, np.inf
            for th_cand in np.arange(0, 180, 2.0):
                try:
                    dc, _ = _dC_at_theta(u0, v0, ell, r, th_cand, grad)
                    if abs(dc) < best_absdc:
                        best_absdc = abs(dc); best_th = th_cand
                except Exception:
                    pass
            theta0 = best_th
        else:
            theta0 = prev_th
        th, dY = nested_newton(u0, v0, ell, r, theta0, grad)
        prev_th = th
        dJ_hk, dC, dh = eval_delta(u0, v0, ell, r, th, dY)
        th_rad = np.deg2rad(th)
        d = _compute_d(r, th, ell)
        results.append({'r': r, 'theta': th, 'dY': dY, 'dh': dh,
                        'Y_add': Y0 + dY, 'Y_sub': Y0 - dY})
    return results

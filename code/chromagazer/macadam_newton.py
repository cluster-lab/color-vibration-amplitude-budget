"""
MacAdam楕円補正のコアライブラリ。
1次解析解 + Sequential Newton法で (θ*, δY*)(r) を高速に算出。

距離は方向依存: d = r × r_ellipse(θ) で、どの方向でも r JND。
"""

import numpy as np
import colour

# MacAdam楕円データ (x, y, a, b, theta_deg)
ELLIPSE_XY = [
    [0.160, 0.057, 0.00085, 0.00035,  62.5],[0.187, 0.118, 0.00220, 0.00055,  77.0],
    [0.253, 0.125, 0.00250, 0.00050,  55.5],[0.150, 0.680, 0.00960, 0.00230, 105.0],
    [0.131, 0.521, 0.00470, 0.00200, 112.5],[0.212, 0.550, 0.00580, 0.00230, 100.0],
    [0.258, 0.450, 0.00500, 0.00200,  92.0],[0.152, 0.365, 0.00380, 0.00190, 110.0],
    [0.280, 0.385, 0.00400, 0.00150,  75.5],[0.380, 0.498, 0.00440, 0.00120,  70.0],
    [0.160, 0.200, 0.00210, 0.00095, 104.0],[0.228, 0.250, 0.00310, 0.00090,  72.0],
    [0.305, 0.323, 0.00230, 0.00090,  58.0],[0.385, 0.393, 0.00380, 0.00160,  65.5],
    [0.472, 0.399, 0.00320, 0.00140,  51.0],[0.527, 0.350, 0.00260, 0.00130,  20.0],
    [0.475, 0.300, 0.00290, 0.00110,  28.5],[0.510, 0.236, 0.00240, 0.00120,  29.5],
    [0.596, 0.283, 0.00260, 0.00130,  13.0],[0.344, 0.284, 0.00230, 0.00090,  60.0],
    [0.390, 0.237, 0.00250, 0.00100,  47.0],[0.441, 0.198, 0.00280, 0.00095,  34.5],
    [0.278, 0.223, 0.00240, 0.00055,  57.5],[0.300, 0.163, 0.00290, 0.00060,  54.0],
    [0.365, 0.153, 0.00360, 0.00095,  40.0],
]

# CIECAM02 viewing conditions
_SURROUND = colour.appearance.InductionFactors_CIECAM02(1, 0.69, 1)
Y0 = 0.4
XYZ_W = np.array([0.3127 * 1.0 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) * 1.0 / 0.3290])
L_A = 64.0
Y_B = 20.0


def xy_to_uv(x, y):
    d = -2*x + 12*y + 3
    return 4*x/d, 6*y/d

def uv_to_xy(u, v):
    d = 2*u - 8*v + 4
    return 3*u/d, 2*v/d

def xy_to_XYZ(x, y, Y):
    if y <= 0:
        return np.array([0., 0., 0.])
    return np.array([x*Y/y, Y, (1-x-y)*Y/y])


def transform_ellipse(x, y, a, b, theta_deg):
    """xy楕円パラメータをuv空間に変換。
    xy楕円上の点を直接uv変換し、最遠点方向を長軸とする（Jacobianの180°曖昧性を回避）。
    """
    th = np.deg2rad(theta_deg)
    u0, v0 = xy_to_uv(x, y)

    best_dist, best_angle = 0, 0
    worst_dist = np.inf
    n_sample = 360
    for t in np.linspace(0, 2*np.pi, n_sample, endpoint=False):
        xe = x + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
        ye = y + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)
        ue, ve = xy_to_uv(xe, ye)
        d = np.sqrt((ue - u0)**2 + (ve - v0)**2)
        if d > best_dist:
            best_dist = d
            best_angle = np.arctan2(ve - v0, ue - u0)
        if d < worst_dist:
            worst_dist = d

    a_uv = best_dist
    b_uv = worst_dist
    angle_uv = np.rad2deg(best_angle)

    return u0, v0, a_uv, b_uv, angle_uv


# --- 方向依存の楕円半径 ---

def ellipse_radius(theta_deg, a_uv, b_uv, angle_uv):
    """uv空間での楕円半径 r_ellipse(θ)。θ方向に1 JNDの距離を返す。"""
    delta = np.deg2rad(theta_deg - angle_uv)
    return a_uv * b_uv / np.sqrt((b_uv * np.cos(delta))**2 + (a_uv * np.sin(delta))**2)


def _compute_d(r, theta, ell):
    """r JND分のuv距離を算出。ell = (a_uv, b_uv, angle_uv)"""
    return r * ellipse_radius(theta, *ell)


# --- CIECAM02 ---

def cam02_JCh(u, v, Y):
    """(u,v,Y) -> (J, C, h)"""
    x, y = uv_to_xy(u, v)
    XYZ = xy_to_XYZ(x, y, Y)
    s = colour.appearance.XYZ_to_CIECAM02(XYZ*100, XYZ_W*100, L_A, Y_B, _SURROUND)
    return float(s.J), float(s.C), float(s.h)


def cam02_JC(u, v, Y):
    J, C, _ = cam02_JCh(u, v, Y)
    return J, C


def eval_delta(u0, v0, ell, r, theta, dY):
    """色ペアの ΔJ, ΔC, Δh を算出。ell = (a_uv, b_uv, angle_uv)"""
    th = np.deg2rad(theta)
    d = _compute_d(r, theta, ell)
    ua, va = u0 + d*np.cos(th), v0 + d*np.sin(th)
    us, vs = u0 - d*np.cos(th), v0 - d*np.sin(th)
    Ja, Ca, ha = cam02_JCh(ua, va, Y0 + dY)
    Js, Cs, hs = cam02_JCh(us, vs, Y0 - dY)
    dh = abs(ha - hs)
    if dh > 180:
        dh = 360 - dh
    return Ja - Js, Ca - Cs, dh


def compute_jacobian(u0, v0, eps_uv=1e-5, eps_Y=1e-4):
    """中心点での ∂(J,C)/∂(u,v,Y)"""
    jac = np.zeros((2, 3))
    for i, (du, dv, dYi) in enumerate([(eps_uv,0,0), (0,eps_uv,0), (0,0,eps_Y)]):
        Jp, Cp = cam02_JC(u0+du, v0+dv, Y0+dYi)
        Jm, Cm = cam02_JC(u0-du, v0-dv, Y0-dYi)
        eps = [eps_uv, eps_uv, eps_Y][i]
        jac[0, i] = (Jp - Jm) / (2 * eps)
        jac[1, i] = (Cp - Cm) / (2 * eps)
    return jac


def first_order_theta(jac):
    """1次解析解 θ* (rによらない定数)"""
    j_u, j_v, j_Y = jac[0]
    c_u, c_v, c_Y = jac[1]
    alpha = c_u - c_Y * j_u / j_Y
    beta  = c_v - c_Y * j_v / j_Y
    return np.rad2deg(np.arctan2(-alpha, beta)) % 180


def first_order_dY(r, ell, theta, jac):
    """1次解析解 δY*(r)"""
    j_u, j_v, j_Y = jac[0]
    th = np.deg2rad(theta)
    d = _compute_d(r, theta, ell)
    return -d * (j_u * np.cos(th) + j_v * np.sin(th)) / j_Y


def newton_refine(u0, v0, ell, r, theta0, dY0, max_iter=5, eps_th=0.1, eps_dY=1e-5):
    """Newton法で ΔJ=0, ΔC=0 を解く"""
    theta, dY = theta0, dY0
    for _ in range(max_iter):
        dJ, dC, _ = eval_delta(u0, v0, ell, r, theta, dY)
        if abs(dJ) + abs(dC) < 0.01:
            break
        dJ_pth, dC_pth, _ = eval_delta(u0, v0, ell, r, theta + eps_th, dY)
        dJ_mth, dC_mth, _ = eval_delta(u0, v0, ell, r, theta - eps_th, dY)
        dJ_pdY, dC_pdY, _ = eval_delta(u0, v0, ell, r, theta, dY + eps_dY)
        dJ_mdY, dC_mdY, _ = eval_delta(u0, v0, ell, r, theta, dY - eps_dY)
        J_mat = np.array([
            [(dJ_pth - dJ_mth) / (2*eps_th), (dJ_pdY - dJ_mdY) / (2*eps_dY)],
            [(dC_pth - dC_mth) / (2*eps_th), (dC_pdY - dC_mdY) / (2*eps_dY)],
        ])
        try:
            step = np.linalg.solve(J_mat, np.array([dJ, dC]))
            theta = (theta - step[0]) % 180
            dY -= step[1]
        except np.linalg.LinAlgError:
            break
    return theta, dY


def _make_result(r, theta, dY, u0, v0, ell, dJ, dC, dh):
    """結果dictを生成"""
    th_rad = np.deg2rad(theta)
    d = _compute_d(r, theta, ell)
    return {
        'r': r, 'theta': theta, 'dY': dY,
        'dJ': dJ, 'dC': dC, 'dh': dh,
        'cost': abs(dJ) + abs(dC),
        'u_add': u0 + d*np.cos(th_rad), 'v_add': v0 + d*np.sin(th_rad),
        'u_sub': u0 - d*np.cos(th_rad), 'v_sub': v0 - d*np.sin(th_rad),
        'Y_add': Y0 + dY, 'Y_sub': Y0 - dY,
    }


def solve_sequential(u0, v0, ell, jac, r_values):
    """逐次Newton追跡で全rの (θ*, δY*) を算出。ell = (a_uv, b_uv, angle_uv)"""
    theta1 = first_order_theta(jac)
    results = []
    prev_th, prev_dY = None, None

    for r in r_values:
        if prev_th is None:
            dY0 = first_order_dY(r, ell, theta1, jac)
            th, dY = newton_refine(u0, v0, ell, r, theta1, dY0)
        else:
            th, dY = newton_refine(u0, v0, ell, r, prev_th, prev_dY)
        prev_th, prev_dY = th, dY

        dJ, dC, dh = eval_delta(u0, v0, ell, r, th, dY)
        results.append(_make_result(r, th, dY, u0, v0, ell, dJ, dC, dh))
    return results, theta1


# --- 2段階最適化: ΔJ=0 (via dY) 拘束下で |ΔC| 最小化 (via θ) ---

def newton_1d_dY_cam02(u0, v0, ell, r, theta, dY0, max_iter=10, eps_dY=1e-5):
    """1D Newton: θ固定で dY のみ調整し ΔJ=0 を解く"""
    dY = dY0
    for _ in range(max_iter):
        dJ, _, _ = eval_delta(u0, v0, ell, r, theta, dY)
        if abs(dJ) < 0.001:
            break
        dJ_p, _, _ = eval_delta(u0, v0, ell, r, theta, dY + eps_dY)
        dJ_m, _, _ = eval_delta(u0, v0, ell, r, theta, dY - eps_dY)
        deriv = (dJ_p - dJ_m) / (2 * eps_dY)
        if abs(deriv) < 1e-12:
            break
        dY -= dJ / deriv
    return dY


def _dC_at_theta_cam02(u0, v0, ell, r, theta, jac):
    """θを与えて内側Newton (dY→ΔJ=0) を解き、残ったΔCを返す"""
    dY0 = first_order_dY(r, ell, theta, jac)
    dY = newton_1d_dY_cam02(u0, v0, ell, r, theta, dY0)
    _, dC, _ = eval_delta(u0, v0, ell, r, theta, dY)
    return dC, dY


def nested_newton_cam02(u0, v0, ell, r, theta0, jac,
                        max_iter=15, eps_th=0.05):
    """ネスト型Newton (CAM02版): 外側θでΔC=0, 内側dYでΔJ=0"""
    theta = theta0
    for _ in range(max_iter):
        dC, dY = _dC_at_theta_cam02(u0, v0, ell, r, theta, jac)
        if abs(dC) < 0.01:
            break
        dC_p, _ = _dC_at_theta_cam02(u0, v0, ell, r, theta + eps_th, jac)
        dC_m, _ = _dC_at_theta_cam02(u0, v0, ell, r, theta - eps_th, jac)
        deriv = (dC_p - dC_m) / (2 * eps_th)
        if abs(deriv) < 1e-12:
            break
        theta = (theta - dC / deriv) % 180
    dY0 = first_order_dY(r, ell, theta, jac)
    dY = newton_1d_dY_cam02(u0, v0, ell, r, theta, dY0)
    return theta, dY


def solve_sequential_twostage(u0, v0, ell, jac, r_values):
    """2段階ネスト型Newton (CAM02版): ΔJ=0拘束下で|ΔC|最小化"""
    theta1 = first_order_theta(jac)
    results = []
    prev_th = None

    for r in r_values:
        theta0 = theta1 if prev_th is None else prev_th
        th, dY = nested_newton_cam02(u0, v0, ell, r, theta0, jac)
        prev_th = th

        dJ, dC, dh = eval_delta(u0, v0, ell, r, th, dY)
        results.append(_make_result(r, th, dY, u0, v0, ell, dJ, dC, dh))
    return results, theta1

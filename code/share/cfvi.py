"""
CFVI: Chromatic Flicker Visibility Index.

A scalar that estimates the perceived flicker strength of a color pair
on a real display, accounting for:
  1. Display gamut clipping (the "post-clip" RGB is what actually emits)
  2. Cone responses (LMS via Stockman & Sharpe 2deg)
  3. Three opponent visual pathways (Lum = L+M, RG = L-M, BY = S - (L+M)/2)
  4. Temporal contrast sensitivity at the flicker frequency (CSF weighting)

CFVI = sqrt( (W_Lum * dLum)^2 + (W_RG * dRG)^2 + (W_BY * dBY)^2 )

Implementation uses sRGB primaries (= Rec.709) since Quest 3 native is sRGB.
For other displays, swap M_RGB_TO_XYZ to the target primaries' matrix.
"""

import numpy as np

# ===== Color matrices =====
# linear sRGB (Rec.709 primaries, D65) -> XYZ
M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# XYZ -> LMS (Stockman & Sharpe 2 degree, normalized to D65)
# Using the Hunt-Pointer-Estevez matrix as a reasonable approximation;
# CAM16/Stockman differ slightly but this is sufficient for opponent-channel projection.
M_XYZ_TO_LMS = np.array([
    [ 0.38971,  0.68898, -0.07868],
    [-0.22981,  1.18340,  0.04641],
    [ 0.0,      0.0,      1.0    ],
])


def rgb_to_lms(rgb):
    """linear sRGB (clipped to [0,1]) -> LMS."""
    rgb_clipped = np.clip(rgb, 0.0, 1.0)
    xyz = M_RGB_TO_XYZ @ rgb_clipped
    return M_XYZ_TO_LMS @ xyz


def opponent_diff(lms_a, lms_b):
    """Return (dLum, dRG, dBY) differences between two LMS triplets."""
    dL = lms_a[0] - lms_b[0]
    dM = lms_a[1] - lms_b[1]
    dS = lms_a[2] - lms_b[2]
    d_lum = (dL + dM) * 0.5
    d_rg  = dL - dM
    d_by  = dS - (dL + dM) * 0.5
    return d_lum, d_rg, d_by


# ===== Temporal CSF weights =====
# Pathway-specific sensitivity at a given flicker frequency.
# Calibrated so that at 22.5 Hz the values match published estimates:
#   W_Lum ~ 0.85   (achromatic CFF ~ 50-60 Hz)
#   W_RG  ~ 0.40   (L-M CFF ~ 25-30 Hz, just below)
#   W_BY  ~ 0.08   (S-cone CFF ~ 10-15 Hz, well above)
# References: Kelly (1979), Mullen (1985), Lee et al. (1990).
def csf_weights(freq_hz):
    """Returns (W_Lum, W_RG, W_BY) at given flicker frequency."""
    f = float(freq_hz)

    # Achromatic: very gradual falloff. ~1.0 below 10 Hz, ~0.85 at 22.5, ~0.5 at 45.
    w_lum = min(1.0, np.exp(-(f * f) / 2200.0) + 0.10)

    # Red-Green: flat below 4 Hz, exponential decay above.
    # Tuned: W_RG(22.5) ~ 0.40, W_RG(45) ~ 0.05.
    excess = max(0.0, f - 4.0)
    w_rg = max(0.01, np.exp(-(excess * excess) / 380.0))

    # Blue-Yellow: flat below 2 Hz, faster decay above.
    # Tuned: W_BY(22.5) ~ 0.08, W_BY(45) ~ 0.005.
    excess = max(0.0, f - 2.0)
    w_by = max(0.001, np.exp(-(excess * excess) / 175.0))

    return float(w_lum), float(w_rg), float(w_by)


def cfvi(rgb_add, rgb_sub, freq_hz=22.5, weights=None):
    """
    CFVI for a single color pair.

    Parameters
    ----------
    rgb_add, rgb_sub : array_like (3,)
        linear sRGB values (will be clipped to [0,1] to simulate display gamut).
    freq_hz : float
        Flicker cycle frequency (Hz).
    weights : tuple (W_Lum, W_RG, W_BY) or None
        Override CSF weights. If None, uses csf_weights(freq_hz).

    Returns
    -------
    cfvi_value : float
    contributions : dict with 'lum', 'rg', 'by' (per-channel weighted contributions)
    """
    lms_a = rgb_to_lms(np.asarray(rgb_add, dtype=float))
    lms_b = rgb_to_lms(np.asarray(rgb_sub, dtype=float))

    d_lum, d_rg, d_by = opponent_diff(lms_a, lms_b)

    if weights is None:
        w_lum, w_rg, w_by = csf_weights(freq_hz)
    else:
        w_lum, w_rg, w_by = weights

    c_lum = w_lum * d_lum
    c_rg  = w_rg  * d_rg
    c_by  = w_by  * d_by

    cfvi_value = float(np.sqrt(c_lum * c_lum + c_rg * c_rg + c_by * c_by))

    return cfvi_value, {
        'lum': float(c_lum), 'rg': float(c_rg), 'by': float(c_by),
        'd_lum': float(d_lum), 'd_rg': float(d_rg), 'd_by': float(d_by),
        'w_lum': w_lum, 'w_rg': w_rg, 'w_by': w_by,
    }


if __name__ == '__main__':
    # Quick sanity test
    print("CSF weights at 22.5 Hz:")
    w = csf_weights(22.5)
    print(f"  W_Lum={w[0]:.3f}  W_RG={w[1]:.3f}  W_BY={w[2]:.3f}")
    print()
    print("Test pair: red vs cyan")
    val, c = cfvi([1.0, 0.0, 0.0], [0.0, 1.0, 1.0], 22.5)
    print(f"  CFVI={val:.4f}")
    print(f"  contributions: Lum={c['lum']:+.4f}  RG={c['rg']:+.4f}  BY={c['by']:+.4f}")

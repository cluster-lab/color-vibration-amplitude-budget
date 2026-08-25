import numpy as np

# --- sRGB(linear) <-> XYZ (D65) ---
M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)

# --- XYZ -> LMS : Hunt-Pointer-Estevez (normalized to D65) ---
M_HPE = np.array([
    [ 0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340,  0.04641],
    [ 0.00000, 0.00000,  1.00000]])
M_LMS2XYZ = np.linalg.inv(M_HPE)

def lin2xyz(rgb): return M_RGB2XYZ @ rgb
def xyz2lin(xyz): return M_XYZ2RGB @ xyz
def xyz2lms(xyz): return M_HPE @ xyz
def lms2xyz(lms): return M_LMS2XYZ @ lms
def lin2lms(rgb): return xyz2lms(lin2xyz(rgb))
def lms2lin(lms): return xyz2lin(lms2xyz(lms))

CENTER = np.array([0.4,0.4,0.4])        # linear sRGB neutral gray, Y=0.4
LMS0 = lin2lms(CENTER)

def decompose(c_add, c_sub):
    """色ペア(linear sRGB)を3経路コーンコントラストに分解。
    背景=ペアの平均。dL=(L_add-L_sub)/2 (amplitude). 戻り値 (Lum,RG,S,dY)。"""
    c0 = 0.5*(c_add+c_sub)
    L0,M0,S0 = lin2lms(c0)
    La,Ma,Sa = lin2lms(c_add)
    Ls,Ms,Ss = lin2lms(c_sub)
    cL = 0.5*(La-Ls)/L0
    cM = 0.5*(Ma-Ms)/M0
    cS = 0.5*(Sa-Ss)/S0
    Lum = cL+cM
    RG  = cL-cM
    Scn = cS-(cL+cM)/2.0
    Ya = lin2xyz(c_add)[1]; Yb = lin2xyz(c_sub)[1]
    dY = 0.5*(Ya-Yb)
    return Lum,RG,Scn,dY

def synth(Lum,RG,S, center=CENTER):
    """目標 (Lum,RG,S) をグレー中心まわりに作る。戻り値 add,sub (linear sRGB)。"""
    L0,M0,S0 = lin2lms(center)
    cL = 0.5*(Lum+RG)
    cM = 0.5*(Lum-RG)
    cS = S + 0.5*(cL+cM)   # S = cS-(cL+cM)/2  ->  cS = S+(cL+cM)/2 = S+Lum/2
    dL = cL*L0; dM=cM*M0; dS=cS*S0
    add = lms2lin(np.array([L0+dL,M0+dM,S0+dS]))
    sub = lms2lin(np.array([L0-dL,M0-dM,S0-dS]))
    return add,sub

def in_srgb(c, tol=1e-6):
    return bool(np.all(c>=-tol) and np.all(c<=1+tol))

if __name__ == "__main__":
    # Self-check: synthesizing the pair for |RG| = 0.02, |S| = 0.04 must give
    # back the linear values the paper quotes for that condition, and
    # decomposing the pair must return the amplitudes it was built from.
    add, sub = synth(0.0, 0.02, 0.04)
    print("check: condition |RG|=0.02, |S|=0.04")
    print("   synthesized  add:", np.round(add, 3), " sub:", np.round(sub, 3))
    print("   paper        add: [0.454 0.380 0.422] sub: [0.346 0.420 0.378]")
    print("   decomposed  :", tuple(round(x, 4) for x in decompose(add, sub)))

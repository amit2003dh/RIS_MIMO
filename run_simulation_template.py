"""
RIS-Aided Uplink Communication System - Function Calling Template
This script demonstrates the correct sequence to call functions from your notebook.

INSTRUCTIONS:
1. Run your notebook cells to define all functions in memory
2. Then use this script to understand the calling sequence
3. OR copy the function definitions from your notebook into the marked sections below
"""

import numpy as np
from scipy.linalg import dft

# =============================================================================
# GLOBAL PARAMETERS (Set these before running)
# =============================================================================
N = 64              # Number of BS antennas (ULA)
M = 64              # Number of RIS elements (UPA: Nv x Nh)
L = 16              # Number of selected beams
K = 2               # Number of users
Nk = 4              # Number of antennas per user (ULA)

# Channel parameters
Mk = 10             # Number of paths for user-RIS channel
Mo = 10             # Number of paths for BS-RIS channel
Gt = 10.0           # Transmit antenna gain (dB)
Gr = 10.0           # Receive antenna gain (dB)

# Physical parameters
fo = 28e9           # Carrier frequency (Hz)
lb = 3e8 / fo       # Wavelength (m)
dv = lb / 2         # Vertical antenna spacing (UPA)
dh = lb / 2         # Horizontal antenna spacing (UPA)
du = lb / 2         # User antenna spacing (ULA)
db = lb / 2         # BS antenna spacing (ULA)

# Power and noise
Pmax_dbm = 20       # Maximum transmit power per user (dBm)
Pmax = 10 ** (Pmax_dbm / 10 - 3)  # Convert to linear scale (Watts)
sigSQR = 1e-10      # Noise power (linear scale)

# Position parameters
x_ris, y_ris, z_ris = 50, 0, 10  # RIS position (m)
x_user_center, y_user_center, z_user_center = 0, 50, 1.5  # User center position (m)

# =============================================================================
# FUNCTION DEFINITIONS (Copy from your notebook)
# =============================================================================
# STEERING VECTORS
def cal_stering_vec_v(th, Nk, lb, du):
    k = np.arange(Nk)
    b_p_s = (2 * np.pi * du / lb) * np.sin(th)
    v = np.exp(1j * k * b_p_s)
    return v.reshape(-1, 1)

def cal_stering_vec_u(th, N, lb, db):
    k = np.arange(N)
    u = np.exp(1j * k * (2 * np.pi * db / lb) * np.sin(th))
    return u.reshape(-1, 1)

def cal_stering_vec_array_response(th, phi, Nv, Nh, dv, dh, lb):
    k = np.arange(Nv)
    beta_v = (2 * np.pi * dv / lb) * np.cos(th)
    a_v = np.exp(1j * k * beta_v).reshape(-1, 1)
    l = np.arange(Nh)
    beta_h = (2 * np.pi * dh / lb) * np.sin(th) * np.cos(phi)
    a_h = np.exp(1j * l * beta_h).reshape(-1, 1)
    a = np.kron(a_v, a_h)
    return a

# PATH LOSS
def cal_dk(x_user_center, y_user_center, z_user_center, x_ris, y_ris, z_ris, r=50):
    radius_k = np.random.uniform(0, r)
    angle_k = np.random.uniform(0, 2 * np.pi)
    x_user = x_user_center + radius_k * np.cos(angle_k)
    y_user = y_user_center + radius_k * np.sin(angle_k)
    z_user = z_user_center
    dk = ((x_user - x_ris)**2 + (y_user - y_ris)**2 + (z_user - z_ris)**2)**0.5
    dk = dk / 1000
    return dk

def cal_mu_k(fo, x_user_center, y_user_center, z_user_center, x_ris, y_ris, z_ris):
    dk = cal_dk(x_user_center, y_user_center, z_user_center, x_ris, y_ris, z_ris)
    f0_ghz = fo / 1e9
    mu_k = (10**(-9.25) * (f0_ghz * dk)**(-2))
    return mu_k

def cal_mu_o(fo, x_ris, y_ris, z_ris):
    # Simplified - use fixed distance for BS-RIS
    d0 = ((x_ris)**2 + (y_ris)**2 + (z_ris)**2)**0.5 / 1000
    f0_ghz = fo / 1e9
    mu_o = (10**(-9.25) * (f0_ghz * d0)**(-2))
    return mu_o

# CHANNEL MATRICES
def cal_Rk_matrix(Gt, Nv, Nh, Nk, dv, du, dh, lambda_val, Mk,
                  fo_hz, xuc, yuc, zuc, xris, yris, zris):
    if Mk == 0:
        return None
    num_tx = Nv * Nh
    Rk_sum = np.zeros((num_tx, Nk), dtype=complex)
    mu_K = cal_mu_k(fo_hz, xuc, yuc, zuc, xris, yris, zris)
    for _ in range(Mk):
        th_aoa = np.random.uniform(0, 2*np.pi)
        ph_aoa = np.random.uniform(0, 2*np.pi)
        th_aod = np.random.uniform(0, 2*np.pi)
        beta = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
        a_vec = cal_stering_vec_array_response(th_aoa, ph_aoa, Nv, Nh, dv, dh, lambda_val)
        v_vec = cal_stering_vec_v(th_aod, Nk, lambda_val, du)
        Rk_sum += beta * (a_vec @ v_vec.conj().T)
    return np.sqrt(mu_K * num_tx / Mk) * Gt * Rk_sum

def cal_G_matrix(mu_o, Gr, lambda_val, Nh, Nv, N_rx, M_o, dv, db, dh):
    if M_o == 0:
        return None
    num_rx = N_rx
    num_tx = Nv * Nh
    G_sum = np.zeros((num_rx, num_tx), dtype=complex)
    for _ in range(M_o):
        theta_aoa = np.random.uniform(0, 2*np.pi)
        phi_aod = np.random.uniform(0, 2*np.pi)
        theta_aod = np.random.uniform(0, 2*np.pi)
        beta = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
        a_vec = cal_stering_vec_array_response(theta_aod, phi_aod, Nv, Nh, dv, dh, lambda_val)
        u_vec = cal_stering_vec_u(theta_aoa, N_rx, lambda_val, db)
        G_sum += beta * (u_vec @ a_vec.conj().T)
    return np.sqrt(mu_o * num_tx / M_o) * Gr * G_sum

def create_phi_matrix(theta_vector):
    phi_vector = np.exp(1j * theta_vector)
    return np.diag(phi_vector)

# WMMSE HELPERS
def calc_Q(G, Phi, Rk_list, wk_list):
    N = G.shape[0]
    K = len(wk_list)
    U = dft(N, scale='sqrtn')
    Q = np.zeros((N, K), dtype=complex)
    for k in range(K):
        Hk = G @ Phi @ Rk_list[k]
        Q[:, k] = (U.conj().T @ Hk @ wk_list[k]).ravel()
    return Q

def calc_Ue(S, Q, sigma_sq):
    L = S.shape[0]
    SQ = S @ Q
    Mmat = SQ @ SQ.conj().T + sigma_sq * np.eye(L)
    return np.linalg.solve(Mmat, SQ)

def calc_We(Ue, S, Q):
    K = Q.shape[1]
    Ee = np.eye(K) - Ue.conj().T @ S @ Q
    Ee = Ee + 1e-8 * np.eye(K)
    return np.linalg.inv(Ee)

def calc_rate(G, Phi, S, Rk_list, wk_list, sigma_sq):
    L = S.shape[0]
    N = G.shape[0]
    U = dft(N, scale='sqrtn')
    Q = calc_Q(G, Phi, Rk_list, wk_list)
    SQ = S @ Q
    arg = np.eye(L) + (1.0/sigma_sq) * SQ @ SQ.conj().T
    sign, ld = np.linalg.slogdet(arg)
    return float(np.real(ld / np.log(2))) if sign > 0 else 0.0

# BISECTION HELPERS
def _pw(Bk, ck, lam):
    nk = Bk.shape[0]
    mat = Bk + lam * np.eye(nk) + 1e-10 * np.eye(nk)
    try:
        w = np.linalg.solve(mat, ck)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(mat) @ ck
    return float(np.real(np.vdot(w.ravel(), w.ravel())))

def _bisect(Bk, ck, P, tol=1e-9, max_it=200):
    if _pw(Bk, ck, 0.0) <= P + tol:
        return 0.0
    lo, hi = 0.0, 1.0
    while _pw(Bk, ck, hi) > P:
        hi *= 2.0
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        if _pw(Bk, ck, mid) > P:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

# STEP B: BEAMFORMER UPDATE
def step_B(G, Phi, Rk_list, We, Ue, S):
    N_ = G.shape[0]
    K_ = len(Rk_list)
    Nk_ = Rk_list[0].shape[1]
    U_ = dft(N_, scale='sqrtn')
    T = U_.conj().T @ S.conj().T @ Ue
    Hk = [G @ Phi @ Rk_list[k] for k in range(K_)]
    wk_new = []
    for k in range(K_):
        Bk = Hk[k].conj().T @ T
        Ak = Bk @ We @ Bk.conj().T
        ak = Bk @ We[:, k:k+1]
        if np.linalg.norm(ak) < 1e-15:
            w = (np.random.randn(Nk_, 1) + 1j * np.random.randn(Nk_, 1))
            wk_new.append(w * np.sqrt(Pmax / np.linalg.norm(w)**2))
            continue
        lam = _bisect(Ak, ak, Pmax)
        mat = Ak + lam * np.eye(Nk_) + 1e-10 * np.eye(Nk_)
        try:
            w = np.linalg.solve(mat, ak)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(mat) @ ak
        wk_new.append(w)
    return wk_new

# STEP C: RIS PHASE UPDATE
def step_C(G, Rk_list, wk_list, We, Ue, S, theta_init, max_iter=100, tol=1e-4):
    N_, M_ris = G.shape
    K_ = len(wk_list)
    U_ = dft(N_, scale='sqrtn')
    T = U_.conj().T @ S.conj().T @ Ue
    D = np.column_stack([(Rk_list[k] @ wk_list[k]).ravel() for k in range(K_)])
    TW = T @ We
    b = np.array([np.dot(G[:, m].conj(), TW @ D[m, :].conj()) for m in range(M_ris)])
    C1 = G.conj().T @ TW @ T.conj().T @ G
    C2 = D @ D.conj().T
    C = C1 * C2.T
    def _obj(v):
        return np.real(np.vdot(v, C @ v)) - 2.0 * np.real(np.vdot(b, v))
    theta = theta_init.copy()
    pe = np.exp(1j * theta)
    prev_obj = _obj(pe)
    for t in range(max_iter):
        for m in range(M_ris):
            cross = np.dot(C[:, m], pe.conj()) - C[m, m] * pe[m].conj()
            theta[m] = -np.angle(b[m].conj() - cross)
            pe[m] = np.exp(1j * theta[m])
        curr_obj = _obj(pe)
        if abs(curr_obj - prev_obj) < tol:
            break
        prev_obj = curr_obj
    return theta, np.diag(pe)

# STEP D: BEAM SELECTION
def step_D_GS(G, Phi, Rk_list, wk_list, sigma_sq, L_val, N_val, S_init=None, max_rounds=5):
    if S_init is not None:
        alpha = [int(np.argmax(np.abs(S_init[l]))) for l in range(L_val)]
        S = S_init.copy()
    else:
        alpha = list(np.random.choice(N_val, L_val, replace=False))
        S = np.zeros((L_val, N_val), dtype=complex)
        for i, b in enumerate(alpha):
            S[i, b] = 1.0
    for _ in range(max_rounds):
        changed = False
        for l in range(L_val):
            best_r, best_x = -np.inf, alpha[l]
            beta = [n for n in range(N_val) if n not in alpha or n == alpha[l]]
            for x in beta:
                S_try = S.copy()
                S_try[l] = 0.0
                S_try[l, x] = 1.0
                r = calc_rate(G, Phi, S_try, Rk_list, wk_list, sigma_sq)
                if r > best_r:
                    best_r = r
                    best_x = x
            if best_x != alpha[l]:
                alpha[l] = best_x
                S[l] = 0.0
                S[l, best_x] = 1.0
                changed = True
        if not changed:
            break
    return S

# MAIN BCD ALGORITHM
def run_BCD(G, Rk_list, sigma_sq, L_val, N_val, M_val, K_val, Nk_val,
            max_iter=150, tol=1e-6, verbose=True):
    # Initialize w
    wk_list = []
    for _ in range(K_val):
        w = (np.random.randn(Nk_val, 1) + 1j * np.random.randn(Nk_val, 1)) / np.sqrt(2 * Nk_val)
        nrm = float(np.real(np.vdot(w.ravel(), w.ravel())))
        wk_list.append(w * np.sqrt(Pmax / max(nrm, 1e-15)))
    
    # Initialize Phi
    theta = np.random.uniform(0, 2 * np.pi, M_val)
    Phi = np.diag(np.exp(1j * theta))
    
    # Initialize S
    beams = list(np.random.choice(N_val, L_val, replace=False))
    S = np.zeros((L_val, N_val), dtype=complex)
    for i, b in enumerate(beams):
        S[i, b] = 1.0
    
    # Bootstrap Ue and We
    Q = calc_Q(G, Phi, Rk_list, wk_list)
    Ue = calc_Ue(S, Q, sigma_sq)
    We = calc_We(Ue, S, Q)
    
    hist = []
    best_rate = -np.inf
    best_S = S.copy()
    best_wk = [w.copy() for w in wk_list]
    best_Phi = Phi.copy()
    
    for j in range(max_iter):
        # Refresh Q, then compute consistent (Ue, We) pair
        Q = calc_Q(G, Phi, Rk_list, wk_list)
        Ue = calc_Ue(S, Q, sigma_sq)
        We = calc_We(Ue, S, Q)
        
        # Update w
        wk_list = step_B(G, Phi, Rk_list, We, Ue, S)
        
        # Update Phi
        theta, Phi = step_C(G, Rk_list, wk_list, We, Ue, S, theta, max_iter=200)
        
        # Update S
        S_new = step_D_GS(G, Phi, Rk_list, wk_list, sigma_sq, L_val, N_val, S_init=S)
        r_new = calc_rate(G, Phi, S_new, Rk_list, wk_list, sigma_sq)
        r_curr = calc_rate(G, Phi, S, Rk_list, wk_list, sigma_sq)
        
        if r_new >= r_curr:
            S = S_new
            rate = r_new
        else:
            rate = r_curr
        
        if rate > best_rate:
            best_rate = rate
            best_S = S.copy()
            best_wk = [w.copy() for w in wk_list]
            best_Phi = Phi.copy()
        
        hist.append(rate)
        
        if verbose and (j % 10 == 0 or j == max_iter - 1):
            print(f"  iter {j+1:3d}  rate = {rate:.4f} bps/Hz")
        
        if j > 5 and abs(hist[-1] - hist[-2]) < tol and abs(hist[-2] - hist[-3]) < tol and abs(hist[-3] - hist[-4]) < tol:
            if verbose:
                print(f"  Converged at iter {j+1}")
            break
    
    return hist

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================
def initialize_channels():
    """Generate channel matrices G and Rk_list"""
    print("Step 1: Initializing channels...")
    ris_Nv = int(np.sqrt(M))
    ris_Nh = M // ris_Nv
    
    # Generate G matrix
    mu_o = cal_mu_o(fo, x_ris, y_ris, z_ris)
    G = cal_G_matrix(mu_o, Gr, lb, ris_Nh, ris_Nv, N, Mo, dv, db, dh)
    print(f"  G matrix shape: {G.shape}")
    
    # Generate Rk matrices
    Rk_list = []
    for k in range(K):
        x_uc = x_user_center + np.random.uniform(-10, 10)
        y_uc = y_user_center + np.random.uniform(-10, 10)
        z_uc = z_user_center
        Rk = cal_Rk_matrix(Gt, ris_Nv, ris_Nh, Nk, dv, du, dh, lb, Mk,
                          fo, x_uc, y_uc, z_uc, x_ris, y_ris, z_ris)
        Rk_list.append(Rk)
        print(f"  R{k+1} matrix shape: {Rk.shape}")
    
    return G, Rk_list

def run_single_simulation():
    """Run a single simulation with BCD algorithm"""
    print("=" * 60)
    print("RIS-Aided Uplink Communication Simulation")
    print("=" * 60)
    
    # Initialize channels
    G, Rk_list = initialize_channels()
    
    # Run BCD algorithm
    print("\nStep 2: Running BCD algorithm...")
    print(f"  Parameters: K={K}, N={N}, M={M}, L={L}, Nk={Nk}")
    print(f"  Pmax per user: {Pmax_dbm} dBm")
    
    rate_history = run_BCD(
        G=G,
        Rk_list=Rk_list,
        sigma_sq=sigSQR,
        L_val=L,
        N_val=N,
        M_val=M,
        K_val=K,
        Nk_val=Nk,
        max_iter=150,
        tol=1e-6,
        verbose=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)
    print(f"Final sum-rate: {rate_history[-1]:.4f} bps/Hz")
    print(f"Convergence after {len(rate_history)} iterations")
    print(f"Initial rate: {rate_history[0]:.4f} bps/Hz")
    print(f"Improvement: {rate_history[-1] - rate_history[0]:.4f} bps/Hz")
    
    return rate_history

def run_multiple_realizations(num_realizations=10):
    """Run multiple simulations with different channel realizations"""
    print("=" * 60)
    print(f"Running {num_realizations} Realizations")
    print("=" * 60)
    
    final_rates = []
    for real in range(num_realizations):
        print(f"\n--- Realization {real + 1} ---")
        G, Rk_list = initialize_channels()
        rate_history = run_BCD(
            G, Rk_list, sigSQR, L, N, M, K, Nk,
            max_iter=150, tol=1e-6, verbose=False
        )
        final_rate = rate_history[-1]
        final_rates.append(final_rate)
        print(f"  Final rate: {final_rate:.4f} bps/Hz")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Statistics Across Realizations")
    print("=" * 60)
    print(f"Mean: {np.mean(final_rates):.4f} bps/Hz")
    print(f"Std:  {np.std(final_rates):.4f} bps/Hz")
    print(f"Min:  {np.min(final_rates):.4f} bps/Hz")
    print(f"Max:  {np.max(final_rates):.4f} bps/Hz")
    
    return final_rates

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Option 1: Run single simulation
    print("Running single simulation...")
    rate_history = run_single_simulation()
    
    # Option 2: Run multiple realizations (uncomment to use)
    # print("\n\nRunning multiple realizations...")
    # final_rates = run_multiple_realizations(num_realizations=10)

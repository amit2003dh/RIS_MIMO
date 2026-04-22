"""
Main Simulation Script for RIS-Aided Uplink Communication System
This script demonstrates the correct function calling sequence to run the BCD algorithm.
"""

import numpy as np
from scipy.linalg import dft

# =============================================================================
# GLOBAL PARAMETERS (must be set before running)
# =============================================================================
# System dimensions
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

# Position parameters (example)
x_ris, y_ris, z_ris = 50, 0, 10  # RIS position (m)
x_user_center, y_user_center, z_user_center = 0, 50, 1.5  # User center position (m)

# =============================================================================
# STEP 1: Initialize Channels
# =============================================================================
def initialize_channels():
    """
    Generate the channel matrices G and Rk_list.
    This corresponds to the channel modeling phase.
    """
    print("Step 1: Initializing channels...")
    
    # RIS UPA dimensions
    ris_Nv = int(np.sqrt(M))
    ris_Nh = M // ris_Nv
    
    # Generate G matrix (BS-RIS channel)
    mu_o = cal_mu_o(fo, x_ris, y_ris, z_ris)
    G = cal_G_matrix(mu_o, Gr, lb, ris_Nh, ris_Nv, N, Mo, dv, db, dh)
    print(f"  G matrix shape: {G.shape}")
    
    # Generate Rk matrices (User-RIS channels for each user)
    Rk_list = []
    for k in range(K):
        # Random user position around center
        x_uc = x_user_center + np.random.uniform(-10, 10)
        y_uc = y_user_center + np.random.uniform(-10, 10)
        z_uc = z_user_center
        
        Rk = cal_Rk_matrix(Gt, ris_Nv, ris_Nh, Nk, dv, du, dh, lb, Mk,
                          fo, x_uc, y_uc, z_uc, x_ris, y_ris, z_ris)
        Rk_list.append(Rk)
        print(f"  R{k+1} matrix shape: {Rk.shape}")
    
    return G, Rk_list

# =============================================================================
# STEP 2: Run BCD Algorithm
# =============================================================================
def run_simulation():
    """
    Main function to run the complete simulation.
    """
    print("=" * 60)
    print("RIS-Aided Uplink Communication Simulation")
    print("=" * 60)
    
    # Step 1: Initialize channels
    G, Rk_list = initialize_channels()
    
    # Step 2: Run BCD algorithm (Algorithm 3)
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
    
    # Step 3: Display results
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)
    print(f"Final sum-rate: {rate_history[-1]:.4f} bps/Hz")
    print(f"Convergence after {len(rate_history)} iterations")
    print(f"Initial rate: {rate_history[0]:.4f} bps/Hz")
    print(f"Improvement: {rate_history[-1] - rate_history[0]:.4f} bps/Hz")
    
    return rate_history

# =============================================================================
# STEP 3: Multiple Realizations
# =============================================================================
def run_multiple_realizations(num_realizations=10):
    """
    Run the simulation multiple times with different channel realizations.
    """
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
    rate_history = run_simulation()
    
    # Option 2: Run multiple realizations (uncomment to use)
    # print("\n\nRunning multiple realizations...")
    # final_rates = run_multiple_realizations(num_realizations=10)

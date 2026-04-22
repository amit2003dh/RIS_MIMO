# RIS-Aided Uplink Communication System - Function Call Documentation

This document explains how functions call each other in the RIS-aided uplink communication system simulation.

## Table of Contents
1. [Overview](#overview)
2. [Function Call Hierarchy](#function-call-hierarchy)
3. [Detailed Call Flow](#detailed-call-flow)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Function Descriptions](#function-descriptions)

---

## Overview

The simulation implements **Algorithm 3 (Block Coordinate Descent - BCD)** from the research paper. The main function `run_BCD` orchestrates the iterative optimization of three variables:
- **w** (active beamformers) - updated by `step_B`
- **Phi** (RIS phase shifts) - updated by `step_C`
- **S** (beam selection matrix) - updated by `step_D_GS`

The algorithm uses the **WMMSE (Weighted Minimum Mean Square Error)** framework to transform the non-convex sum-rate maximization problem into a series of convex subproblems.

---

## Function Call Hierarchy

```
run_single_simulation()
├── initialize_channels()
│   ├── cal_mu_o()
│   │   └── cal_dk()
│   ├── cal_G_matrix()
│   │   ├── cal_stering_vec_array_response()
│   │   └── cal_stering_vec_u()
│   └── cal_Rk_matrix() [called K times]
│       ├── cal_mu_k()
│       │   └── cal_dk()
│       ├── cal_stering_vec_array_response()
│       └── cal_stering_vec_v()
│
└── run_BCD()
    ├── calc_Q()
    │   └── dft() [from scipy]
    ├── calc_Ue()
    ├── calc_We()
    ├── step_B()
    │   ├── dft() [from scipy]
    │   ├── _bisect()
    │   │   └── _pw()
    │   └── np.linalg.solve() / np.linalg.pinv()
    ├── step_C()
    │   ├── dft() [from scipy]
    │   └── np.linalg.solve() / np.linalg.pinv()
    ├── step_D_GS()
    │   └── calc_rate()
    │       ├── calc_Q()
    │       │   └── dft() [from scipy]
    │       └── dft() [from scipy]
    └── calc_rate()
        ├── calc_Q()
        │   └── dft() [from scipy]
        └── dft() [from scipy]
```

---

## Detailed Call Flow

### Phase 1: Channel Initialization

**Function:** `initialize_channels()`

**Purpose:** Generate the channel matrices G (BS-RIS) and Rk_list (User-RIS for each user).

**Call Sequence:**
```
initialize_channels()
├── cal_mu_o(fo, x_ris, y_ris, z_ris)
│   └── cal_dk()  # Calculate distance
│
├── cal_G_matrix(mu_o, Gr, lb, ris_Nh, ris_Nv, N, Mo, dv, db, dh)
│   └── [Loop Mo times]
│       ├── cal_stering_vec_array_response(theta_aod, phi_aod, Nv, Nh, dv, dh, lb)
│       └── cal_stering_vec_u(theta_aoa, N_rx, lambda_val, db)
│
└── [Loop K times for each user]
    ├── cal_mu_k(fo, x_uc, y_uc, z_uc, x_ris, y_ris, z_ris)
    │   └── cal_dk()  # Calculate distance
    │
    └── cal_Rk_matrix(Gt, ris_Nv, ris_Nh, Nk, dv, du, dh, lb, Mk, ...)
        └── [Loop Mk times]
            ├── cal_stering_vec_array_response(th_aoa, ph_aoa, Nv, Nh, dv, dh, lb)
            └── cal_stering_vec_v(th_aod, Nk, lambda_val, du)
```

**Output:** 
- `G`: BS-RIS channel matrix (N × M)
- `Rk_list`: List of K User-RIS channel matrices (each M × Nk)

---

### Phase 2: BCD Algorithm Main Loop

**Function:** `run_BCD(G, Rk_list, sigma_sq, L_val, N_val, M_val, K_val, Nk_val, ...)`

**Purpose:** Iteratively optimize beamformers, RIS phases, and beam selection.

**Initialization Phase (runs once at start):**
```
run_BCD()
├── [Initialize wk_list]
│   └── Generate random beamformers scaled to Pmax
│
├── [Initialize theta and Phi]
│   ├── theta = random uniform [0, 2π]
│   └── Phi = diag(exp(1j * theta))
│
├── [Initialize S]
│   ├── Select L random beams from N
│   └── S = selection matrix (L × N)
│
└── [Bootstrap WMMSE variables]
    ├── calc_Q(G, Phi, Rk_list, wk_list)
    │   └── U = dft(N, scale='sqrtn')  # Unitary DFT
    │   └── Q[:, k] = U^H @ Hk @ wk[k]
    │
    ├── calc_Ue(S, Q, sigma_sq)
    │   └── Ue = (S @ Q @ Q^H @ S^H + σ²I)^(-1) @ (S @ Q)
    │
    └── calc_We(Ue, S, Q)
        └── We = (I - Ue^H @ S @ Q)^(-1)
```

**Iterative Phase (repeats until convergence):**
```
For each iteration j = 1 to max_iter:
├── [Refresh Q, Ue, We with current variables]
│   ├── calc_Q(G, Phi, Rk_list, wk_list)
│   ├── calc_Ue(S, Q, sigma_sq)
│   └── calc_We(Ue, S, Q)
│
├── [Update beamformers w]
│   └── step_B(G, Phi, Rk_list, We, Ue, S)
│       ├── U = dft(N, scale='sqrtn')
│       ├── T = U^H @ S^H @ Ue
│       ├── Hk = [G @ Phi @ Rk_list[k] for k in range(K)]
│       └── [For each user k]
│           ├── Bk = Hk[k]^H @ T
│           ├── Ak = Bk @ We @ Bk^H
│           ├── ak = Bk @ We[:, k]
│           ├── lam = _bisect(Ak, ak, Pmax)
│           │   └── _pw(Bk, ck, lam)  # Power calculation
│           └── wk = (Ak + lam*I)^(-1) @ ak
│
├── [Update RIS phases Phi]
│   └── step_C(G, Rk_list, wk_list, We, Ue, S, theta_init)
│       ├── U = dft(N, scale='sqrtn')
│       ├── T = U^H @ S^H @ Ue
│       ├── D = [Rk_list[k] @ wk_list[k] for k in range(K)]
│       ├── TW = T @ We
│       ├── b = [G[:, m]^H @ TW @ D[m, :]^H for m in range(M)]
│       ├── C1 = G^H @ TW @ T^H @ G
│       ├── C2 = D @ D^H
│       ├── C = C1 * C2^T
│       └── [Iterative element-wise update]
│           └── theta[m] = -angle(b[m]^H - C[:, m] @ pe[m])
│
├── [Update beam selection S]
│   └── step_D_GS(G, Phi, Rk_list, wk_list, sigma_sq, L, N, S_init)
│       └── [Greedy search]
│           └── [For each beam l]
│               └── [For each candidate beam x]
│                   ├── S_try = S with beam l replaced by x
│                   └── r = calc_rate(G, Phi, S_try, Rk_list, wk_list, sigma_sq)
│                       ├── calc_Q(G, Phi, Rk_list, wk_list)
│                       └── U = dft(N, scale='sqrtn')
│
└── [Calculate current rate]
    └── calc_rate(G, Phi, S, Rk_list, wk_list, sigma_sq)
        ├── calc_Q(G, Phi, Rk_list, wk_list)
        └── U = dft(N, scale='sqrtn')
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     GLOBAL PARAMETERS                             │
│  N, M, L, K, Nk, Mk, Mo, Gt, Gr, fo, lb, Pmax, sigSQR, positions │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  initialize_channels()                            │
│  Input: Global parameters                                        │
│  Output: G (N×M), Rk_list [K matrices of M×Nk]                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       run_BCD()                                  │
│  Input: G, Rk_list, sigma_sq, L, N, M, K, Nk                    │
│  Output: rate_history (list of sum-rates per iteration)         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   calc_Q()      │  │   calc_Ue()     │  │   calc_We()     │
│  Input: G, Phi, │  │  Input: S, Q,   │  │  Input: Ue, S, Q │
│  Rk_list, wk    │  │  sigma_sq       │  │                 │
│  Output: Q (N×K)│  │  Output: Ue     │  │  Output: We     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    step_B()    │  │    step_C()     │  │  step_D_GS()    │
│  Input: G, Phi, │  │  Input: G, Rk,  │  │  Input: G, Phi,│
│  Rk, We, Ue, S │  │  wk, We, Ue, S  │  │  Rk, wk, sigma  │
│  Output: wk_new│  │  Output: theta, │  │  Output: S_new  │
│                │  │  Phi_new       │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      calc_rate()                                  │
│  Input: G, Phi, S, Rk_list, wk_list, sigma_sq                    │
│  Output: sum_rate (bps/Hz)                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Function Descriptions

### Helper Functions (Lowest Level)

#### Steering Vectors
- **`cal_stering_vec_v(th, Nk, lb, du)`**: Calculates steering vector for user ULA
- **`cal_stering_vec_u(th, N, lb, db)`**: Calculates steering vector for BS ULA
- **`cal_stering_vec_array_response(th, phi, Nv, Nh, dv, dh, lb)`**: Calculates steering vector for RIS UPA

#### Path Loss
- **`cal_dk(...)`**: Calculates distance between user and RIS
- **`cal_mu_k(...)`**: Calculates path loss coefficient for user-RIS channel
- **`cal_mu_o(...)`**: Calculates path loss coefficient for BS-RIS channel

#### Channel Generation
- **`cal_Rk_matrix(...)`**: Generates user-RIS channel matrix (Paper Eq. 4)
  - Calls: `cal_mu_k`, `cal_stering_vec_array_response`, `cal_stering_vec_v`
- **`cal_G_matrix(...)`**: Generates BS-RIS channel matrix (Paper Eq. 5)
  - Calls: `cal_stering_vec_array_response`, `cal_stering_vec_u`

#### RIS Phase
- **`create_phi_matrix(theta_vector)`**: Creates diagonal RIS phase shift matrix

### WMMSE Helper Functions

#### Signal Processing
- **`calc_Q(G, Phi, Rk_list, wk_list)`**: Calculates beam-domain signal matrix (Paper Eq. 10)
  - Input: G, Phi, Rk_list, wk_list
  - Output: Q (N × K)
  - Calls: `dft()` from scipy
  - Formula: Qk = U^H @ Hk @ wk

- **`calc_Ue(S, Q, sigma_sq)`**: Calculates MMSE receive combiner (Paper Step 4)
  - Input: S, Q, sigma_sq
  - Output: Ue (L × K)
  - Formula: Ue = (S @ Q @ Q^H @ S^H + σ²I)^(-1) @ (S @ Q)

- **`calc_We(Ue, S, Q)`**: Calculates WMMSE weight matrix (Paper Step 3)
  - Input: Ue, S, Q
  - Output: We (K × K)
  - Formula: We = (I - Ue^H @ S @ Q)^(-1)

- **`calc_rate(G, Phi, S, Rk_list, wk_list, sigma_sq)`**: Calculates uplink sum-rate (Paper Eq. 3)
  - Input: G, Phi, S, Rk_list, wk_list, sigma_sq
  - Output: sum_rate (bps/Hz)
  - Calls: `calc_Q`, `dft()`
  - Formula: R = log det(I_L + (1/σ²) @ S @ Q @ Q^H @ S^H)

### Bisection Helpers (for power constraint)

- **`_pw(Bk, ck, lam)`**: Calculates power ||w||² for given Lagrangian multiplier
  - Formula: w = (Bk + lam*I)^(-1) @ ck, then power = ||w||²

- **`_bisect(Bk, ck, P, tol, max_it)`**: Finds Lagrangian multiplier via bisection
  - Calls: `_pw` iteratively
  - Ensures ||w||² ≤ Pmax

### Update Steps (Core BCD Algorithm)

#### Step B: Beamformer Update
- **`step_B(G, Phi, Rk_list, We, Ue, S)`**: Updates active beamformers (Paper Eq. 13-15)
  - Input: G, Phi, Rk_list, We, Ue, S
  - Output: wk_new (list of K beamformers)
  - Calls: `dft()`, `_bisect`, `_pw`, `np.linalg.solve`
  - For each user k:
    1. Compute Bk = Hk^H @ U^H @ S^H @ Ue
    2. Compute Ak = Bk @ We @ Bk^H
    3. Compute ak = Bk @ We[:, k]
    4. Find lam via bisection such that ||wk||² ≤ Pmax
    5. Compute wk = (Ak + lam*I)^(-1) @ ak

#### Step C: RIS Phase Update
- **`step_C(G, Rk_list, wk_list, We, Ue, S, theta_init)`**: Updates RIS phases (Algorithm 1)
  - Input: G, Rk_list, wk_list, We, Ue, S, theta_init
  - Output: theta (new phases), Phi (new phase matrix)
  - Calls: `dft()`
  - Uses Element-Wise Block Coordinate Descent (EWBCD)
  - Iteratively updates each RIS element phase to minimize objective

#### Step D: Beam Selection Update
- **`step_D_GS(G, Phi, Rk_list, wk_list, sigma_sq, L, N, S_init)`**: Updates beam selection (Algorithm 2)
  - Input: G, Phi, Rk_list, wk_list, sigma_sq, L, N, S_init
  - Output: S_new (new beam selection matrix)
  - Calls: `calc_rate`
  - Uses Greedy Search (GS) algorithm
  - For each beam position, tries all candidates and keeps best

### Main Algorithm

#### run_BCD
- **`run_BCD(G, Rk_list, sigma_sq, L_val, N_val, M_val, K_val, Nk_val, ...)`**: Main BCD algorithm (Algorithm 3)
  - Input: G, Rk_list, sigma_sq, L, N, M, K, Nk, max_iter, tol, verbose
  - Output: rate_history (list of sum-rates per iteration)
  - Calls: `calc_Q`, `calc_Ue`, `calc_We`, `step_B`, `step_C`, `step_D_GS`, `calc_rate`
  - Algorithm:
    1. Initialize wk_list, theta, Phi, S
    2. Bootstrap Ue and We
    3. For each iteration:
       - Refresh Q, Ue, We
       - Update wk via step_B
       - Update Phi via step_C
       - Update S via step_D_GS
       - Calculate rate
       - Check convergence
    4. Return rate history

### Top-Level Functions

#### initialize_channels
- **`initialize_channels()`**: Generates all channel matrices
  - Input: None (uses global parameters)
  - Output: G, Rk_list
  - Calls: `cal_mu_o`, `cal_G_matrix`, `cal_mu_k`, `cal_Rk_matrix`

#### run_single_simulation
- **`run_single_simulation()`**: Runs one complete simulation
  - Input: None (uses global parameters)
  - Output: rate_history
  - Calls: `initialize_channels`, `run_BCD`

#### run_multiple_realizations
- **`run_multiple_realizations(num_realizations)`**: Runs multiple simulations
  - Input: num_realizations
  - Output: final_rates (list of final rates)
  - Calls: `initialize_channels`, `run_BCD` (multiple times)
  - Computes statistics: mean, std, min, max

---

## Key Observations

1. **Initialization Phase**: Channels are generated once per simulation using random parameters (AoA, AoD, path coefficients).

2. **BCD Loop**: The main algorithm iteratively updates three variables in a fixed order:
   - First: Update beamformers (w)
   - Second: Update RIS phases (Phi)
   - Third: Update beam selection (S)

3. **WMMSE Framework**: Before each update, the algorithm refreshes the WMMSE variables (Q, Ue, We) to ensure consistency.

4. **Power Constraint**: The beamformer update uses bisection to ensure each user's power does not exceed Pmax (per-user constraint).

5. **Convergence**: The algorithm checks convergence by monitoring the change in sum-rate over iterations.

6. **Unitary DFT**: The corrected implementation uses unitary DFT (`scale='sqrtn'`) for consistency with the paper.

---

## Running the Simulation

### Single Simulation
```python
from run_simulation_template import run_single_simulation

rate_history = run_single_simulation()
print(f"Final rate: {rate_history[-1]:.4f} bps/Hz")
```

### Multiple Realizations
```python
from run_simulation_template import run_multiple_realizations

final_rates = run_multiple_realizations(num_realizations=10)
print(f"Mean rate: {np.mean(final_rates):.4f} bps/Hz")
```

### Custom Parameters
Modify the global parameters at the top of `run_simulation_template.py`:
```python
K = 4               # Change to 4 users
Pmax_dbm = 23       # Increase power to 23 dBm
L = 32              # Select 32 beams instead of 16
```

---

## References

- Paper Eq. (3): Sum-rate formula
- Paper Eq. (4): User-RIS channel model
- Paper Eq. (5): BS-RIS channel model
- Paper Eq. (10): Beam-domain signal matrix
- Paper Eq. (13-15): Beamformer update
- Algorithm 1: RIS phase optimization (EWBCD)
- Algorithm 2: Beam selection (Greedy Search)
- Algorithm 3: Main BCD algorithm

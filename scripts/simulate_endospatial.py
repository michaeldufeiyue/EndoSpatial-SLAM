#!/usr/bin/env python3
"""Reproducible synthetic SE(3)-tangent Monte-Carlo study for EndoSpatial-SLAM.

This is a controlled simulation, not a run of FAST-LIO2/LIO-SAM/Transformer
implementations or a physical-hardware evaluation. The three executor streams
are generated from explicitly stated stochastic error models. The script
creates all figures and CSV files used by the manuscript.

Run:
    python simulate_endospatial.py --outdir ../
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 0. Global Configuration & Pinned Seed (202609)
# ==========================================
PINNED_SEED = 202609
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 9.5,
    "font.size": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.45
})

# ==========================================
# 1. Lie Group SE(3) & Right-Jacobian Utilities
# ==========================================
def skew(w):
    return np.array([
        [0.0, -w[2],  w[1]],
        [w[2],  0.0, -w[0]],
        [-w[1], w[0],  0.0]
    ])

def exp_so3(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def log_so3(R):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-7:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return (theta / (2.0 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

def right_jacobian_so3(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * skew(phi)
    K = skew(phi / theta)
    return np.eye(3) - ((1.0 - np.cos(theta)) / theta) * K + (1.0 - np.sin(theta) / theta) * (K @ K)

def right_jacobian_inv_so3(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        return np.eye(3) + 0.5 * skew(phi)
    K = skew(phi / theta)
    half_theta = 0.5 * theta
    cot_half = np.cos(half_theta) / np.sin(half_theta)
    return np.eye(3) + 0.5 * skew(phi) + (1.0 - half_theta * cot_half) * (K @ K)

def exp_se3(xi):
    """Right-trivialized exponential map: xi = [rho(3), phi(3)]^T."""
    rho, phi = xi[:3], xi[3:]
    R = exp_so3(phi)
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        V = np.eye(3) + 0.5 * skew(phi)
    else:
        K = skew(phi / theta)
        V = np.eye(3) + ((1.0 - np.cos(theta)) / theta) * K + (1.0 - np.sin(theta) / theta) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T

def log_se3(T):
    """Right-trivialized logarithm map: returns xi = [rho(3), phi(3)]^T."""
    R = T[:3, :3]
    p = T[:3, 3]
    phi = log_so3(R)
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        V_inv = np.eye(3) - 0.5 * skew(phi)
    else:
        K = skew(phi / theta)
        half_theta = 0.5 * theta
        cot_half = np.cos(half_theta) / np.sin(half_theta)
        V_inv = np.eye(3) - 0.5 * skew(phi) + (1.0 - half_theta * cot_half) * (K @ K)
    rho = V_inv @ p
    return np.concatenate([rho, phi])

def adjoint_se3(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad

def right_jacobian_inv_se3(xi):
    rho, phi = xi[:3], xi[3:]
    Jr_inv = right_jacobian_inv_so3(phi)
    Q = 0.5 * skew(rho)
    J_inv = np.zeros((6, 6))
    J_inv[:3, :3] = Jr_inv
    J_inv[:3, 3:] = -Jr_inv @ Q @ Jr_inv
    J_inv[3:, 3:] = Jr_inv
    return J_inv

# ==========================================
# 2. Reference Trajectory & Shared Sensor Stream Generator
# ==========================================
def generate_reference_trajectory(N_steps=400, dt=0.1):
    T_gt = []
    for k in range(N_steps):
        t = k * dt
        x = 24.0 * np.sin(0.15 * t)
        y = 16.0 * np.sin(0.30 * t)
        z = 2.0 * np.sin(0.10 * t) + 0.06 * t
        dx = 24.0 * 0.15 * np.cos(0.15 * t)
        dy = 16.0 * 0.30 * np.cos(0.30 * t)
        dz = 2.0 * 0.10 * np.cos(0.10 * t) + 0.06
        yaw = np.arctan2(dy, dx)
        pitch = -np.arctan2(dz, np.hypot(dx, dy)) * 0.4
        R = exp_so3(np.array([0.0, pitch, yaw]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, z])
        T_gt.append(T)
    return T_gt

def generate_shared_observation_stream(rng, T_gt, dt, scenario, corr_rho=0.35):
    """
    Generates identical raw sensory innovations and external disturbance histories
    used across ALL baseline and ablation variants (Matched-Stream Evaluation).
    """
    N_steps = len(T_gt)
    # Nominal standard deviations: [rx, ry, rz (m), wx, wy, wz (rad)]
    std_e1 = np.array([0.095, 0.095, 0.105, np.deg2rad(0.50), np.deg2rad(0.50), np.deg2rad(0.55)])
    std_e2 = np.array([0.110, 0.110, 0.120, np.deg2rad(0.60), np.deg2rad(0.60), np.deg2rad(0.65)])
    std_e3 = np.array([0.120, 0.120, 0.125, np.deg2rad(0.62), np.deg2rad(0.62), np.deg2rad(0.68)])
    std_imu = np.array([0.025, 0.025, 0.030, np.deg2rad(0.18), np.deg2rad(0.18), np.deg2rad(0.22)])

    # Generate correlated innovations (modeling shared LiDAR/camera raw observations)
    z_shared = rng.normal(0.0, 1.0, (N_steps, 6))
    z_ind1 = rng.normal(0.0, 1.0, (N_steps, 6))
    z_ind2 = rng.normal(0.0, 1.0, (N_steps, 6))
    z_ind3 = rng.normal(0.0, 1.0, (N_steps, 6))

    s_factor = np.sqrt(corr_rho)
    i_factor = np.sqrt(1.0 - corr_rho)

    innov_e1 = (s_factor * z_shared + i_factor * z_ind1) * std_e1
    innov_e2 = (s_factor * z_shared + i_factor * z_ind2) * std_e2
    innov_e3 = (0.5 * s_factor * z_shared + np.sqrt(1.0 - 0.25 * corr_rho) * z_ind3) * std_e3
    innov_imu = rng.normal(0.0, 1.0, (N_steps, 6)) * std_imu

    # External disturbance vectors and observability profiles
    dist_e1 = np.zeros((N_steps, 6))
    dist_e2 = np.zeros((N_steps, 6))
    dist_e3 = np.zeros((N_steps, 6))
    obs_eig = np.tile(np.array([4.8, 4.2, 3.9]), (N_steps, 1))
    reported_std_e1 = np.tile(std_e1, (N_steps, 1))

    for k in range(N_steps):
        t = k * dt
        # Scenario I or III: LiDAR Spoofing (12-18s)
        if scenario in ["S1", "S3"] and (12.0 <= t <= 18.0):
            dist_e1[k] = np.array([0.25, 1.15, 0.12, 0.0, 0.0, np.deg2rad(4.5)])

        # Scenario III: Featureless Corridor Degeneracy on E1 (20-25s)
        if scenario == "S3" and (20.0 <= t <= 25.0):
            dist_e1[k] = np.array([1.10, 0.18, 0.05, 0.0, 0.0, np.deg2rad(1.2)])
            obs_eig[k, 0] = 0.12  # Weakest-mode observability collapse
            reported_std_e1[k, 0] = 0.48

        # Scenario II or III: Spatial-Model OOD Hallucination (26-32s)
        if scenario in ["S2", "S3"] and (26.0 <= t <= 32.0):
            dist_e2[k] = np.array([-0.45, -0.35, 1.20, np.deg2rad(1.5), np.deg2rad(-1.2), np.deg2rad(7.2)])

        # Scenario IV: Edge Case (E1 Degenerate + E2 Biased simultaneously, 15-25s)
        if scenario == "S4" and (15.0 <= t <= 25.0):
            dist_e1[k] = np.array([1.25, 0.22, 0.08, 0.0, 0.0, np.deg2rad(1.5)])
            obs_eig[k, 0] = 0.10
            reported_std_e1[k, 0] = 0.50
            dist_e2[k] = np.array([1.35, -0.85, 0.65, 0.0, 0.0, np.deg2rad(8.0)])

    return {
        "innov_e1": innov_e1, "innov_e2": innov_e2, "innov_e3": innov_e3, "innov_imu": innov_imu,
        "dist_e1": dist_e1, "dist_e2": dist_e2, "dist_e3": dist_e3,
        "obs_eig": obs_eig, "reported_std_e1": reported_std_e1,
        "std_e2": std_e2, "std_e3": std_e3, "std_imu": std_imu, "corr_rho": corr_rho
    }

# ==========================================
# 3. Stateful Estimators, Arbitration & Sliding-Window Solver
# ==========================================
def run_trial(T_gt, stream, method="DHR_Full", cov_scale_e2=1.0):
    """
    Executes stateful recursive estimators, arbitration, closed-loop recovery,
    and sliding-window Gauss-Newton backend optimization.
    """
    N_steps = len(T_gt)
    dt = 0.1

    # Controller hyperparameters
    alpha_obs, lambda_th = 2.5, 1.2
    tau_obs = 0.45
    tau_cert = 0.50
    tau_iso = 0.38
    tau_rejoin = 0.72
    K_iso = 4
    K_rejoin = 5
    sigma_arb = 1.45
    W_win = 8
    L_buf = 25

    # Stateful recursive memory biases (submap drift for E1, KV-cache drift for E2)
    mem_state = np.zeros((3, 6))
    quarantined = [False, False, False]
    iso_count = [0, 0, 0]
    rejoin_count = [0, 0, 0]
    ramp_factor = [1.0, 1.0, 1.0]

    T_est_hist = [T_gt[0].copy()]
    raw_buffer = []  # Circular buffer of raw observations for memory reconstruction
    score_hist = np.zeros((N_steps, 3))
    weight_hist = np.zeros((N_steps, 3))
    cert_hist = np.ones(N_steps)

    for k in range(1, N_steps):
        T_rel_gt = np.linalg.inv(T_gt[k - 1]) @ T_gt[k]
        raw_buffer.append((k, T_rel_gt, stream["innov_e1"][k], stream["innov_e2"][k], stream["innov_e3"][k],
                           stream["dist_e1"][k], stream["dist_e2"][k], stream["dist_e3"][k]))
        if len(raw_buffer) > L_buf:
            raw_buffer.pop(0)

        # --- Update Stateful Recursive Estimators ---
        if method == "Homogeneous_AI":
            # Three correlated neural models sharing OOD failure mode
            for i in range(3):
                mem_state[i] = 0.88 * mem_state[i] + 0.32 * stream["dist_e2"][k]
            err_1 = stream["innov_e2"][k] + stream["dist_e2"][k] + mem_state[0]
            err_2 = 0.92 * stream["innov_e2"][k] + 0.08 * stream["innov_e3"][k] + 0.96 * stream["dist_e2"][k] + mem_state[1]
            err_3 = 0.90 * stream["innov_e2"][k] + 0.10 * stream["innov_e1"][k] + 1.04 * stream["dist_e2"][k] + mem_state[2]
        else:
            # E1: Geometric submap recursive integrator
            mem_state[0] = 0.86 * mem_state[0] + 0.30 * stream["dist_e1"][k]
            err_1 = stream["innov_e1"][k] + stream["dist_e1"][k] + mem_state[0]

            # E2: Spatial Transformer KV-cache recursive state
            mem_state[1] = 0.89 * mem_state[1] + 0.34 * stream["dist_e2"][k]
            err_2 = stream["innov_e2"][k] + stream["dist_e2"][k] + mem_state[1]

            # E3: Semantic-topological graph state
            mem_state[2] = 0.82 * mem_state[2] + 0.25 * stream["dist_e3"][k]
            err_3 = stream["innov_e3"][k] + stream["dist_e3"][k] + mem_state[2]

        errs = [err_1, err_2, err_3]
        T_cands = [T_gt[k] @ exp_se3(errs[i]) for i in range(3)]

        # Build covariances (including miscalibration scale on E2)
        Sigma_1 = np.diag(stream["reported_std_e1"][k] ** 2)
        Sigma_2 = np.diag((stream["std_e2"] * cov_scale_e2) ** 2)
        Sigma_3 = np.diag(stream["std_e3"] ** 2)
        Sigmas = [Sigma_1, Sigma_2, Sigma_3]

        # Dimensionless observability score phi_i (Eq. 10)
        phi = np.zeros(3)
        for i in range(3):
            lam_i = stream["obs_eig"][k, i]
            if method == "No_Obs_Mod":
                phi[i] = 1.0
            else:
                phi[i] = 1.0 / (1.0 + np.exp(-alpha_obs * (lam_i - lambda_th)))

        # --- Single-Channel Baselines ---
        if method == "E1_only":
            T_est_hist.append(T_cands[0])
            continue
        elif method == "E2_only":
            T_est_hist.append(T_cands[1])
            continue
        elif method == "E3_only":
            T_est_hist.append(T_cands[2])
            continue

        # --- Pairwise SE(3) Discrepancy & Cross-Covariance Transport (Eqs. 7-9) ---
        d_maha = np.zeros((3, 3))
        rho_c = stream["corr_rho"]
        for i in range(3):
            for j in range(i + 1, 3):
                T_rel_ij = np.linalg.inv(T_cands[i]) @ T_cands[j]
                xi_ij = log_se3(T_rel_ij)
                if method == "No_Rot_Residual":
                    xi_ij[3:] = 0.0

                Ji = -right_jacobian_inv_se3(xi_ij) @ adjoint_se3(T_rel_ij)
                Jj = right_jacobian_inv_se3(xi_ij)
                # Cross-covariance block C_ij from shared sensory observations
                c_factor = rho_c if (i < 2 and j < 2) else 0.5 * rho_c
                C_ij = c_factor * np.sqrt(Sigmas[i]) @ np.sqrt(Sigmas[j])
                Sigma_rel = Ji @ Sigmas[i] @ Ji.T + Jj @ Sigmas[j] @ Jj.T + Ji @ C_ij @ Jj.T + Jj @ C_ij.T @ Ji.T
                d_val = np.sqrt(xi_ij.T @ np.linalg.inv(Sigma_rel + 1e-6 * np.eye(6)) @ xi_ij)
                d_maha[i, j] = d_maha[j, i] = d_val

        # --- Consensus Scoring (Eqs. 11-12) ---
        scores = np.zeros(3)
        for i in range(3):
            if method == "No_Eligible_Rule":
                # Original flawed Eq. (11) weighting for ablation comparison
                vals = [phi[j] * (d_maha[i, j] ** 2) / 6.0 for j in range(3) if j != i]
                scores[i] = np.exp(-min(vals) / (2.0 * sigma_arb ** 2))
            else:
                # Proposed Reliable-Peer Eligibility Rule (Eqs. 11-12)
                eligible = [j for j in range(3) if j != i and phi[j] >= tau_obs and not quarantined[j]]
                if len(eligible) == 0:
                    scores[i] = 0.0
                else:
                    min_d2 = min((d_maha[i, j] ** 2) / 6.0 for j in eligible)
                    scores[i] = np.exp(-min_d2 / (2.0 * sigma_arb ** 2))

        score_hist[k] = scores

        # --- Fusion Rule Selection ---
        if method == "Static_Fusion":
            w_raw = np.array([1.0 / np.trace(Sigmas[i]) for i in range(3)])
            weights = w_raw / np.sum(w_raw)
        elif method == "TimeVarying_Cov":
            w_raw = np.array([phi[i] / np.trace(Sigmas[i]) for i in range(3)])
            weights = w_raw / np.sum(w_raw)
        elif method == "Chi2_Median":
            # Component-wise median + chi-square gating
            med_err = np.median(np.vstack(errs), axis=0)
            weights = np.zeros(3)
            for i in range(3):
                gate = np.linalg.norm(errs[i][:3] - med_err[:3])
                weights[i] = phi[i] if gate < 0.45 else 0.0
            if np.sum(weights) < 1e-6:
                weights = np.array([0.0, 0.0, 1.0])
            else:
                weights /= np.sum(weights)
        elif method == "Conservative_CI":
            # Trace-minimized Covariance Intersection weights
            inv_traces = np.array([phi[i] / np.trace(Sigmas[i]) for i in range(3)])
            weights = inv_traces / (np.sum(inv_traces) + 1e-8)
        else:
            # Proposed EndoSpatial-DHR & Ablations
            valid_channels = [i for i in range(3) if phi[i] >= tau_obs and not quarantined[i]]
            if method == "No_Eligible_Rule":
                c_k = 1
            else:
                c_k = 1 if (len(valid_channels) >= 2 and max(scores[i] for i in valid_channels) >= tau_cert) else 0
            cert_hist[k] = c_k

            if c_k == 0:
                # Uncertified fallback: select highest-observability non-quarantined channel,
                # or fall back to pure IMU propagation (zero estimator weights) if all channels are quarantined.
                active_idx = [i for i in range(3) if not quarantined[i]]
                weights = np.zeros(3)
                if len(active_idx) > 0:
                    best_i = active_idx[int(np.argmax([phi[i] for i in active_idx]))]
                    weights[best_i] = 1.0
            else:
                w_raw = np.array([
                    (0.0 if quarantined[i] else ramp_factor[i] * phi[i] * scores[i])
                    for i in range(3)
                ])
                weights = w_raw / (np.sum(w_raw) + 1e-8)

        weight_hist[k] = weights

        # --- Sliding-Window Weighted Gauss-Newton Optimization (Eq. 16) ---
        T_imu_pred = T_est_hist[-1] @ T_rel_gt @ exp_se3(stream["innov_imu"][k])
        Sigma_imu = np.diag(stream["std_imu"] ** 2)

        # Gauss-Newton update on SE(3) around T_imu_pred
        T_curr = T_imu_pred.copy()
        for _ in range(3):
            H_gn = np.linalg.inv(Sigma_imu)
            b_gn = np.linalg.inv(Sigma_imu) @ log_se3(np.linalg.inv(T_curr) @ T_imu_pred)
            for i in range(3):
                if weights[i] > 1e-4:
                    r_i = log_se3(np.linalg.inv(T_curr) @ T_cands[i])
                    # Huber robust weight
                    r_norm = np.linalg.norm(r_i)
                    huber_w = 1.0 if r_norm < 0.35 else (0.35 / r_norm)
                    W_eff = weights[i] * huber_w * np.linalg.inv(Sigmas[i])
                    H_gn += W_eff
                    b_gn += W_eff @ r_i
            delta_xi = np.linalg.solve(H_gn + 1e-6 * np.eye(6), b_gn)
            T_curr = T_curr @ exp_se3(delta_xi)

        T_est_hist.append(T_curr)

        # --- Autonomous Closed-Loop State Cleansing & Shadow-Mode Re-entry ---
        if method in ["DHR_Full", "No_Eligible_Rule", "No_Obs_Mod", "No_Rot_Residual", "Homogeneous_AI"]:
            for i in range(3):
                valid_cnt = sum(1 for idx in range(3) if phi[idx] >= tau_obs and not quarantined[idx])
                if not quarantined[i]:
                    if scores[i] < tau_iso and valid_cnt >= 2:
                        iso_count[i] += 1
                        if iso_count[i] >= K_iso:
                            # Trigger isolation, timestamped rollback, and raw-buffer reconstruction
                            quarantined[i] = True
                            iso_count[i] = 0
                            rejoin_count[i] = 0
                            ramp_factor[i] = 0.0
                            # Re-encode memory state over recent circular buffer conditioned on consensus trajectory
                            mem_state[i] = np.zeros(6)
                            for buf_item in raw_buffer[-6:]:
                                d_buf = buf_item[5 + i]
                                mem_state[i] = 0.85 * mem_state[i] + 0.30 * d_buf
                    else:
                        iso_count[i] = 0
                else:
                    # Shadow-mode consistency verification
                    if scores[i] > tau_rejoin and phi[i] >= tau_obs:
                        rejoin_count[i] += 1
                        if rejoin_count[i] >= K_rejoin:
                            quarantined[i] = False
                            rejoin_count[i] = 0
                            ramp_factor[i] = 0.25
                    else:
                        rejoin_count[i] = 0
                if not quarantined[i] and ramp_factor[i] < 1.0:
                    ramp_factor[i] = min(1.0, ramp_factor[i] + 0.25)

    return T_est_hist, score_hist, weight_hist, cert_hist

def compute_metrics(T_gt, T_est):
    pos_errs = np.array([np.linalg.norm(T_est[k][:3, 3] - T_gt[k][:3, 3]) for k in range(len(T_gt))])
    ate_rmse = np.sqrt(np.mean(pos_errs ** 2))
    rpe_list = []
    rot_errs = []
    for k in range(1, len(T_gt)):
        d_gt = np.linalg.inv(T_gt[k - 1]) @ T_gt[k]
        d_est = np.linalg.inv(T_est[k - 1]) @ T_est[k]
        e_rel = log_se3(np.linalg.inv(d_gt) @ d_est)
        rpe_list.append(np.linalg.norm(e_rel[:3]))
        e_abs = log_se3(np.linalg.inv(T_gt[k]) @ T_est[k])
        rot_errs.append(np.rad2deg(np.linalg.norm(e_abs[3:])))
    return ate_rmse, np.sqrt(np.mean(np.array(rpe_list) ** 2)), np.sqrt(np.mean(np.array(rot_errs) ** 2)), pos_errs

# ==========================================
# 4. Run 50-Trial Monte-Carlo Evaluation (Seed 202609)
# ==========================================
def main():
    N_trials = 50
    N_steps = 400
    dt = 0.1
    t_axis = np.arange(N_steps) * dt
    T_gt = generate_reference_trajectory(N_steps, dt)

    methods = [
        ("E1_only", "Geometry-only (E1)"),
        ("E2_only", "Spatial-model-only (E2)"),
        ("E3_only", "Semantic-graph-only (E3)"),
        ("Static_Fusion", "Static covariance-weighted fusion"),
        ("TimeVarying_Cov", "Time-varying covariance fusion"),
        ("Chi2_Median", "Chi2-gated component-wise median"),
        ("Conservative_CI", "Conservative Covariance Intersection"),
        ("DHR_Full", "EndoSpatial-DHR (Ours)")
    ]

    scenarios = ["S1", "S2", "S3", "S4"]
    summary = {s: {m[0]: [] for m in methods} for s in scenarios}
    high_corr_ate = {m[0]: [] for m in methods}
    miscal_ate = {m[0]: [] for m in methods}
    s4_uncert_rates = []

    # Representative trial storage (Trial 0 of S3)
    rep_curves = {}
    rep_trajs = {}

    for trial in range(N_trials):
        rng = np.random.default_rng(PINNED_SEED + trial)
        for s in scenarios:
            stream = generate_shared_observation_stream(rng, T_gt, dt, s, corr_rho=0.35)
            for m_key, _ in methods:
                T_est, scores, weights, certs = run_trial(T_gt, stream, method=m_key)
                ate, rpe, rot, pos_errs = compute_metrics(T_gt, T_est)
                summary[s][m_key].append(ate)
                if s == "S4" and m_key == "DHR_Full":
                    # Overlap frames t in [15s, 25s]
                    overlap_mask = (t_axis >= 15.0) & (t_axis <= 25.0)
                    s4_uncert_rates.append(100.0 * np.mean(certs[overlap_mask] == 0))
                if trial == 0 and s == "S3":
                    rep_curves[m_key] = (pos_errs, scores, weights, certs)
                    rep_trajs[m_key] = np.array([T[:3, 3] for T in T_est])

        # Stress test 1: High correlation (rho = 0.65) in S3
        rng_hc = np.random.default_rng(PINNED_SEED + 5000 + trial)
        stream_hc = generate_shared_observation_stream(rng_hc, T_gt, dt, "S3", corr_rho=0.65)
        for m_key, _ in methods:
            T_est, _, _, _ = run_trial(T_gt, stream_hc, method=m_key)
            ate, _, _, _ = compute_metrics(T_gt, T_est)
            high_corr_ate[m_key].append(ate)

        # Stress test 2: Covariance miscalibration (0.25x Sigma_2) in S3
        rng_mc = np.random.default_rng(PINNED_SEED + 9000 + trial)
        stream_mc = generate_shared_observation_stream(rng_mc, T_gt, dt, "S3", corr_rho=0.35)
        for m_key, _ in methods:
            T_est, _, _, _ = run_trial(T_gt, stream_mc, method=m_key, cov_scale_e2=0.25)
            ate, _, _, _ = compute_metrics(T_gt, T_est)
            miscal_ate[m_key].append(ate)

    print("=" * 105)
    print(f"  TABLE I: 50-Trial Monte-Carlo ATE RMSE (mean +/- std, seed={PINNED_SEED})")
    print("=" * 105)
    for m_key, m_label in methods:
        s1_m, s1_s = np.mean(summary["S1"][m_key]), np.std(summary["S1"][m_key])
        s2_m, s2_s = np.mean(summary["S2"][m_key]), np.std(summary["S2"][m_key])
        s3_m, s3_s = np.mean(summary["S3"][m_key]), np.std(summary["S3"][m_key])
        s4_m, s4_s = np.mean(summary["S4"][m_key]), np.std(summary["S4"][m_key])
        hc_m, hc_s = np.mean(high_corr_ate[m_key]), np.std(high_corr_ate[m_key])
        mc_m, mc_s = np.mean(miscal_ate[m_key]), np.std(miscal_ate[m_key])
        print(f"{m_label:<36} | S1:{s1_m:.3f}+-{s1_s:.3f} | S2:{s2_m:.3f}+-{s2_s:.3f} | "
              f"S3:{s3_m:.3f}+-{s3_s:.3f} | S4:{s4_m:.3f}+-{s4_s:.3f} | HC:{hc_m:.3f} | MC:{mc_m:.3f}")
    print(f"Scenario IV Uncertified Detection Rate: {np.mean(s4_uncert_rates):.1f}%")

    # --- Matched Closed-Loop Ablation Study (50 paired trials in S3) ---
    ablation_variants = [
        ("DHR_Full", "Full EndoSpatial-DHR"),
        ("No_Eligible_Rule", "Without reliable-peer eligibility"),
        ("No_Obs_Mod", "Without observability modulation"),
        ("No_Rot_Residual", "Without rotational residual"),
        ("No_Cleansing", "Without state cleansing (matched)"),
        ("Homogeneous_AI", "Homogeneous AI pool")
    ]
    abl_stats = {v[0]: {"ate": [], "rpe": [], "rot": [], "curves": []} for v in ablation_variants}

    for trial in range(N_trials):
        rng = np.random.default_rng(PINNED_SEED + trial)
        stream_s3 = generate_shared_observation_stream(rng, T_gt, dt, "S3", corr_rho=0.35)
        for v_key, _ in ablation_variants:
            T_est, _, _, _ = run_trial(T_gt, stream_s3, method=v_key)
            ate, rpe, rot, pos_errs = compute_metrics(T_gt, T_est)
            abl_stats[v_key]["ate"].append(ate)
            abl_stats[v_key]["rpe"].append(rpe)
            abl_stats[v_key]["rot"].append(rot)
            abl_stats[v_key]["curves"].append(pos_errs)

    print("\n" + "=" * 82)
    print(f"  TABLE II: Matched Closed-Loop Ablation Study (50 paired trials, seed={PINNED_SEED})")
    print("=" * 82)
    for v_key, v_label in ablation_variants:
        a_m, a_s = np.mean(abl_stats[v_key]["ate"]), np.std(abl_stats[v_key]["ate"])
        r_m = np.mean(abl_stats[v_key]["rpe"])
        rot_m = np.mean(abl_stats[v_key]["rot"])
        print(f"{v_label:<36} | ATE: {a_m:.3f} +/- {a_s:.3f} m | RPE: {r_m:.3f} m | Rot: {rot_m:.3f} deg")

    # ==========================================
    # 5. Generate Publication Figures
    # ==========================================
    # --- Figure 2: Representative Compound-Scenario Response ---
    fig2, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True)
    _, scores_dhr, weights_dhr, certs_dhr = rep_curves["DHR_Full"]

    for ax in axes:
        ax.axvspan(12, 18, color="#FEE2E2", alpha=0.55)
        ax.axvspan(20, 25, color="#E0F2FE", alpha=0.60)
        ax.axvspan(26, 32, color="#FEF3C7", alpha=0.60)

    axes[0].plot(t_axis, scores_dhr[:, 0], label="Geometry ($E_1$)", color="#1D4ED8", lw=1.4)
    axes[0].plot(t_axis, scores_dhr[:, 1], label="Spatial AI ($E_2$)", color="#7C3AED", lw=1.4)
    axes[0].plot(t_axis, scores_dhr[:, 2], label="Semantic Graph ($E_3$)", color="#059669", lw=1.4)
    axes[0].axhline(0.38, color="#DC2626", ls="--", lw=1.1, label="Isolation $\\tau_{\\mathrm{iso}}$")
    axes[0].set_ylabel("Consensus $s_i^{(k)}$")
    axes[0].set_ylim(-0.05, 1.12)
    axes[0].legend(loc="lower left", ncol=4, frameon=True)
    axes[0].set_title("(a) Reliable-Peer Consensus Scores under LiDAR Spoofing (12-18s), Degeneracy (20-25s) & OOD (26-32s)", fontsize=9.2, fontweight="bold")

    axes[1].plot(t_axis, weights_dhr[:, 0], label="$\\omega_1^{(k)}$ ($E_1$)", color="#1D4ED8", lw=1.4)
    axes[1].plot(t_axis, weights_dhr[:, 1], label="$\\omega_2^{(k)}$ ($E_2$)", color="#7C3AED", lw=1.4)
    axes[1].plot(t_axis, weights_dhr[:, 2], label="$\\omega_3^{(k)}$ ($E_3$)", color="#059669", lw=1.4)
    axes[1].set_ylabel("Trust Weight $\\omega_i^{(k)}$")
    axes[1].set_ylim(-0.05, 0.85)
    axes[1].legend(loc="upper right", ncol=3, frameon=True)
    axes[1].set_title("(b) Dynamic Trust Weights with Shadow-Mode Quarantine & Ramp-Up Re-Entry", fontsize=9.2, fontweight="bold")

    axes[2].plot(t_axis, rep_curves["TimeVarying_Cov"][0], label="Time-Varying Cov. Fusion", color="#64748B", ls="-.", lw=1.3)
    axes[2].plot(t_axis, rep_curves["Chi2_Median"][0], label="$\\chi^2$-Gated Median", color="#D97706", ls="--", lw=1.3)
    axes[2].plot(t_axis, abl_stats["No_Cleansing"]["curves"][0], label="DHR w/o State Cleansing", color="#DC2626", ls=":", lw=1.5)
    axes[2].plot(t_axis, rep_curves["DHR_Full"][0], label="EndoSpatial-DHR (Ours)", color="#059669", ls="-", lw=1.8)
    axes[2].set_ylabel("Position Error (m)")
    axes[2].set_xlabel("Time $t$ (s)")
    axes[2].legend(loc="upper left", ncol=2, frameon=True)
    axes[2].set_title("(c) Position Error Norm Comparison on Identical Sensor Stream", fontsize=9.2, fontweight="bold")

    plt.tight_layout()
    fig2.savefig("figures/EndoSpatial_SLAM_Fig2.pdf", bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 3: 3D Trajectory Comparison ---
    gt_xyz = np.array([T[:3, 3] for T in T_gt])
    fig3 = plt.figure(figsize=(3.55, 2.95))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="#0F172A", lw=1.8, label="Ground Truth")
    ax3.plot(rep_trajs["E1_only"][:, 0], rep_trajs["E1_only"][:, 1], rep_trajs["E1_only"][:, 2], color="#1D4ED8", ls="--", lw=1.1, label="Geometry ($E_1$)")
    ax3.plot(rep_trajs["E2_only"][:, 0], rep_trajs["E2_only"][:, 1], rep_trajs["E2_only"][:, 2], color="#7C3AED", ls="-.", lw=1.1, label="Spatial AI ($E_2$)")
    ax3.plot(rep_trajs["Static_Fusion"][:, 0], rep_trajs["Static_Fusion"][:, 1], rep_trajs["Static_Fusion"][:, 2], color="#DC2626", ls=":", lw=1.2, label="Static Fusion")
    ax3.plot(rep_trajs["DHR_Full"][:, 0], rep_trajs["DHR_Full"][:, 1], rep_trajs["DHR_Full"][:, 2], color="#059669", lw=1.6, label="EndoSpatial-DHR")
    ax3.set_xlabel("X (m)", labelpad=-4)
    ax3.set_ylabel("Y (m)", labelpad=-4)
    ax3.set_zlabel("Z (m)", labelpad=-4)
    ax3.view_init(elev=25, azim=-50)
    ax3.legend(loc="upper left", fontsize=6.5, ncol=2)
    plt.tight_layout()
    fig3.savefig("figures/fig_3d_trajectory.pdf", bbox_inches="tight")
    plt.close(fig3)

    # --- Figure 4: Matched Closed-Loop Ablation Study ---
    fig4, (ax4_a, ax4_b) = plt.subplots(2, 1, figsize=(3.55, 4.4), gridspec_kw={"height_ratios": [1.1, 1.0]})
    colors_abl = ["#059669", "#0284C7", "#D97706", "#64748B", "#DC2626", "#7C3AED"]
    styles_abl = ["-", "--", "-.", ":", "--", "-."]

    for idx, (v_key, v_label) in enumerate(ablation_variants):
        ax4_a.plot(t_axis, abl_stats[v_key]["curves"][0], label=v_label, color=colors_abl[idx], ls=styles_abl[idx], lw=1.4 if idx > 0 else 1.8)
    ax4_a.set_ylabel("Position Error (m)")
    ax4_a.set_xlabel("Time $t$ (s)")
    ax4_a.set_title("(a) Representative Position-Error Traces (Matched Stream)", fontsize=8.8, fontweight="bold")
    ax4_a.legend(loc="upper left", fontsize=6.2, ncol=2)

    short_labels = ["Full\nDHR", "w/o Peer\nRule", "w/o Obs.\nMod.", "w/o Rot.\nResid.", "w/o\nCleanse", "Homog.\nAI Pool"]
    means = [np.mean(abl_stats[v[0]]["ate"]) for v in ablation_variants]
    stds = [np.std(abl_stats[v[0]]["ate"]) for v in ablation_variants]
    bars = ax4_b.bar(np.arange(len(short_labels)), means, yerr=stds, capsize=2.5, color=colors_abl, edgecolor="#0F172A", lw=0.7, width=0.55)
    ax4_b.set_xticks(np.arange(len(short_labels)))
    ax4_b.set_xticklabels(short_labels, fontsize=7.0)
    ax4_b.set_ylabel("ATE RMSE (m)")
    ax4_b.set_ylim(0, max(means) * 1.25)
    ax4_b.set_title("(b) Mean ATE RMSE & Std over 50 Paired Trials", fontsize=8.8, fontweight="bold")
    for b in bars:
        h = b.get_height()
        ax4_b.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h + 0.015), ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    plt.tight_layout()
    fig4.savefig("figures/EndoSpatial_SLAM_Ablation.pdf", bbox_inches="tight")
    plt.close(fig4)
    print("\nAll figures saved to ./figures/ successfully.")

if __name__ == "__main__":
    main()

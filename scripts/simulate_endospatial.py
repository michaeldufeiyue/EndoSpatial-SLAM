#!/usr/bin/env python3
"""Reproducible synthetic SE(3)-tangent Monte-Carlo study for EndoSpatial-SLAM.

This is a controlled simulation, not a run of FAST-LIO2/LIO-SAM/Transformer
implementations or a physical-hardware evaluation. The three executor streams
are generated from explicitly stated stochastic error models. The script
creates all figures and CSV files used by the revised manuscript.

Run:
    python simulate_endospatial.py --outdir ../
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261030
N_TRIALS = 50
DT = 0.1
DURATION = 40.0
TIMES = np.arange(0.0, DURATION + 1e-9, DT)
N = len(TIMES)
K_ISO = 3
TAU_ISO = 0.25
TAU_REJOIN = 0.58
SIGMA_ARB = 1.5
ALPHA_OBS = 12.0
LAMBDA_TH = 0.25

# Tangent perturbation order: [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z].
# Position scales are metres; rotation scales are radians.
BASE_STD = np.array([
    [0.10, 0.10, 0.10, np.deg2rad(0.50), np.deg2rad(0.50), np.deg2rad(0.50)],
    [0.12, 0.12, 0.12, np.deg2rad(0.70), np.deg2rad(0.70), np.deg2rad(0.70)],
    [0.13, 0.13, 0.13, np.deg2rad(0.65), np.deg2rad(0.65), np.deg2rad(0.65)],
])
REF_STD = np.array([0.10, 0.10, 0.10, np.deg2rad(0.70), np.deg2rad(0.70), np.deg2rad(0.70)])
REF_COV_SQRT = np.diag(REF_STD)

SCENARIOS = {
    "I": {"lidar": True, "corridor": False, "patch": False},
    "II": {"lidar": False, "corridor": False, "patch": True},
    "III": {"lidar": True, "corridor": True, "patch": True},
}


def ground_truth() -> np.ndarray:
    """Synthetic smooth 3-D figure-eight reference trajectory."""
    u = 2.0 * np.pi * TIMES / DURATION
    xyz = np.column_stack((
        9.0 * np.sin(u),
        5.0 * np.sin(2.0 * u),
        0.8 * np.sin(0.5 * u) + 0.25 * np.cos(3.0 * u),
    ))
    return xyz


def scenario_masks(scenario: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = SCENARIOS[scenario]
    lidar = (TIMES >= 12.0) & (TIMES <= 18.0) if spec["lidar"] else np.zeros(N, bool)
    corridor = (TIMES >= 20.0) & (TIMES <= 25.0) if spec["corridor"] else np.zeros(N, bool)
    patch = (TIMES >= 26.0) & (TIMES <= 32.0) if spec["patch"] else np.zeros(N, bool)
    return lidar, corridor, patch


def synthesize_trial(scenario: str, rng: np.random.Generator,
                     clean_memory: bool = True) -> Dict[str, np.ndarray]:
    """Generate candidate estimates and diagonal local covariance models."""
    lidar, corridor, patch = scenario_masks(scenario)
    err = rng.normal(size=(N, 3, 6)) * BASE_STD[None, :, :]
    cov = np.zeros((N, 3, 6, 6), dtype=float)

    # E1: LiDAR spoofing causes a lateral false displacement. Without state
    # cleansing, a residual local-map bias remains after the attack interval.
    if np.any(lidar):
        err[lidar, 0, 1] += 1.10
        if not clean_memory:
            post = TIMES > 18.0
            err[post, 0, 1] += 0.34

    # E1: corridor degeneracy weakens the x-direction information and creates
    # a bounded random-walk drift. The covariance is inflated consistently.
    if np.any(corridor):
        idx = np.flatnonzero(corridor)
        rw = np.cumsum(rng.normal(0.0, 0.035, size=len(idx)))
        rw -= rw[0]
        # A slowly growing longitudinal drift is intentionally paired with
        # inflated covariance: the residual alone is not a reliable alarm.
        ramp = np.linspace(0.0, 1.15, len(idx))
        err[idx, 0, 0] += rw + ramp
        std = BASE_STD[0].copy()
        std[0] = 0.48
        cov[idx, 0] = np.diag(std ** 2)

    # A turn-related gyro bias episode provides an orientation-dominant case:
    # a translation-only voter cannot detect this inconsistency.
    if scenario == "III":
        turn = (TIMES >= 5.0) & (TIMES <= 9.0)
        err[turn, 0, 5] += np.deg2rad(8.0)

    # E2: an image/feature hallucination creates an upward and yaw error.
    if np.any(patch):
        err[patch, 1, 2] += 1.20
        err[patch, 1, 5] += np.deg2rad(7.0)
        if not clean_memory:
            post = TIMES > 32.0
            err[post, 1, 2] += 0.32
            err[post, 1, 5] += np.deg2rad(1.8)

    # Nominal covariance for each estimator; E1's covariance is inflated only
    # in the weak corridor direction. E2 deliberately remains overconfident
    # during the OOD patch interval to model miscalibrated uncertainty.
    for i in range(3):
        cov[:, i] = np.diag(BASE_STD[i] ** 2)
    if np.any(corridor):
        std = BASE_STD[0].copy()
        std[0] = 0.48
        cov[corridor, 0] = np.diag(std ** 2)

    # Add small independent temporal correlation to make curves smoother but
    # preserve reproducibility. This term is shared only within each executor.
    for i in range(3):
        corr = np.zeros((N, 6))
        for k in range(1, N):
            corr[k] = 0.72 * corr[k - 1] + rng.normal(size=6) * BASE_STD[i] * 0.18
        err[:, i, :] += corr

    return {"errors": err, "cov": cov, "lidar": lidar, "corridor": corridor, "patch": patch}


def observability(cov: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Dimensionless weakest-mode information score in [0,1].

    All simulated covariance matrices are diagonal, so the minimum eigenvalue
    of the whitened information matrix is obtained without a costly eigensolve.
    """
    if not enabled:
        return np.ones((N, 3))
    variances = np.diagonal(cov, axis1=2, axis2=3)
    ratios = (REF_STD[None, None, :] ** 2) / np.maximum(variances, 1e-12)
    lam_min = np.min(ratios, axis=2)
    z = np.clip(ALPHA_OBS * (lam_min - LAMBDA_TH), -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def manifold_pair_distance(ei: np.ndarray, ej: np.ndarray,
                           ci: np.ndarray, cj: np.ndarray,
                           use_rotation: bool = True) -> float:
    """First-order right-tangent Mahalanobis distance, normalized by dof.

    The synthetic estimators are generated with independent innovations, so
    cross-covariance terms are zero in this simulation. The paper states the
    general cross-covariance expression and its independence approximation.
    """
    sl = slice(0, 6) if use_rotation else slice(0, 3)
    residual = (ei - ej)[sl]
    cov_rel = ci[sl, sl] + cj[sl, sl]
    d2 = float(residual.T @ np.linalg.solve(cov_rel + 1e-10*np.eye(len(residual)), residual))
    return max(d2 / len(residual), 0.0)


def dhr_weights(errors: np.ndarray, cov: np.ndarray, use_observability: bool = True,
                use_rotation: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trimmed nearest-neighbour consensus for a 3-executor pool."""
    phi = observability(cov, enabled=use_observability)
    dof = 6 if use_rotation else 3
    dims = slice(0, dof)
    # Vectorized diagonal-covariance Mahalanobis distances for the three pairs.
    var = np.diagonal(cov, axis1=2, axis2=3)[:, :, :dof]
    pairs = ((0, 1), (0, 2), (1, 2))
    dist = np.zeros((N, 3, 3))
    for i, j in pairs:
        delta = errors[:, i, dims] - errors[:, j, dims]
        denom = var[:, i, :] + var[:, j, :]
        d2 = np.sum(delta * delta / np.maximum(denom, 1e-12), axis=1) / dof
        dist[:, i, j] = dist[:, j, i] = d2
    scores = np.zeros((N, 3))
    for i in range(3):
        candidates = [phi[:, j] * dist[:, i, j] for j in range(3) if j != i]
        d_trim = np.minimum(candidates[0], candidates[1])
        scores[:, i] = np.exp(-d_trim / (2.0 * SIGMA_ARB ** 2))
    raw = phi * scores
    total = raw.sum(axis=1, keepdims=True)
    weights = np.divide(raw, total, out=np.zeros_like(raw), where=total > 1e-12)
    return scores, weights, phi


def run_fusion(data: Dict[str, np.ndarray], mode: str) -> Dict[str, np.ndarray]:
    errors, cov = data["errors"], data["cov"]
    if mode == "static":
        # Fixed inverse-trace weights; no temporal arbitration or state reset.
        precision = np.array([1.0 / np.trace(cov[0, i]) for i in range(3)])
        w = np.tile(precision / precision.sum(), (N, 1))
        scores = np.ones((N, 3))
    elif mode == "dhr_no_obs":
        scores, w, _ = dhr_weights(errors, cov, use_observability=False)
    elif mode == "dhr_euclidean":
        scores, w, _ = dhr_weights(errors, cov, use_observability=True, use_rotation=False)
    elif mode in ("dhr", "dhr_no_clean"):
        scores, w, _ = dhr_weights(errors, cov, use_observability=True)
    else:
        raise ValueError(mode)

    # Stateful isolation/re-entry controller. In this simulation, the cleaned
    # case removes post-attack memory bias in synthesize_trial(); here we model
    # the finite dwell times and a shadow re-entry period.
    if mode.startswith("dhr"):
        isolated = np.zeros(3, dtype=bool)
        bad_count = np.zeros(3, dtype=int)
        good_count = np.zeros(3, dtype=int)
        for k in range(N):
            for i in range(3):
                if scores[k, i] < TAU_ISO:
                    bad_count[i] += 1
                    good_count[i] = 0
                else:
                    bad_count[i] = 0
                    good_count[i] += 1
                if bad_count[i] >= K_ISO:
                    isolated[i] = True
                if isolated[i] and good_count[i] >= 5:
                    isolated[i] = False
                    bad_count[i] = 0
            if np.any(isolated):
                w[k, isolated] = 0.0
                sw = w[k].sum()
                if sw > 1e-12:
                    w[k] /= sw
                else:
                    # No trustworthy executor: do not silently force a vote.
                    # The synthetic evaluator marks this frame as unavailable.
                    w[k] = 0.0

    fused = np.einsum("ni,nij->nj", w, errors)
    unavailable = (w.sum(axis=1) < 1e-9)
    # Hold the previous estimate only for plotting continuity; unavailable
    # frames are counted separately and penalized in continuity statistics.
    for k in range(N):
        if unavailable[k]:
            fused[k] = fused[k - 1] if k else np.zeros(6)
    return {"error": fused, "weights": w, "scores": scores, "unavailable": unavailable}


def metrics(error: np.ndarray, unavailable: np.ndarray | None = None) -> Dict[str, float]:
    pos = error[:, :3]
    rot = error[:, 3:]
    ate = float(np.sqrt(np.mean(np.sum(pos ** 2, axis=1))))
    rpe = float(np.sqrt(np.mean(np.sum(np.diff(pos, axis=0) ** 2, axis=1))))
    rot_rmse = float(np.rad2deg(np.sqrt(np.mean(np.sum(rot ** 2, axis=1)))))
    continuity = 100.0 if unavailable is None else 100.0 * (1.0 - float(np.mean(unavailable)))
    return {"ate_rmse_m": ate, "rpe_m_per_frame": rpe,
            "rotation_rmse_deg": rot_rmse, "availability_percent": continuity}


def method_errors(data: Dict[str, np.ndarray], scenario: str) -> Dict[str, Dict[str, np.ndarray]]:
    e = data["errors"]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i, name in enumerate(("E1_geometry", "E2_spatial", "E3_graph")):
        out[name] = {"error": e[:, i, :], "weights": np.eye(3)[i][None, :].repeat(N, axis=0),
                     "scores": np.ones((N, 3)), "unavailable": np.zeros(N, bool)}
    out["Static_ensemble"] = run_fusion(data, "static")
    out["DHR_no_observability"] = run_fusion(data, "dhr_no_obs")
    out["DHR_no_manifold_rotation"] = run_fusion(data, "dhr_euclidean")
    out["DHR_no_cleansing"] = run_fusion(data, "dhr_no_clean")
    out["EndoSpatial_DHR"] = run_fusion(data, "dhr")
    return out


def homogeneous_ai_pool(data: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    """Three near-identical learned estimators sharing the same OOD bias."""
    rng = np.random.default_rng(seed)
    source = data["errors"][:, 1, :]
    replicas = np.repeat(source[:, None, :], 3, axis=1)
    jitter = rng.normal(size=replicas.shape) * BASE_STD[1][None, None, :] * 0.12
    replicas = replicas + jitter
    cov = np.repeat(data["cov"][:, 1:2, :, :], 3, axis=1)
    scores, weights, _ = dhr_weights(replicas, cov, use_observability=True)
    fused = np.einsum("ni,nij->nj", weights, replicas)
    return {"error": fused, "weights": weights, "scores": scores,
            "unavailable": np.zeros(N, dtype=bool)}

def simulate_all() -> Tuple[dict, dict, dict]:
    rng = np.random.default_rng(SEED)
    scenario_trial_metrics: Dict[str, Dict[str, list]] = {}
    representative = {}
    ablation_runs = []
    for scenario in SCENARIOS:
        scenario_trial_metrics[scenario] = {}
        for trial in range(N_TRIALS):
            # Independent seeds are generated by the master RNG; reproducible.
            seed = int(rng.integers(0, 2**32 - 1))
            trial_rng = np.random.default_rng(seed)
            data_clean = synthesize_trial(scenario, trial_rng, clean_memory=True)
            methods = method_errors(data_clean, scenario)
            for name, result in methods.items():
                mm = metrics(result["error"], result["unavailable"])
                scenario_trial_metrics[scenario].setdefault(name, []).append(mm)
            if scenario == "III":
                # Paired ablations use the same random seed and disturbance draw.
                data_no_clean = synthesize_trial(scenario, np.random.default_rng(seed), clean_memory=False)
                res_no_clean = run_fusion(data_no_clean, "dhr_no_clean")
                homo = homogeneous_ai_pool(data_clean, seed + 1)
                if trial == 0:
                    representative[scenario] = (data_clean, methods, data_no_clean)
                ablation_runs.append({
                    "Full_DHR": metrics(methods["EndoSpatial_DHR"]["error"], methods["EndoSpatial_DHR"]["unavailable"]),
                    "No_observability": metrics(methods["DHR_no_observability"]["error"], methods["DHR_no_observability"]["unavailable"]),
                    "No_manifold_rotation": metrics(methods["DHR_no_manifold_rotation"]["error"], methods["DHR_no_manifold_rotation"]["unavailable"]),
                    "No_state_cleansing": metrics(res_no_clean["error"], res_no_clean["unavailable"]),
                    "Homogeneous_AI_pool": metrics(homo["error"], homo["unavailable"]),
                })
            elif trial == 0:
                representative[scenario] = (data_clean, methods, data_clean)
    return scenario_trial_metrics, representative, {"ablation_runs": ablation_runs}


def mean_std(rows: list, key: str) -> Tuple[float, float]:
    a = np.array([x[key] for x in rows], dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def write_csvs(outdir: Path, scenario_metrics: dict, ablation: dict) -> dict:
    resultdir = outdir / "results"
    resultdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (resultdir / "scenario_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "ATE_RMSE_mean_m", "ATE_RMSE_std_m",
                    "RPE_mean_m_per_frame", "rotation_RMSE_mean_deg", "availability_mean_percent"])
        for scenario, methods in scenario_metrics.items():
            summary[scenario] = {}
            # The main scenario table contains only the three individual
            # executors, static fusion, and the complete DHR method. Ablation
            # variants are reported exclusively in ablation_comparison.csv.
            main_methods = ("E1_geometry", "E2_spatial", "E3_graph",
                            "Static_ensemble", "EndoSpatial_DHR")
            for name in main_methods:
                rows = methods[name]
                ate_m, ate_s = mean_std(rows, "ate_rmse_m")
                rpe_m, _ = mean_std(rows, "rpe_m_per_frame")
                rot_m, _ = mean_std(rows, "rotation_rmse_deg")
                av_m, _ = mean_std(rows, "availability_percent")
                summary[scenario][name] = {"ate_mean": ate_m, "ate_std": ate_s,
                                           "rpe_mean": rpe_m, "rotation_mean": rot_m,
                                           "availability_mean": av_m}
                w.writerow([scenario, name, f"{ate_m:.6f}", f"{ate_s:.6f}",
                            f"{rpe_m:.6f}", f"{rot_m:.6f}", f"{av_m:.4f}"])
    ab_rows = ablation["ablation_runs"]
    with (resultdir / "ablation_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "ATE_RMSE_mean_m", "ATE_RMSE_std_m", "RPE_mean_m_per_frame",
                    "rotation_RMSE_mean_deg", "availability_mean_percent"])
        ab_summary = {}
        for key in ("Full_DHR", "No_observability", "No_manifold_rotation", "No_state_cleansing", "Homogeneous_AI_pool"):
            rows = [x[key] for x in ab_rows]
            ate_m, ate_s = mean_std(rows, "ate_rmse_m")
            rpe_m, _ = mean_std(rows, "rpe_m_per_frame")
            rot_m, _ = mean_std(rows, "rotation_rmse_deg")
            av_m, _ = mean_std(rows, "availability_percent")
            ab_summary[key] = {"ate_mean": ate_m, "ate_std": ate_s, "rpe_mean": rpe_m,
                               "rotation_mean": rot_m, "availability_mean": av_m}
            w.writerow([key, f"{ate_m:.6f}", f"{ate_s:.6f}", f"{rpe_m:.6f}",
                        f"{rot_m:.6f}", f"{av_m:.4f}"])
    with (resultdir / "simulation_summary.json").open("w") as f:
        json.dump({"seed": SEED, "trials": N_TRIALS, "dt_s": DT,
                   "duration_s": DURATION, "scenario_summary": summary,
                   "ablation_summary": ab_summary,
                   "scope_note": "Synthetic SE(3)-tangent error simulation only; not real dataset or hardware measurements."},
                  f, indent=2)
    return {"scenario": summary, "ablation": ab_summary}


def plot_performance(outdir: Path, rep: dict) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    data, methods, data_no_clean = rep["III"]
    t = TIMES
    # Representative-run traces; summary statistics are reported separately.
    fig, axs = plt.subplots(3, 1, figsize=(8.4, 8.4), sharex=True)
    labels = ["E1 (geometry)", "E2 (spatial model)", "E3 (semantic graph)"]
    for i, label in enumerate(labels):
        axs[0].plot(t, methods["EndoSpatial_DHR"]["scores"][:, i], label=label, lw=1.4)
    axs[0].axhline(TAU_ISO, ls="--", lw=1.0, label=r"isolation threshold $\tau_{iso}$")
    axs[0].set_ylabel("Consensus score")
    axs[0].set_ylim(0, 1.03)
    axs[0].legend(ncol=2, fontsize=8, loc="lower left")
    axs[0].grid(alpha=0.25)

    for i, label in enumerate(labels):
        axs[1].plot(t, methods["EndoSpatial_DHR"]["weights"][:, i], label=label, lw=1.4)
    axs[1].set_ylabel("Trust weight")
    axs[1].set_ylim(0, 1.03)
    axs[1].legend(ncol=3, fontsize=8, loc="upper right")
    axs[1].grid(alpha=0.25)

    no_clean_result = run_fusion(data_no_clean, "dhr_no_clean")
    for key, label in [("Static_ensemble", "Static ensemble"),
                       ("no_clean", "DHR without state cleansing"),
                       ("EndoSpatial_DHR", "EndoSpatial-DHR")]:
        result = no_clean_result if key == "no_clean" else methods[key]
        err = result["error"][:, :3]
        axs[2].plot(t, np.linalg.norm(err, axis=1), label=label, lw=1.5)
    axs[2].set_ylabel("Position error (m)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(fontsize=8, loc="upper left")
    axs[2].grid(alpha=0.25)
    for ax in axs:
        ax.axvspan(12, 18, color="tab:red", alpha=0.08)
        ax.axvspan(20, 25, color="tab:gray", alpha=0.12)
        ax.axvspan(26, 32, color="tab:orange", alpha=0.08)
    axs[0].text(15, 0.05, "LiDAR spoofing", ha="center", fontsize=7)
    axs[1].text(22.5, 0.05, "degeneracy", ha="center", fontsize=7)
    axs[2].text(29, axs[2].get_ylim()[1] * 0.80, "visual OOD", ha="center", fontsize=7)
    fig.suptitle("Synthetic compound-threat response (one representative trial)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "EndoSpatial_SLAM_Fig2.pdf", bbox_inches="tight")
    fig.savefig(figdir / "EndoSpatial_SLAM_Fig2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(outdir: Path, rep: dict) -> None:
    figdir = outdir / "figures"
    gt = ground_truth()
    data, methods, data_no_clean = rep["III"]
    fig = plt.figure(figsize=(8.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "k--", lw=2.0, label="Ground truth")
    colors = [("E1_geometry", "Geometry-only (E1)"),
              ("E2_spatial", "Spatial model (E2)"),
              ("Static_ensemble", "Static ensemble"),
              ("EndoSpatial_DHR", "EndoSpatial-DHR")]
    for key, label in colors:
        est = gt + methods[key]["error"][:, :3]
        ax.plot(est[:, 0], est[:, 1], est[:, 2], lw=1.2, label=label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Synthetic 3-D figure-eight trajectory under compound disturbances")
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=25, azim=-58)
    fig.tight_layout()
    fig.savefig(figdir / "fig_3d_trajectory.pdf", bbox_inches="tight")
    fig.savefig(figdir / "fig_3d_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(outdir: Path, rep: dict, ablation: dict) -> None:
    figdir = outdir / "figures"
    data, methods, noclean_data = rep["III"]
    t = TIMES
    variants = {
        "Full_DHR": methods["EndoSpatial_DHR"],
        "No_observability": methods["DHR_no_observability"],
        "No_manifold_rotation": methods["DHR_no_manifold_rotation"],
        "No_state_cleansing": run_fusion(noclean_data, "dhr_no_clean"),
        "Homogeneous_AI_pool": homogeneous_ai_pool(data, SEED + 56),
    }
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.7))
    style = [("Full_DHR", "Full DHR"), ("No_observability", "w/o observability"),
             ("No_manifold_rotation", "w/o rotational residual"),
             ("No_state_cleansing", "w/o state cleansing")]
    for key, label in style:
        err = np.linalg.norm(variants[key]["error"][:, :3], axis=1)
        axs[0].plot(t, err, lw=1.4, label=label)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position error (m)")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=7)
    names = ["Full DHR", "w/o observability", "w/o manifold", "w/o cleansing", "homogeneous pool"]
    keys = ["Full_DHR", "No_observability", "No_manifold_rotation", "No_state_cleansing", "Homogeneous_AI_pool"]
    vals, errs = [], []
    for k in keys:
        rows = [x[k]["ate_rmse_m"] for x in ablation["ablation_runs"]]
        vals.append(np.mean(rows))
        errs.append(np.std(rows, ddof=1))
    x = np.arange(len(names))
    axs[1].bar(x, vals, yerr=errs, capsize=3)
    axs[1].set_xticks(x, names, rotation=18, ha="right", fontsize=7)
    axs[1].set_ylabel("ATE RMSE (m)")
    axs[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic ablation under compound disturbances (50 trials)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "EndoSpatial_SLAM_Ablation.pdf", bbox_inches="tight")
    fig.savefig(figdir / "EndoSpatial_SLAM_Ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    scenario_metrics, representative, ablation = simulate_all()
    summary = write_csvs(outdir, scenario_metrics, ablation)
    plot_performance(outdir, representative)
    plot_trajectory(outdir, representative)
    plot_ablation(outdir, representative, ablation)
    print(json.dumps(summary, indent=2))
    print(f"Generated figures in: {outdir / 'figures'}")
    print(f"Generated tables in: {outdir / 'results'}")

if __name__ == "__main__":
    main()

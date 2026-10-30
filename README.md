# EndoSpatial-SLAM revised manuscript and reproducibility package

This package contains a revised LaTeX manuscript, the figures embedded in it, and the Python code used to generate the synthetic curves and tables.

## Scope and scientific status

**Important:** all experimental data in the manuscript are generated synthetically by the supplied Monte-Carlo simulator in the six-dimensional local pose-error space. No real-world dataset, physical robot, or embedded computing platform is used. The simulation does not ingest raw LiDAR, camera, or IMU recordings, and it does not validate a complete SLAM implementation or establish a worst-case security guarantee.

The synthetic experiment uses a 40 s 3-D figure-eight reference trajectory sampled at 10 Hz, 50 trials per scenario, and a fixed random seed (`20261030`). It injects prescribed LiDAR spoofing, corridor-degeneracy, and spatial-model OOD bias episodes. In the state-cleansing ablation, the post-attack bias is retained to model persistent estimator-memory contamination. In the full-method stream, that injected bias is reset after the disturbance episode. This is an error-stream abstraction: actual timestamped voxel rollback, Transformer KV-cache reconstruction, and their computational cost are **not** simulated.

## Package layout

- `EndoSpatial_SLAM_revised.tex` — revised LaTeX source.
- `EndoSpatial_SLAM_revised.pdf` — compiled manuscript.
- `figures/*.pdf` — vector figures used by the manuscript.
- `figures/*.png` — raster previews of the same figures.
- `scripts/simulate_endospatial.py` — generates synthetic Monte-Carlo results, CSV tables, and performance/trajectory/ablation curves.
- `scripts/draw_architecture.py` — generates the system architecture diagram.
- `results/scenario_comparison.csv` — per-scenario estimator comparison (mean and standard deviation over 50 trials).
- `results/ablation_comparison.csv` — ablation metrics for the compound scenario.
- `results/simulation_summary.json` — machine-readable summaries, random seed, and scope note.

## Reproduction

Requirements: Python 3.10+ with `numpy` and `matplotlib`; a LaTeX environment with IEEEtran, `algorithm2e`, `amsmath`, `graphicx`, `booktabs`, and `multirow` to compile the paper.

From this directory:

```bash
python scripts/simulate_endospatial.py --outdir .
python scripts/draw_architecture.py
latexmk -pdf -interaction=nonstopmode EndoSpatial_SLAM_revised.tex
```

The Python script writes its figures to `figures/` and its CSV/JSON outputs to `results/`. The reported summary values in the LaTeX tables are rounded from these generated CSV files.

## Methodological notes

1. The synthetic estimator errors are represented as local six-dimensional perturbations ordered as `[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]`, with translations in metres and rotations in radians.
2. Innovations are independent across estimator channels in the current simulation, with temporal correlation within each channel. Therefore the simulation uses the zero-cross-covariance special case of the relative covariance propagation. The manuscript provides the general cross-covariance expression and states that shared-sensor cross-correlation must be estimated or handled conservatively in a real implementation.
3. The observability model uses diagonal covariance matrices and a dimensionless weakest-mode information score. It is a scalar observability proxy, not a full eigenvector-wise directional arbitration implementation.
4. The comparison labels `E1`, `E2`, and `E3` identify simulated estimator streams with different prescribed error models. They do not denote executable reimplementations of published SLAM systems.
5. All reported simulated availability values are 100% in this model, so the manuscript deliberately does not claim operational-survival or hardware-latency performance.

## Before external submission

A submission making real-world performance claims would require integration of executable estimator implementations, documented sensor recordings and calibration files, a reproducible attack-injection pipeline, covariance-calibration experiments, and hardware profiling. In particular, the mathematical covariance/correlation model and the state-cleansing mechanism must be validated on real sensor data before drawing real-world security conclusions.

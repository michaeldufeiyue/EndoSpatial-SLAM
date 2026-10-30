#!/usr/bin/env python3
"""Create the architecture diagram used by the revised EndoSpatial-SLAM paper."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6.0))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')


def box(x, y, w, h, title, body, face='#f5f7fa', edge='#34495e', title_color='#16202a'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.10',
                       linewidth=1.2, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    ax.text(x+w/2, y+h-0.25, title, ha='center', va='top', fontsize=9.5,
            fontweight='bold', color=title_color)
    ax.text(x+w/2, y+h/2-0.08, body, ha='center', va='center', fontsize=8,
            color='#283747', linespacing=1.35)
    return p


def arrow(x1, y1, x2, y2, text=None, rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                        linewidth=1.15, color='#34495e',
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2+0.12, text, ha='center', va='bottom', fontsize=7)

box(0.25, 2.20, 1.55, 1.55, 'Sensor streams', 'LiDAR scans\nRGB / depth images\nHigh-rate IMU', face='#edf3f8')
box(2.20, 4.05, 2.15, 1.35, r'$E_1$: geometric-inertial', 'IMU deskew + point-to-plane\nICP / local submap', face='#eef6f3')
box(2.20, 2.30, 2.15, 1.35, r'$E_2$: spatial model', 'Sparse-window Transformer\npose + predictive covariance', face='#f5eff8')
box(2.20, 0.55, 2.15, 1.35, r'$E_3$: semantic-topological', 'Instance graph + robust\nregistration', face='#fbf4e9')
box(5.00, 2.20, 2.10, 1.55, 'SE(3) consensus', 'Relative Log-map residuals\nCross-covariance propagation\nObservability-aware scores', face='#eef3fb')
box(7.75, 2.20, 1.80, 1.55, 'Factor graph', 'IMU factor once\nWeighted estimator factors\nSliding-window LM', face='#f2f5f5')
box(10.10, 2.20, 1.60, 1.55, 'State output', 'Pose / velocity\nSanitized local map\nUncertainty', face='#edf5ef')
box(5.15, 0.25, 4.35, 1.05, 'Feedback and recovery controller',
    'Isolate  →  rollback / re-encode  →  shadow re-entry  →  dynamic reconfiguration',
    face='#fcf1ee', edge='#a65d46')

arrow(1.80, 3.00, 2.20, 4.65)
arrow(1.80, 3.00, 2.20, 2.98)
arrow(1.80, 3.00, 2.20, 1.22)
arrow(4.35, 4.72, 5.00, 3.45)
arrow(4.35, 2.98, 5.00, 2.98)
arrow(4.35, 1.22, 5.00, 2.48)
arrow(7.10, 2.98, 7.75, 2.98)
arrow(9.55, 2.98, 10.10, 2.98)
arrow(8.65, 2.20, 8.10, 1.30, 'fault / drift', rad=0.08)

ax.text(6.05, 5.65, 'EndoSpatial-SLAM: heterogeneous estimation and feedback recovery',
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(6.0, 5.28, 'Shared timestamped inputs; estimator memories remain isolated',
        ha='center', va='center', fontsize=8.5, color='#566573')
fig.tight_layout()
fig.savefig(OUT / 'EndoSpatial_SLAM_Arch.pdf', bbox_inches='tight')
fig.savefig(OUT / 'EndoSpatial_SLAM_Arch.png', dpi=240, bbox_inches='tight')

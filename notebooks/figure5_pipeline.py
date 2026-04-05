"""
Figure 5: Pipeline Flowchart (Clean Mermaid-style)

A cleaner, less visually cluttered version of the pipeline diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Setup
ROOT = Path(__file__).parent.parent
plt.style.use(ROOT / "plot" / "mplstyle.txt")

# Colors (muted, professional)
COLORS = {
    'config': '#f5f5f5',
    'config_border': '#999999',
    'training': '#e3f2fd',
    'training_border': '#1976d2',
    'obs': '#fff3e0',
    'obs_border': '#f57c00',
    'inference': '#f3e5f5',
    'inference_border': '#7b1fa2',
    'output': '#e8f5e9',
    'output_border': '#388e3c',
    'arrow': '#555555',
    'text': '#333333',
}

def draw_box(ax, x, y, w, h, text, color, border_color, fontsize=9, bold=False):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], weight=weight, zorder=3)
    return box

def draw_cylinder(ax, x, y, w, h, text, color, border_color, fontsize=8):
    """Draw a cylinder (database) shape with text."""
    # Main body
    rect = FancyBboxPatch(
        (x - w/2, y - h/2 + 0.02), w, h - 0.04,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(rect)
    # Top ellipse
    ellipse = mpatches.Ellipse((x, y + h/2 - 0.02), w, 0.06,
                                facecolor=color, edgecolor=border_color,
                                linewidth=1.5, zorder=3)
    ax.add_patch(ellipse)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], zorder=4)

def draw_arrow(ax, start, end, color=None, style='->', connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=1.5,
        connectionstyle=connectionstyle,
        zorder=1,
        mutation_scale=12
    )
    ax.add_patch(arrow)

def draw_section_label(ax, x, y, text, color):
    """Draw a section label."""
    ax.text(x, y, text, ha='center', va='center', fontsize=8,
            color=color, weight='bold', style='italic')

# Create figure - wider to accommodate better layout
fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.set_aspect('equal')
ax.axis('off')

# === LAYOUT POSITIONS ===
# Row 1: Configs
config_y = 5.8
# Row 2: Scripts (gen_*)
script_y = 4.7
# Row 3: Intermediate data
data_y = 3.7
# Row 4: Training script / Inference
train_script_y = 2.7
# Row 5: Trained inferrer / Inference boxes
inferrer_y = 1.7
# Row 6: Outputs
output_y = 0.6

# Column positions
col_sim = 1.8      # Simulation/Training column
col_obs = 5.0      # Observation column
col_sbi = 7.0      # SBI column
col_mcmc = 9.2     # MCMC column

# === SECTION: Configs (top) ===
draw_box(ax, col_sim, config_y, 1.5, 0.45, 'simulation\nconfig', COLORS['config'], COLORS['config_border'], fontsize=8)
draw_box(ax, col_obs, config_y, 1.5, 0.45, 'observation\nconfig', COLORS['config'], COLORS['config_border'], fontsize=8)
draw_box(ax, col_mcmc, config_y, 1.5, 0.45, 'inference\nconfig', COLORS['config'], COLORS['config_border'], fontsize=8)

# === SECTION: Training Pipeline (left) ===
draw_section_label(ax, col_sim, script_y + 0.55, 'TRAINING (one-time)', COLORS['training_border'])

draw_box(ax, col_sim, script_y, 1.8, 0.4, 'gen_simulations.py', COLORS['training'], COLORS['training_border'], fontsize=8, bold=True)
draw_cylinder(ax, col_sim, data_y, 1.5, 0.5, 'M-c pairs +\nprofiles', COLORS['training'], COLORS['training_border'], fontsize=7)
draw_box(ax, col_sim, train_script_y, 1.8, 0.4, 'train_inferrer.py', COLORS['training'], COLORS['training_border'], fontsize=8, bold=True)
draw_cylinder(ax, col_sim, inferrer_y, 1.4, 0.5, 'trained\ninferrer', COLORS['training'], COLORS['training_border'], fontsize=7)

# Training vertical arrows (straight down)
draw_arrow(ax, (col_sim, config_y - 0.25), (col_sim, script_y + 0.22))
draw_arrow(ax, (col_sim, script_y - 0.22), (col_sim, data_y + 0.28))
draw_arrow(ax, (col_sim, data_y - 0.28), (col_sim, train_script_y + 0.22))
draw_arrow(ax, (col_sim, train_script_y - 0.22), (col_sim, inferrer_y + 0.28))

# === SECTION: Observations (middle) ===
draw_section_label(ax, col_obs, script_y + 0.55, 'OBSERVATIONS', COLORS['obs_border'])

draw_box(ax, col_obs, script_y, 1.8, 0.4, 'gen_observations.py', COLORS['obs'], COLORS['obs_border'], fontsize=8, bold=True)
draw_cylinder(ax, col_obs, data_y, 1.5, 0.5, 'observed\nprofiles', COLORS['obs'], COLORS['obs_border'], fontsize=7)

# Observation vertical arrows
draw_arrow(ax, (col_obs, config_y - 0.25), (col_obs, script_y + 0.22))
draw_arrow(ax, (col_obs, script_y - 0.22), (col_obs, data_y + 0.28))

# === SECTION: Inference ===
draw_section_label(ax, (col_sbi + col_mcmc) / 2, inferrer_y + 0.65, 'INFERENCE', COLORS['inference_border'])

draw_box(ax, col_sbi, inferrer_y, 1.3, 0.5, 'SBI', COLORS['inference'], COLORS['inference_border'], fontsize=10, bold=True)
draw_box(ax, col_mcmc, inferrer_y, 1.3, 0.5, 'MCMC', COLORS['inference'], COLORS['inference_border'], fontsize=10, bold=True)

# Arrows INTO inference boxes
# trained inferrer -> SBI (horizontal then down)
draw_arrow(ax, (col_sim + 0.75, inferrer_y), (col_sbi - 0.68, inferrer_y))

# observed profiles -> SBI (diagonal)
draw_arrow(ax, (col_obs + 0.5, data_y - 0.15), (col_sbi - 0.3, inferrer_y + 0.28), connectionstyle="arc3,rad=-0.15")

# observed profiles -> MCMC (diagonal)
draw_arrow(ax, (col_obs + 0.6, data_y - 0.2), (col_mcmc - 0.4, inferrer_y + 0.28), connectionstyle="arc3,rad=-0.2")

# inference config -> MCMC (straight down)
draw_arrow(ax, (col_mcmc, config_y - 0.25), (col_mcmc, inferrer_y + 0.28))

# === SECTION: Outputs (bottom) ===
draw_section_label(ax, col_obs, output_y + 0.55, 'OUTPUTS', COLORS['output_border'])

draw_cylinder(ax, col_sbi - 0.8, output_y, 1.2, 0.45, 'SBI\nposteriors', COLORS['output'], COLORS['output_border'], fontsize=7)
draw_cylinder(ax, col_mcmc - 0.8, output_y, 1.2, 0.45, 'MCMC\nposteriors', COLORS['output'], COLORS['output_border'], fontsize=7)
draw_box(ax, col_mcmc + 0.7, output_y, 1.4, 0.45, 'plot_chains.py', COLORS['output'], COLORS['output_border'], fontsize=7, bold=True)

# Arrows from inference to outputs (straight down)
draw_arrow(ax, (col_sbi, inferrer_y - 0.28), (col_sbi - 0.8, output_y + 0.25))
draw_arrow(ax, (col_mcmc, inferrer_y - 0.28), (col_mcmc - 0.8, output_y + 0.25))

# Arrows from posteriors to plot_chains
draw_arrow(ax, (col_sbi - 0.8 + 0.65, output_y), (col_mcmc + 0.7 - 0.73, output_y))
draw_arrow(ax, (col_mcmc - 0.8 + 0.65, output_y), (col_mcmc + 0.7 - 0.73, output_y))

# Add a subtle background grouping
from matplotlib.patches import Rectangle

# Training background
train_bg = Rectangle((col_sim - 1.1, inferrer_y - 0.4), 2.2, 4.0,
                       facecolor=COLORS['training'], edgecolor='none', alpha=0.25, zorder=0)
ax.add_patch(train_bg)

# Observation background
obs_bg = Rectangle((col_obs - 1.0, data_y - 0.4), 2.0, 2.0,
                    facecolor=COLORS['obs'], edgecolor='none', alpha=0.25, zorder=0)
ax.add_patch(obs_bg)

plt.tight_layout()

# Save
out_path = ROOT / "outputs"
fig.savefig(out_path / "figure5_pipeline.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_path / "figure5_pipeline.png", bbox_inches="tight", dpi=150)
print(f"Saved to: {out_path / 'figure5_pipeline.pdf'}")

plt.show()

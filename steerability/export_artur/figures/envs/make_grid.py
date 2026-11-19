"""Assemble the 2x3 environment figure from the per-environment renders in this folder."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
ROWS = [
    ("ScrewLightbulb", ["lamp_training_1_s1_start_0", "lamp_training_2_s1_start_1", "lamp_held_out_s1_mid"]),
    ("CoffeePressButton", ["coffee_training_1_kitchen0_L8S6", "coffee_training_2_kitchen1_L4S8", "coffee_held_out_kitchen11_L1S4"]),
]
COLS = ["Training environment 1", "Training environment 2", "Held-out environment"]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
fig.patch.set_facecolor("white")
for r, (task, stems) in enumerate(ROWS):
    for c, stem in enumerate(stems):
        ax = axes[r, c]
        ax.imshow(Image.open(HERE / f"{stem}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if r == 0:
            ax.set_title(COLS[c], fontsize=12, color="#0b0b0b", pad=8)
        if c == 0:
            ax.set_ylabel(task, fontsize=12, color="#0b0b0b", labelpad=8)
fig.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.005, wspace=0.03, hspace=0.03)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"env_grid.{ext}", dpi=200, facecolor="white")
print("wrote env_grid.png / env_grid.pdf")

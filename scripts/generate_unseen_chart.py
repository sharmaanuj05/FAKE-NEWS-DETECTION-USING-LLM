"""
scripts/generate_unseen_chart.py — Generates polished Right vs Wrong per class performance chart
using real saved results from reports/comparison.json on the Unseen-Source Test.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comp_file = os.path.join(BASE_DIR, "reports", "comparison.json")

with open(comp_file) as f:
    comp_data = json.load(f)

unseen_res = comp_data["benchmarks"]["Unseen-Source Test (LIAR)"]["new_model"]
# Unseen stats:
# Total 1283 items
# Real class (840 items):
num_real = unseen_res["num_real"]
false_fake = unseen_res["false_fake_count"]
correct_real = num_real - false_fake

# Fake class (443 items):
num_fake = unseen_res["num_fake"]
false_real = unseen_res["false_real_count"]
correct_fake = num_fake - false_real

classes = ["Real Claims (840)", "Fake Claims (443)"]
correct = [correct_real, correct_fake]
wrong = [false_fake, false_real]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#1e293b")

x = np.arange(len(classes))
width = 0.35

rects1 = ax.bar(x - width/2, correct, width, label="Correct Prediction", color="#10b981", edgecolor="#047857", linewidth=1.2)
rects2 = ax.bar(x + width/2, wrong, width, label="Misclassification", color="#ef4444", edgecolor="#b91c1c", linewidth=1.2)

ax.set_ylabel("Number of Claims", fontsize=12, color="#cbd5e1", fontweight="bold")
ax.set_title("Unseen-Source Test Performance (1,283 Real Claims Across Domains)", fontsize=13, color="#f8fafc", fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11, color="#e2e8f0", fontweight="600")
ax.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=10, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_color("#334155")

def autolabel(rects, totals):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        pct = (height / totals[idx % len(totals)]) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#f8fafc")

totals = [num_real, num_fake]
autolabel(rects1, totals)
autolabel(rects2, totals)

ax.set_ylim(0, max(num_real, num_fake) * 1.15)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "reports", "unseen_performance_chart.png")
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Chart saved to {out_path}")

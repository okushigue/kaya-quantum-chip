# kaya_bs_figure_paper.py
import warnings
import numpy as np               # <-- ADD
import matplotlib.pyplot as plt
import seaborn as sns
from kaya_boson_sampling_v2 import KayaBosonSampling
# ---------- CONFIGURAÇÃO PAPER ----------
plt.rcdefaults()
plt.rcParams.update({
    "text.usetex": False,               # True se tiver LaTeX instalado
    "font.family": "serif",
    "font.serif": ["Liberation Serif"],
    "font.size": 12,
    "figure.figsize": (7, 3.5),         # 1-coluna IEEE
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

warnings.filterwarnings("ignore", category=UserWarning)  # remove avisos de categoria

# ---------- GERA DADOS ----------
bs = KayaBosonSampling(n_photons=3, n_modes=8)
counts = bs.simulate(shots=15_000)          # mais amostras → curva suave

# ---------- PLOT ----------
top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
states, freq = zip(*top_items)
probs = np.array(freq) / sum(counts.values())

colors = sns.color_palette("colorblind", len(states))

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(states, probs, color=sns.color_palette("colorblind", len(states)),
              edgecolor="black", linewidth=0.5)
ax.set_ylim(0, max(probs) * 1.15)
ax.margins(y=0)
ax.set_xlabel("Output string", fontsize=12)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title("Kaya boson-sampling (3 photons, 8 modes)", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right", fontsize=10)
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{p:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("kaya_bs_paper.png", dpi=300, bbox_inches="tight")
print("Figura paper salva: kaya_bs_paper.png")

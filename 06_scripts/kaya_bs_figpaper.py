# kaya_bs_figpaper.py
import matplotlib.pyplot as plt
import seaborn as sns
from kaya_boson_sampling_v2 import KayaBosonSampling, CFG

CFG["MAX_PLOT_STATES"] = 20          # mais estados
CFG["PALETTE"] = sns.color_palette("tab10")

bs = KayaBosonSampling(n_photons=3, n_modes=8)
counts = bs.simulate(shots=10_000)   # mais shots → curva mais suave
fname = bs.plot(counts)

# Ajusta fonte para paper
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.figsize": (7, 3.5),
    "figure.dpi": 300,
})
plt.savefig("kaya_bs_paper.png", dpi=300, bbox_inches="tight")
print("Figura para paper salva: kaya_bs_paper.png")

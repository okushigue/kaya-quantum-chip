# kaya_phase_estimation_v2.py
import os
import time
import logging
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("KayaPE")

plt.rcdefaults()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 11,
    "figure.figsize": (7, 3.5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ---------- CLASSE PRINCIPAL ----------
class KayaPhaseEstimation:
    def __init__(self, n_photons: int = 2):
        self.n = n_photons
        log.info("Kaya Phase Estimation | NOON state N=%d", self.n)

    # ---------- CIRCUITOS ----------
    def _noon_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.x(0)          # |10⟩
        qc.h(0); qc.h(1) # BS1
        return qc

    def _apply_phase(self, qc: QuantumCircuit, phi: float) -> QuantumCircuit:
        qc.p(phi * self.n, 0)  # Nϕ
        return qc

    def _recombine(self, qc: QuantumCircuit) -> QuantumCircuit:
        qc.h(0); qc.h(1)  # BS2
        return qc

    # ---------- PROBABILIDADE ----------
    def _detection_prob(self, state: Statevector) -> float:
        return sum(abs(state.data[i])**2 for i in range(len(state.data)) if i & 1)

    # ---------- SWEEP COM REPETIÇÕES ----------
    def phase_sweep(self, phases: np.ndarray, shots: int = 1024, reps: int = 10):
        log.info("Running phase sweep | %d phases × %d reps", len(phases), reps)
        q_probs = np.zeros((reps, len(phases)))
        c_probs = np.zeros((reps, len(phases)))

        for r in range(reps):
            for i, phi in enumerate(phases):
                # Quantum
                qc = self._noon_circuit()
                qc = self._apply_phase(qc, phi)
                qc = self._recombine(qc)
                q_probs[r, i] = self._detection_prob(Statevector.from_instruction(qc))

                # Classical (single photon)
                qc_c = QuantumCircuit(2)
                qc_c.x(0); qc_c.h(0); qc_c.h(1)
                qc_c.p(phi, 0)
                qc_c.h(0); qc_c.h(1)
                c_probs[r, i] = self._detection_prob(Statevector.from_instruction(qc_c))

        q_mean = q_probs.mean(axis=0)
        q_std  = q_probs.std(axis=0)
        c_mean = c_probs.mean(axis=0)
        c_std  = c_probs.std(axis=0)
        return q_mean, q_std, c_mean, c_std

    # ---------- PLOT PAPER ----------
    def plot_paper(self, phases: np.ndarray, q_mean, q_std, c_mean, c_std):
        fig, ax = plt.subplots()
        ax.errorbar(phases, q_mean, yerr=q_std, fmt='o-', color='tab:blue',
                    capsize=3, label=f'Quantum N={self.n}', linewidth=1.2, markersize=4)
        ax.errorbar(phases, c_mean, yerr=c_std, fmt='s-', color='tab:red',
                    capsize=3, label='Classical N=1', linewidth=1.2, markersize=4)

        # curvas teóricas
        ax.plot(phases, 0.5*(1 + np.cos(self.n * phases)), '--', color='tab:blue', alpha=0.7)
        ax.plot(phases, 0.5*(1 + np.cos(phases)), '--', color='tab:red', alpha=0.7)

        ax.set_xlabel("Phase φ (rad)")
        ax.set_ylabel("Detection probability")
        ax.set_title(f"NOON-state phase estimation | N={self.n}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.margins(x=0)

        fname = f"kaya_pe_N{self.n}_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        log.info("Paper figure saved: %s", fname)

        # CSV
        csv = fname.replace(".png", ".csv")
        pd.DataFrame({"phase": phases, "q_mean": q_mean, "q_std": q_std,
                      "c_mean": c_mean, "c_std": c_std}).to_csv(csv, index=False)
        log.info("Data saved: %s", csv)
        return fname, csv

    # ---------- PIPELINE COMPLETO ----------
    def run(self, n_points: int = 50, reps: int = 10):
        phases = np.linspace(0, 2*np.pi, n_points)
        q_mean, q_std, c_mean, c_std = self.phase_sweep(phases, reps=reps)
        fig, csv = self.plot_paper(phases, q_mean, q_std, c_mean, c_std)

        visibility = (q_mean.max() - q_mean.min()) / (q_mean.max() + q_mean.min())
        delta_phi = 1 / self.n
        return {
            "n_photons": self.n,
            "visibility": visibility,
            "delta_phi_heisenberg": delta_phi,
            "figure": fig,
            "csv": csv,
        }


# ---------- BATERIA ----------
def benchmark():
    configs = [2, 3, 4, 5]
    results = {}
    for n in configs:
        log.info("Benchmark N=%d", n)
        results[n] = KayaPhaseEstimation(n).run()
    print("\n----- RESUMO -----")
    print("N  | visibility | Δφ (Heisenberg)")
    for n, r in results.items():
        print(f"{n}  |   {r['visibility']:.3f}    |  {r['delta_phi_heisenberg']:.4f}")
    return results


# ---------- MAIN ----------
if __name__ == "__main__":
    print("🌊 Kaya Phase-Estimation v2 | Paper-Ready")
    # single run
    r = KayaPhaseEstimation(3).run(n_points=60, reps=10)
    print(f"Visibility: {r['visibility']:.3f} | Δφ: {r['delta_phi_heisenberg']:.4f} rad")
    # benchmark
    if input("\nRodar benchmark completo? (y/n): ").lower() == "y":
        benchmark()

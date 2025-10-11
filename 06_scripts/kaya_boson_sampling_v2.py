# kaya_boson_sampling_v2.py
import os
import json
import math
import itertools
import logging
import datetime as dt
import time
from typing import Tuple, Dict, List
import numpy as np
from scipy.linalg import qr
from scipy.stats import chisquare
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIGURAÇÕES ----------
CFG = {
    "SHOTS": 2000,
    "TOL_UNITARY": 1e-10,
    "MAX_PLOT_STATES": 15,
    "PALETTE": sns.color_palette("colorblind"),
    "LOG_LEVEL": logging.INFO,          # mude para DEBUG se quiser verbose
}

logging.basicConfig(
    level=CFG["LOG_LEVEL"],
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("KayaBS")

# ---------- CLASSE PRINCIPAL ----------
class KayaBosonSampling:
    def __init__(self, n_photons: int = 3, n_modes: int = 6):
        self.n = n_photons
        self.m = n_modes
        self.U: np.ndarray | None = None
        log.info("Kaya Boson Sampling | %d photons in %d modes", self.n, self.m)

    # ---------- UTILITÁRIOS ----------
    def random_unitary(self) -> np.ndarray:
        X = np.random.randn(self.m, self.m) + 1j * np.random.randn(self.m, self.m)
        Q, R = qr(X)
        d = np.diag(R)
        Q = Q @ np.diag(d / np.abs(d))
        self.U = Q
        return Q

    def validate_unitary(self, U: np.ndarray | None = None) -> bool:
        U = U if U is not None else self.U
        Id = np.eye(self.m, dtype=complex)
        err = max(
            np.max(np.abs(U @ U.conj().T - Id)),
            np.max(np.abs(U.conj().T @ U - Id)),
        )
        ok = err < CFG["TOL_UNITARY"]
        log.debug("Unitary check | err=%.2e | %s", err, ok)
        return ok

    # ---------- PERMANENT ----------
    def permanent(self, A: np.ndarray) -> Tuple[complex, float]:
        t0 = time.perf_counter()
        if A.shape[0] == 0:
            return 1.0, 0.0
        if A.shape[0] <= 7:
            perm = self._permanent_exact(A)
        else:
            perm = self._permanent_ryser(A)
        return perm, time.perf_counter() - t0

    def _permanent_exact(self, A: np.ndarray) -> complex:
        n = A.shape[0]
        total = 0.0
        for p in itertools.permutations(range(n)):
            prod = 1.0
            for i in range(n):
                prod *= A[i, p[i]]
            total += prod
        return complex(total)

    def _permanent_ryser(self, A: np.ndarray) -> complex:
        n = A.shape[0]
        total = 0.0
        for r in range(n + 1):
            for cols in itertools.combinations(range(n), r):
                rows_sum = A[:, cols].sum(axis=1)
                prod = np.prod(rows_sum)
                total += (-1) ** (n - r) * prod
        return complex(total)

    # ---------- ESTADOS ----------
    def output_states(self) -> List[Tuple[int, ...]]:
        states = []
        for comb in itertools.combinations_with_replacement(range(self.m), self.n):
            st = [0] * self.m
            for mode in comb:
                st[mode] += 1
            states.append(tuple(st))
        return states

    # ---------- PROBABILIDADES ----------
    def probabilities(self, states: List[Tuple[int, ...]]) -> List[float]:
        if self.U is None:
            self.random_unitary()
        probs = []
        for st in states:
            if sum(st) != self.n:
                probs.append(0.0)
                continue
            cols = [i for i, cnt in enumerate(st) for _ in range(cnt)]
            sub = self.U[: self.n, cols]
            perm, _ = self.permanent(sub)
            norm = np.prod([math.factorial(c) for c in st])
            probs.append(abs(perm) ** 2 / norm)
        total = sum(probs)
        if total <= 0:
            log.warning("Total prob zero – usando uniforme")
            return [1.0 / len(states)] * len(states)
        return [p / total for p in probs]

    # ---------- SIMULAÇÃO ----------
    def simulate(self, shots: int = CFG["SHOTS"]) -> Dict[str, int]:
        states = self.output_states()
        probs = self.probabilities(states)
        idx = np.random.choice(len(states), size=shots, p=probs)
        counts = {}
        for i in idx:
            key = "".join(map(str, states[i]))
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ---------- MÉTRICAS ----------
    @staticmethod
    def entropy(counts: Dict[str, int]) -> float:
        vals = np.array(list(counts.values()))
        probs = vals / vals.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    @staticmethod
    def tvd(counts: Dict[str, int], expected_probs: List[float], states: List[str]) -> float:
        total = sum(counts.values())
        emp = np.array([counts.get(s, 0) / total for s in states])
        return 0.5 * np.abs(emp - expected_probs).sum()

    # ---------- PLOT ----------
    def plot(self, counts: Dict[str, int], save: bool = True) -> str | None:
        try:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: CFG["MAX_PLOT_STATES"]]
            states, freq = zip(*top)
            probs = np.array(freq) / sum(counts.values())

            plt.figure(figsize=(14, 6))
            bars = plt.bar(states, probs, color=CFG["PALETTE"][: len(states)])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Probability")
            plt.title(f"Kaya Boson Sampling | {self.n} photons, {self.m} modes")
            plt.grid(axis="y", alpha=0.3)
            for bar, p in zip(bars, probs):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f"{p:.3f}", ha="center", va="bottom", fontsize=8)

            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"kaya_bs_{self.n}p_{self.m}m_{ts}.png"
            if save:
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                pd.DataFrame({"state": states, "prob": probs, "counts": freq}).to_csv(
                    fname.replace(".png", ".csv"), index=False
                )
                log.info("Plot+CSV saved: %s", fname)
            plt.close()
            return fname
        except Exception as e:
            log.error("Plot error: %s", e)
            return None

    # ---------- PIPELINE COMPLETO ----------
    def run(self, shots: int = CFG["SHOTS"]) -> dict:
        log.info("Running full pipeline...")
        t0 = time.perf_counter()
        self.random_unitary()
        self.validate_unitary()
        states = self.output_states()
        probs = self.probabilities(states)
        counts = self.simulate(shots)
        ent = self.entropy(counts)
        max_ent = np.log2(len(states))
        tvd_val = self.tvd(counts, probs, ["".join(map(str, s)) for s in states])
        chi2, pval = chisquare(list(counts.values()))
        plot_file = self.plot(counts)

        return {
            "n_photons": self.n,
            "n_modes": self.m,
            "entropy": ent,
            "max_entropy": max_ent,
            "entropy_ratio": ent / max_ent,
            "tvd": tvd_val,
            "chi2_pvalue": pval,
            "coverage": len(counts) / len(states),
            "computation_time": time.perf_counter() - t0,
            "plot_file": plot_file or "",
        }


# ---------- BATERIA DE TESTES ----------
def benchmark() -> None:
    configs = [(2, 4), (3, 6), (3, 8), (4, 8)]
    results = {}
    for p, m in configs:
        log.info("Benchmarking (%d,%d)", p, m)
        results[f"{p}_{m}"] = KayaBosonSampling(p, m).run()

    # Salva JSON consolidado
    out_json = f"kaya_bs_benchmark_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_json)

    # Tabela resumo
    print("\n----- RESUMO -----")
    print("cfg | entropy | ratio | TVD | pval χ² | coverage")
    for k, v in results.items():
        print(f"{k:>5} | {v['entropy']:5.2f} | {v['entropy_ratio']:5.3f} | "
              f"{v['tvd']:5.3f} | {v['chi2_pvalue']:7.4f} | {v['coverage']:5.1%}")


# ---------- MAIN ----------
if __name__ == "__main__":
    benchmark()

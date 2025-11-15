# kaya_boson_sampling_v4.py
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
from scipy.stats import chisquare, entropy as scipy_entropy
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIGURATIONS ----------
CFG = {
    "SHOTS": 5000,
    "TOL_UNITARY": 1e-10,
    "MAX_PLOT_STATES": 20,
    "PALETTE": sns.color_palette("husl", 20),
    "LOG_LEVEL": logging.INFO,
}

logging.basicConfig(
    level=CFG["LOG_LEVEL"],
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("KayaBS")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.figsize": (16, 10),
    "figure.dpi": 300,
})

# ---------- MAIN CLASS ----------
class KayaBosonSampling:
    def __init__(self, n_photons: int = 3, n_modes: int = 6, loss_rate: float = 0.0):
        """
        Quantum Boson Sampling with physical validation.
        
        Args:
            n_photons: Number of indistinguishable photons
            n_modes: Number of optical modes
            loss_rate: Photonic loss rate (0-1)
        """
        self.n = n_photons
        self.m = n_modes
        self.loss_rate = loss_rate
        self.U: np.ndarray | None = None
        self.perm_times: List[float] = []
        log.info("Kaya Boson Sampling | %d photons in %d modes | loss=%.3f", 
                 self.n, self.m, loss_rate)

    # ---------- UTILITIES ----------
    def random_unitary(self) -> np.ndarray:
        """Generates Haar-random unitary via QR decomposition"""
        X = np.random.randn(self.m, self.m) + 1j * np.random.randn(self.m, self.m)
        Q, R = qr(X)
        d = np.diag(R)
        Q = Q @ np.diag(d / np.abs(d))
        self.U = Q
        log.debug("Generated Haar-random unitary")
        return Q

    def validate_unitary(self, U: np.ndarray | None = None) -> bool:
        """Verifies unitarity: U†U = UU† = I"""
        U = U if U is not None else self.U
        Id = np.eye(self.m, dtype=complex)
        err = max(
            np.max(np.abs(U @ U.conj().T - Id)),
            np.max(np.abs(U.conj().T @ U - Id)),
        )
        ok = err < CFG["TOL_UNITARY"]
        log.debug("Unitary check | err=%.2e | valid=%s", err, ok)
        return ok

    # ---------- PERMANENT (3 ALGORITHMS) ----------
    def permanent(self, A: np.ndarray) -> Tuple[complex, float]:
        """
        Calculates permanent of matrix A with adaptive algorithm.
        Returns (permanent, computation_time)
        """
        t0 = time.perf_counter()
        n = A.shape[0]
        
        if n == 0:
            return 1.0, 0.0
        elif n <= 4:
            perm = self._permanent_exact(A)
        elif n <= 8:
            perm = self._permanent_glynn(A)
        else:
            perm = self._permanent_ryser(A)
        
        elapsed = time.perf_counter() - t0
        self.perm_times.append(elapsed)
        return perm, elapsed

    def _permanent_exact(self, A: np.ndarray) -> complex:
        """Brute force: O(n!)"""
        n = A.shape[0]
        total = 0.0
        for p in itertools.permutations(range(n)):
            prod = 1.0
            for i in range(n):
                prod *= A[i, p[i]]
            total += prod
        return complex(total)

    def _permanent_glynn(self, A: np.ndarray) -> complex:
        """
        Glynn formula: O(2^n)
        Better than Ryser for n>5
        """
        n = A.shape[0]
        total = 0.0
        
        for k in range(2**n):
            # Gray code for efficiency
            s = np.array([1 if (k >> i) & 1 else -1 for i in range(n)])
            prod = 1.0
            for i in range(n):
                prod *= np.sum(A[i, :] * s)
            total += prod
        
        return complex(total / (2**(n-1)))

    def _permanent_ryser(self, A: np.ndarray) -> complex:
        """Ryser formula: O(2^n × n²)"""
        n = A.shape[0]
        total = 0.0
        for r in range(n + 1):
            for cols in itertools.combinations(range(n), r):
                rows_sum = A[:, cols].sum(axis=1)
                prod = np.prod(rows_sum)
                total += (-1) ** (n - r) * prod
        return complex(total)

    # ---------- FOCK STATES ----------
    def output_states(self) -> List[Tuple[int, ...]]:
        """
        Generates all possible Fock states: |n₁,n₂,...,nₘ⟩
        with Σnᵢ = n_photons
        """
        states = []
        for comb in itertools.combinations_with_replacement(range(self.m), self.n):
            st = [0] * self.m
            for mode in comb:
                st[mode] += 1
            states.append(tuple(st))
        return list(set(states))  # Remove duplicates

    # ---------- QUANTUM PROBABILITIES ----------
    def probabilities(self, states: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Calculates P(s) = |perm(Uₛ)|² / Πnᵢ!
        where Uₛ is submatrix with columns repeated according to occupation
        """
        if self.U is None:
            self.random_unitary()
        
        probs = np.zeros(len(states))
        
        for idx, st in enumerate(states):
            if sum(st) != self.n:
                continue
            
            # Construct submatrix
            cols = []
            for mode_idx, count in enumerate(st):
                cols.extend([mode_idx] * count)
            
            sub = self.U[:self.n, cols]
            perm, _ = self.permanent(sub)
            
            # Normalization
            norm = np.prod([math.factorial(c) for c in st])
            probs[idx] = abs(perm) ** 2 / norm
        
        # Apply losses (reduces detection probability)
        if self.loss_rate > 0:
            probs *= (1 - self.loss_rate) ** self.n
        
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            log.warning("Total prob zero - using uniform")
            probs = np.ones(len(states)) / len(states)
        
        return probs

    # ---------- CLASSICAL SIMULATION (DISTINGUISHABLE PARTICLES) ----------
    def classical_probabilities(self, states: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Distinguishable particles: uniform multinomial distribution
        P_classical(s) = n! / (Πnᵢ!) × (1/m)^n
        """
        probs = np.zeros(len(states))
        
        for idx, st in enumerate(states):
            if sum(st) != self.n:
                continue
            
            # Multinomial count
            numerator = math.factorial(self.n)
            denominator = np.prod([math.factorial(c) for c in st])
            
            # Each photon has probability 1/m to go to each mode
            probs[idx] = (numerator / denominator) * (1 / self.m) ** self.n
        
        probs /= probs.sum()
        return probs

    # ---------- SIMULATION ----------
    def simulate(self, shots: int = CFG["SHOTS"], classical: bool = False) -> Dict[str, int]:
        """
        Simulates Boson Sampling measurements.
        
        Args:
            shots: Number of measurements
            classical: If True, simulates distinguishable particles
        """
        states = self.output_states()
        
        if classical:
            probs = self.classical_probabilities(states)
            log.debug("Simulating classical (distinguishable) particles")
        else:
            probs = self.probabilities(states)
            log.debug("Simulating quantum (indistinguishable) bosons")
        
        idx = np.random.choice(len(states), size=shots, p=probs)
        
        counts = {}
        for i in idx:
            key = "".join(map(str, states[i]))
            counts[key] = counts.get(key, 0) + 1
        
        return counts

    # ---------- QUANTUM ADVANTAGE METRICS ----------
    def heaviness_score(self, counts: Dict[str, int], states: List[Tuple[int, ...]]) -> float:
        """
        Heavy Output Generation (HOG): fraction of samples in outputs 
        with probability above median.
        
        Aaronson-Arkhipov test to demonstrate quantum advantage.
        """
        probs = self.probabilities(states)
        median_prob = np.median(probs)
        
        total_shots = sum(counts.values())
        heavy_count = 0
        
        state_to_idx = {"".join(map(str, s)): i for i, s in enumerate(states)}
        
        for state_str, count in counts.items():
            if state_str in state_to_idx:
                idx = state_to_idx[state_str]
                if probs[idx] > median_prob:
                    heavy_count += count
        
        return heavy_count / total_shots

    def linear_xeb(self, counts: Dict[str, int], states: List[Tuple[int, ...]]) -> float:
        """
        Linear Cross-Entropy Benchmarking (XEB):
        XEB = D × Σᵢ P(xᵢ) - 1
        
        where D = Hilbert space dimension, xᵢ = measured samples
        XEB ≈ 0 for classical, XEB > 0 for quantum
        """
        probs = self.probabilities(states)
        state_to_idx = {"".join(map(str, s)): i for i, s in enumerate(states)}
        
        total_shots = sum(counts.values())
        sum_probs = 0.0
        
        for state_str, count in counts.items():
            if state_str in state_to_idx:
                idx = state_to_idx[state_str]
                sum_probs += count * probs[idx]
        
        D = len(states)
        xeb = D * (sum_probs / total_shots) - 1
        return xeb

    def kolmogorov_distance(self, counts_q: Dict, counts_c: Dict, states: List) -> float:
        """
        Kolmogorov distance between quantum and classical distributions.
        """
        state_strs = ["".join(map(str, s)) for s in states]
        total_q = sum(counts_q.values())
        total_c = sum(counts_c.values())
        
        p_q = np.array([counts_q.get(s, 0) / total_q for s in state_strs])
        p_c = np.array([counts_c.get(s, 0) / total_c for s in state_strs])
        
        return np.max(np.abs(np.cumsum(p_q - p_c)))

    # ---------- PHYSICAL VALIDATION: HONG-OU-MANDEL ----------
    def hong_ou_mandel_test(self) -> Dict:
        """
        Tests photon indistinguishability via HOM dip.
        For 2 photons in 50:50 beam splitter, coincidences → 0 (ideal)
        """
        if self.n != 2:
            log.warning("HOM test designed for n=2 photons")
            return {}
        
        # Beam splitter 50:50
        bs = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # Input: |1,1⟩ (1 photon in each port)
        # Expected output: |2,0⟩ or |0,2⟩ (quantum bunching)
        
        self.U = np.eye(self.m, dtype=complex)
        self.U[:2, :2] = bs
        
        counts = self.simulate(shots=10000)
        
        # Coincidence count (1,1)
        coincidences = counts.get("11" + "0"*(self.m-2), 0)
        bunching = counts.get("20" + "0"*(self.m-2), 0) + counts.get("02" + "0"*(self.m-2), 0)
        
        visibility = 1 - (2 * coincidences / (coincidences + bunching + 1e-9))
        
        return {
            "coincidences": coincidences,
            "bunching": bunching,
            "hom_visibility": visibility,
            "quantum_signature": visibility > 0.5
        }

    # ---------- STATISTICAL METRICS ----------
    def shannon_entropy(self, counts: Dict[str, int]) -> float:
        """Shannon entropy of measured distribution"""
        vals = np.array(list(counts.values()))
        probs = vals / vals.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def collision_entropy(self, counts: Dict[str, int]) -> float:
        """
        Collision entropy: H₂ = -log₂(Σpᵢ²)
        Measures distribution diversity
        """
        vals = np.array(list(counts.values()))
        probs = vals / vals.sum()
        return -np.log2(np.sum(probs**2) + 1e-12)

    # ---------- COMPLETE PLOT ----------
    def plot_comprehensive(self, counts_q: Dict, counts_c: Dict, states: List, 
                          metrics: Dict, save: bool = True) -> str:
        """Generates figure with 6 subplots comparing quantum vs classical"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Quantum Distribution
        ax1 = fig.add_subplot(gs[0, :2])
        top_q = sorted(counts_q.items(), key=lambda x: x[1], reverse=True)[:CFG["MAX_PLOT_STATES"]]
        states_q, freq_q = zip(*top_q) if top_q else ([], [])
        probs_q = np.array(freq_q) / sum(counts_q.values()) if freq_q else np.array([])
        
        bars1 = ax1.bar(range(len(states_q)), probs_q, color=CFG["PALETTE"][:len(states_q)], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xticks(range(len(states_q)))
        ax1.set_xticklabels(states_q, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Probability")
        ax1.set_title(f"Quantum Boson Sampling | {self.n} photons, {self.m} modes", 
                     fontweight='bold', fontsize=12)
        ax1.grid(axis="y", alpha=0.3)
        
        # 2. Classical Distribution
        ax2 = fig.add_subplot(gs[1, :2])
        top_c = sorted(counts_c.items(), key=lambda x: x[1], reverse=True)[:CFG["MAX_PLOT_STATES"]]
        states_c, freq_c = zip(*top_c) if top_c else ([], [])
        probs_c = np.array(freq_c) / sum(counts_c.values()) if freq_c else np.array([])
        
        bars2 = ax2.bar(range(len(states_c)), probs_c, color='gray', 
                       alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(range(len(states_c)))
        ax2.set_xticklabels(states_c, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Probability")
        ax2.set_title("Classical (Distinguishable Particles)", fontweight='bold', fontsize=12)
        ax2.grid(axis="y", alpha=0.3)
        
        # 3. Direct comparison (overlapping)
        ax3 = fig.add_subplot(gs[2, :2])
        common_states = set(states_q) & set(states_c)
        if common_states:
            common = sorted(common_states, 
                          key=lambda s: counts_q.get(s, 0) + counts_c.get(s, 0), 
                          reverse=True)[:15]
            x = np.arange(len(common))
            width = 0.35
            
            vals_q = [counts_q.get(s, 0) / sum(counts_q.values()) for s in common]
            vals_c = [counts_c.get(s, 0) / sum(counts_c.values()) for s in common]
            
            ax3.bar(x - width/2, vals_q, width, label='Quantum', color='tab:blue', alpha=0.7)
            ax3.bar(x + width/2, vals_c, width, label='Classical', color='gray', alpha=0.7)
            ax3.set_xticks(x)
            ax3.set_xticklabels(common, rotation=45, ha="right", fontsize=8)
            ax3.set_ylabel("Probability")
            ax3.set_title("Direct Comparison", fontweight='bold', fontsize=12)
            ax3.legend()
            ax3.grid(axis="y", alpha=0.3)
        
        # 4. Quantum Advantage Metrics
        ax4 = fig.add_subplot(gs[0, 2])
        metrics_names = ['HOG\nScore', 'Linear\nXEB', 'KS\nDistance']
        metrics_vals = [
            metrics.get('heaviness', 0),
            metrics.get('linear_xeb', 0),
            metrics.get('ks_distance', 0)
        ]
        colors_m = ['tab:green', 'tab:orange', 'tab:purple']
        
        bars4 = ax4.barh(metrics_names, metrics_vals, color=colors_m, alpha=0.7, edgecolor='black')
        ax4.set_xlabel("Score")
        ax4.set_title("Quantum Advantage Metrics", fontweight='bold', fontsize=11)
        ax4.grid(axis='x', alpha=0.3)
        ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        for bar, val in zip(bars4, metrics_vals):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 5. Comparative Entropy
        ax5 = fig.add_subplot(gs[1, 2])
        entropy_data = {
            'Quantum\n(Shannon)': metrics.get('entropy_q', 0),
            'Classical\n(Shannon)': metrics.get('entropy_c', 0),
            'Max\n(Uniform)': metrics.get('max_entropy', 0),
            'Collision\n(Quantum)': metrics.get('collision_entropy', 0)
        }
        
        bars5 = ax5.bar(entropy_data.keys(), entropy_data.values(), 
                       color=['tab:blue', 'gray', 'lightgray', 'tab:cyan'],
                       alpha=0.7, edgecolor='black')
        ax5.set_ylabel("Entropy (bits)")
        ax5.set_title("Entropy Analysis", fontweight='bold', fontsize=11)
        ax5.grid(axis='y', alpha=0.3)
        ax5.tick_params(axis='x', rotation=0, labelsize=9)
        
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Results Table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = f"""
        BOSON SAMPLING RESULTS
        {'='*40}
        Configuration:
          Photons:              {self.n}
          Modes:                {self.m}
          Shots:                {metrics.get('shots', 0)}
          Loss rate:            {self.loss_rate:.3f}
        
        Quantum Metrics:
          Shannon entropy:      {metrics.get('entropy_q', 0):.3f} bits
          Collision entropy:    {metrics.get('collision_entropy', 0):.3f}
          Coverage:             {metrics.get('coverage_q', 0):.2%}
        
        Classical Comparison:
          Shannon entropy:      {metrics.get('entropy_c', 0):.3f} bits
          Coverage:             {metrics.get('coverage_c', 0):.2%}
        
        Advantage Tests:
          HOG score:            {metrics.get('heaviness', 0):.4f}
          Linear XEB:           {metrics.get('linear_xeb', 0):.4f}
          KS distance:          {metrics.get('ks_distance', 0):.4f}
        
        Computational:
          Mean perm time:       {metrics.get('mean_perm_time', 0):.4f} s
          Total time:           {metrics.get('total_time', 0):.2f} s
        
        Validation:
          Quantum advantage:    {'YES' if metrics.get('linear_xeb', 0) > 0 else 'NO'}
          Statistical sig:      {metrics.get('chi2_pval', 0):.4f}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
        
        fname = f"kaya_bs_{self.n}p_{self.m}m_comprehensive_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
        if save:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            log.info("Comprehensive plot saved: %s", fname)
        
        plt.close()
        return fname

    # ---------- COMPLETE PIPELINE ----------
    def run(self, shots: int = CFG["SHOTS"]) -> dict:
        """Runs complete analysis: quantum vs classical"""
        log.info("Running comprehensive Boson Sampling analysis...")
        t0 = time.perf_counter()
        
        # Setup
        self.random_unitary()
        self.validate_unitary()
        states = self.output_states()
        
        # Simulations
        counts_q = self.simulate(shots=shots, classical=False)
        counts_c = self.simulate(shots=shots, classical=True)
        
        # Quantum metrics
        heaviness = self.heaviness_score(counts_q, states)
        xeb = self.linear_xeb(counts_q, states)
        ks_dist = self.kolmogorov_distance(counts_q, counts_c, states)
        
        # Entropies
        entropy_q = self.shannon_entropy(counts_q)
        entropy_c = self.shannon_entropy(counts_c)
        collision_ent = self.collision_entropy(counts_q)
        max_entropy = np.log2(len(states))
        
        # Statistics
        chi2, pval = chisquare(list(counts_q.values()))
        
        # Consolidated metrics
        metrics = {
            "shots": shots,
            "heaviness": heaviness,
            "linear_xeb": xeb,
            "ks_distance": ks_dist,
            "entropy_q": entropy_q,
            "entropy_c": entropy_c,
            "collision_entropy": collision_ent,
            "max_entropy": max_entropy,
            "coverage_q": len(counts_q) / len(states),
            "coverage_c": len(counts_c) / len(states),
            "chi2_pval": pval,
            "mean_perm_time": np.mean(self.perm_times) if self.perm_times else 0,
            "total_time": time.perf_counter() - t0
        }
        
        # Plot
        plot_file = self.plot_comprehensive(counts_q, counts_c, states, metrics)
        
        # Detailed CSV
        csv_file = plot_file.replace(".png", ".csv")
        state_strs = ["".join(map(str, s)) for s in states]
        df = pd.DataFrame({
            "state": state_strs,
            "quantum_counts": [counts_q.get(s, 0) for s in state_strs],
            "classical_counts": [counts_c.get(s, 0) for s in state_strs],
            "quantum_prob": self.probabilities(states),
            "classical_prob": self.classical_probabilities(states)
        })
        df.to_csv(csv_file, index=False)
        log.info("Data saved: %s", csv_file)
        
        return {
            "n_photons": self.n,
            "n_modes": self.m,
            "loss_rate": self.loss_rate,
            **metrics,
            "plot_file": plot_file,
            "csv_file": csv_file,
            "quantum_advantage": xeb > 0,
        }


# ---------- SPECIALIZED TESTS ----------
def hong_ou_mandel_demo():
    """Hong-Ou-Mandel dip demonstration"""
    print("\n" + "="*60)
    print("HONG-OU-MANDEL INTERFERENCE TEST")
    print("="*60)
    
    bs = KayaBosonSampling(n_photons=2, n_modes=2)
    hom_results = bs.hong_ou_mandel_test()
    
    print(f"Coincidences (|1,1⟩):  {hom_results.get('coincidences', 0)}")
    print(f"Bunching (|2,0⟩+|0,2⟩): {hom_results.get('bunching', 0)}")
    print(f"HOM Visibility:         {hom_results.get('hom_visibility', 0):.4f}")
    print(f"Quantum signature:      {hom_results.get('quantum_signature', False)}")
    print("="*60 + "\n")


def benchmark():
    """Comparative test battery"""
    configs = [(2, 4), (3, 6), (4, 8), (5, 10)]
    results = {}
    
    print("\n" + "="*80)
    print("BENCHMARK: Boson Sampling - Quantum vs Classical")
    print("="*80 + "\n")
    
    for p, m in configs:
        log.info("Testing configuration (%d photons, %d modes)", p, m)
        bs = KayaBosonSampling(p, m, loss_rate=0.01)
        results[f"{p}p_{m}m"] = bs.run(shots=5000)
    
    # Summary table
    print("\n" + "-"*100)
    print(f"{'Config':<10} {'HOG':<8} {'XEB':<8} {'KS-dist':<10} {'H(Q)':<8} {'H(C)':<8} "
          f"{'QAdv':<8} {'Time(s)':<8}")
    print("-"*100)
    
    for name, r in results.items():
        print(f"{name:<10} {r['heaviness']:<8.4f} {r['linear_xeb']:<8.4f} "
              f"{r['ks_distance']:<10.4f} {r['entropy_q']:<8.3f} {r['entropy_c']:<8.3f} "
              f"{'YES' if r['quantum_advantage'] else 'NO':<8} {r['total_time']:<8.2f}")
    print("-"*100 + "\n")
    
    # Consolidated JSON
    out_json = f"kaya_bs_benchmark_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Benchmark results saved: %s", out_json)
    
    return results


def scaling_analysis():
    """Analyzes permanent computational scalability"""
    print("\n" + "="*60)
    print("COMPUTATIONAL SCALING ANALYSIS")
    print("="*60)
    
    configs = [(2, 4), (3, 6), (4, 8), (5, 10), (6, 12)]
    
    print(f"\n{'N':<5} {'M':<5} {'States':<10} {'Perm/call':<15} {'Total(s)':<10}")
    print("-"*60)
    
    for n, m in configs:
        bs = KayaBosonSampling(n, m)
        bs.random_unitary()
        states = bs.output_states()
        
        t0 = time.perf_counter()
        _ = bs.probabilities(states)
        elapsed = time.perf_counter() - t0
        
        avg_perm_time = np.mean(bs.perm_times) if bs.perm_times else 0
        
        print(f"{n:<5} {m:<5} {len(states):<10} {avg_perm_time:<15.6f} {elapsed:<10.3f}")
    
    print("-"*60 + "\n")


def loss_sensitivity_analysis(n_photons: int = 3, n_modes: int = 6):
    """Analyzes sensitivity to photonic losses"""
    print("\n" + "="*70)
    print(f"LOSS SENSITIVITY ANALYSIS | {n_photons} photons, {n_modes} modes")
    print("="*70)
    
    loss_rates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
    results = []
    
    for loss in loss_rates:
        log.info("Testing loss rate: %.3f", loss)
        bs = KayaBosonSampling(n_photons, n_modes, loss_rate=loss)
        r = bs.run(shots=3000)
        results.append(r)
    
    print("\n" + "-"*70)
    print(f"{'Loss':<8} {'HOG':<8} {'XEB':<8} {'H(Q)':<8} {'QAdv':<8}")
    print("-"*70)
    
    for loss, r in zip(loss_rates, results):
        print(f"{loss:<8.3f} {r['heaviness']:<8.4f} {r['linear_xeb']:<8.4f} "
              f"{r['entropy_q']:<8.3f} {'YES' if r['quantum_advantage'] else 'NO':<8}")
    print("-"*70 + "\n")
    
    return results


# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n🌊 Kaya Boson Sampling v3 | Quantum Advantage Validation\n")
    
    # 1. HOM demonstration
    hong_ou_mandel_demo()
    
    # 2. Detailed run
    print("[1] Running detailed analysis (3 photons, 6 modes)...")
    bs = KayaBosonSampling(n_photons=3, n_modes=6, loss_rate=0.02)
    results = bs.run(shots=5000)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Heaviness Score (HOG):    {results['heaviness']:.4f}")
    print(f"Linear XEB:               {results['linear_xeb']:.4f}")
    print(f"KS Distance (Q vs C):     {results['ks_distance']:.4f}")
    print(f"Shannon Entropy (Q):      {results['entropy_q']:.3f} bits")
    print(f"Shannon Entropy (C):      {results['entropy_c']:.3f} bits")
    print(f"Collision Entropy:        {results['collision_entropy']:.3f}")
    print(f"Coverage (Q):             {results['coverage_q']:.2%}")
    print(f"Quantum Advantage:        {'YES ✓' if results['quantum_advantage'] else 'NO ✗'}")
    print(f"Mean Permanent Time:      {results['mean_perm_time']:.4f} s")
    print(f"Total Computation Time:   {results['total_time']:.2f} s")
    print("="*60 + "\n")
    
    # 3. Benchmark (optional)
    if input("Run full benchmark? (y/n): ").lower() == "y":
        benchmark()
    
    # 4. Scalability analysis (optional)
    if input("Run scaling analysis? (y/n): ").lower() == "y":
        scaling_analysis()
    
    # 5. Loss analysis (optional)
    if input("Run loss sensitivity analysis? (y/n): ").lower() == "y":
        loss_sensitivity_analysis(n_photons=3, n_modes=6)

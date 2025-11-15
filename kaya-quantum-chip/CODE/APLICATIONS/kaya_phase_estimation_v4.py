# kaya_phase_estimation_v4.py
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
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("KayaPE")

plt.rcdefaults()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 11,
    "figure.figsize": (12, 8),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ---------- CLASSE PRINCIPAL ----------
class KayaPhaseEstimation:
    def __init__(self, n_photons: int = 2, loss_rate: float = 0.0, dephasing_rate: float = 0.0):
        """
        Estimação de fase quântica com estados NOON.
        
        Args:
            n_photons: Número de fotões no estado NOON
            loss_rate: Taxa de perda fotônica (0-1)
            dephasing_rate: Taxa de defasamento (0-1)
        """
        self.n = n_photons
        self.loss_rate = loss_rate
        self.dephasing_rate = dephasing_rate
        log.info("Kaya Phase Estimation | N=%d | loss=%.3f | dephasing=%.3f", 
                 self.n, loss_rate, dephasing_rate)

    # ---------- CIRCUITOS ----------
    def _noon_circuit(self) -> QuantumCircuit:
        """Prepara estado NOON aproximado com beam splitters"""
        qc = QuantumCircuit(2)
        qc.x(0)          # |10⟩
        qc.h(0); qc.h(1) # BS1 (simplificado)
        return qc

    def _apply_phase(self, qc: QuantumCircuit, phi: float) -> QuantumCircuit:
        """Aplica fase Nφ no braço superior"""
        qc.p(phi * self.n, 0)
        return qc

    def _apply_noise(self, prob: float) -> float:
        """Aplica modelo de ruído fenomenológico"""
        # Perda fotônica: reduz visibilidade
        visibility_loss = (1 - self.loss_rate) ** self.n
        # Defasamento: reduz contraste
        visibility_dephase = np.exp(-self.dephasing_rate * self.n)
        
        # Aplica à probabilidade
        noise_factor = visibility_loss * visibility_dephase
        return 0.5 + noise_factor * (prob - 0.5)

    def _recombine(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Segundo beam splitter"""
        qc.h(0); qc.h(1)
        return qc

    # ---------- PROBABILIDADE ----------
    def _detection_prob(self, state: Statevector) -> float:
        """Probabilidade de detecção no detector 1 (qubit 1 = |1⟩)"""
        return sum(abs(state.data[i])**2 for i in range(len(state.data)) if i & 1)

    def _simulate_measurement(self, prob: float, shots: int) -> tuple:
        """Simula medições com ruído binomial"""
        # Aplica ruído ao valor teórico
        noisy_prob = self._apply_noise(prob)
        # Simula shots independentes
        counts = np.random.binomial(shots, noisy_prob)
        measured_prob = counts / shots
        # Incerteza binomial
        uncertainty = np.sqrt(noisy_prob * (1 - noisy_prob) / shots)
        return measured_prob, uncertainty

    # ---------- MÉTRICAS QUÂNTICAS ----------
    def quantum_fisher_information(self) -> float:
        """
        QFI para estados NOON ideais: F_Q = N²
        Com perdas: F_Q ≈ N² × (1-loss)^N
        """
        ideal_qfi = self.n ** 2
        loss_factor = (1 - self.loss_rate) ** self.n
        return ideal_qfi * loss_factor

    def cramer_rao_bound(self, shots: int) -> float:
        """
        Limite de Cramér-Rao: Δφ_min = 1/√(M × F_Q)
        onde M = número de medições
        """
        qfi = self.quantum_fisher_information()
        return 1 / np.sqrt(shots * qfi)

    def shot_noise_limit(self, shots: int) -> float:
        """
        Limite clássico (SQL): Δφ_SQL = 1/√M
        """
        return 1 / np.sqrt(shots)

    def heisenberg_limit(self, shots: int) -> float:
        """
        Limite de Heisenberg: Δφ_HL = 1/(N × √M)
        """
        return 1 / (self.n * np.sqrt(shots))

    # ---------- SWEEP COM MEDIÇÕES REALISTAS ----------
    def phase_sweep(self, phases: np.ndarray, shots: int = 1024, reps: int = 10):
        """
        Varre fases com repetições independentes e ruído realista.
        
        Returns:
            q_data: (reps, n_phases) - medições quânticas
            c_data: (reps, n_phases) - medições clássicas (N=1)
        """
        log.info("Running phase sweep | %d phases × %d reps × %d shots", 
                 len(phases), reps, shots)
        
        q_data = np.zeros((reps, len(phases)))
        c_data = np.zeros((reps, len(phases)))
        q_uncertainty = np.zeros((reps, len(phases)))
        c_uncertainty = np.zeros((reps, len(phases)))

        for r in range(reps):
            for i, phi in enumerate(phases):
                # Quantum NOON
                qc = self._noon_circuit()
                qc = self._apply_phase(qc, phi)
                qc = self._recombine(qc)
                prob_q = self._detection_prob(Statevector.from_instruction(qc))
                q_data[r, i], q_uncertainty[r, i] = self._simulate_measurement(prob_q, shots)

                # Classical (N=1, sem enhancement)
                qc_c = QuantumCircuit(2)
                qc_c.x(0); qc_c.h(0); qc_c.h(1)
                qc_c.p(phi, 0)
                qc_c.h(0); qc_c.h(1)
                prob_c = self._detection_prob(Statevector.from_instruction(qc_c))
                
                # Aplica ruído equivalente ao caso clássico
                temp_loss = self.loss_rate
                temp_dephase = self.dephasing_rate
                self.loss_rate = temp_loss if temp_loss > 0 else 0
                self.dephasing_rate = temp_dephase if temp_dephase > 0 else 0
                temp_n = self.n
                self.n = 1  # Classical case
                c_data[r, i], c_uncertainty[r, i] = self._simulate_measurement(prob_c, shots)
                self.n = temp_n

        q_mean = q_data.mean(axis=0)
        q_std = q_data.std(axis=0)
        c_mean = c_data.mean(axis=0)
        c_std = c_data.std(axis=0)
        
        return q_mean, q_std, c_mean, c_std

    # ---------- ANÁLISE DE VISIBILIDADE ----------
    def analyze_visibility(self, phases: np.ndarray, probs: np.ndarray) -> dict:
        """
        Calcula visibilidade: V = (P_max - P_min)/(P_max + P_min)
        e ajusta curva sinusoidal
        """
        p_max = probs.max()
        p_min = probs.min()
        visibility = (p_max - p_min) / (p_max + p_min) if (p_max + p_min) > 0 else 0
        
        # Fit sinusoidal: P(φ) = A + B*cos(N*φ + φ0)
        def model(phi, A, B, phi0):
            return A + B * np.cos(self.n * phi + phi0)
        
        try:
            popt, pcov = curve_fit(model, phases, probs, 
                                   p0=[0.5, 0.5, 0], 
                                   bounds=([0, 0, -np.pi], [1, 1, np.pi]))
            perr = np.sqrt(np.diag(pcov))
            fit_quality = np.corrcoef(probs, model(phases, *popt))[0, 1] ** 2  # R²
        except:
            popt = [0.5, 0.5, 0]
            perr = [0, 0, 0]
            fit_quality = 0
        
        return {
            "visibility": visibility,
            "fit_params": popt,
            "fit_errors": perr,
            "r_squared": fit_quality
        }

    # ---------- PLOT COMPLETO ----------
    def plot_comprehensive(self, phases: np.ndarray, q_mean, q_std, c_mean, c_std, shots: int):
        """Gera figura com 4 subplots: comparação, visibilidade, limites, e resíduos"""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Comparação Quantum vs Classical
        ax1 = fig.add_subplot(gs[0, :])
        ax1.errorbar(phases, q_mean, yerr=q_std, fmt='o-', color='tab:blue',
                    capsize=3, label=f'Quantum N={self.n}', linewidth=1.5, markersize=5, alpha=0.7)
        ax1.errorbar(phases, c_mean, yerr=c_std, fmt='s-', color='tab:red',
                    capsize=3, label='Classical N=1', linewidth=1.5, markersize=5, alpha=0.7)
        
        # Curvas teóricas
        theory_q = 0.5 * (1 + np.cos(self.n * phases))
        theory_c = 0.5 * (1 + np.cos(phases))
        ax1.plot(phases, theory_q, '--', color='tab:blue', alpha=0.5, linewidth=2, label='Theory (Quantum)')
        ax1.plot(phases, theory_c, '--', color='tab:red', alpha=0.5, linewidth=2, label='Theory (Classical)')
        
        ax1.set_xlabel("Phase φ (rad)")
        ax1.set_ylabel("Detection probability")
        ax1.set_title(f"NOON-state phase estimation | N={self.n}, shots={shots}, loss={self.loss_rate:.3f}")
        ax1.legend(ncol=2)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # 2. Análise de visibilidade
        ax2 = fig.add_subplot(gs[1, 0])
        q_analysis = self.analyze_visibility(phases, q_mean)
        c_analysis = self.analyze_visibility(phases, c_mean)
        
        categories = ['Quantum\n(measured)', 'Classical\n(measured)', 'Quantum\n(ideal)', 'Classical\n(ideal)']
        visibilities = [q_analysis['visibility'], c_analysis['visibility'], 1.0, 1.0]
        colors_vis = ['tab:blue', 'tab:red', 'lightblue', 'lightcoral']
        
        bars = ax2.bar(categories, visibilities, color=colors_vis, alpha=0.7, edgecolor='black')
        ax2.set_ylabel("Visibility V")
        ax2.set_title("Contrast Analysis")
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal')
        ax2.grid(axis='y', alpha=0.3)
        
        # Adiciona valores sobre as barras
        for bar, val in zip(bars, visibilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Comparação de limites de incerteza
        ax3 = fig.add_subplot(gs[1, 1])
        crb = self.cramer_rao_bound(shots)
        sql = self.shot_noise_limit(shots)
        hl = self.heisenberg_limit(shots)
        qfi = self.quantum_fisher_information()
        
        limits = ['Cramér-Rao\n(achieved)', 'Shot Noise\n(classical)', 'Heisenberg\n(ideal)']
        values = [crb, sql, hl]
        colors_lim = ['tab:green', 'tab:orange', 'tab:purple']
        
        bars2 = ax3.bar(limits, values, color=colors_lim, alpha=0.7, edgecolor='black')
        ax3.set_ylabel("Δφ (rad)")
        ax3.set_title(f"Uncertainty Bounds | QFI={qfi:.2f}")
        ax3.set_yscale('log')
        ax3.grid(axis='y', alpha=0.3, which='both')
        
        for bar, val in zip(bars2, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Resíduos
        ax4 = fig.add_subplot(gs[2, 0])
        theory_q_vals = 0.5 * (1 + np.cos(self.n * phases))
        residuals_q = q_mean - theory_q_vals
        ax4.scatter(phases, residuals_q, color='tab:blue', alpha=0.6, s=30, label='Quantum')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.fill_between(phases, -q_std, q_std, color='tab:blue', alpha=0.2)
        ax4.set_xlabel("Phase φ (rad)")
        ax4.set_ylabel("Residuals")
        ax4.set_title("Model Fit Quality")
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Tabela de métricas
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        metrics_text = f"""
        QUANTUM METRICS (N={self.n})
        {'='*35}
        Visibility:           {q_analysis['visibility']:.4f}
        R² (fit quality):     {q_analysis['r_squared']:.4f}
        QFI:                  {qfi:.2f}
        Cramér-Rao (Δφ):      {crb:.5f} rad
        
        CLASSICAL COMPARISON (N=1)
        {'='*35}
        Visibility:           {c_analysis['visibility']:.4f}
        Shot Noise Limit:     {sql:.5f} rad
        
        QUANTUM ADVANTAGE
        {'='*35}
        Factor (SQL/CRB):     {sql/crb:.2f}×
        Heisenberg scaling:   1/N = {1/self.n:.4f}
        
        NOISE PARAMETERS
        {'='*35}
        Photon loss:          {self.loss_rate:.4f}
        Dephasing:            {self.dephasing_rate:.4f}
        Shots per point:      {shots}
        """
        
        ax5.text(0.1, 0.95, metrics_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fname = f"kaya_pe_N{self.n}_comprehensive_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        log.info("Comprehensive figure saved: %s", fname)
        return fname

    # ---------- PIPELINE COMPLETO ----------
    def run(self, n_points: int = 50, shots: int = 1024, reps: int = 10):
        """Executa análise completa"""
        phases = np.linspace(0, 2*np.pi, n_points)
        q_mean, q_std, c_mean, c_std = self.phase_sweep(phases, shots=shots, reps=reps)
        fig = self.plot_comprehensive(phases, q_mean, q_std, c_mean, c_std, shots)
        
        # Análises
        q_analysis = self.analyze_visibility(phases, q_mean)
        c_analysis = self.analyze_visibility(phases, c_mean)
        qfi = self.quantum_fisher_information()
        crb = self.cramer_rao_bound(shots)
        sql = self.shot_noise_limit(shots)
        hl = self.heisenberg_limit(shots)
        
        # CSV detalhado
        csv = fig.replace(".png", ".csv")
        theory_q = 0.5 * (1 + np.cos(self.n * phases))
        theory_c = 0.5 * (1 + np.cos(phases))
        
        df = pd.DataFrame({
            "phase_rad": phases,
            "quantum_mean": q_mean,
            "quantum_std": q_std,
            "quantum_theory": theory_q,
            "classical_mean": c_mean,
            "classical_std": c_std,
            "classical_theory": theory_c,
        })
        df.to_csv(csv, index=False)
        log.info("Data saved: %s", csv)
        
        return {
            "n_photons": self.n,
            "shots": shots,
            "reps": reps,
            "quantum_visibility": q_analysis['visibility'],
            "classical_visibility": c_analysis['visibility'],
            "quantum_fisher_info": qfi,
            "cramer_rao_bound": crb,
            "shot_noise_limit": sql,
            "heisenberg_limit": hl,
            "quantum_advantage": sql / crb,
            "fit_r_squared": q_analysis['r_squared'],
            "figure": fig,
            "csv": csv,
        }


# ---------- BATERIA DE TESTES ----------
def benchmark(shots: int = 2048, loss_rate: float = 0.0):
    """Compara diferentes valores de N"""
    configs = [2, 3, 4, 5]
    results = {}
    
    print("\n" + "="*70)
    print("BENCHMARK: NOON-state Phase Estimation")
    print("="*70)
    
    for n in configs:
        log.info("Benchmark N=%d", n)
        results[n] = KayaPhaseEstimation(n, loss_rate=loss_rate).run(
            n_points=60, shots=shots, reps=10
        )
    
    print("\n" + "-"*70)
    print(f"{'N':<4} {'Vis_Q':<8} {'QFI':<8} {'CRB':<10} {'SQL':<10} {'Adv':<8}")
    print("-"*70)
    for n, r in results.items():
        print(f"{n:<4} {r['quantum_visibility']:<8.4f} {r['quantum_fisher_info']:<8.2f} "
              f"{r['cramer_rao_bound']:<10.6f} {r['shot_noise_limit']:<10.6f} "
              f"{r['quantum_advantage']:<8.2f}×")
    print("-"*70)
    
    return results


# ---------- ANÁLISE DE RUÍDO ----------
def noise_analysis(n_photons: int = 3, shots: int = 1024):
    """Analisa efeito de perdas fotônicas"""
    loss_rates = [0.0, 0.01, 0.05, 0.1, 0.2]
    results = []
    
    print("\n" + "="*70)
    print(f"NOISE ANALYSIS: N={n_photons} photons")
    print("="*70)
    
    for loss in loss_rates:
        log.info("Loss rate = %.3f", loss)
        r = KayaPhaseEstimation(n_photons, loss_rate=loss).run(
            n_points=40, shots=shots, reps=5
        )
        results.append(r)
    
    print("\n" + "-"*70)
    print(f"{'Loss':<8} {'Vis':<8} {'QFI':<8} {'CRB':<10} {'Adv':<8}")
    print("-"*70)
    for loss, r in zip(loss_rates, results):
        print(f"{loss:<8.3f} {r['quantum_visibility']:<8.4f} {r['quantum_fisher_info']:<8.2f} "
              f"{r['cramer_rao_bound']:<10.6f} {r['quantum_advantage']:<8.2f}×")
    print("-"*70)
    
    return results


# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n🌊 Kaya Phase-Estimation v3 | Scientific Analysis\n")
    
    # Single run detalhado
    print("[1] Running detailed analysis (N=3)...")
    r = KayaPhaseEstimation(n_photons=3, loss_rate=0.02, dephasing_rate=0.01).run(
        n_points=60, shots=2048, reps=15
    )
    
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Quantum Visibility:    {r['quantum_visibility']:.4f}")
    print(f"Classical Visibility:  {r['classical_visibility']:.4f}")
    print(f"QFI:                   {r['quantum_fisher_info']:.2f}")
    print(f"Cramér-Rao Bound:      {r['cramer_rao_bound']:.6f} rad")
    print(f"Shot Noise Limit:      {r['shot_noise_limit']:.6f} rad")
    print(f"Heisenberg Limit:      {r['heisenberg_limit']:.6f} rad")
    print(f"Quantum Advantage:     {r['quantum_advantage']:.2f}×")
    print(f"Fit Quality (R²):      {r['fit_r_squared']:.4f}")
    print(f"{'='*50}\n")
    
    # Benchmark opcional
    if input("Executar benchmark completo? (y/n): ").lower() == "y":
        benchmark(shots=2048, loss_rate=0.02)
    
    # Análise de ruído opcional
    if input("Executar análise de ruído? (y/n): ").lower() == "y":
        noise_analysis(n_photons=3, shots=2048)

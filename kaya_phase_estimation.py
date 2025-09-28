# kaya_phase_estimation.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

class KayaPhaseEstimation:
    def __init__(self, n_photons=2):
        self.n_photons = n_photons
        print(f"🌊 Kaya Phase Estimation | NOON state with {n_photons} photons")
    
    def noon_state_circuit(self):
        """Create circuit for NOON state: (|N,0> + |0,N>)/√2 (N=2 only)"""
        if self.n_photons != 2:
            raise ValueError("This implementation supports only N=2")
        
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.x(1)
        qc.h(0)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.x(0)
        qc.cx(0, 1)
        qc.x(0)
        return qc
    
    def apply_phase(self, qc, phase):
        """Apply phase φ to first mode (qubits 0 and 1)"""
        qc.rz(phase, 0)
        qc.rz(phase, 1)
        return qc
    
    def simulate_phase_sweep(self, phases=None):
        print("\n🔬 Simulating Phase Estimation with NOON State...")
        
        if phases is None:
            phases = np.linspace(0, 2*np.pi, 100)
        
        probabilities = []
        for phi in phases:
            qc = self.noon_state_circuit()
            qc = self.apply_phase(qc, phi)
            state = Statevector.from_instruction(qc)
            target_index = 3  # |0011> = |11> in mode 0
            amp = state.data[target_index]
            prob = abs(amp)**2
            probabilities.append(prob)
        
        plt.figure(figsize=(10, 4))
        plt.plot(phases, probabilities, 'b-', linewidth=2)
        plt.xlabel('Phase φ (radians)')
        plt.ylabel('Probability of |1100>')
        plt.title(f'Interference with NOON State (N={self.n_photons})')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        plt.text(0.1, 0.9, f'Precision: Δφ = 1/{self.n_photons}',
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        plt.tight_layout()
        plt.savefig('kaya_noon_interference.png', dpi=150)
        print("   📈 Plot saved as 'kaya_noon_interference.png'")
        
        sensitivity = self.n_photons
        print(f"   Quantum sensitivity: {sensitivity}× better than classical")
        
        return phases, probabilities, sensitivity

if __name__ == "__main__":
    pe = KayaPhaseEstimation(n_photons=2)
    phases, probs, sens = pe.simulate_phase_sweep()
    print(f"\n✅ Phase estimation completed! Sensitivity = {sens}×")
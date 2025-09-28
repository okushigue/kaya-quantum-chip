# kaya_boson_sampling.py
import numpy as np
from scipy.linalg import qr
import math
from itertools import permutations, combinations_with_replacement

class KayaBosonSampling:
    def __init__(self, n_photons=3, n_modes=6):
        self.n_photons = n_photons
        self.n_modes = n_modes
        print(f"🔬 Kaya Boson Sampling | {n_photons} photons in {n_modes} modes")
    
    def random_unitary(self):
        """Generate random unitary matrix (photonic interferometer)"""
        X = np.random.randn(self.n_modes, self.n_modes) + 1j * np.random.randn(self.n_modes, self.n_modes)
        Q, R = qr(X)
        d = np.diag(R)
        ph = d / np.abs(d)
        Q = Q @ np.diag(ph)
        return Q
    
    def permanent(self, matrix):
        """Calculate matrix permanent (exact for small matrices)"""
        n = matrix.shape[0]
        if n == 0:
            return 1
        perm_sum = 0
        for p in permutations(range(n)):
            prod = 1
            for i in range(n):
                prod *= matrix[i, p[i]]
            perm_sum += prod
        return perm_sum
    
    def simulate(self, shots=5000):
        """Simulate boson sampling using classical permanent calculation"""
        print("\n🚀 Simulating Boson Sampling...")
        
        U = self.random_unitary()
        print(f"   Interferometer unitary generated ({self.n_modes}x{self.n_modes})")
        
        output_states = []
        for comb in combinations_with_replacement(range(self.n_modes), self.n_photons):
            state = [0]*self.n_modes
            for mode in comb:
                state[mode] += 1
            output_states.append(tuple(state))
        
        print(f"   Total possible output states: {len(output_states)}")
        
        probabilities = []
        for state in output_states:
            if sum(state) != self.n_photons:
                probabilities.append(0.0)
                continue
                
            cols = []
            for mode_idx, count in enumerate(state):
                cols.extend([mode_idx] * count)
            
            if len(cols) != self.n_photons:
                probabilities.append(0.0)
                continue
                
            subU = U[:self.n_photons, :][:, cols]
            perm = self.permanent(subU)
            norm_factor = 1.0
            for occupation in state:
                norm_factor *= math.factorial(occupation)
                
            prob = abs(perm)**2 / norm_factor
            probabilities.append(prob)
        
        total_prob = sum(probabilities)
        if total_prob == 0:
            print("   ❌ Error: Total probability zero!")
            return {}, 0.0
            
        probabilities = [p / total_prob for p in probabilities]
        
        sampled_indices = np.random.choice(len(output_states), size=shots, p=probabilities)
        counts = {}
        for idx in sampled_indices:
            state_str = ''.join(f'{s}' for s in output_states[idx])
            counts[state_str] = counts.get(state_str, 0) + 1
        
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("   Top 5 results:")
        for state, count in sorted_counts:
            print(f"     {state}: {count} ({count/shots:.1%})")
        
        probs = np.array(list(counts.values())) / shots
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        print(f"   Sampling entropy: {entropy:.2f} bits")
        
        return counts, entropy

if __name__ == "__main__":
    bs = KayaBosonSampling(n_photons=3, n_modes=6)
    counts, entropy = bs.simulate(shots=2000)
    print("\n✅ Boson Sampling completed!")
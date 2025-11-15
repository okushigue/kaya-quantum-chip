"""
Kaya Quantum Key Distribution - BB84 Protocol Enhanced Implementation
======================================================================
Realistic simulation of BB84 QKD protocol with:
- Channel loss and detector efficiency
- Multi-photon noise and dark counts
- Eavesdropping detection
- Privacy amplification
- Statistical analysis

Author: Jefferson M. Okushigue
Version: 2.0 (Enhanced)
"""

import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import hashlib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class QKDParameters:
    """BB84 QKD system parameters based on realistic implementations."""
    n_bits: int = 256
    wavelength_nm: float = 1550.0  # Telecom C-band
    
    # Channel parameters
    channel_loss_db_km: float = 0.2  # Fiber attenuation
    fiber_length_km: float = 50.0
    
    # Detector parameters
    detector_efficiency: float = 0.85  # SNSPD typical
    dark_count_rate: float = 1e-6  # Hz (very low for SNSPDs)
    timing_jitter_ps: float = 50.0  # Detector timing resolution
    
    # Noise sources
    state_preparation_error: float = 0.01  # 1% imperfection
    basis_misalignment_error: float = 0.015  # 1.5% angular error
    detector_readout_error: float = 0.02  # 2% POVM error
    
    # Security parameters
    qber_threshold: float = 0.11  # 11% - theoretical limit
    privacy_amplification: bool = True
    error_correction: bool = True
    
    @property
    def total_channel_loss(self) -> float:
        """Calculate total channel transmission."""
        loss_db = self.channel_loss_db_km * self.fiber_length_km
        return 10 ** (-loss_db / 10)
    
    @property
    def expected_detection_rate(self) -> float:
        """Expected photon detection rate."""
        return self.total_channel_loss * self.detector_efficiency

class BB84Protocol:
    """
    Enhanced BB84 Quantum Key Distribution protocol simulator.
    
    Implements the complete BB84 workflow:
    1. State preparation (Alice)
    2. Quantum channel transmission
    3. Measurement (Bob)
    4. Classical sifting
    5. Error estimation (QBER)
    6. Privacy amplification
    7. Security analysis
    """
    
    def __init__(self, params: Optional[QKDParameters] = None):
        self.params = params or QKDParameters()
        self.simulator = AerSimulator()
        self.results = {}
        
        # Statistics tracking
        self.transmission_stats = {
            'prepared': 0,
            'detected': 0,
            'sifted': 0,
            'errors': 0
        }
        
        self._print_configuration()
    
    def _print_configuration(self):
        """Display system configuration."""
        print("\n" + "="*70)
        print("🔐 KAYA BB84 QKD PROTOCOL - Enhanced Simulation")
        print("="*70)
        print(f"  Wavelength: {self.params.wavelength_nm} nm")
        print(f"  Fiber length: {self.params.fiber_length_km} km")
        print(f"  Channel loss: {self.params.channel_loss_db_km} dB/km")
        print(f"  Total transmission: {self.params.total_channel_loss*100:.2f}%")
        print(f"  Detector efficiency: {self.params.detector_efficiency*100:.1f}%")
        print(f"  Expected detection: {self.params.expected_detection_rate*100:.2f}%")
        print(f"  QBER threshold: {self.params.qber_threshold*100:.1f}%")
        print("="*70 + "\n")
    
    def _prepare_quantum_state(self, bit: int, basis: int) -> QuantumCircuit:
        """
        Prepare quantum state for BB84 protocol.
        
        Args:
            bit: 0 or 1 (information bit)
            basis: 0 (Z-basis) or 1 (X-basis)
        
        Returns:
            QuantumCircuit with prepared state
        """
        qc = QuantumCircuit(1, 1)
        
        # Encode bit in computational basis
        if bit == 1:
            qc.x(0)
        
        # Apply basis choice (Hadamard for X-basis)
        if basis == 1:
            qc.h(0)
        
        # State preparation errors (realistic imperfections)
        if random.random() < self.params.state_preparation_error:
            # Small rotation error
            error_angle = random.gauss(0, 0.05)
            qc.ry(error_angle, 0)
        
        return qc
    
    def _apply_channel_noise(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Apply realistic quantum channel noise.
        
        Models:
        - Polarization rotation
        - Phase noise
        - Birefringence
        """
        # Phase noise from fiber birefringence
        if random.random() < 0.02:
            phase_noise = random.gauss(0, 0.1)
            qc.rz(phase_noise, 0)
        
        # Basis misalignment (imperfect PBS/interferometers)
        if random.random() < self.params.basis_misalignment_error:
            misalignment = random.gauss(0, 0.03)
            qc.ry(misalignment, 0)
        
        return qc
    
    def _measure_quantum_state(self, qc: QuantumCircuit, basis: int) -> int:
        """
        Measure quantum state in specified basis.
        
        Args:
            qc: Quantum circuit with prepared state
            basis: Measurement basis (0=Z, 1=X)
        
        Returns:
            Measurement outcome (0 or 1)
        """
        # Apply measurement basis
        if basis == 1:
            qc.h(0)
        
        # Detector readout error
        if random.random() < self.params.detector_readout_error:
            qc.x(0)  # Flip result
        
        qc.measure(0, 0)
        
        # Execute measurement
        job = self.simulator.run(qc, shots=1, memory=True)
        result = job.result().get_memory()[0]
        
        return int(result)
    
    def _apply_detection_effects(self) -> Optional[bool]:
        """
        Simulate photon detection with realistic effects.
        
        Returns:
            True if photon detected, False if lost, None for dark count
        """
        # Channel loss
        if random.random() > self.params.total_channel_loss:
            return False
        
        # Detector efficiency
        if random.random() > self.params.detector_efficiency:
            return False
        
        # Dark count (false detection)
        if random.random() < self.params.dark_count_rate:
            return None  # Signal dark count
        
        return True
    
    def run_protocol(self) -> Dict:
        """
        Execute complete BB84 protocol.
        
        Returns:
            Dictionary with protocol results and statistics
        """
        print("📡 Executing BB84 Quantum Key Distribution...")
        
        n_bits = self.params.n_bits
        
        # Step 1: Alice prepares random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(n_bits)]
        
        # Bob chooses random measurement bases
        bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
        
        # Step 2: Quantum transmission
        print("  🔄 Quantum transmission phase...")
        
        raw_key_alice = []
        raw_key_bob = []
        detected_indices = []
        dark_counts = 0
        
        for i in range(n_bits):
            # Alice prepares state
            qc = self._prepare_quantum_state(alice_bits[i], alice_bases[i])
            
            # Channel noise
            qc = self._apply_channel_noise(qc)
            
            # Detection
            detection = self._apply_detection_effects()
            
            if detection is None:
                # Dark count - random result
                dark_counts += 1
                bob_bit = random.randint(0, 1)
                raw_key_alice.append(alice_bits[i])
                raw_key_bob.append(bob_bit)
                detected_indices.append(i)
            elif detection:
                # Successful detection
                bob_bit = self._measure_quantum_state(qc.copy(), bob_bases[i])
                raw_key_alice.append(alice_bits[i])
                raw_key_bob.append(bob_bit)
                detected_indices.append(i)
            # else: photon lost (no detection)
        
        self.transmission_stats['prepared'] = n_bits
        self.transmission_stats['detected'] = len(raw_key_alice)
        
        detection_rate = len(raw_key_alice) / n_bits
        
        print(f"  ✓ Photons prepared: {n_bits}")
        print(f"  ✓ Photons detected: {len(raw_key_alice)} ({detection_rate:.1%})")
        print(f"  ✓ Dark counts: {dark_counts}")
        
        # Step 3: Sifting (basis reconciliation)
        print("\n  🔍 Sifting phase (basis reconciliation)...")
        
        sifted_alice = []
        sifted_bob = []
        
        for i in range(len(raw_key_alice)):
            idx = detected_indices[i]
            if alice_bases[idx] == bob_bases[idx]:
                sifted_alice.append(raw_key_alice[i])
                sifted_bob.append(raw_key_bob[i])
        
        self.transmission_stats['sifted'] = len(sifted_alice)
        sifting_efficiency = len(sifted_alice) / len(raw_key_alice) if raw_key_alice else 0
        
        print(f"  ✓ Sifted key length: {len(sifted_alice)} bits")
        print(f"  ✓ Sifting efficiency: {sifting_efficiency:.1%}")
        
        if len(sifted_alice) == 0:
            return {'error': 'No bits survived sifting', 'secure': False}
        
        # Step 4: Error estimation (QBER)
        print("\n  🛡️  Security analysis...")
        
        errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
        qber = errors / len(sifted_alice)
        
        self.transmission_stats['errors'] = errors
        
        # Security assessment
        secure, security_level = self._assess_security(qber)
        
        print(f"  ✓ Bit errors: {errors}/{len(sifted_alice)}")
        print(f"  ✓ QBER: {qber*100:.3f}%")
        print(f"  ✓ Security: {security_level}")
        
        # Step 5: Privacy amplification
        if secure and self.params.privacy_amplification:
            final_key = self._privacy_amplification(sifted_alice, qber)
            key_rate = self._calculate_key_rate(len(sifted_alice), len(final_key), n_bits)
        else:
            final_key = ''
            key_rate = 0.0
        
        # Step 6: Results compilation
        self.results = {
            'n_bits_prepared': n_bits,
            'n_bits_detected': len(raw_key_alice),
            'detection_rate': detection_rate,
            'n_bits_sifted': len(sifted_alice),
            'sifting_efficiency': sifting_efficiency,
            'errors': errors,
            'qber': qber,
            'secure': secure,
            'security_level': security_level,
            'final_key_length': len(final_key),
            'final_key': final_key[:128] if final_key else '',  # First 128 bits
            'key_rate_bits_per_photon': key_rate,
            'dark_counts': dark_counts,
            'parameters': {
                'fiber_length_km': self.params.fiber_length_km,
                'channel_loss_db': self.params.channel_loss_db_km * self.params.fiber_length_km,
                'detector_efficiency': self.params.detector_efficiency
            }
        }
        
        self._print_key_summary()
        
        return self.results
    
    def _assess_security(self, qber: float) -> Tuple[bool, str]:
        """
        Assess security based on QBER.
        
        Security levels based on information-theoretic analysis:
        - QBER < 2%: Highly secure (no eavesdropping)
        - QBER < 5%: Secure (acceptable noise)
        - QBER < 8%: Marginally secure (high noise)
        - QBER < 11%: Borderline (approaching theoretical limit)
        - QBER ≥ 11%: Insecure (possible eavesdropping)
        """
        if qber >= self.params.qber_threshold:
            return False, "❌ INSECURE - QBER exceeds threshold"
        elif qber < 0.02:
            return True, "✅ HIGHLY SECURE (QBER < 2%)"
        elif qber < 0.05:
            return True, "✅ SECURE (QBER < 5%)"
        elif qber < 0.08:
            return True, "⚠️  MARGINALLY SECURE (QBER < 8%)"
        else:
            return True, "🚨 BORDERLINE (QBER < 11%)"
    
    def _privacy_amplification(self, sifted_key: List[int], qber: float) -> str:
        """
        Apply privacy amplification using universal hash functions.
        
        Uses SHA-256 to compress the key and remove Eve's information.
        Output length based on QBER and information-theoretic security.
        """
        # Convert to bit string
        key_str = ''.join(map(str, sifted_key))
        
        # Apply SHA-256 hash
        hashed = hashlib.sha256(key_str.encode()).hexdigest()
        
        # Calculate secure key length based on QBER
        # Using simplified formula: r = 1 - 2*h(QBER)
        # where h(x) is binary entropy
        if qber > 0 and qber < 0.5:
            h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
            key_rate = max(0, 1 - 2 * h_qber)
        else:
            key_rate = 0
        
        # Determine output length (bits)
        target_length = int(len(sifted_key) * key_rate)
        target_length = min(target_length, 256)  # Cap at 256 bits
        
        # Convert hash to binary and truncate
        binary_key = bin(int(hashed, 16))[2:].zfill(256)
        
        return binary_key[:target_length] if target_length > 0 else ''
    
    def _calculate_key_rate(self, sifted_bits: int, final_bits: int, 
                           prepared_bits: int) -> float:
        """Calculate secure key rate (bits per prepared photon)."""
        return final_bits / prepared_bits if prepared_bits > 0 else 0.0
    
    def _print_key_summary(self):
        """Print summary of generated key."""
        if not self.results.get('final_key'):
            return
        
        key = self.results['final_key']
        key_length = self.results['final_key_length']
        
        print(f"\n  🔑 Secure Key Generated:")
        print(f"  ✓ Key length: {key_length} bits")
        print(f"  ✓ Key rate: {self.results['key_rate_bits_per_photon']:.4f} bits/photon")
        
        if len(key) >= 32:
            print(f"  ✓ Key preview: {key[:32]}...{key[-32:]}")
        else:
            print(f"  ✓ Key: {key}")
    
    def simulate_eavesdropping(self, intercept_prob: float = 0.25) -> Dict:
        """
        Simulate intercept-resend eavesdropping attack.
        
        CRITICAL FIX: The attack needs to properly track which photons were 
        intercepted and create errors when Alice/Bob/Eve bases don't align.
        
        Theory: When Eve intercepts with probability p_eve:
        - If Alice, Bob, Eve all use same basis → no error
        - If Alice/Bob match but Eve differs → 50% error rate
        - Expected QBER ≈ p_eve * 0.25 (averaged over all basis combinations)
        
        Args:
            intercept_prob: Probability Eve intercepts each photon
        
        Returns:
            Results showing induced QBER and detection
        """
        print(f"\n🕵️  Simulating Eavesdropping Attack (p_intercept={intercept_prob:.0%})...")
        
        n_bits = self.params.n_bits
        
        alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(n_bits)]
        bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
        
        # Track Eve's actions
        eve_intercepted = [False] * n_bits
        eve_bases = [None] * n_bits
        eve_bits = [None] * n_bits
        
        raw_key_alice = []
        raw_key_bob = []
        detected_indices = []
        intercepts = 0
        
        for i in range(n_bits):
            qc = None
            
            # Eve's intercept-resend attack
            if random.random() < intercept_prob:
                intercepts += 1
                eve_intercepted[i] = True
                
                # Eve measures in random basis
                eve_bases[i] = random.randint(0, 1)
                
                # Eve intercepts and measures
                qc_eve = self._prepare_quantum_state(alice_bits[i], alice_bases[i])
                # CRITICAL: Don't apply channel noise to Eve's measurement
                # (she's at Alice's end, before channel)
                eve_bits[i] = self._measure_quantum_state(qc_eve, eve_bases[i])
                
                # Eve resends the state she measured, in HER basis
                # This is where the attack introduces errors
                qc = self._prepare_quantum_state(eve_bits[i], eve_bases[i])
                
                # NOW apply channel effects (Eve to Bob)
                qc = self._apply_channel_noise(qc)
            else:
                # No interception - normal transmission
                qc = self._prepare_quantum_state(alice_bits[i], alice_bases[i])
                qc = self._apply_channel_noise(qc)
            
            # Bob's detection
            detection = self._apply_detection_effects()
            
            if detection:
                # Bob measures in his chosen basis
                bob_bit = self._measure_quantum_state(qc, bob_bases[i])
                raw_key_alice.append(alice_bits[i])
                raw_key_bob.append(bob_bit)
                detected_indices.append(i)
        
        # Sifting - only keep bits where Alice and Bob used same basis
        sifted_alice = []
        sifted_bob = []
        sifted_eve_intercepted = []
        sifted_eve_bases = []
        
        for i in range(len(raw_key_alice)):
            idx = detected_indices[i]
            if alice_bases[idx] == bob_bases[idx]:
                sifted_alice.append(raw_key_alice[i])
                sifted_bob.append(raw_key_bob[i])
                sifted_eve_intercepted.append(eve_intercepted[idx])
                sifted_eve_bases.append(eve_bases[idx])
        
        if len(sifted_alice) > 0:
            errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
            induced_qber = errors / len(sifted_alice)
            
            # Analyze errors in detail
            errors_from_eve = 0
            intercepted_count = sum(sifted_eve_intercepted)
            
            for j in range(len(sifted_alice)):
                if sifted_alice[j] != sifted_bob[j] and sifted_eve_intercepted[j]:
                    errors_from_eve += 1
            
            secure, level = self._assess_security(induced_qber)
            
            print(f"  ✓ Intercepts: {intercepts}/{n_bits} ({intercepts/n_bits:.1%})")
            print(f"  ✓ Detected photons: {len(raw_key_alice)}")
            print(f"  ✓ Sifted bits: {len(sifted_alice)}")
            print(f"  ✓ Intercepted (in sifted): {intercepted_count}/{len(sifted_alice)} ({intercepted_count/len(sifted_alice)*100:.1f}%)")
            print(f"  ✓ Total errors: {errors}/{len(sifted_alice)}")
            print(f"  ✓ Errors from Eve: {errors_from_eve}")
            print(f"  ✓ Induced QBER: {induced_qber*100:.3f}%")
            print(f"  ✓ Security status: {level}")
            
            # Theoretical QBER from intercept-resend
            # When Eve intercepts with prob p, and Alice/Bob match bases:
            # - 50% of time Eve's basis matches → no error
            # - 50% of time Eve's basis differs → 25% error (on average)
            # Total expected QBER ≈ p * 0.125 in sifted bits
            theoretical_qber = intercept_prob * 0.125
            print(f"  ✓ Expected QBER (theory): {theoretical_qber*100:.3f}%")
            
            # More detailed analysis
            print(f"\n  📊 Attack Analysis:")
            if errors_from_eve > 0:
                print(f"    ✓ Attack DETECTED: {errors_from_eve} errors attributable to Eve")
            else:
                print(f"    ⚠️  No errors from interception (statistical fluctuation or small sample)")
            
            if induced_qber > theoretical_qber * 2:
                print(f"    🚨 QBER higher than expected - possible eavesdropping!")
            elif induced_qber < theoretical_qber * 0.5:
                print(f"    ⚠️  QBER lower than expected - small sample size")
            else:
                print(f"    ✓ QBER consistent with {intercept_prob*100:.0f}% interception")
            
            return {
                'intercept_probability': intercept_prob,
                'intercepts_total': intercepts,
                'intercepts_sifted': intercepted_count,
                'total_errors': errors,
                'errors_from_eve': errors_from_eve,
                'induced_qber': induced_qber,
                'theoretical_qber': theoretical_qber,
                'detected': induced_qber > 0.05,  # Practical threshold
                'security_level': level,
                'sifted_bits': len(sifted_alice)
            }
        
        return {'error': 'No successful detections'}
    
    def run_statistical_analysis(self, n_trials: int = 10) -> Dict:
        """
        Run multiple trials for statistical analysis.
        
        Args:
            n_trials: Number of independent protocol runs
        
        Returns:
            Statistical summary of all trials
        """
        print(f"\n📊 Running Statistical Analysis ({n_trials} trials)...")
        
        results_list = []
        qber_values = []
        key_rates = []
        detection_rates = []
        secure_count = 0
        
        for trial in range(n_trials):
            result = self.run_protocol()
            
            if 'qber' in result:
                results_list.append(result)
                qber_values.append(result['qber'])
                key_rates.append(result['key_rate_bits_per_photon'])
                detection_rates.append(result['detection_rate'])
                
                if result['secure']:
                    secure_count += 1
        
        if not results_list:
            return {'error': 'No successful trials'}
        
        stats = {
            'n_trials': n_trials,
            'success_rate': secure_count / n_trials,
            'avg_qber': np.mean(qber_values),
            'std_qber': np.std(qber_values),
            'avg_key_rate': np.mean(key_rates),
            'std_key_rate': np.std(key_rates),
            'avg_detection_rate': np.mean(detection_rates),
            'std_detection_rate': np.std(detection_rates),
            'detailed_results': results_list
        }
        
        print(f"\n  📈 Statistical Summary:")
        print(f"  ✓ Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  ✓ Avg QBER: {stats['avg_qber']*100:.3f}% ± {stats['std_qber']*100:.3f}%")
        print(f"  ✓ Avg key rate: {stats['avg_key_rate']:.4f} ± {stats['std_key_rate']:.4f} bits/photon")
        print(f"  ✓ Avg detection: {stats['avg_detection_rate']*100:.2f}% ± {stats['std_detection_rate']*100:.2f}%")
        
        return stats
    
    def plot_results(self, stats: Dict, filename: str = "kaya_qkd_analysis.png"):
        """Generate comprehensive performance plots."""
        if 'detailed_results' not in stats:
            return
        
        results = stats['detailed_results']
        n_trials = len(results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Kaya BB84 QKD Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: QBER evolution
        qbers = [r['qber']*100 for r in results]
        ax1.plot(range(n_trials), qbers, 'bo-', alpha=0.7, linewidth=2, markersize=6)
        ax1.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='Highly Secure (2%)')
        ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Secure (5%)')
        ax1.axhline(y=11, color='red', linestyle='--', label='Threshold (11%)')
        ax1.set_xlabel('Trial Number', fontsize=11)
        ax1.set_ylabel('QBER (%)', fontsize=11)
        ax1.set_title('Quantum Bit Error Rate', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detection rates
        det_rates = [r['detection_rate']*100 for r in results]
        ax2.plot(range(n_trials), det_rates, 'go-', alpha=0.7, linewidth=2, markersize=6)
        ax2.axhline(y=stats['avg_detection_rate']*100, color='red', linestyle='--',
                   label=f"Average: {stats['avg_detection_rate']*100:.1f}%")
        ax2.set_xlabel('Trial Number', fontsize=11)
        ax2.set_ylabel('Detection Rate (%)', fontsize=11)
        ax2.set_title('Photon Detection Efficiency', fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Key rates
        key_rates = [r['key_rate_bits_per_photon'] for r in results]
        ax3.plot(range(n_trials), key_rates, 'mo-', alpha=0.7, linewidth=2, markersize=6)
        ax3.axhline(y=stats['avg_key_rate'], color='red', linestyle='--',
                   label=f"Average: {stats['avg_key_rate']:.4f}")
        ax3.set_xlabel('Trial Number', fontsize=11)
        ax3.set_ylabel('Key Rate (bits/photon)', fontsize=11)
        ax3.set_title('Secure Key Generation Rate', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Histogram of QBER distribution
        ax4.hist(qbers, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=stats['avg_qber']*100, color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats['avg_qber']*100:.2f}%")
        ax4.set_xlabel('QBER (%)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('QBER Distribution', fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n  💾 Analysis plot saved: {filename}")
    
    def export_results(self, filename: str = "kaya_qkd_results.json"):
        """Export results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  💾 Results exported: {filename}")


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("🔬 KAYA BB84 QKD - Enhanced Quantum Cryptography Simulation")
    print("="*70)
    
    # Test 1: Standard operation
    print("\n1️⃣  STANDARD OPERATION (50 km fiber):")
    params1 = QKDParameters(
        n_bits=256,
        fiber_length_km=50,
        channel_loss_db_km=0.2,
        detector_efficiency=0.85
    )
    qkd1 = BB84Protocol(params1)
    result1 = qkd1.run_protocol()
    
    # Test 2: Long distance
    print("\n2️⃣  LONG DISTANCE (100 km fiber):")
    params2 = QKDParameters(
        n_bits=512,
        fiber_length_km=100,
        channel_loss_db_km=0.2,
        detector_efficiency=0.90
    )
    qkd2 = BB84Protocol(params2)
    result2 = qkd2.run_protocol()
    
    # Test 3: Statistical analysis
    print("\n3️⃣  STATISTICAL ANALYSIS:")
    stats = qkd1.run_statistical_analysis(n_trials=10)
    qkd1.plot_results(stats)
    
    # Test 4: Eavesdropping simulation
    print("\n4️⃣  EAVESDROPPING ATTACK SIMULATION:")
    attack_result = qkd1.simulate_eavesdropping(intercept_prob=0.30)
    
    # Export results
    qkd1.export_results()
    
    print("\n" + "="*70)
    print("✅ Kaya BB84 QKD Simulation Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  • kaya_qkd_analysis.png")
    print("  • kaya_qkd_results.json")
    print("\n🔗 GitHub: https://github.com/okushigue/kaya-quantum-chip")
    print("📧 Contact: okushigue@gmail.com\n")

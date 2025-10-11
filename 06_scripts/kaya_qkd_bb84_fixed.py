# kaya_qkd_bb84_fixed.py
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import hashlib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class KayaQKDBB84:
    def __init__(self, n_bits: int = 100, channel_loss: float = 0.1, 
                 detector_eff: float = 0.9, dark_count_rate: float = 1e-6,
                 include_realistic_noise: bool = True):
        self.n_bits = n_bits
        self.channel_loss = channel_loss
        self.detector_eff = detector_eff
        self.dark_count_rate = dark_count_rate
        self.include_realistic_noise = include_realistic_noise
        self.simulator = AerSimulator()
        self._results = {}
        print(f"🔐 Kaya QKD BB84 Protocol | Bits: {n_bits}, Loss: {channel_loss:.1%}, Efficiency: {detector_eff:.1%}")
    
    def _apply_realistic_noise(self, quantum_circuit: QuantumCircuit, bit: int, basis: int) -> QuantumCircuit:
        """Aplica ruídos realistas encontrados em implementações práticas"""
        if not self.include_realistic_noise:
            return quantum_circuit
            
        # Ruído de preparação de estado (0.5-1.5%)
        if random.random() < 0.01:
            quantum_circuit.x(0)  # Flip bit com pequena probabilidade
        
        # Ruído de fase (imperfeições no interferômetro)
        if random.random() < 0.015:
            phase_error = random.gauss(0, 0.05)  # Pequeno desvio de fase
            quantum_circuit.rz(phase_error, 0)
        
        # Ruído de desalinhamento de bases (0.5-2%)
        if random.random() < 0.01:
            misalignment = random.gauss(0, 0.02)
            quantum_circuit.rz(misalignment, 0)
        
        return quantum_circuit
    
    def _apply_channel_effects(self, quantum_state: int) -> Optional[int]:
        """Aplica efeitos realistas do canal quântico"""
        # Perda no canal
        if random.random() < self.channel_loss:
            return None
        
        # Dark counts (falsas detecções)
        if random.random() < self.dark_count_rate:
            return random.randint(0, 1)
        
        # Eficiência do detector
        if random.random() > self.detector_eff:
            return None
        
        # Ruído de detecção (1-3% de erro de leitura)
        if self.include_realistic_noise and random.random() < 0.02:
            return 1 - quantum_state  # Flip do bit
        
        return quantum_state
    
    def _create_quantum_state(self, bit: int, basis: int) -> QuantumCircuit:
        """Cria o estado quântico baseado no bit e base escolhidos"""
        qc = QuantumCircuit(1, 1)
        
        # Preparação do estado
        if bit == 1:
            qc.x(0)
        
        # Aplicação da base
        if basis == 1:  # Base X (Hadamard)
            qc.h(0)
        
        # Aplica ruído realista
        qc = self._apply_realistic_noise(qc, bit, basis)
        
        return qc
    
    def _measure_quantum_state(self, qc: QuantumCircuit, basis: int) -> int:
        """Mede o estado quântico na base especificada"""
        # Aplica a base de medição
        if basis == 1:  # Base X
            qc.h(0)
        
        # Ruído de medição (detectores imperfeitos)
        if self.include_realistic_noise and random.random() < 0.01:
            qc.x(0)  # Pequeno erro de medição
        
        qc.measure(0, 0)
        
        # Executa a simulação
        job = self.simulator.run(qc, shots=1, memory=True)
        result = job.result().get_memory()[0]
        
        return int(result)
    
    def run_protocol(self) -> Dict:
        """Executa o protocolo BB84 completo"""
        print("\n📡 Executing BB84 Quantum Key Distribution Protocol...")
        
        # Geração aleatória de bits e bases
        alice_bits = [random.randint(0, 1) for _ in range(self.n_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(self.n_bits)]  # 0=Z, 1=X
        bob_bases = [random.randint(0, 1) for _ in range(self.n_bits)]
        
        raw_key_alice = []
        raw_key_bob = []
        detected_indices = []
        
        print("   🔄 Transmitting photons through quantum channel...")
        
        for i in range(self.n_bits):
            # Alice prepara o estado quântico
            qc = self._create_quantum_state(alice_bits[i], alice_bases[i])
            
            # Bob mede o estado
            try:
                bob_bit = self._measure_quantum_state(qc, bob_bases[i])
                
                # Aplica efeitos do canal
                final_bit = self._apply_channel_effects(bob_bit)
                
                if final_bit is not None:
                    raw_key_alice.append(alice_bits[i])
                    raw_key_bob.append(final_bit)
                    detected_indices.append(i)
                    
            except Exception as e:
                continue
        
        # Estatísticas de transmissão
        transmitted = self.n_bits
        detected = len(raw_key_alice)
        detection_rate = detected / transmitted if transmitted > 0 else 0
        
        print(f"   📊 Transmission Statistics:")
        print(f"      • Photons transmitted: {transmitted}")
        print(f"      • Photons detected: {detected} ({detection_rate:.1%})")
        print(f"      • Channel efficiency: {(1-self.channel_loss)*self.detector_eff:.1%}")
        
        # Sincronização de bases (sifting)
        sifted_alice = []
        sifted_bob = []
        matching_bases_indices = []
        
        for i in range(len(raw_key_alice)):
            idx = detected_indices[i]
            if alice_bases[idx] == bob_bases[idx]:
                sifted_alice.append(raw_key_alice[i])
                sifted_bob.append(raw_key_bob[i])
                matching_bases_indices.append(idx)
        
        sifting_efficiency = len(sifted_alice) / detected if detected > 0 else 0
        
        print(f"   🔍 Sifting Process:")
        print(f"      • Bits after sifting: {len(sifted_alice)}")
        print(f"      • Sifting efficiency: {sifting_efficiency:.1%}")
        
        # Cálculo do QBER (Quantum Bit Error Rate)
        if len(sifted_alice) > 0:
            errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
            qber = errors / len(sifted_alice)
            
            # Estimativa de eavesdropping
            security_status, security_message = self._estimate_security(qber)
            
            print(f"   🛡️  Security Analysis:")
            print(f"      • Errors detected: {errors}/{len(sifted_alice)}")
            print(f"      • QBER: {qber:.3%}")
            print(f"      • Secure threshold: < 11.0%")
            print(f"      • Status: {security_message}")
            
            # Amplificação de privacidade
            final_key = self._privacy_amplification(sifted_alice)
            key_entropy = self._calculate_key_entropy(final_key)
            
            # Armazena resultados
            self._results = {
                'transmitted_bits': transmitted,
                'detected_bits': detected,
                'detection_rate': detection_rate,
                'sifted_bits': len(sifted_alice),
                'sifting_efficiency': sifting_efficiency,
                'errors': errors,
                'qber': qber,
                'secure': security_status,
                'security_message': security_message,
                'final_key': final_key,
                'key_length': len(final_key),
                'key_entropy': key_entropy,
                'alice_bases': alice_bases,
                'bob_bases': bob_bases,
                'matching_bases_count': len(matching_bases_indices),
                'noise_included': self.include_realistic_noise
            }
            
            # Exibe resumo da chave
            self._display_key_summary(final_key)
            
            return self._results
        else:
            error_msg = "No bits survived the quantum channel and detection process"
            print(f"   ❌ {error_msg}")
            return {'error': error_msg, 'secure': False}
    
    def _estimate_security(self, qber: float) -> Tuple[bool, str]:
        """Estima a segurança baseado no QBER"""
        if qber < 0.02:
            return True, "✅ HIGHLY SECURE (QBER < 2%)"
        elif qber < 0.05:
            return True, "✅ SECURE (QBER < 5%)"
        elif qber < 0.08:
            return True, "⚠️  MARGINALLY SECURE (QBER < 8%)"
        elif qber < 0.11:
            return True, "🚨 BORDERLINE (QBER < 11%)"
        else:
            return False, "❌ INSECURE - Possible eavesdropping detected"
    
    def _privacy_amplification(self, sifted_key: List[int]) -> str:
        """Aplica amplificação de privacidade para reduzir informação do eavesdropper"""
        if not sifted_key:
            return ""
        
        # Converte lista de bits para string
        key_str = ''.join(map(str, sifted_key))
        
        # Aplica hash SHA-256 para amplificação de privacidade
        hashed = hashlib.sha256(key_str.encode()).hexdigest()
        
        # Converte hexadecimal para binário (apenas primeiros 128 bits para chave AES-128)
        binary_key = bin(int(hashed[:32], 16))[2:].zfill(128)
        
        return binary_key
    
    def _calculate_key_entropy(self, key: str) -> float:
        """Calcula a entropia da chave final"""
        if not key:
            return 0.0
        
        # Conta a frequência de 0s e 1s
        count_0 = key.count('0')
        count_1 = key.count('1')
        total = len(key)
        
        # Calcula entropia de Shannon
        p0 = count_0 / total
        p1 = count_1 / total
        
        entropy = 0.0
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        
        return entropy
    
    def _display_key_summary(self, final_key: str):
        """Exibe resumo da chave gerada"""
        if not final_key:
            return
        
        print(f"   🔑 Final Key Summary:")
        print(f"      • Key length: {len(final_key)} bits")
        print(f"      • Key entropy: {self._results['key_entropy']:.3f} bits/bit")
        print(f"      • First 32 bits: {final_key[:32]}")
        print(f"      • Last 32 bits: {final_key[-32:]}")
        
        # Verifica aleatoriedade
        if self._results['key_entropy'] > 0.98:
            randomness = "EXCELLENT"
        elif self._results['key_entropy'] > 0.95:
            randomness = "HIGH"
        elif self._results['key_entropy'] > 0.90:
            randomness = "MODERATE"
        else:
            randomness = "LOW"
            
        print(f"      • Randomness quality: {randomness}")
        print(f"      • Realistic noise: {'ENABLED' if self.include_realistic_noise else 'DISABLED'}")
    
    def run_multiple_trials(self, n_trials: int = 10) -> Dict:
        """Executa múltiplos trials para análise estatística"""
        print(f"\n📊 Running {n_trials} statistical trials...")
        
        results = []
        secure_count = 0
        qber_values = []
        key_lengths = []
        detection_rates = []
        
        for trial in range(n_trials):
            print(f"   Trial {trial + 1}/{n_trials}...")
            result = self.run_protocol()
            
            if 'qber' in result:
                results.append(result)
                qber_values.append(result['qber'])
                key_lengths.append(result['key_length'])
                detection_rates.append(result['detection_rate'])
                
                if result['secure']:
                    secure_count += 1
        
        if results:
            security_probability = secure_count / n_trials
            avg_qber = np.mean(qber_values)
            avg_key_length = np.mean(key_lengths)
            avg_detection_rate = np.mean(detection_rates)
            
            stats = {
                'trials_completed': n_trials,
                'security_probability': security_probability,
                'average_qber': avg_qber,
                'average_key_length': avg_key_length,
                'average_detection_rate': avg_detection_rate,
                'qber_std': np.std(qber_values),
                'reliability': f"{security_probability:.1%} secure executions",
                'detailed_results': results
            }
            
            print(f"\n   📈 Statistical Summary:")
            print(f"      • Security probability: {security_probability:.1%}")
            print(f"      • Average QBER: {avg_qber:.3%} ± {np.std(qber_values):.3%}")
            print(f"      • Average detection rate: {avg_detection_rate:.1%}")
            print(f"      • Average key length: {avg_key_length:.0f} bits")
            print(f"      • Realistic noise: {'ENABLED' if self.include_realistic_noise else 'DISABLED'}")
            
            return stats
        else:
            return {'error': 'No successful trials'}
    
    def plot_performance(self, stats: Dict):
        """Gera gráficos de performance"""
        if 'detailed_results' not in stats:
            return
        
        results = stats['detailed_results']
        qbers = [r['qber'] for r in results]
        detection_rates = [r['detection_rate'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico do QBER
        ax1.plot(range(len(qbers)), qbers, 'bo-', alpha=0.7, label='QBER')
        ax1.axhline(y=0.02, color='green', linestyle='--', alpha=0.7, label='Highly Secure (2%)')
        ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Secure (5%)')
        ax1.axhline(y=0.11, color='red', linestyle='--', label='Security threshold (11%)')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('QBER')
        ax1.set_title('QBER Evolution Across Trials')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico das taxas de detecção
        ax2.plot(range(len(detection_rates)), detection_rates, 'go-', alpha=0.7, label='Detection Rate')
        ax2.axhline(y=stats['average_detection_rate'], color='red', linestyle='--', 
                   label=f'Average: {stats["average_detection_rate"]:.1%}')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Detection Rate')
        ax2.set_title('Photon Detection Rate Across Trials')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qkd_performance_analysis.png', dpi=150, bbox_inches='tight')
        print("   💾 Performance plot saved as 'qkd_performance_analysis.png'")
        plt.close()  # Evita o warning de non-interactive
    
    def simulate_eavesdropping_attack(self, eavesdrop_probability: float = 0.3):
        """Simula um ataque de eavesdropping (intercept-and-resend)"""
        print(f"\n🕵️  Simulating Eavesdropping Attack (intercept probability: {eavesdrop_probability:.0%})...")
        
        # Backup da configuração original
        original_noise_setting = self.include_realistic_noise
        self.include_realistic_noise = False  # Desabilita ruído para isolar o ataque
        
        alice_bits = [random.randint(0, 1) for _ in range(self.n_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(self.n_bits)]
        bob_bases = [random.randint(0, 1) for _ in range(self.n_bits)]
        
        raw_key_alice = []
        raw_key_bob = []
        detected_indices = []
        eavesdrop_count = 0
        
        for i in range(self.n_bits):
            # Eve intercepta com probabilidade especificada
            if random.random() < eavesdrop_probability:
                eavesdrop_count += 1
                # Eve escolhe uma base aleatória para medir
                eve_basis = random.randint(0, 1)
                
                # Eve mede o estado
                qc_eve = self._create_quantum_state(alice_bits[i], alice_bases[i])
                eve_bit = self._measure_quantum_state(qc_eve, eve_basis)
                
                # Eve reenvia o estado na base que ela mediu
                qc_resent = self._create_quantum_state(eve_bit, eve_basis)
            else:
                # Sem eavesdropping - estado original
                qc_resent = self._create_quantum_state(alice_bits[i], alice_bases[i])
            
            # Bob mede o estado (que pode ter sido modificado por Eve)
            bob_bit = self._measure_quantum_state(qc_resent, bob_bases[i])
            final_bit = self._apply_channel_effects(bob_bit)
            
            if final_bit is not None:
                raw_key_alice.append(alice_bits[i])
                raw_key_bob.append(final_bit)
                detected_indices.append(i)
        
        # Processamento normal
        sifted_alice = []
        sifted_bob = []
        
        for i in range(len(raw_key_alice)):
            idx = detected_indices[i]
            if alice_bases[idx] == bob_bases[idx]:
                sifted_alice.append(raw_key_alice[i])
                sifted_bob.append(raw_key_bob[i])
        
        if len(sifted_alice) > 0:
            errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
            qber = errors / len(sifted_alice)
            
            security_status, security_message = self._estimate_security(qber)
            
            print(f"   🕵️  Eavesdropping Results:")
            print(f"      • Eavesdropping attempts: {eavesdrop_count}/{self.n_bits} ({eavesdrop_count/self.n_bits:.1%})")
            print(f"      • Induced QBER: {qber:.3%}")
            print(f"      • Security status: {security_message}")
            
            # Restaura configuração original
            self.include_realistic_noise = original_noise_setting
            
            return {
                'eavesdrop_probability': eavesdrop_probability,
                'eavesdrop_attempts': eavesdrop_count,
                'induced_qber': qber,
                'secure': security_status,
                'detected_bits': len(raw_key_alice),
                'sifted_bits': len(sifted_alice)
            }
        
        # Restaura configuração original
        self.include_realistic_noise = original_noise_setting
        return {'error': 'No successful detections during eavesdropping simulation'}
    
    def get_detailed_report(self) -> Dict:
        """Retorna relatório detalhado dos resultados"""
        return self._results

# Exemplo de uso avançado
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Kaya QKD BB84 - Advanced Quantum Cryptography Simulator")
    print("=" * 60)
    
    # Teste com ruído realista
    print("\n1. TESTE COM RUÍDO REALISTA:")
    qkd_realistic = KayaQKDBB84(
        n_bits=256, 
        channel_loss=0.2, 
        detector_eff=0.85,
        include_realistic_noise=True
    )
    result1 = qkd_realistic.run_protocol()
    
    # Teste sem ruído (ideal)
    print("\n2. TESTE IDEAL (SEM RUÍDO):")
    qkd_ideal = KayaQKDBB84(
        n_bits=256,
        channel_loss=0.1,
        detector_eff=0.95,
        include_realistic_noise=False
    )
    result2 = qkd_ideal.run_protocol()
    
    # Teste estatístico
    print("\n3. ANÁLISE ESTATÍSTICA:")
    stats = qkd_realistic.run_multiple_trials(n_trials=5)
    
    # Gera gráficos
    qkd_realistic.plot_performance(stats)
    
    # Simulação de ataque
    print("\n4. SIMULAÇÃO DE ATAQUE EAVESDROPPING:")
    attack_result = qkd_realistic.simulate_eavesdropping_attack(eavesdrop_probability=0.4)
    
    print("\n" + "=" * 60)
    print("✅ Kaya QKD BB84 Simulation Completed Successfully!")
    print("=" * 60)

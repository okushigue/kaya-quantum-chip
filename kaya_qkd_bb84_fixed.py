# kaya_qkd_bb84_fixed.py
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class KayaQKDBB84:
    def __init__(self, n_bits=100, channel_loss=0.1, detector_eff=0.9):
        self.n_bits = n_bits
        self.channel_loss = channel_loss
        self.detector_eff = detector_eff
        self.simulator = AerSimulator()
        print(f"🔐 Kaya QKD BB84 | {n_bits} bits, loss={channel_loss:.0%}")
    
    def run_protocol(self):
        print("\n📡 Executing BB84 Protocol...")
        
        alice_bits = [random.randint(0, 1) for _ in range(self.n_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(self.n_bits)]  # 0=Z, 1=X
        bob_bases = [random.randint(0, 1) for _ in range(self.n_bits)]
        
        raw_key_alice = []
        raw_key_bob = []
        detected_indices = []
        
        for i in range(self.n_bits):
            if random.random() < self.channel_loss:
                continue
            
            qc = QuantumCircuit(1, 1)
            if alice_bits[i] == 1:
                qc.x(0)
            if alice_bases[i] == 1:
                qc.h(0)
            if bob_bases[i] == 1:
                qc.h(0)
            qc.measure(0, 0)
            
            job = self.simulator.run(qc, shots=1)
            result = job.result().get_counts()
            if result:
                bob_bit = int(list(result.keys())[0])
                if random.random() < self.detector_eff:
                    raw_key_alice.append(alice_bits[i])
                    raw_key_bob.append(bob_bit)
                    detected_indices.append(i)
        
        print(f"   Photons transmitted: {self.n_bits}")
        print(f"   Photons detected: {len(raw_key_alice)} ({len(raw_key_alice)/self.n_bits:.1%})")
        
        sifted_alice = []
        sifted_bob = []
        for i in range(len(raw_key_alice)):
            if alice_bases[detected_indices[i]] == bob_bases[detected_indices[i]]:
                sifted_alice.append(raw_key_alice[i])
                sifted_bob.append(raw_key_bob[i])
        
        print(f"   Bits after sifting: {len(sifted_alice)}")
        
        if len(sifted_alice) > 0:
            errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
            qber = errors / len(sifted_alice)
            print(f"   QBER: {qber:.1%} (secure threshold: <11%)")
            
            is_secure = qber < 0.11
            final_key = ''.join(map(str, sifted_alice[:16])) + "..." if len(sifted_alice) > 16 else ''.join(map(str, sifted_alice))
            print(f"   Final key (first 16 bits): {final_key}")
            print(f"   {'✅ Secure' if is_secure else '❌ Insecure'}")
            
            return {
                'detected': len(raw_key_alice),
                'sifted': len(sifted_alice),
                'qber': qber,
                'secure': is_secure,
                'key': ''.join(map(str, sifted_alice))
            }
        else:
            print("   ❌ No bits survived!")
            return {'error': 'no detections'}
    
if __name__ == "__main__":
    qkd = KayaQKDBB84(n_bits=200, channel_loss=0.2, detector_eff=0.85)
    result = qkd.run_protocol()
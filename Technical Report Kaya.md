# Technical Report: Kaya Photonic Quantum Chip

**Date:** September 28, 2025, 18:12  

## Summary
This report presents simulation results for the Kaya photonic quantum chip, focusing on three main applications: **Boson Sampling** (quantum computing), **QKD BB84** (quantum cryptography), and **Phase Estimation with NOON State** (quantum metrology). All experiments demonstrated performance compatible with integrated photonic hardware implementations.

---

## Detailed Results by Experiment

### 1. Boson Sampling
| Indicator         | Value  | Interpretation                     |
|-------------------|--------|------------------------------------|
| Number of Photons | 3      | Indistinguishable photons          |
| Number of Modes   | 6      | Optical channels on the chip       |
| Possible States   | 56     | Quantum combinatorics             |
| Entropy           | 4.79 bits | Distribution complexity          |

### 2. Quantum Key Distribution (BB84)
| Parameter         | Value         | Specification                     |
|-------------------|---------------|-----------------------------------|
| Transmitted Bits  | 200           | Initial sequence length           |
| Channel Loss      | 20%           | Realistic optical loss            |
| Detected Photons  | 130 (65.0%)   | Total efficiency                  |
| Sifted Bits       | 70            | Basis matching                    |
| QBER              | 0.0%          | < 11% (secure)                    |
| Status            | **Secure**    | Security verification             |

### 3. Phase Estimation with NOON State
| Parameter               | Value          | Quantum Advantage                 |
|-------------------------|----------------|-----------------------------------|
| Number of Photons (N)   | 2              | Quantum resource                  |
| Sensitivity             | 2×             | Surpasses classical limit         |
| Phase Precision         | Δφ = 1/2       | Heisenberg limit                  |
| Applications            | Sensors, gyroscopes | On-chip measurement           |

---

## Recommended Technical Specifications for the Kaya Chip
| Component        | Specification                     | Justification                              |
|------------------|-----------------------------------|--------------------------------------------|
| Platform         | SiN or LiNbO₃                     | Low optical loss (< 0.1 dB/cm)             |
| Optical Modes    | ≥ 6                               | Supports boson sampling                    |
| Photon Source    | Integrated SPDC                   | Generation of entangled photon pairs       |
| Detector         | SNSPD                             | Efficiency > 90%, low noise                |
| Modulator        | Thermo-optic or electro-optic     | Phase control for QKD/NOON                 |
| Total Loss       | < 3 dB                            | Feasibility for quantum algorithms         |

---

## Conclusion
The simulation experiments confirm that the Kaya chip is **viable** for the three pillars of photonic quantum technology:  
✅ **Computing** (Boson Sampling),  
✅ **Communication** (QKD),  
✅ **Metrology** (NOON State).  

The results align with cutting-edge implementations, providing a solid foundation for integrated photonic hardware development. **Recommendations:**  
- Proceed with circuit layout design;  
- Establish partnerships with specialized photonic foundries.

#!/usr/bin/env python3
"""
🌊 KAYA 1024-MODE HARD MODE CHALLENGE
Difficult classification task to demonstrate noise advantage

Challenges:
1. Similar chaotic systems (variants of same attractor)
2. Reduced features via PCA (1024 → 100)
3. Increased noise levels (0% - 50%)
4. Rigorous cross-validation (10-fold)
5. More samples (600)

Author: Jefferson M. Okushigue
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("KAYA-HARD")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🔥 KAYA 1024-MODE HARD MODE CHALLENGE 🔥                ║
║                                                              ║
║   "When Easy Isn't Enough - Proving Noise Advantage"        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# 1. SPARSE PHOTONIC CHIP (Same as before)
# ============================================================================

class SparsePhotonicChip:
    """1024-mode photonic chip with sparse representation"""
    
    def __init__(self, n_modes=1024, loss_db=8.0, crosstalk=0.10, bs_quality=0.95):
        self.n_modes = n_modes
        self.loss_db = loss_db
        self.loss = 10 ** (-loss_db / 10)
        self.crosstalk = crosstalk
        self.bs_quality = bs_quality
        self.rng = np.random.default_rng(42)
        
        self.U_base_sparse = self._generate_sparse_haar_unitary(n_modes)
        self.phases = np.zeros(n_modes)
    
    def _generate_sparse_haar_unitary(self, n, density=0.05):
        """Generate sparse Haar-random unitary"""
        U = sparse.eye(n, dtype=complex, format='lil')
        n_layers = int(np.log2(n))
        n_rotations = int(n * density * n_layers)
        
        for layer in range(n_layers):
            stride = 2 ** layer
            for _ in range(n_rotations // n_layers):
                i = self.rng.integers(0, n - stride)
                j = i + stride
                
                theta = self.rng.uniform(0, 2*np.pi)
                phi = self.rng.uniform(0, 2*np.pi)
                
                c = np.cos(theta)
                s = np.sin(theta) * np.exp(1j * phi)
                
                U_temp = U.tocsr()
                row_i = U_temp.getrow(i).toarray().flatten()
                row_j = U_temp.getrow(j).toarray().flatten()
                
                new_row_i = c * row_i + s * row_j
                new_row_j = -np.conj(s) * row_i + c * row_j
                
                U[i, :] = new_row_i
                U[j, :] = new_row_j
        
        return U.tocsr()
    
    def set_phases(self, signal):
        """Configure phase shifters"""
        if len(signal) == 0:
            signal = np.zeros(1)
        norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-12)
        self.phases = 2 * np.pi * norm[:self.n_modes]
        if len(self.phases) < self.n_modes:
            self.phases = np.pad(self.phases, (0, self.n_modes - len(self.phases)))
        return self.phases
    
    def apply_transformation(self, input_state):
        """Apply sparse unitary with noise"""
        phase_diag = sparse.diags(np.exp(1j * self.phases), format='csr')
        
        if self.crosstalk > 0:
            main_diag = np.ones(self.n_modes)
            off_diag = np.full(self.n_modes - 1, self.crosstalk)
            crosstalk_matrix = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
            phase_diag = crosstalk_matrix @ phase_diag @ crosstalk_matrix.T
        
        U_dagger = self.U_base_sparse.conj().T
        output_state = self.U_base_sparse @ (phase_diag @ (U_dagger @ input_state))
        
        if sparse.issparse(output_state):
            output_state = output_state.toarray().flatten()
        
        output_state *= np.sqrt(self.loss * self.bs_quality)
        
        noise_level = 0.02 * (1 - self.bs_quality * 0.5)
        noise = self.rng.normal(0, noise_level, self.n_modes) + 1j * self.rng.normal(0, noise_level, self.n_modes)
        output_state += noise
        
        norm = np.linalg.norm(output_state)
        if norm > 1e-12:
            output_state /= norm
        
        return output_state
    
    def sample_output(self, samples=500):
        """Sample output distribution"""
        input_state = np.zeros(self.n_modes, dtype=complex)
        input_state[0] = 1.0
        output_state = self.apply_transformation(input_state)
        probs = np.abs(output_state) ** 2
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0, None)
        total = np.sum(probs)
        if total < 1e-12:
            probs = np.ones(self.n_modes) / self.n_modes
        else:
            probs /= total
        counts = self.rng.multinomial(samples, probs)
        return counts / samples


# ============================================================================
# 2. HARD CHAOTIC SYSTEMS - Similar Variants
# ============================================================================

class HardChaoticSystems:
    """
    HARD MODE: Generate similar chaotic systems
    - Lorenz vs Lorenz-variant (Chen attractor)
    - Rössler vs Rössler-hyperchaos
    - Logistic r=3.9 vs r=3.95 (very close parameters)
    """
    
    @staticmethod
    def lorenz_standard(n=1024, dt=0.01, noise=0.10):
        """Standard Lorenz (σ=10, ρ=28, β=8/3)"""
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        x[0], y[0], z[0] = 0.1, 0.0, 0.0
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        for i in range(1, n):
            noise_i = noise * np.random.randn()
            dx = sigma * (y[i-1] - x[i-1])
            dy = x[i-1] * (rho - z[i-1]) - y[i-1]
            dz = x[i-1] * y[i-1] - beta * z[i-1]
            x[i] = x[i-1] + dx * dt + noise_i
            y[i] = y[i-1] + dy * dt + noise_i
            z[i] = z[i-1] + dz * dt + noise_i
        return x
    
    @staticmethod
    def chen_attractor(n=1024, dt=0.01, noise=0.10):
        """Chen attractor (similar to Lorenz but different dynamics)"""
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        x[0], y[0], z[0] = 0.1, 0.0, 0.0
        a, b, c = 35.0, 3.0, 28.0  # Chen parameters
        
        for i in range(1, n):
            noise_i = noise * np.random.randn()
            dx = a * (y[i-1] - x[i-1])
            dy = (c - a) * x[i-1] - x[i-1] * z[i-1] + c * y[i-1]
            dz = x[i-1] * y[i-1] - b * z[i-1]
            x[i] = x[i-1] + dx * dt + noise_i
            y[i] = y[i-1] + dy * dt + noise_i
            z[i] = z[i-1] + dz * dt + noise_i
        return x
    
    @staticmethod
    def rossler_standard(n=1024, dt=0.05, noise=0.10):
        """Standard Rössler (a=0.2, b=0.2, c=5.7)"""
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        x[0], y[0], z[0] = 1.0, 1.0, 1.0
        a, b, c = 0.2, 0.2, 5.7
        
        for i in range(1, n):
            noise_i = noise * np.random.randn()
            dx = -y[i-1] - z[i-1]
            dy = x[i-1] + a * y[i-1]
            dz = b + z[i-1] * (x[i-1] - c)
            x[i] = x[i-1] + dx * dt + noise_i
            y[i] = y[i-1] + dy * dt + noise_i
            z[i] = z[i-1] + dz * dt + noise_i
        return x
    
    @staticmethod
    def rossler_hyperchaos(n=1024, dt=0.05, noise=0.10):
        """Rössler hyperchaos (different c parameter, more complex)"""
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        x[0], y[0], z[0] = 1.0, 1.0, 1.0
        a, b, c = 0.25, 0.3, 4.5  # Different parameters
        
        for i in range(1, n):
            noise_i = noise * np.random.randn()
            dx = -y[i-1] - z[i-1]
            dy = x[i-1] + a * y[i-1]
            dz = b + z[i-1] * (x[i-1] - c)
            x[i] = x[i-1] + dx * dt + noise_i
            y[i] = y[i-1] + dy * dt + noise_i
            z[i] = z[i-1] + dz * dt + noise_i
        return x
    
    @staticmethod
    def logistic_39(n=1024, noise=0.10):
        """Logistic map r=3.9 (deeply chaotic)"""
        x = np.zeros(n)
        x[0] = 0.4
        r = 3.9
        for i in range(1, n):
            noise_i = noise * np.random.randn()
            x[i] = r * x[i-1] * (1 - x[i-1]) + noise_i
            x[i] = np.clip(x[i], 0, 1)
        return x
    
    @staticmethod
    def logistic_395(n=1024, noise=0.10):
        """Logistic map r=3.95 (VERY similar to 3.9)"""
        x = np.zeros(n)
        x[0] = 0.4
        r = 3.95
        for i in range(1, n):
            noise_i = noise * np.random.randn()
            x[i] = r * x[i-1] * (1 - x[i-1]) + noise_i
            x[i] = np.clip(x[i], 0, 1)
        return x


# ============================================================================
# 3. FEATURE EXTRACTION WITH PCA
# ============================================================================

class QuantumFeatureExtractor:
    """Extract quantum features with dimensionality reduction"""
    
    def __init__(self, chip):
        self.chip = chip
        self.n_modes = chip.n_modes
    
    def extract_features(self, signal, n_chunks=8):
        """Extract 1024D features (will be reduced by PCA later)"""
        features = []
        
        # Classical features (16D)
        classical = [
            np.mean(signal), np.std(signal), np.median(signal),
            np.percentile(signal, 25), np.percentile(signal, 75),
            np.min(signal), np.max(signal), np.ptp(signal),
        ]
        classical.extend([
            self._safe_stat(lambda x: np.mean(x**2), signal),
            self._safe_stat(lambda x: np.mean(x**3), signal),
            self._safe_stat(lambda x: np.mean(np.abs(np.diff(x))), signal),
            self._safe_stat(lambda x: np.std(np.diff(x)), signal),
        ])
        
        # Spectral
        fft = np.abs(np.fft.fft(signal))[:len(signal)//2]
        classical.extend([np.mean(fft[:10]), np.max(fft), np.argmax(fft) / len(fft), np.sum(fft**2)])
        features.extend(classical)
        
        # Quantum features
        chunk_size = len(signal) // n_chunks
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else len(signal)
            chunk = signal[start:end]
            
            self.chip.set_phases(chunk)
            probs = self.chip.sample_output(samples=500)
            
            chunk_features = [
                np.mean(probs), np.std(probs), np.max(probs),
                self._safe_entropy(probs), np.sum(probs**2),
                np.sum(probs > 0.001) / len(probs),
            ]
            features.extend(chunk_features)
        
        features = np.array(features, dtype=np.float32)
        if len(features) < self.n_modes:
            features = np.pad(features, (0, self.n_modes - len(features)))
        return features[:self.n_modes]
    
    def _safe_stat(self, func, data, default=0.0):
        try:
            result = func(data)
            if np.isnan(result) or np.isinf(result):
                return default
            return float(result)
        except:
            return default
    
    def _safe_entropy(self, probs, default=0.0):
        try:
            probs_clean = probs[probs > 0]
            if len(probs_clean) == 0:
                return default
            return float(entropy(probs_clean))
        except:
            return default


# ============================================================================
# 4. HARD MODE EXPERIMENT
# ============================================================================

def run_hard_mode_experiment(n_samples=600, noise_levels=None, n_components=100):
    """
    HARD MODE: Similar systems + PCA reduction + CV
    """
    
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    print("\n" + "="*70)
    print("🔥 HARD MODE CHALLENGE - Similar Chaotic Systems")
    print("="*70)
    print(f"Task: Distinguish VERY similar chaotic variants")
    print(f"  Class 0: Lorenz (σ=10, ρ=28, β=8/3)")
    print(f"  Class 1: Chen (a=35, b=3, c=28) - Similar to Lorenz!")
    print(f"  Class 2: Rössler (a=0.2, b=0.2, c=5.7)")
    print(f"Difficulty: Chen is VERY close to Lorenz in phase space")
    print(f"PCA: 1024D → {n_components}D (dimensionality reduction)")
    print("="*70)
    
    results = []
    chaos = HardChaoticSystems()
    
    for noise_level in noise_levels:
        print(f"\n🎯 Testing noise level: {noise_level*100:.1f}%")
        print("-"*70)
        
        start_time = time.time()
        
        # Initialize chip
        chip = SparsePhotonicChip(n_modes=1024, loss_db=8.0, crosstalk=0.10, bs_quality=0.95)
        extractor = QuantumFeatureExtractor(chip)
        
        # Generate HARD dataset
        print(f"  Generating {n_samples} samples (similar systems)...")
        X, y = [], []
        samples_per_class = n_samples // 3
        
        for i in range(n_samples):
            if i < samples_per_class:
                signal = chaos.lorenz_standard(n=1024, noise=noise_level)
                label = 0
            elif i < 2 * samples_per_class:
                signal = chaos.chen_attractor(n=1024, noise=noise_level)  # SIMILAR to Lorenz!
                label = 1
            else:
                signal = chaos.rossler_standard(n=1024, noise=noise_level)
                label = 2
            
            features = extractor.extract_features(signal)
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Clean NaN and Inf values
        print(f"  Cleaning data (removing NaN/Inf)...")
        print(f"    Before: {X.shape}")
        
        # Replace NaN and Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for remaining issues
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        print(f"    NaN values: {n_nan}")
        print(f"    Inf values: {n_inf}")
        print(f"    After: {X.shape}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PCA reduction: 1024 → n_components
        print(f"  Applying PCA: 1024D → {n_components}D...")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"    Explained variance: {explained_var:.1%}")
        
        # Train
        print("  Training Random Forest...")
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
        clf.fit(X_train_pca, y_train)
        
        # Cross-validation
        print("  Running 10-fold cross-validation...")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_pca, y_train, cv=cv, n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Test
        y_pred = clf.predict(X_test_pca)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        elapsed = time.time() - start_time
        
        print(f"\n  ✅ Results:")
        print(f"     CV Accuracy:   {cv_mean:.1%} ± {cv_std:.1%}")
        print(f"     Test Accuracy: {test_accuracy:.1%}")
        print(f"  ⏱️  Time: {elapsed:.1f}s")
        
        results.append({
            'noise_level': noise_level,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_accuracy,
            'pca_components': n_components,
            'explained_variance': explained_var,
            'time_seconds': elapsed
        })
    
    return pd.DataFrame(results)


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_hard_mode_results(df):
    """Visualize hard mode results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔥 KAYA HARD MODE - Similar Chaotic Systems\n1024D → 100D via PCA',
                 fontsize=16, fontweight='bold')
    
    # 1. CV Accuracy with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(df['noise_level']*100, df['cv_mean']*100, yerr=df['cv_std']*100,
                 fmt='o-', linewidth=3, markersize=10, capsize=5, color='#667eea', label='CV Mean ± Std')
    ax1.plot(df['noise_level']*100, df['test_accuracy']*100, 's--', linewidth=2, markersize=8,
             color='#f093fb', label='Test Accuracy')
    baseline_cv = df['cv_mean'].iloc[0] * 100
    ax1.axhline(y=baseline_cv, color='red', linestyle='--', alpha=0.5, label=f'Baseline CV: {baseline_cv:.1f}%')
    ax1.set_xlabel('Noise Level (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Performance vs Noise (Similar Systems)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Optimal point
    optimal_idx = df['cv_mean'].idxmax()
    optimal_noise = df.loc[optimal_idx, 'noise_level'] * 100
    optimal_cv = df.loc[optimal_idx, 'cv_mean'] * 100
    ax1.scatter([optimal_noise], [optimal_cv], color='green', s=300, marker='*',
                zorder=5, edgecolors='black', linewidths=2)
    
    # 2. Improvement over baseline
    ax2 = axes[0, 1]
    baseline_cv = df['cv_mean'].iloc[0]
    improvements = (df['cv_mean'] - baseline_cv) * 100
    colors = ['red' if x < 0 else 'green' for x in improvements]
    ax2.bar(df['noise_level']*100, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Noise Level (%)', fontsize=12)
    ax2.set_ylabel('CV Improvement (%)', fontsize=12)
    ax2.set_title('Noise Advantage (Relative to 0%)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. PCA Explained Variance
    ax3 = axes[1, 0]
    ax3.plot(df['noise_level']*100, df['explained_variance']*100, 'D-', 
             linewidth=2, markersize=8, color='#f093fb')
    ax3.set_xlabel('Noise Level (%)', fontsize=12)
    ax3.set_ylabel('Explained Variance (%)', fontsize=12)
    ax3.set_title('PCA Information Retention (1024D → 100D)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    optimal_row = df.loc[optimal_idx]
    baseline_test = df['test_accuracy'].iloc[0] * 100
    optimal_test = optimal_row['test_accuracy'] * 100
    test_improvement = optimal_test - baseline_test
    
    summary_text = f"""
    🔥 HARD MODE SUMMARY
    {'='*45}
    
    Challenge:     Similar chaotic variants
    • Lorenz vs Chen (VERY similar!)
    • Rössler (different family)
    
    Dimensionality: 1024D → 100D (PCA)
    Samples:        {len(df)*200} total
    
    📊 BASELINE (0% noise)
    {'='*45}
    CV Accuracy:    {df['cv_mean'].iloc[0]*100:.2f}% ± {df['cv_std'].iloc[0]*100:.2f}%
    Test Accuracy:  {baseline_test:.2f}%
    
    🎯 OPTIMAL ({optimal_noise:.0f}% noise)
    {'='*45}
    CV Accuracy:    {optimal_cv:.2f}% ± {optimal_row['cv_std']*100:.2f}%
    Test Accuracy:  {optimal_test:.2f}%
    
    💪 IMPROVEMENT
    {'='*45}
    CV Gain:        +{improvements.max():.2f}%
    Test Gain:      +{test_improvement:.2f}%
    
    ⚡ EFFICIENCY
    {'='*45}
    Avg Time:       {df['time_seconds'].mean():.1f}s
    PCA Variance:   {optimal_row['explained_variance']*100:.1f}%
    
    ✅ CONCLUSION
    {'='*45}
    {'Noise advantage CONFIRMED!' if improvements.max() > 2 else 'Marginal noise effect'}
    {'Optimal noise ~' + f"{optimal_noise:.0f}%" if improvements.max() > 2 else 'Problem remains hard'}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'kaya_1024_hard_mode_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {filename}")
    
    return filename


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("\n🚀 Starting KAYA HARD MODE experiment...")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run experiment
    df_results = run_hard_mode_experiment(n_samples=600, n_components=100)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'kaya_1024_hard_mode_{timestamp}.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved: {csv_filename}")
    
    # Visualize
    plot_filename = plot_hard_mode_results(df_results)
    
    # Summary
    print("\n" + "="*70)
    print("✅ HARD MODE COMPLETE!")
    print("="*70)
    print(f"\n📊 Results:")
    print(df_results.to_string(index=False))
    
    optimal_idx = df_results['cv_mean'].idxmax()
    optimal_row = df_results.loc[optimal_idx]
    baseline_cv = df_results['cv_mean'].iloc[0]
    improvement = (optimal_row['cv_mean'] - baseline_cv) * 100
    
    print(f"\n🎯 Key Findings:")
    print(f"   Baseline CV:      {baseline_cv*100:.2f}%")
    print(f"   Optimal noise:    {optimal_row['noise_level']*100:.1f}%")
    print(f"   Peak CV:          {optimal_row['cv_mean']*100:.2f}%")
    print(f"   Improvement:      {'+' if improvement > 0 else ''}{improvement:.2f}%")
    
    if improvement > 2:
        print(f"\n🎉 NOISE ADVANTAGE CONFIRMED!")
        print(f"   Similar systems + PCA reduction revealed SR effect")
    else:
        print(f"\n⚠️  Marginal or no noise advantage")
        print(f"   Task may still be too easy, or need more extreme noise")
    
    print(f"\n📁 Output files:")
    print(f"   • {csv_filename}")
    print(f"   • {plot_filename}")
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🔥 Hard Mode: Testing limits of noise advantage 🔥         ║
║                                                              ║
║        KAYA 1024 - When Easy Mode Isn't Enough              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    return df_results


if __name__ == "__main__":
    results = main()
    print("\n🎉 Hard mode execution complete!")
    print("📊 Check CSV and PNG files for detailed results.")

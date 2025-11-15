# kaya_validation_rigorous.py
# Additional tests to validate the real robustness of the KAYA system

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import your original classes here
# from kaya_caos7 import KayaQuantumExtreme, ChaoticAttractorGenerator

class KayaValidationSuite:
    """Rigorous test suite to validate the KAYA system"""
    
    def __init__(self, kaya_system):
        self.kaya = kaya_system
        
    def test_1_no_data_augmentation(self):
        """Test WITHOUT data augmentation to see real accuracy"""
        print("\n" + "="*70)
        print("TEST 1: No Data Augmentation (Real Balancing)")
        print("="*70)
        
        X, y = self.kaya.generate_dataset(n_samples=900)
        
        # Split without artificial balancing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        # Only scaling, NO augmentation
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test with Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"✓ Accuracy WITHOUT augmentation: {acc:.1%}")
        print(f"  Training classes: {np.bincount(y_train)}")
        print(f"  Test classes: {np.bincount(y_test)}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Lorenz', 'Logistic', 'Rössler']))
        
        return acc
    
    def test_2_stratified_cross_validation(self, n_folds=10):
        """Stratified cross-validation with multiple folds"""
        print("\n" + "="*70)
        print(f"TEST 2: Stratified Cross-Validation ({n_folds} folds)")
        print("="*70)
        
        X, y = self.kaya.generate_dataset(n_samples=1200)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Models to test
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, random_state=42
            )
        }
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            scores = cross_validate(
                model, X_scaled, y, cv=cv, 
                scoring=['accuracy', 'f1_weighted'],
                n_jobs=-1, return_train_score=True
            )
            
            print(f"\n{name}:")
            print(f"  ├─ Training Accuracy: {scores['train_accuracy'].mean():.1%} ± {scores['train_accuracy'].std():.1%}")
            print(f"  ├─ Validation Accuracy: {scores['test_accuracy'].mean():.1%} ± {scores['test_accuracy'].std():.1%}")
            print(f"  └─ F1-Score: {scores['test_f1_weighted'].mean():.3f} ± {scores['test_f1_weighted'].std():.3f}")
            
            # Check for overfitting
            gap = scores['train_accuracy'].mean() - scores['test_accuracy'].mean()
            if gap > 0.05:
                print(f"  ⚠️  Possible overfitting detected (gap: {gap:.1%})")
            else:
                print(f"  ✓ Good generalization (gap: {gap:.1%})")
    
    def test_3_different_seeds(self, n_trials=5):
        """Test with different seeds to verify stability"""
        print("\n" + "="*70)
        print(f"TEST 3: Stability with Different Seeds ({n_trials} trials)")
        print("="*70)
        
        accuracies = []
        
        for trial in range(n_trials):
            # Generate new dataset with different seed
            self.kaya.chip.rng = np.random.default_rng(seed=42 + trial)
            X, y = self.kaya.generate_dataset(n_samples=800)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42 + trial
            )
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            
            acc = accuracy_score(y_test, rf.predict(X_test_scaled))
            accuracies.append(acc)
            print(f"  Trial {trial+1}: {acc:.1%}")
        
        print(f"\n📊 Statistics:")
        print(f"  ├─ Mean: {np.mean(accuracies):.1%}")
        print(f"  ├─ Standard Deviation: {np.std(accuracies):.1%}")
        print(f"  ├─ Minimum: {np.min(accuracies):.1%}")
        print(f"  └─ Maximum: {np.max(accuracies):.1%}")
        
        if np.std(accuracies) < 0.05:
            print("  ✓ Stable results across different seeds")
        else:
            print("  ⚠️  High variance between seeds")
        
        return accuracies
    
    def test_4_larger_dataset(self):
        """Test with significantly larger dataset"""
        print("\n" + "="*70)
        print("TEST 4: Larger Dataset (2400 samples)")
        print("="*70)
        
        X, y = self.kaya.generate_dataset(n_samples=2400)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        acc = accuracy_score(y_test, rf.predict(X_test_scaled))
        print(f"\n✓ Accuracy with larger dataset: {acc:.1%}")
        
        return acc
    
    def test_5_feature_analysis(self):
        """Analyze feature importance"""
        print("\n" + "="*70)
        print("TEST 5: Feature Importance Analysis")
        print("="*70)
        
        X, y = self.kaya.generate_dataset(n_samples=900)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Top 10 most important features
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print("\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. Feature {idx:3d}: {importances[idx]:.4f}")
        
        # Importance distribution analysis
        top_10_sum = importances[indices].sum()
        print(f"\n📊 Sum of top 10 features: {top_10_sum:.1%}")
        
        if top_10_sum > 0.5:
            print("  ⚠️  Model too dependent on few features")
        else:
            print("  ✓ Well-distributed importance")
        
        return importances

def run_complete_suite():
    """Runs all validation tests"""
    print("\n" + "="*70)
    print("🧪 KAYA QUANTUM - RIGOROUS VALIDATION SUITE")
    print("="*70)
    print("This suite will test the real robustness of the system without artificial optimizations")
    
    # Import and initialize your KAYA system here
    # from kaya_caos7 import KayaQuantumExtreme
    # kaya = KayaQuantumExtreme(n_modes=256, noise_level=0.10, crosstalk=0.05, loss_db=7.0)
    # validator = KayaValidationSuite(kaya)
    
    print("\n⚠️  IMPORTANT: Import KayaQuantumExtreme before running!")
    print("Example:")
    print("  from kaya_caos7 import KayaQuantumExtreme")
    print("  kaya = KayaQuantumExtreme(n_modes=256, noise_level=0.10, crosstalk=0.05, loss_db=7.0)")
    print("  validator = KayaValidationSuite(kaya)")
    print("  validator.test_1_no_data_augmentation()")
    print("  validator.test_2_stratified_cross_validation()")
    print("  validator.test_3_different_seeds()")
    print("  validator.test_4_larger_dataset()")
    print("  validator.test_5_feature_analysis()")
    
    # Uncomment when you have the system imported:
    # validator.test_1_no_data_augmentation()
    # validator.test_2_stratified_cross_validation()
    # validator.test_3_different_seeds()
    # validator.test_4_larger_dataset()
    # validator.test_5_feature_analysis()
    
    print("\n" + "="*70)
    print("🎯 VALIDATION SUITE COMPLETED")
    print("="*70)

if __name__ == "__main__":
    run_complete_suite()

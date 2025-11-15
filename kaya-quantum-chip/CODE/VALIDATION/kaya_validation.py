# kaya_validation_rigorous.py
# Additional tests to validate the real robustness of the KAYA system

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import original classes from the file
import sys
sys.path.append('.')

try:
    from kaya_caos7 import KayaQuantumExtreme, DisruptiveKayaChip, ChaoticAttractorGenerator
    KAYA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Error importing kaya_caos7: {e}")
    print("Make sure kaya_caos7.py is in the same directory!")
    KAYA_AVAILABLE = False

class KayaValidationSuite:
    """Rigorous test suite to validate the KAYA system"""
    
    def __init__(self, kaya_system):
        self.kaya = kaya_system
        
    def test_1_no_data_augmentation(self):
        """Test WITHOUT data augmentation to see real accuracy"""
        print("\n" + "="*70)
        print("📊 TEST 1: No Data Augmentation (Real Balancing)")
        print("="*70)
        print("Objective: Measure accuracy without artificially increasing data")
        
        X, y = self.kaya.generate_dataset(n_samples=900)
        
        # Split without artificial balancing
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
        
        print(f"\n✓ Accuracy WITHOUT augmentation: {acc:.1%}")
        print(f"  Training classes: {np.bincount(y_train)}")
        print(f"  Test classes: {np.bincount(y_test)}")
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Lorenz', 'Logistic', 'Rössler'],
                                   digits=3))
        
        # Evaluate result
        if acc >= 0.95:
            status = "🚀 EXCELLENT - Very robust system!"
        elif acc >= 0.85:
            status = "✅ GOOD - Solid performance"
        elif acc >= 0.75:
            status = "⚠️  ACCEPTABLE - Can improve"
        else:
            status = "❌ LOW - Needs optimization"
        
        print(f"\n{status}")
        return acc
    
    def test_2_stratified_cross_validation(self, n_folds=10):
        """Stratified cross-validation with multiple folds"""
        print("\n" + "="*70)
        print(f"📊 TEST 2: Stratified Cross-Validation ({n_folds} folds)")
        print("="*70)
        print("Objective: Evaluate generalization with rigorous cross-validation")
        
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
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Testing {name}...")
            scores = cross_validate(
                model, X_scaled, y, cv=cv, 
                scoring=['accuracy', 'f1_weighted'],
                n_jobs=-1, return_train_score=True
            )
            
            train_acc = scores['train_accuracy'].mean()
            test_acc = scores['test_accuracy'].mean()
            gap = train_acc - test_acc
            
            print(f"  ├─ Training Accuracy: {train_acc:.1%} ± {scores['train_accuracy'].std():.1%}")
            print(f"  ├─ Validation Accuracy: {test_acc:.1%} ± {scores['test_accuracy'].std():.1%}")
            print(f"  ├─ F1-Score: {scores['test_f1_weighted'].mean():.3f} ± {scores['test_f1_weighted'].std():.3f}")
            print(f"  └─ Gap (Train-Val): {gap:.1%}")
            
            # Check for overfitting
            if gap > 0.05:
                print(f"     ⚠️  Possible overfitting detected!")
            else:
                print(f"     ✓ Good generalization")
            
            results[name] = {
                'test_acc': test_acc,
                'std': scores['test_accuracy'].std(),
                'gap': gap
            }
        
        # Best model
        best_model = max(results, key=lambda k: results[k]['test_acc'])
        print(f"\n🏆 Best model: {best_model} ({results[best_model]['test_acc']:.1%})")
        
        return results
    
    def test_3_different_seeds(self, n_trials=5):
        """Test with different seeds to verify stability"""
        print("\n" + "="*70)
        print(f"📊 TEST 3: Stability with Different Seeds ({n_trials} trials)")
        print("="*70)
        print("Objective: Verify if results are consistent with different initializations")
        
        accuracies = []
        
        for trial in range(n_trials):
            print(f"\n🔄 Trial {trial+1}/{n_trials}...")
            
            # Create new instance with different seed
            kaya_temp = KayaQuantumExtreme(
                n_modes=256,
                noise_level=self.kaya.noise_level,
                crosstalk=self.kaya.crosstalk,
                loss_db=self.kaya.chip.loss_db
            )
            # Change seed
            kaya_temp.chip.rng = np.random.default_rng(seed=100 + trial * 17)
            
            X, y = kaya_temp.generate_dataset(n_samples=800)
            
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
            print(f"  Accuracy: {acc:.1%}")
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\n📊 Final Statistics:")
        print(f"  ├─ Mean: {mean_acc:.1%}")
        print(f"  ├─ Standard Deviation: {std_acc:.1%}")
        print(f"  ├─ Minimum: {np.min(accuracies):.1%}")
        print(f"  ├─ Maximum: {np.max(accuracies):.1%}")
        print(f"  └─ Range: {(np.max(accuracies) - np.min(accuracies)):.1%}")
        
        if std_acc < 0.03:
            print("\n  ✓ Very stable results!")
        elif std_acc < 0.05:
            print("\n  ✓ Stable results")
        else:
            print("\n  ⚠️  High variance between seeds")
        
        return accuracies
    
    def test_4_larger_dataset(self):
        """Test with significantly larger dataset"""
        print("\n" + "="*70)
        print("📊 TEST 4: Larger Dataset (2400 samples)")
        print("="*70)
        print("Objective: Verify performance with more test data")
        
        print("\nGenerating large dataset (may take a few minutes)...")
        X, y = self.kaya.generate_dataset(n_samples=2400)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        print(f"  ├─ Training: {X_train.shape[0]} samples")
        print(f"  └─ Test: {X_test.shape[0]} samples (3x larger than original)")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\nTraining model...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        y_pred = rf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Accuracy with larger dataset: {acc:.1%}")
        
        # Performance by class
        print("\n📋 Performance by Class:")
        for cls, name in enumerate(['Lorenz', 'Logistic', 'Rössler']):
            mask = y_test == cls
            acc_cls = accuracy_score(y_test[mask], y_pred[mask])
            support = np.sum(mask)
            print(f"  {name}: {acc_cls:.1%} ({support} samples)")
        
        return acc
    
    def test_5_feature_analysis(self):
        """Analyze feature importance"""
        print("\n" + "="*70)
        print("📊 TEST 5: Feature Importance Analysis")
        print("="*70)
        print("Objective: Identify which features are most discriminative")
        
        X, y = self.kaya.generate_dataset(n_samples=900)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("\nTraining model for feature analysis...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Top 15 most important features
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        print("\n🏆 Top 15 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. Feature {idx:3d}: {importances[idx]:.4f} {'█' * int(importances[idx] * 100)}")
        
        # Importance distribution analysis
        top_10_sum = importances[indices[:10]].sum()
        top_20_sum = importances[indices[:20] if len(indices) >= 20 else indices].sum()
        
        print(f"\n📊 Importance Concentration:")
        print(f"  ├─ Top 10 features: {top_10_sum:.1%}")
        print(f"  └─ Top 20 features: {top_20_sum:.1%}")
        
        if top_10_sum > 0.6:
            print("\n  ⚠️  Model too dependent on few features")
        elif top_10_sum > 0.4:
            print("\n  ⚠️  Moderate concentration in few features")
        else:
            print("\n  ✓ Importance well distributed among features")
        
        return importances

def run_complete_suite():
    """Runs all validation tests"""
    print("\n" + "="*70)
    print("🧪 KAYA QUANTUM - RIGOROUS VALIDATION SUITE")
    print("="*70)
    print("This suite will test the real robustness of the system without artificial optimizations")
    
    if not KAYA_AVAILABLE:
        print("\n❌ Could not import KayaQuantumExtreme!")
        print("Make sure kaya_caos7.py is in the same directory.")
        return
    
    # Initialize KAYA system with moderate parameters
    print("\n🚀 Initializing KAYA system...")
    print("   Parameters: Noise ±10%, Crosstalk 5%, Loss 7.0dB")
    
    kaya = KayaQuantumExtreme(
        n_modes=256,
        noise_level=0.10,
        crosstalk=0.05,
        loss_db=7.0
    )
    
    validator = KayaValidationSuite(kaya)
    
    # Store results
    final_results = {}
    
    # Run all tests
    try:
        print("\n" + "🔬 STARTING TEST BATTERY..." + "\n")
        
        final_results['test1'] = validator.test_1_no_data_augmentation()
        final_results['test2'] = validator.test_2_stratified_cross_validation()
        final_results['test3'] = validator.test_3_different_seeds()
        final_results['test4'] = validator.test_4_larger_dataset()
        final_results['test5'] = validator.test_5_feature_analysis()
        
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "="*70)
    print("🏆 FINAL VALIDATION SUMMARY")
    print("="*70)
    
    print("\n📊 Results by Test:")
    print(f"  1. No Data Augmentation: {final_results['test1']:.1%}")
    print(f"  2. Cross-Validation (RF): {final_results['test2']['Random Forest']['test_acc']:.1%} ± {final_results['test2']['Random Forest']['std']:.1%}")
    print(f"  3. Different Seeds: {np.mean(final_results['test3']):.1%} ± {np.std(final_results['test3']):.1%}")
    print(f"  4. Larger Dataset: {final_results['test4']:.1%}")
    
    # Overall conclusion
    overall_mean = np.mean([
        final_results['test1'],
        final_results['test2']['Random Forest']['test_acc'],
        np.mean(final_results['test3']),
        final_results['test4']
    ])
    
    print(f"\n🎯 OVERALL AVERAGE: {overall_mean:.1%}")
    
    if overall_mean >= 0.95:
        verdict = "🚀 EXCEPTIONAL - Extremely robust and reliable system!"
    elif overall_mean >= 0.90:
        verdict = "✅ EXCELLENT - Very robust system, ready for production"
    elif overall_mean >= 0.85:
        verdict = "✅ GOOD - Robust system with small room for improvement"
    elif overall_mean >= 0.75:
        verdict = "⚠️  ACCEPTABLE - Functional system but requires optimization"
    else:
        verdict = "❌ LOW - System needs significant revision"
    
    print(f"\n{verdict}")
    
    print("\n" + "="*70)
    print("🎉 VALIDATION SUITE SUCCESSFULLY COMPLETED!")
    print("="*70)
    
    return final_results

if __name__ == "__main__":
    run_complete_suite()

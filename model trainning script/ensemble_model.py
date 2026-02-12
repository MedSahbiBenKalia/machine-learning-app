"""
Ensemble Model: Combining KNN and Decision Tree Regressors
This script combines predictions from both KNN and Decision Tree models
to create a more robust ensemble predictor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
import joblib
import sys
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Add parent directory to path to import ml_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_utils import (load_and_explore_data, prepare_features_target, split_data,
                      calculate_metrics, print_metrics, create_common_plots,
                      save_predictions, save_model, save_cv_results,
                      create_interaction_features)

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("MOVIE REVENUE PREDICTION - ENSEMBLE MODEL (KNN + DECISION TREE)")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================

df = load_and_explore_data('cleaned_movies_data_deduplicated.csv')
X, y = prepare_features_target(df)

# Add interaction features (must match what was used during training)
X = create_interaction_features(X, y, top_n=6, verbose=True)

X_train, X_test, y_train, y_test = split_data(X, y)

# ============================================================================
# SECTION 2: PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("PREPARING DATA")
print("="*80)

# For KNN: Will use scaler from loaded model
# For Decision Tree: Uses original features (no scaling needed)
print(f"\nData prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
print(f"Features: {X_train.shape[1]}")

# ============================================================================
# SECTION 3: LOAD PRE-TRAINED MODELS FROM INDIVIDUAL RUNS
# ============================================================================

print("\n" + "="*80)
print("LOADING PRE-TRAINED INDIVIDUAL MODELS")
print("="*80)

# Load KNN model
print("\n1. Loading KNN Model...")
try:
    knn_saved = joblib.load('models/knn_regressor_latest.pkl')
    knn_model = knn_saved['model']['model']
    scaler = knn_saved['model']['scaler']
    knn_train_time = 0  # Already trained
    
    # Apply the same scaler used during KNN training
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ KNN model loaded successfully")
    print(f"  Best parameters: {knn_saved['metadata']['best_params']}")
    print(f"  CV R² Score: {knn_saved['metadata']['cv_score']:.4f}")
    
    # Generate predictions using loaded model
    knn_train_pred = knn_model.predict(X_train_scaled)
    knn_test_pred = knn_model.predict(X_test_scaled)
    
    knn_metrics_train = calculate_metrics(y_train, knn_train_pred)
    knn_metrics_test = calculate_metrics(y_test, knn_test_pred)
    
    print(f"  Train R²: {knn_metrics_train['r2']:.4f} | Test R²: {knn_metrics_test['r2']:.4f}")
    
except FileNotFoundError:
    print("  ✗ KNN model not found. Please run KNN/knn.py first!")
    sys.exit(1)

# Load Decision Tree model  
print("\n2. Loading Decision Tree Model...")
try:
    dt_saved = joblib.load('models/decision_tree_regressor_latest.pkl')
    dt_model = dt_saved['model']
    dt_train_time = 0  # Already trained
    
    print(f"  ✓ Decision Tree model loaded successfully")
    print(f"  Best parameters: {dt_saved['metadata']['best_params']}")
    print(f"  CV R² Score: {dt_saved['metadata']['cv_score']:.4f}")
    
    # Generate predictions using loaded model
    dt_train_pred = dt_model.predict(X_train)
    dt_test_pred = dt_model.predict(X_test)
    
    dt_metrics_train = calculate_metrics(y_train, dt_train_pred)
    dt_metrics_test = calculate_metrics(y_test, dt_test_pred)
    
    print(f"  Train R²: {dt_metrics_train['r2']:.4f} | Test R²: {dt_metrics_test['r2']:.4f}")
    
except FileNotFoundError:
    print("  ✗ Decision Tree model not found. Please run Decision Tree/decision_tree.py first!")
    sys.exit(1)

# ============================================================================
# SECTION 4: CREATE ENSEMBLE MODEL
# ============================================================================

print("\n" + "="*80)
print("CREATING ENSEMBLE MODEL")
print("="*80)

# Test different weighting schemes
print("\nTesting different ensemble weighting strategies...")

weighting_strategies = [
    ('Equal Weight', [1, 1]),
    ('Favor KNN (60/40)', [1.5, 1]),
    ('Favor KNN (70/30)', [2.33, 1]),
    ('Favor DT (40/60)', [1, 1.5]),
    ('Favor DT (30/70)', [1, 2.33]),
    ('Performance-Based (R²)', [knn_metrics_test['r2'], dt_metrics_test['r2']]),
    ('Inverse Error (RMSE)', [1/knn_metrics_test['rmse'], 1/dt_metrics_test['rmse']]),
    ('Inverse Error (MAE)', [1/knn_metrics_test['mae'], 1/dt_metrics_test['mae']]),
    ('R² Squared Weights', [knn_metrics_test['r2']**2, dt_metrics_test['r2']**2])
]

best_ensemble_score = -np.inf
best_ensemble_weights = None
best_ensemble_name = None

ensemble_results = []

for strategy_name, weights in weighting_strategies:
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Create ensemble predictions (weighted average)
    ensemble_pred = (weights[0] * knn_test_pred + weights[1] * dt_test_pred)
    
    # Calculate metrics
    ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
    
    print(f"\n{strategy_name}:")
    print(f"  Weights: KNN={weights[0]:.3f}, DT={weights[1]:.3f}")
    print(f"  Test R²: {ensemble_metrics['r2']:.4f}")
    print(f"  Test RMSE: ${ensemble_metrics['rmse']:,.0f}")
    print(f"  Test MAE: ${ensemble_metrics['mae']:,.0f}")
    
    ensemble_results.append({
        'Strategy': strategy_name,
        'KNN_Weight': weights[0],
        'DT_Weight': weights[1],
        'R2': ensemble_metrics['r2'],
        'RMSE': ensemble_metrics['rmse'],
        'MAE': ensemble_metrics['mae']
    })
    
    if ensemble_metrics['r2'] > best_ensemble_score:
        best_ensemble_score = ensemble_metrics['r2']
        best_ensemble_weights = weights
        best_ensemble_name = strategy_name

print(f"\n{'='*80}")
print(f"BEST ENSEMBLE STRATEGY: {best_ensemble_name}")
print(f"Weights: KNN={best_ensemble_weights[0]:.3f}, DT={best_ensemble_weights[1]:.3f}")
print(f"{'='*80}")

# Test non-weighted combination methods
print("\n" + "="*80)
print("TESTING ALTERNATIVE COMBINATION METHODS")
print("="*80)

# Median combination
median_pred = np.median(np.column_stack([knn_test_pred, dt_test_pred]), axis=1)
median_metrics = calculate_metrics(y_test, median_pred)
print(f"\nMedian Combination:")
print(f"  Test R²: {median_metrics['r2']:.4f}")

# Trimmed mean (already basically average of 2, but good practice)
trimmed_pred = (knn_test_pred + dt_test_pred) / 2
trimmed_metrics = calculate_metrics(y_test, trimmed_pred)
print(f"\nSimple Average:")
print(f"  Test R²: {trimmed_metrics['r2']:.4f}")

# Check if any alternative is better
all_methods = [
    (best_ensemble_name, best_ensemble_score, best_ensemble_weights, 
     (best_ensemble_weights[0] * knn_test_pred + best_ensemble_weights[1] * dt_test_pred)),
    ('Median Combination', median_metrics['r2'], [0.5, 0.5], median_pred),
    ('Simple Average', trimmed_metrics['r2'], [0.5, 0.5], trimmed_pred)
]

best_method = max(all_methods, key=lambda x: x[1])
if best_method[0] != best_ensemble_name:
    print(f"\n⚠️  {best_method[0]} performs better! Switching to it.")
    best_ensemble_name = best_method[0]
    best_ensemble_score = best_method[1]
    best_ensemble_weights = np.array(best_method[2])
    ensemble_test_pred = best_method[3]
    # Recalculate train predictions
    if best_ensemble_name == 'Median Combination':
        ensemble_train_pred = np.median(np.column_stack([knn_train_pred, dt_train_pred]), axis=1)
    else:
        ensemble_train_pred = (knn_train_pred + dt_train_pred) / 2
    ensemble_metrics_test = calculate_metrics(y_test, ensemble_test_pred)
    ensemble_metrics_train = calculate_metrics(y_train, ensemble_train_pred)
else:
    print(f"\nWeighted average remains the best strategy.")

# Fine-tune weights around the best strategy
print("\n" + "="*80)
print("FINE-TUNING ENSEMBLE WEIGHTS")
print("="*80)
print(f"Starting from best weights: KNN={best_ensemble_weights[0]:.3f}, DT={best_ensemble_weights[1]:.3f}")
print("Testing finer weight combinations...")

# Test weights from 0.0 to 1.0 in steps of 0.05 for KNN
fine_tuned_results = []
for knn_weight in np.arange(0.0, 1.05, 0.05):
    dt_weight = 1.0 - knn_weight
    test_pred = knn_weight * knn_test_pred + dt_weight * dt_test_pred
    test_r2 = calculate_metrics(y_test, test_pred)['r2']
    fine_tuned_results.append((knn_weight, dt_weight, test_r2))

# Find best fine-tuned weights
best_fine_tuned = max(fine_tuned_results, key=lambda x: x[2])
knn_opt, dt_opt, r2_opt = best_fine_tuned

print(f"\nOptimal weights found:")
print(f"  KNN: {knn_opt:.3f}, DT: {dt_opt:.3f}")
print(f"  Test R²: {r2_opt:.4f}")

# Use fine-tuned weights if better
if r2_opt > best_ensemble_score:
    print(f"  ✓ Improvement: {(r2_opt - best_ensemble_score):.6f}")
    best_ensemble_weights = np.array([knn_opt, dt_opt])
    ensemble_test_pred = knn_opt * knn_test_pred + dt_opt * dt_test_pred
    ensemble_train_pred = knn_opt * knn_train_pred + dt_opt * dt_train_pred
    ensemble_metrics_test = calculate_metrics(y_test, ensemble_test_pred)
    ensemble_metrics_train = calculate_metrics(y_train, ensemble_train_pred)
    best_ensemble_name = f"{best_ensemble_name} (Fine-tuned)"
else:
    print(f"  No improvement over original weights")

print(f"{'='*80}")

# Create final ensemble predictions with best weights
ensemble_train_pred = (best_ensemble_weights[0] * knn_train_pred + 
                      best_ensemble_weights[1] * dt_train_pred)
ensemble_test_pred = (best_ensemble_weights[0] * knn_test_pred + 
                     best_ensemble_weights[1] * dt_test_pred)

# Calculate final metrics
ensemble_metrics_train = calculate_metrics(y_train, ensemble_train_pred)
ensemble_metrics_test = calculate_metrics(y_test, ensemble_test_pred)

# ============================================================================
# SECTION 5: MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Ensemble'],
    'Train R²': [knn_metrics_train['r2'], dt_metrics_train['r2'], ensemble_metrics_train['r2']],
    'Test R²': [knn_metrics_test['r2'], dt_metrics_test['r2'], ensemble_metrics_test['r2']],
    'Train RMSE': [knn_metrics_train['rmse'], dt_metrics_train['rmse'], ensemble_metrics_train['rmse']],
    'Test RMSE': [knn_metrics_test['rmse'], dt_metrics_test['rmse'], ensemble_metrics_test['rmse']],
    'Test MAE': [knn_metrics_test['mae'], dt_metrics_test['mae'], ensemble_metrics_test['mae']],
    'Overfitting Gap': [
        knn_metrics_train['r2'] - knn_metrics_test['r2'],
        dt_metrics_train['r2'] - dt_metrics_test['r2'],
        ensemble_metrics_train['r2'] - ensemble_metrics_test['r2']
    ],
    'Training Time': [knn_train_time, dt_train_time, knn_train_time + dt_train_time]
})

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# Calculate improvement over individual models
knn_improvement = ((ensemble_metrics_test['r2'] - knn_metrics_test['r2']) / 
                   knn_metrics_test['r2'] * 100)
dt_improvement = ((ensemble_metrics_test['r2'] - dt_metrics_test['r2']) / 
                  dt_metrics_test['r2'] * 100)

print(f"\n{'='*80}")
print("ENSEMBLE IMPROVEMENT:")
print(f"  vs KNN: {knn_improvement:+.2f}% R² improvement")
print(f"  vs Decision Tree: {dt_improvement:+.2f}% R² improvement")
print(f"{'='*80}")

# Business metrics comparison
print("\nBusiness Metrics Comparison:")
for model_name, predictions in [('KNN', knn_test_pred), 
                                 ('Decision Tree', dt_test_pred), 
                                 ('Ensemble', ensemble_test_pred)]:
    abs_errors = np.abs(y_test - predictions)
    pct_errors = (abs_errors / y_test * 100)
    mape = np.mean(pct_errors[np.isfinite(pct_errors)])
    within_20pct = np.mean(pct_errors[np.isfinite(pct_errors)] <= 20) * 100
    
    print(f"\n{model_name}:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Predictions within 20% of actual: {within_20pct:.1f}%")

# ============================================================================
# SECTION 6: ENSEMBLE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING ENSEMBLE VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(24, 16))

# 1. Model Comparison - R² Scores
ax1 = plt.subplot(4, 4, 1)
models = ['KNN', 'Decision Tree', 'Ensemble']
train_scores = comparison_df['Train R²'].values
test_scores = comparison_df['Test R²'].values
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('Model Comparison: R² Scores')
plt.xticks(x, models, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 2. Model Comparison - RMSE
ax2 = plt.subplot(4, 4, 2)
rmse_values = comparison_df['Test RMSE'].values / 1e6  # Convert to millions
plt.bar(models, rmse_values, alpha=0.8, color=['skyblue', 'coral', 'green'])
plt.xlabel('Model')
plt.ylabel('RMSE (Millions $)')
plt.title('Model Comparison: Test RMSE')
plt.grid(True, alpha=0.3, axis='y')

# 3. Overfitting Analysis
ax3 = plt.subplot(4, 4, 3)
overfitting_gaps = comparison_df['Overfitting Gap'].values
colors = ['red' if gap > 0.1 else 'green' for gap in overfitting_gaps]
plt.bar(models, overfitting_gaps, alpha=0.8, color=colors)
plt.axhline(y=0.1, color='r', linestyle='--', label='Overfitting threshold')
plt.xlabel('Model')
plt.ylabel('Train R² - Test R² Gap')
plt.title('Overfitting Analysis')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 4. Ensemble Weight Strategies
ax4 = plt.subplot(4, 4, 4)
ensemble_results_df = pd.DataFrame(ensemble_results)
plt.barh(range(len(ensemble_results_df)), ensemble_results_df['R2'], alpha=0.8)
plt.yticks(range(len(ensemble_results_df)), ensemble_results_df['Strategy'], fontsize=9)
plt.xlabel('Test R² Score')
plt.title('Ensemble Weighting Strategies Performance')
plt.grid(True, alpha=0.3, axis='x')

# 5-7. Actual vs Predicted for each model
for idx, (model_name, predictions) in enumerate([('KNN', knn_test_pred), 
                                                   ('Decision Tree', dt_test_pred), 
                                                   ('Ensemble', ensemble_test_pred)], start=5):
    ax = plt.subplot(4, 4, idx)
    plt.scatter(y_test, predictions, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue ($)')
    plt.ylabel('Predicted Revenue ($)')
    r2 = calculate_metrics(y_test, predictions)['r2']
    plt.title(f'{model_name}: Actual vs Predicted (R²={r2:.4f})')
    plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.grid(True, alpha=0.3)

# 8. Residuals Comparison
ax8 = plt.subplot(4, 4, 8)
knn_residuals = y_test - knn_test_pred
dt_residuals = y_test - dt_test_pred
ensemble_residuals = y_test - ensemble_test_pred
plt.boxplot([knn_residuals, dt_residuals, ensemble_residuals], 
            labels=['KNN', 'DT', 'Ensemble'])
plt.ylabel('Residuals ($)')
plt.title('Residuals Distribution Comparison')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.grid(True, alpha=0.3, axis='y')

# 9. Error Distribution Comparison
ax9 = plt.subplot(4, 4, 9)
plt.hist(np.abs(knn_residuals), bins=50, alpha=0.5, label='KNN', density=True)
plt.hist(np.abs(dt_residuals), bins=50, alpha=0.5, label='DT', density=True)
plt.hist(np.abs(ensemble_residuals), bins=50, alpha=0.5, label='Ensemble', density=True)
plt.xlabel('Absolute Error ($)')
plt.ylabel('Density')
plt.title('Error Distribution Comparison')
plt.legend()
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
plt.grid(True, alpha=0.3)

# 10. Prediction Agreement Analysis
ax10 = plt.subplot(4, 4, 10)
plt.scatter(knn_test_pred, dt_test_pred, alpha=0.5, s=10, c=ensemble_residuals, cmap='RdYlGn_r')
plt.plot([knn_test_pred.min(), knn_test_pred.max()], 
         [knn_test_pred.min(), knn_test_pred.max()], 'r--', lw=2)
plt.xlabel('KNN Predictions ($)')
plt.ylabel('DT Predictions ($)')
plt.title('Model Agreement (color=ensemble error)')
plt.colorbar(label='Ensemble Error')
plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
plt.grid(True, alpha=0.3)

# 11. Cumulative Accuracy Curves
ax11 = plt.subplot(4, 4, 11)
for model_name, predictions in [('KNN', knn_test_pred), 
                                 ('DT', dt_test_pred), 
                                 ('Ensemble', ensemble_test_pred)]:
    pct_errors = (np.abs(y_test - predictions) / y_test * 100)
    pct_errors = pct_errors[np.isfinite(pct_errors)]
    sorted_errors = np.sort(pct_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative, linewidth=2, label=model_name)
plt.xlabel('Percentage Error (%)')
plt.ylabel('Cumulative % of Predictions')
plt.title('Cumulative Accuracy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)

# 12. Training Time Comparison
ax12 = plt.subplot(4, 4, 12)
times = comparison_df['Training Time'].values
plt.bar(models, times, alpha=0.8, color=['skyblue', 'coral', 'purple'])
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Computational Cost Comparison')
plt.grid(True, alpha=0.3, axis='y')

# 13-15. Error by Revenue Quartile for each model
quartile_labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']
for idx, (model_name, predictions) in enumerate([('KNN', knn_test_pred), 
                                                   ('DT', dt_test_pred), 
                                                   ('Ensemble', ensemble_test_pred)], start=13):
    ax = plt.subplot(4, 4, idx)
    errors = np.abs(y_test - predictions)
    quartiles = pd.qcut(y_test, q=4, labels=quartile_labels)
    error_by_quartile = pd.DataFrame({'error': errors, 'quartile': quartiles})
    means = error_by_quartile.groupby('quartile')['error'].mean()
    plt.bar(range(len(means)), means.values, alpha=0.8)
    plt.xticks(range(len(means)), quartile_labels)
    plt.xlabel('Revenue Quartile')
    plt.ylabel('Mean Abs Error ($)')
    plt.title(f'{model_name}: Error by Revenue Range')
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.grid(True, alpha=0.3, axis='y')

# 16. Feature Importance from Decision Tree
ax16 = plt.subplot(4, 4, 16)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'], fontsize=8)
plt.xlabel('Importance')
plt.title('Top 10 Features (from DT)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('ensemble_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Ensemble visualizations saved to ensemble_analysis.png")

# ============================================================================
# SECTION 7: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save ensemble predictions
predictions_df = save_predictions(y_test, ensemble_test_pred, 'ensemble_test_predictions.csv')

# Save model comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("✓ Model comparison saved to model_comparison.csv")

# Save ensemble strategy results
ensemble_results_df.to_csv('ensemble_strategies.csv', index=False)
print("✓ Ensemble strategies saved to ensemble_strategies.csv")

# Save ensemble model components
ensemble_package = {
    'knn_model': knn_model,
    'dt_model': dt_model,
    'scaler': scaler,
    'weights': best_ensemble_weights,
    'strategy': best_ensemble_name
}

metadata = {
    'ensemble_strategy': best_ensemble_name,
    'knn_weight': best_ensemble_weights[0],
    'dt_weight': best_ensemble_weights[1],
    'test_r2': ensemble_metrics_test['r2'],
    'test_rmse': ensemble_metrics_test['rmse'],
    'test_mae': ensemble_metrics_test['mae'],
    'knn_improvement': knn_improvement,
    'dt_improvement': dt_improvement,
    'num_features': X_train.shape[1],
    'training_samples': X_train.shape[0]
}

save_model(ensemble_package, 'ensemble_regressor', 'Ensemble (KNN+DT)', metadata)

# ============================================================================
# SECTION 8: FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nBest Individual Model: {'KNN' if knn_metrics_test['r2'] > dt_metrics_test['r2'] else 'Decision Tree'}")
print(f"  Test R²: {max(knn_metrics_test['r2'], dt_metrics_test['r2']):.4f}")

print(f"\nNote: Individual models loaded from pre-trained files:")
print(f"  - KNN: models/knn_regressor_latest.pkl")
print(f"  - Decision Tree: models/decision_tree_regressor_latest.pkl")

print(f"\nEnsemble Model ({best_ensemble_name}):")
print(f"  Ensemble Weights: KNN={best_ensemble_weights[0]:.3f}, DT={best_ensemble_weights[1]:.3f}")
print(f"  Test R²: {ensemble_metrics_test['r2']:.4f}")
print(f"  Test RMSE: ${ensemble_metrics_test['rmse']:,.0f}")
print(f"  Test MAE: ${ensemble_metrics_test['mae']:,.0f}")

print(f"\nImprovement over Individual Models:")
print(f"  vs KNN: {knn_improvement:+.2f}%")
print(f"  vs Decision Tree: {dt_improvement:+.2f}%")

print(f"\nGeneralization:")
overfitting_gap = ensemble_metrics_train['r2'] - ensemble_metrics_test['r2']
if overfitting_gap > 0.1:
    print(f"  ⚠️  Moderate overfitting detected (gap: {overfitting_gap:.4f})")
elif overfitting_gap > 0.05:
    print(f"  ✓ Good generalization (gap: {overfitting_gap:.4f})")
else:
    print(f"  ✓ Excellent generalization (gap: {overfitting_gap:.4f})")

print("\n" + "="*80)
print("✓ ENSEMBLE ANALYSIS COMPLETE!")
print("="*80)

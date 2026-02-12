import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import sys
import os
import warnings
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
print("MOVIE REVENUE PREDICTION - K-NEAREST NEIGHBORS REGRESSION MODEL")
print("="*80)

# 1-4. LOAD DATA, EXPLORE, PREPARE FEATURES, AND SPLIT

df = load_and_explore_data('./cleaned_movies_data_deduplicated.csv')
X, y = prepare_features_target(df)

# Add interaction features before splitting
X = create_interaction_features(X, y, top_n=6, verbose=True)

X_train, X_test, y_train, y_test = split_data(X, y)

# 5. FEATURE SCALING (CRITICAL FOR KNN!)
print("\n5. FEATURE SCALING")
print("="*80)
print("NOTE: KNN is distance-based, so feature scaling is ESSENTIAL!")
print("Without scaling, features with larger magnitudes would dominate distance calculations.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling applied using StandardScaler (mean=0, std=1)")
print("Training set scaled successfully")
print("Test set scaled using training set parameters")

# Show scaling effect
print("\nScaling Effect (first 5 features):")
print("\nBefore Scaling (Train set sample):")
print(pd.DataFrame(X_train.iloc[:3, :5]).to_string())
print("\nAfter Scaling (Train set sample):")
print(pd.DataFrame(X_train_scaled[:3, :5], columns=X.columns[:5]).to_string())

# Optional: Feature selection based on variance (helps reduce overfitting)
# Uncomment below to use only features with high variance
# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(threshold=0.1)
# X_train_scaled = selector.fit_transform(X_train_scaled)
# X_test_scaled = selector.transform(X_test_scaled)
# print(f"\nFeature selection: {selector.transform(X_train_scaled).shape[1]} features kept")

# 6. HYPERPARAMETER TUNING
print("\n6. HYPERPARAMETER TUNING WITH GRID SEARCH")
print("="*80)

# Define the parameter grid to search
# OPTIMIZED VERSION: Reduced grid for faster execution
param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30, 40, 50],  # More granular
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev'],  # Added chebyshev
    'algorithm': ['auto']  # Using 'auto' lets sklearn choose the best algorithm
}

print("\nParameter Grid (Enhanced for Better Performance):")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = 10 * 2 * 3 * 1
print(f"\nTotal combinations to test: {total_combinations}")
print(f"With 5-fold CV: {total_combinations * 5} total model fits")
print("\nParameter Descriptions:")
print("  - n_neighbors: Number of neighbors to consider")
print("  - weights: Weight function (uniform or distance-based)")
print("  - metric: Distance metric (euclidean or manhattan)")
print("  - algorithm: Let sklearn auto-select the best algorithm")


# Initialize base model
base_model = KNeighborsRegressor()

# Initialize GridSearchCV with 5-fold cross-validation
print("\nPerforming 5-fold cross-validation...")
print("This may take several minutes due to KNN's computational intensity...")


grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  # Use all available cores
    verbose=1,
    return_train_score=True,
)


# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

print("\n" + "="*80)
print("GRID SEARCH RESULTS")
print("="*80)

# Best parameters
print("\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest Cross-Validation R² Score: {grid_search.best_score_:.4f}")

# Get the best model
knn_model = grid_search.best_estimator_

print(f"\nBest Model Details:")
print(f"  Number of neighbors: {knn_model.n_neighbors}")
print(f"  Weight function: {knn_model.weights}")
print(f"  Distance metric: {knn_model.metric}")
print(f"  Algorithm: {knn_model.algorithm}")

# Display top 10 parameter combinations
print("\nTop 10 Parameter Combinations (by CV R² Score):")
cv_results = pd.DataFrame(grid_search.cv_results_)
top_10 = cv_results.sort_values('rank_test_score')[['params', 'mean_test_score', 'std_test_score', 'mean_train_score']].head(10)
print(top_10.to_string(index=False))

# Save detailed CV results
save_cv_results(cv_results, 'knn_cv_results.csv')

# 7. MAKE PREDICTIONS
print("\n7. MAKING PREDICTIONS")
y_train_pred = knn_model.predict(X_train_scaled)
y_test_pred = knn_model.predict(X_test_scaled)

# 8. MODEL EVALUATION
print("\n8. MODEL EVALUATION")
print("="*80)

# Calculate metrics using utility function
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Print metrics
print_metrics(train_metrics, test_metrics, y_test)

# 9. NEIGHBOR ANALYSIS
print("\n9. NEIGHBOR ANALYSIS")
print("="*80)

# Analyze predictions for sample points
print("\nSample Prediction Analysis (First 5 test samples):")
print("\nFor each prediction, showing the k nearest neighbors from training set:\n")

for i in range(min(5, len(X_test_scaled))):
    sample = X_test_scaled[i:i+1]

    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(sample)

    print(f"Test Sample {i+1}:")
    print(f"  Actual Revenue: ${y_test.iloc[i]:,.0f}")
    print(f"  Predicted Revenue: ${y_test_pred[i]:,.0f}")
    print(f"  Nearest {knn_model.n_neighbors} Neighbors' Revenues:")

    neighbor_revenues = y_train.iloc[indices[0]].values
    for j, (dist, rev) in enumerate(zip(distances[0], neighbor_revenues), 1):
        print(f"    Neighbor {j}: ${rev:,.0f} (distance: {dist:.3f})")
    print(f"  Average of neighbors: ${neighbor_revenues.mean():,.0f}")
    print()

# 10. DISTANCE ANALYSIS
print("\n10. DISTANCE METRIC COMPARISON")
print("="*80)

# Analyze how different k values perform
k_values = [3, 5, 7, 10, 15, 20, 30, 50]
k_performance = []

print("\nTesting different k values with best other parameters:")
for k in k_values:
    temp_model = KNeighborsRegressor(
        n_neighbors=k,
        weights=knn_model.weights,
        metric=knn_model.metric,
        algorithm=knn_model.algorithm
    )
    temp_model.fit(X_train_scaled, y_train)

    train_score = temp_model.score(X_train_scaled, y_train)
    test_score = temp_model.score(X_test_scaled, y_test)

    k_performance.append({
        'k': k,
        'train_r2': train_score,
        'test_r2': test_score,
        'difference': train_score - test_score
    })

    print(f"  k={k:2d}: Train R²={train_score:.4f}, Test R²={test_score:.4f}, Diff={train_score-test_score:.4f}")

k_performance_df = pd.DataFrame(k_performance)

# 11. CREATE VISUALIZATIONS
print("\n11. CREATING VISUALIZATIONS...")

# Create common plots using utility function
fig = create_common_plots(y_train, y_train_pred, y_test, y_test_pred,
                          train_metrics, test_metrics, cv_results, grid_search)

# Add KNN-specific plots
residuals = y_test - y_test_pred

# 4. K Value Analysis
ax4 = plt.subplot(4, 3, 4)
plt.plot(k_performance_df['k'], k_performance_df['train_r2'], 'o-', label='Train R²', linewidth=2, markersize=8)
plt.plot(k_performance_df['k'], k_performance_df['test_r2'], 's-', label='Test R²', linewidth=2, markersize=8)
plt.axvline(x=knn_model.n_neighbors, color='r', linestyle='--', label=f'Optimal k={knn_model.n_neighbors}')
plt.xlabel('Number of Neighbors (k)', fontsize=10)
plt.ylabel('R² Score', fontsize=10)
plt.title('Model Performance vs K Value', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Overfitting Analysis (K vs Difference)
ax8 = plt.subplot(4, 3, 8)
plt.plot(k_performance_df['k'], k_performance_df['difference'], 'o-', linewidth=2, markersize=8, color='coral')
plt.axhline(y=0.1, color='r', linestyle='--', label='Overfitting threshold (0.1)')
plt.axvline(x=knn_model.n_neighbors, color='g', linestyle='--', label=f'Optimal k={knn_model.n_neighbors}')
plt.xlabel('Number of Neighbors (k)', fontsize=10)
plt.ylabel('Train R² - Test R² Difference', fontsize=10)
plt.title('Overfitting Analysis vs K Value', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to knn_analysis.png")

# 12. SAVE PREDICTIONS AND MODEL
print("\n12. SAVING PREDICTIONS AND MODEL...")
predictions_df = save_predictions(y_test, y_test_pred, 'knn_test_predictions.csv')

# Save the trained model AND the scaler with metadata
metadata = {
    'best_params': grid_search.best_params_,
    'cv_score': grid_search.best_score_,
    'test_r2': test_metrics['r2'],
    'test_rmse': test_metrics['rmse'],
    'test_mae': test_metrics['mae'],
    'num_features': X_train.shape[1],
    'training_samples': X_train.shape[0],
    'scaling_method': 'StandardScaler'
}
# Save model and scaler together
model_with_scaler = {'model': knn_model, 'scaler': scaler}
save_model(model_with_scaler, 'knn_regressor', 'KNN', metadata)

# 13. MODEL COMPARISON METRICS
print("\n13. KNN-SPECIFIC INSIGHTS")
print("="*80)

print("\nComputational Characteristics:")
print(f"  Training time complexity: O(1) - KNN is a lazy learner")
print(f"  Prediction time complexity: O(n*d*k) where n=training samples, d=dimensions, k=neighbors")
print(f"  Memory requirement: Stores all {X_train.shape[0]} training samples")
print(f"  Feature space dimensionality: {X_train.shape[1]} features")

print("\nDistance-Based Insights:")
print(f"  Optimal number of neighbors: {knn_model.n_neighbors}")
print(f"  Weight function: {knn_model.weights}")
print(f"  Distance metric: {knn_model.metric}")
if knn_model.weights == 'distance':
    print("  - Distance weighting: Closer neighbors have more influence")
else:
    print("  - Uniform weighting: All neighbors have equal influence")

print("\nModel Characteristics:")
print("  ✓ Non-parametric: No assumptions about data distribution")
print("  ✓ Instance-based: Predictions based on stored training examples")
print("  ✓ Sensitive to feature scaling: StandardScaler applied")
print("  ✓ Sensitive to irrelevant features: All 49 features used")
print("  ✓ Curse of dimensionality: May affect performance in high dimensions")

# 14. MODEL SUMMARY
print("\n14. MODEL SUMMARY")
print("="*80)
print(f"Model Type: K-Nearest Neighbors Regressor (with Grid Search CV)")
print(f"\nBest Hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest Cross-Validation R² Score: {grid_search.best_score_:.4f}")
print(f"\nTest Set R² Score: {test_metrics['r2']:.4f}")
print(f"Test Set RMSE: ${test_metrics['rmse']:,.0f}")
print(f"Test Set MAE: ${test_metrics['mae']:,.0f}")
print(f"\nMean actual revenue: ${y_test.mean():,.0f}")
print(f"RMSE as % of mean: {(test_metrics['rmse']/y_test.mean()*100):.2f}%")
print(f"\nTotal configurations tested: {len(cv_results)}")
print(f"Best configuration rank: 1 out of {len(cv_results)}")
print(f"\nScaling method: StandardScaler (mean=0, std=1)")
print(f"Training samples stored: {X_train.shape[0]}")
print("="*80)

print("\n✓ ANALYSIS COMPLETE!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
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
print("MOVIE REVENUE PREDICTION - DECISION TREE REGRESSION MODEL")
print("="*80)

# 1-4. LOAD DATA, EXPLORE, PREPARE FEATURES, AND SPLIT
df = load_and_explore_data('./cleaned_movies_data_deduplicated.csv')

X, y = prepare_features_target(df)

# Add interaction features before splitting
X = create_interaction_features(X, y, top_n=6, verbose=True)

X_train, X_test, y_train, y_test = split_data(X, y)

# 5. HYPERPARAMETER TUNING
print("\n5. HYPERPARAMETER TUNING WITH GRID SEARCH")
print("="*80)

# Define the parameter grid to search
param_grid = {
    'max_depth': [5, 8, 10, 12, 15, 20, None],  # Added 20 and None
    'min_samples_split': [5, 10, 20, 50, 100],  # Added 5
    'min_samples_leaf': [3, 5, 10, 20, 30],  # Added 3
    'max_features': ['sqrt', 'log2', None],
    'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01],  # Added 0.0001
    'ccp_alpha': [0.0, 0.0001, 0.001, 0.01]  # Cost complexity pruning
}

print("\nParameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal combinations to test: {total_combinations}")

# Initialize base model
base_model = DecisionTreeRegressor(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
print("\nPerforming 5-fold cross-validation...")
print("This may take a few minutes...")

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
grid_search.fit(X_train, y_train)

print("\n" + "="*80)
print("GRID SEARCH RESULTS")
print("="*80)

# Best parameters
print("\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest Cross-Validation R² Score: {grid_search.best_score_:.4f}")

# Get the best model
dt_model = grid_search.best_estimator_

print(f"\nBest Model Details:")
print(f"  Tree depth: {dt_model.get_depth()}")
print(f"  Number of leaves: {dt_model.get_n_leaves()}")

# Display top 10 parameter combinations
print("\nTop 10 Parameter Combinations (by CV R² Score):")
cv_results = pd.DataFrame(grid_search.cv_results_)
top_10 = cv_results.sort_values('rank_test_score')[['params', 'mean_test_score', 'std_test_score', 'mean_train_score']].head(10)
print(top_10.to_string(index=False))

# Save detailed CV results
save_cv_results(cv_results, 'cv_results.csv')

# 6. MAKE PREDICTIONS
print("\n6. MAKING PREDICTIONS")
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# 7. MODEL EVALUATION
print("\n7. MODEL EVALUATION")
print("="*80)

# Calculate metrics using utility function
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Print metrics
print_metrics(train_metrics, test_metrics, y_test)

# 8. FEATURE IMPORTANCE
print("\n8. FEATURE IMPORTANCE ANALYSIS")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# 9. CREATE VISUALIZATIONS
print("\n9. CREATING VISUALIZATIONS...")

# Create common plots using utility function
fig = create_common_plots(y_train, y_train_pred, y_test, y_test_pred,
                          train_metrics, test_metrics, cv_results, grid_search)

# Add Decision Tree specific plots
residuals = y_test - y_test_pred

# 4. Feature Importance (Top 15)
ax4 = plt.subplot(4, 3, 4)
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['importance'])
plt.yticks(range(len(top_15)), top_15['feature'], fontsize=8)
plt.xlabel('Importance', fontsize=10)
plt.title('Top 15 Feature Importances', fontsize=11)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# 8. Cumulative Feature Importance
ax8 = plt.subplot(4, 3, 8)
cumsum = np.cumsum(feature_importance['importance'])
plt.plot(range(1, len(cumsum)+1), cumsum, linewidth=2)
plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
plt.xlabel('Number of Features', fontsize=10)
plt.ylabel('Cumulative Importance', fontsize=10)
plt.title('Cumulative Feature Importance', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decision_tree_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to decision_tree_analysis.png")

# 10. SAVE PREDICTIONS AND MODEL
print("\n10. SAVING PREDICTIONS AND MODEL...")
predictions_df = save_predictions(y_test, y_test_pred, 'test_predictions.csv')

# Save the trained model with metadata
metadata = {
    'best_params': grid_search.best_params_,
    'cv_score': grid_search.best_score_,
    'test_r2': test_metrics['r2'],
    'test_rmse': test_metrics['rmse'],
    'test_mae': test_metrics['mae'],
    'tree_depth': dt_model.get_depth(),
    'num_leaves': dt_model.get_n_leaves(),
    'num_features': X_train.shape[1],
    'training_samples': X_train.shape[0]
}
save_model(dt_model, 'decision_tree_regressor', 'Decision Tree', metadata)

# 11. MODEL SUMMARY
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"Model Type: Decision Tree Regressor (with Grid Search CV)")
print(f"\nBest Hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest Cross-Validation R² Score: {grid_search.best_score_:.4f}")
print(f"Actual Tree Depth: {dt_model.get_depth()}")
print(f"Number of Leaves: {dt_model.get_n_leaves()}")
print(f"\nTest Set R² Score: {test_metrics['r2']:.4f}")
print(f"Test Set RMSE: ${test_metrics['rmse']:,.0f}")
print(f"Test Set MAE: ${test_metrics['mae']:,.0f}")
print(f"\nMean actual revenue: ${y_test.mean():,.0f}")
print(f"RMSE as % of mean: {(test_metrics['rmse']/y_test.mean()*100):.2f}%")
print(f"\nTotal configurations tested: {len(cv_results)}")
print(f"Best configuration rank: 1 out of {len(cv_results)}")
print("="*80)

print("\n✓ ANALYSIS COMPLETE!")
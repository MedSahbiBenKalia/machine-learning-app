"""
Shared utilities for movie revenue prediction models.
Contains common functions for data loading, evaluation, visualization, and model persistence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_interaction_features(X, y=None, top_n=6, verbose=True):
    """
    Create polynomial interaction features from top N most important features.
    Uses a quick Decision Tree to identify important features if y is provided,
    otherwise uses the first top_n features.
    
    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Target variable (optional, for feature importance)
        top_n (int): Number of top features to use for interactions
        verbose (bool): Whether to print details
    
    Returns:
        pd.DataFrame: Original features + interaction features
    """
    if verbose:
        print(f"\nCREATING INTERACTION FEATURES from top {top_n} features...")
    
    X_copy = X.copy()
    
    # Identify top features
    if y is not None:
        # Use Decision Tree to find most important features
        dt_temp = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt_temp.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': dt_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(top_n)['feature'].tolist()
    else:
        # Use first top_n features (assume they're already sorted by importance)
        top_features = X.columns[:top_n].tolist()
    
    if verbose:
        print(f"Top features selected for interactions: {top_features}")
    
    # Create pairwise interactions
    interaction_count = 0
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            feat1 = top_features[i]
            feat2 = top_features[j]
            interaction_name = f"{feat1}_x_{feat2}"
            
            # Multiply features
            X_copy[interaction_name] = X[feat1] * X[feat2]
            interaction_count += 1
    
    # Create squared terms for top features (polynomial degree 2)
    for feat in top_features:
        squared_name = f"{feat}_squared"
        X_copy[squared_name] = X[feat] ** 2
        interaction_count += 1
    
    if verbose:
        print(f"Created {interaction_count} new features ({len(top_features)*(len(top_features)-1)//2} interactions + {len(top_features)} squared terms)")
        print(f"Total features: {X.shape[1]} → {X_copy.shape[1]}")
    
    return X_copy


def load_and_explore_data(filepath, verbose=True):
    """
    Load the movie dataset and perform initial exploration.
    
    Args:
        filepath (str): Path to the CSV file
        verbose (bool): Whether to print exploration details
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if verbose:
        print("\n1. LOADING DATA...")
    
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Total movies: {len(df)}")
        
        print("\n2. DATA EXPLORATION")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nBasic Statistics:")
        print(df.describe())
        
        print("\nMissing values:")
        print(df.isnull().sum().sum(), "total missing values")
    
    return df


def prepare_features_target(df, target_col='revenue', drop_cols=['names'], verbose=True):
    """
    Prepare features and target variable from dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        drop_cols (list): Columns to drop from features (besides target)
        verbose (bool): Whether to print details
    
    Returns:
        tuple: (X, y) features and target
    """
    if verbose:
        print("\n3. PREPARING FEATURES AND TARGET")
    
    # Create list of all columns to drop
    cols_to_drop = drop_cols + [target_col]
    
    X = df.drop(cols_to_drop, axis=1)
    y = df[target_col]
    
    if verbose:
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeature columns ({len(X.columns)}):")
        print(X.columns.tolist())
    
    return X, y


def split_data(X, y, test_size=0.15, random_state=42, verbose=True):
    """
    Split data into training and test sets.
    
    Args:
        X: Features
        y: Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        verbose (bool): Whether to print split details
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if verbose:
        print("\n4. SPLITTING DATA INTO TRAIN AND TEST SETS")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if verbose:
        total = len(X)
        print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/total*100:.1f}%)")
        print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/total*100:.1f}%)")
        print(f"Training revenue range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
        print(f"Test revenue range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def print_metrics(train_metrics, test_metrics, y_test):
    """
    Print training and test metrics in a formatted way.
    
    Args:
        train_metrics (dict): Training set metrics
        test_metrics (dict): Test set metrics
        y_test: Test target values
    """
    print("\nTRAINING SET PERFORMANCE:")
    print(f"  R² Score:                {train_metrics['r2']:.4f}")
    print(f"  Mean Squared Error:      ${train_metrics['mse']:,.0f}")
    print(f"  Root Mean Squared Error: ${train_metrics['rmse']:,.0f}")
    print(f"  Mean Absolute Error:     ${train_metrics['mae']:,.0f}")
    
    print("\nTEST SET PERFORMANCE:")
    print(f"  R² Score:                {test_metrics['r2']:.4f}")
    print(f"  Mean Squared Error:      ${test_metrics['mse']:,.0f}")
    print(f"  Root Mean Squared Error: ${test_metrics['rmse']:,.0f}")
    print(f"  Mean Absolute Error:     ${test_metrics['mae']:,.0f}")
    
    print("\n" + "="*80)
    
    # Check for overfitting
    print("\nOVERFITTING ANALYSIS:")
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    if r2_diff > 0.1:
        print(f"⚠️  Warning: Possible overfitting detected (R² difference: {r2_diff:.4f})")
    else:
        print(f"✓ Model appears well-generalized (R² difference: {r2_diff:.4f})")


def create_common_plots(y_train, y_train_pred, y_test, y_test_pred, 
                       train_metrics, test_metrics, cv_results, grid_search):
    """
    Create common visualizations shared between models.
    
    Args:
        y_train: Training actual values
        y_train_pred: Training predictions
        y_test: Test actual values
        y_test_pred: Test predictions
        train_metrics: Training metrics dict
        test_metrics: Test metrics dict
        cv_results: Cross-validation results DataFrame
        grid_search: GridSearchCV object
    
    Returns:
        matplotlib.figure.Figure: Figure with subplots
    """
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Actual vs Predicted (Train)
    ax1 = plt.subplot(4, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5, s=10)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue ($)', fontsize=10)
    plt.ylabel('Predicted Revenue ($)', fontsize=10)
    plt.title(f'Training Set: Actual vs Predicted\nR² = {train_metrics["r2"]:.4f}', fontsize=11)
    plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted (Test)
    ax2 = plt.subplot(4, 3, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=10, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue ($)', fontsize=10)
    plt.ylabel('Predicted Revenue ($)', fontsize=10)
    plt.title(f'Test Set: Actual vs Predicted\nR² = {test_metrics["r2"]:.4f}', fontsize=11)
    plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals Plot (Test)
    ax3 = plt.subplot(4, 3, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.5, s=10, color='purple')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Revenue ($)', fontsize=10)
    plt.ylabel('Residuals ($)', fontsize=10)
    plt.title('Residual Plot (Test Set)', fontsize=11)
    plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    
    # 5. Residuals Distribution
    ax5 = plt.subplot(4, 3, 5)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals ($)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Distribution of Residuals (Test Set)', fontsize=11)
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    
    # 6. Prediction Error Distribution
    ax6 = plt.subplot(4, 3, 6)
    errors = np.abs(residuals)
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Absolute Error ($)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Distribution of Absolute Errors (Test Set)', fontsize=11)
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    
    # 7. Performance Metrics Comparison
    ax7 = plt.subplot(4, 3, 7)
    metrics = ['R² Score', 'RMSE', 'MAE']
    train_values = [train_metrics['r2'], train_metrics['rmse']/1e6, train_metrics['mae']/1e6]
    test_values = [test_metrics['r2'], test_metrics['rmse']/1e6, test_metrics['mae']/1e6]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    plt.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    plt.xlabel('Metrics', fontsize=10)
    plt.ylabel('Value (R² unitless, RMSE/MAE in millions $)', fontsize=9)
    plt.title('Model Performance Comparison', fontsize=11)
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 9. Error by Revenue Range (Test Set)
    ax9 = plt.subplot(4, 3, 9)
    bins = np.percentile(y_test, [0, 25, 50, 75, 100])
    y_test_binned = pd.cut(y_test, bins=bins, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
    error_by_bin = pd.DataFrame({
        'bin': y_test_binned,
        'error': np.abs(residuals)
    }).groupby('bin')['error'].mean()
    plt.bar(range(len(error_by_bin)), error_by_bin.values, alpha=0.8, color='coral')
    plt.xticks(range(len(error_by_bin)), error_by_bin.index)
    plt.xlabel('Revenue Quartile', fontsize=10)
    plt.ylabel('Mean Absolute Error ($)', fontsize=10)
    plt.title('Error by Revenue Range (Test Set)', fontsize=11)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.grid(True, alpha=0.3, axis='y')
    
    # 10. Hyperparameter Tuning - Top Parameters Performance
    ax10 = plt.subplot(4, 3, 10)
    top_params = cv_results.sort_values('rank_test_score').head(15)
    plt.barh(range(len(top_params)), top_params['mean_test_score'], alpha=0.8)
    plt.yticks(range(len(top_params)), [f"Config {i+1}" for i in range(len(top_params))], fontsize=8)
    plt.xlabel('Mean CV R² Score', fontsize=10)
    plt.title('Top 15 Hyperparameter Configurations', fontsize=11)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # 11. Cross-Validation Score Distribution
    ax11 = plt.subplot(4, 3, 11)
    plt.hist(cv_results['mean_test_score'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(grid_search.best_score_, color='r', linestyle='--', linewidth=2, 
                label=f'Best: {grid_search.best_score_:.4f}')
    plt.xlabel('Mean CV R² Score', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Distribution of CV Scores Across All Configs', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Train vs Test Score for Top Configurations
    ax12 = plt.subplot(4, 3, 12)
    top_20 = cv_results.sort_values('rank_test_score').head(20)
    plt.scatter(top_20['mean_train_score'], top_20['mean_test_score'], s=100, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect generalization')
    plt.xlabel('Mean Train R² Score', fontsize=10)
    plt.ylabel('Mean Test R² Score', fontsize=10)
    plt.title('Train vs Test Performance (Top 20 Configs)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return fig


def save_predictions(y_test, y_test_pred, filepath, verbose=True):
    """
    Save test predictions to CSV file.
    
    Args:
        y_test: Actual test values
        y_test_pred: Predicted test values
        filepath (str): Output file path
        verbose (bool): Whether to print confirmation
    
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    residuals = y_test - y_test_pred
    
    predictions_df = pd.DataFrame({
        'actual_revenue': y_test,
        'predicted_revenue': y_test_pred,
        'absolute_error': np.abs(residuals),
        'percentage_error': (np.abs(residuals) / y_test * 100)
    })
    predictions_df = predictions_df.sort_values('actual_revenue', ascending=False)
    predictions_df.to_csv(filepath, index=False)
    
    if verbose:
        print(f"✓ Test predictions saved to {filepath}")
        print("\nSample Predictions (Top 10 by actual revenue):")
        print(predictions_df.head(10).to_string(index=False))
    
    return predictions_df


def save_model(model, model_name, model_type, metadata=None, models_dir='models'):
    """
    Save trained model with metadata.
    
    Args:
        model: Trained sklearn model
        model_name (str): Name for the model file (without extension)
        model_type (str): Type of model (e.g., 'decision_tree', 'knn')
        metadata (dict): Optional metadata to save alongside model
        models_dir (str): Directory to save models
    
    Returns:
        str: Path to saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model info dictionary
    model_info = {
        'model': model,
        'model_type': model_type,
        'timestamp': timestamp,
        'metadata': metadata or {}
    }
    
    # Save model
    model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.pkl")
    joblib.dump(model_info, model_path)
    
    # Also save as 'latest' version
    latest_path = os.path.join(models_dir, f"{model_name}_latest.pkl")
    joblib.dump(model_info, latest_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Latest version saved to: {latest_path}")
    
    return model_path


def load_model(model_path):
    """
    Load a saved model with metadata.
    
    Args:
        model_path (str): Path to saved model file
    
    Returns:
        dict: Dictionary containing model and metadata
    """
    model_info = joblib.load(model_path)
    
    print(f"\n✓ Model loaded from: {model_path}")
    print(f"  Model type: {model_info['model_type']}")
    print(f"  Saved on: {model_info['timestamp']}")
    
    if model_info['metadata']:
        print("  Metadata:")
        for key, value in model_info['metadata'].items():
            print(f"    {key}: {value}")
    
    return model_info


def save_cv_results(cv_results, filepath, verbose=True):
    """
    Save cross-validation results to CSV.
    
    Args:
        cv_results (pd.DataFrame): GridSearchCV results
        filepath (str): Output file path
        verbose (bool): Whether to print confirmation
    """
    cv_results_save = cv_results[['params', 'mean_test_score', 'std_test_score', 
                                   'mean_train_score', 'std_train_score', 'rank_test_score']]
    cv_results_save = cv_results_save.sort_values('rank_test_score')
    cv_results_save.to_csv(filepath, index=False)
    
    if verbose:
        print(f"\n✓ Full CV results saved to {filepath}")


def create_model_comparison(model_results, filepath='model_comparison.csv', verbose=True):
    """
    Create and save a comprehensive model comparison table.
    
    Args:
        model_results (list): List of dictionaries containing model results
            Each dict should have: name, train_metrics, test_metrics, training_time
        filepath (str): Output file path
        verbose (bool): Whether to print comparison
    
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparison_data = []
    
    for result in model_results:
        comparison_data.append({
            'Model': result['name'],
            'Train R²': result['train_metrics']['r2'],
            'Test R²': result['test_metrics']['r2'],
            'Train RMSE': result['train_metrics']['rmse'],
            'Test RMSE': result['test_metrics']['rmse'],
            'Test MAE': result['test_metrics']['mae'],
            'Overfitting Gap': result['train_metrics']['r2'] - result['test_metrics']['r2'],
            'Training Time (s)': result.get('training_time', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(filepath, index=False)
    
    if verbose:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("\n✓ Model comparison saved to", filepath)
    
    return comparison_df

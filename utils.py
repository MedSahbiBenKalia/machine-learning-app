import pandas as pd
import numpy as np
import os
import joblib

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ACTORS_SCORE_FILE = os.path.join(MODELS_DIR, 'actor-star-power.csv')


# =============================================================================
# Model Wrapper Classes — give every model a uniform .predict() interface
# =============================================================================

class DirectModelWrapper:
    """Wraps a plain sklearn estimator (e.g. DecisionTreeRegressor)."""
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


class ScaledModelWrapper:
    """Wraps an sklearn estimator that needs StandardScaler (e.g. KNN)."""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class EnsembleModelWrapper:
    """Wraps the ensemble (KNN + DT) with weighted predictions."""
    def __init__(self, knn_model, dt_model, scaler, weights):
        self.knn_model = knn_model
        self.dt_model = dt_model
        self.scaler = scaler
        self.weights = weights  # [knn_weight, dt_weight]

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        knn_pred = self.knn_model.predict(X_scaled)
        dt_pred = self.dt_model.predict(X)
        return self.weights[0] * knn_pred + self.weights[1] * dt_pred

# Weights configuration
WEIGHTS_3 = [0.5, 0.3, 0.2]
WEIGHTS_2 = [0.625, 0.375]
WEIGHTS_1 = [1.0]

def load_actor_data():
    """Load actor star power data from CSV and return the dataframe and lookup dict."""
    if not os.path.exists(ACTORS_SCORE_FILE):
        print(f"Warning: Actor score file not found at {ACTORS_SCORE_FILE}")
        return {}, 0
        
    df_scores = pd.read_csv(ACTORS_SCORE_FILE)
    global_mean_score = df_scores['Star_Power_Score'].mean()
    actor_score_map = dict(zip(df_scores['Actor_Name'], df_scores['Star_Power_Score']))
    return actor_score_map, global_mean_score

def lookup_actor(actor_name, actor_score_map):
    """Look up a single actor's star power score. Returns None if not found."""
    return actor_score_map.get(actor_name, None)

def calculate_cast_power(actor_names, actor_score_map, global_mean_score):
    """
    Calculate the total Cast Star Power from a list of actor names.
    
    Args:
        actor_names: list of actor name strings (up to 3)
        actor_score_map: dict mapping actor names to scores
        global_mean_score: default score for unknown actors
    
    Returns:
        tuple: (total_power, individual_scores_list)
    """
    # Filter out empty names
    actors = [a.strip() for a in actor_names if a and a.strip()]

    if not actors:
        return global_mean_score, []

    # Get scores, using global mean for unknown actors
    individual = []
    for actor in actors:
        score = actor_score_map.get(actor, global_mean_score)
        found = actor in actor_score_map
        individual.append({
            'name': actor,
            'score': score,
            'found': found
        })

    scores = [item['score'] for item in individual]
    num_actors = len(scores)

    if num_actors >= 3:
        total_power = (scores[0] * WEIGHTS_3[0] +
                       scores[1] * WEIGHTS_3[1] +
                       scores[2] * WEIGHTS_3[2])
    elif num_actors == 2:
        total_power = (scores[0] * WEIGHTS_2[0] +
                       scores[1] * WEIGHTS_2[1])
    elif num_actors == 1:
        total_power = scores[0] * WEIGHTS_1[0]
    else:
        total_power = global_mean_score

    return total_power, individual

def load_model(model_name):
    """
    Load a trained model from the models directory.
    Handles 3 pkl formats:
      - Decision Tree: data['model'] is a DecisionTreeRegressor
      - KNN: data['model'] is {'model': KNeighborsRegressor, 'scaler': StandardScaler}
      - Ensemble: data['model'] is {'knn_model': ..., 'dt_model': ..., 'scaler': ..., 'weights': ...}
    Returns a wrapper object with a .predict(X) method.
    """
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        print(f"  {model_name}: file not found at {model_path}")
        return None

    try:
        data = joblib.load(model_path)
    except Exception as e:
        print(f"  {model_name}: error loading file — {e}")
        return None

    # --- Direct estimator (no wrapper dict) ---
    if hasattr(data, 'predict'):
        print(f"  {model_name}: loaded directly as {type(data).__name__}")
        return DirectModelWrapper(data)

    # --- Dictionary wrapper produced by save_model() in ml_utils.py ---
    if not isinstance(data, dict):
        print(f"  {model_name}: unexpected type {type(data).__name__}, cannot load")
        return None

    print(f"  {model_name}: loaded dict with keys {list(data.keys())}")
    inner = data.get('model')

    if inner is None:
        print(f"  {model_name}: no 'model' key found")
        return None

    # Case 1 — inner is a plain sklearn estimator (Decision Tree)
    if hasattr(inner, 'predict'):
        print(f"  {model_name}: extracted {type(inner).__name__} from data['model']")
        return DirectModelWrapper(inner)

    # Case 2 & 3 — inner is a dict
    if isinstance(inner, dict):
        inner_keys = list(inner.keys())
        print(f"  {model_name}: data['model'] is dict with keys {inner_keys}")

        # Case 2 — KNN: {'model': estimator, 'scaler': scaler}
        if 'model' in inner and 'scaler' in inner:
            estimator = inner['model']
            scaler = inner['scaler']
            if hasattr(estimator, 'predict'):
                print(f"  {model_name}: KNN format — {type(estimator).__name__} + scaler")
                return ScaledModelWrapper(estimator, scaler)

        # Case 3 — Ensemble: {'knn_model': ..., 'dt_model': ..., 'scaler': ..., 'weights': ...}
        if 'knn_model' in inner and 'dt_model' in inner and 'scaler' in inner and 'weights' in inner:
            knn = inner['knn_model']
            dt = inner['dt_model']
            scaler = inner['scaler']
            weights = inner['weights']
            if hasattr(knn, 'predict') and hasattr(dt, 'predict'):
                print(f"  {model_name}: Ensemble format — KNN({type(knn).__name__}) + DT({type(dt).__name__}), weights={weights}")
                return EnsembleModelWrapper(knn, dt, scaler, weights)

        # Fallback: scan for any value with .predict()
        for k, v in inner.items():
            if hasattr(v, 'predict'):
                print(f"  {model_name}: fallback — found estimator in data['model']['{k}']")
                return DirectModelWrapper(v)

    print(f"  {model_name}: could not extract a usable model")
    return None
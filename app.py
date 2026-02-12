from flask import Flask, request, render_template, jsonify
import numpy as np
import os
from utils import load_actor_data, lookup_actor, calculate_cast_power, load_model
import pandas as pd

app = Flask(__name__)

# --- 1. SETUP MODEL & FEATURES ---
# Load all models using utils.load_model helper which looks in models/ directory
MODELS = {}

model_files = {
    'decision_tree': ('Decision Tree', 'decision_tree_regressor_latest.pkl'),
    'ensemble': ('Ensemble', 'ensemble_regressor_latest.pkl'),
    'knn': ('KNN', 'knn_regressor_latest.pkl'),
}

for key, (name, filename) in model_files.items():
    model = load_model(filename)
    if model:
        MODELS[key] = (name, model)
        print(f"{name} model loaded successfully")
    else:
        print(f"Warning: {name} model not loaded")

# Load actor star power data
try:
    actor_score_map, global_mean_score = load_actor_data()
    print(f"Loaded {len(actor_score_map)} actors. Global mean: {global_mean_score:,.0f}")
except Exception as e:
    print(f"Error loading actor data: {e}")
    actor_score_map, global_mean_score = {}, 0

# The exact list of features from your model (DO NOT CHANGE THE ORDER)
FEATURES = [
    # --- 49 base features ---
    'budget_x', 'Cast_Star_Power_Total', 'release-month', 'release-year',
    'country_AU', 'country_CA', 'country_CN', 'country_DE', 'country_ES',
    'country_FR', 'country_GB', 'country_HK', 'country_IT', 'country_JP',
    'country_KR', 'country_MX', 'country_US', 'country_other',
    'genre_Drama', 'genre_Comedy', 'genre_Action', 'genre_Thriller',
    'genre_Adventure', 'genre_Romance', 'genre_Horror', 'genre_Animation',
    'genre_Family', 'genre_Fantasy', 'genre_Crime', 'genre_Science Fiction',
    'genre_Mystery', 'genre_History', 'genre_War', 'genre_Music',
    'genre_Documentary', 'genre_TV Movie', 'genre_Western',
    'orig_lang_English', 'orig_lang_Japanese', 'orig_lang_Spanish, Castilian',
    'orig_lang_Korean', 'orig_lang_French', 'orig_lang_Chinese',
    'orig_lang_Cantonese', 'orig_lang_Italian', 'orig_lang_German',
    'orig_lang_Russian', 'orig_lang_Other', 'is_sequel',
    # --- 15 interaction features ---
    'budget_x_x_Cast_Star_Power_Total', 'budget_x_x_release-year',
    'budget_x_x_genre_Animation', 'budget_x_x_release-month',
    'budget_x_x_genre_Documentary', 'Cast_Star_Power_Total_x_release-year',
    'Cast_Star_Power_Total_x_genre_Animation',
    'Cast_Star_Power_Total_x_release-month',
    'Cast_Star_Power_Total_x_genre_Documentary',
    'release-year_x_genre_Animation', 'release-year_x_release-month',
    'release-year_x_genre_Documentary', 'genre_Animation_x_release-month',
    'genre_Animation_x_genre_Documentary', 'release-month_x_genre_Documentary',
    # --- 6 squared features ---
    'budget_x_squared', 'Cast_Star_Power_Total_squared', 'release-year_squared',
    'genre_Animation_squared', 'release-month_squared', 'genre_Documentary_squared',
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/lookup_actor', methods=['POST'])
def lookup_actor_route():
    """API endpoint to look up a single actor's star power score."""
    data = request.get_json()
    actor_name = data.get('actor_name', '').strip()
    
    if not actor_name:
        return jsonify({'found': False, 'message': 'No actor name provided.'})
    
    score = lookup_actor(actor_name, actor_score_map)
    if score is not None:
        return jsonify({
            'found': True,
            'actor_name': actor_name,
            'score': round(score, 2)
        })
    else:
        return jsonify({
            'found': False,
            'actor_name': actor_name,
            'message': f'Actor "{actor_name}" not found. Global mean ({global_mean_score:,.0f}) will be used.'
        })


@app.route('/calculate_cast_power', methods=['POST'])
def calculate_cast_power_route():
    """API endpoint to calculate total cast star power from actor names."""
    data = request.get_json()
    actors = [
        data.get('actor1', '').strip(),
        data.get('actor2', '').strip(),
        data.get('actor3', '').strip()
    ]
    # Filter out empty
    actors = [a for a in actors if a]
    
    if not actors:
        return jsonify({'error': 'At least one actor name is required.'})
    
    total, details = calculate_cast_power(actors, actor_score_map, global_mean_score)
    
    return jsonify({
        'total_cast_power': round(total, 2),
        'actors': details,
        'global_mean': round(global_mean_score, 2)
    })


@app.route('/predict', methods=['POST'])
def predict():
    # Check which models are loaded
    if not MODELS:
        return render_template('index.html', 
                             prediction_text="Error: No models loaded. Check server logs.")

    # --- 2. INITIALIZE INPUT VECTOR ---
    input_vector = [0.0] * len(FEATURES)

    try:
        # --- 3. MAP NUMERICAL VALUES ---
        input_vector[FEATURES.index('budget_x')] = float(request.form.get('budget'))
        input_vector[FEATURES.index('release-month')] = int(request.form.get('month'))
        input_vector[FEATURES.index('release-year')] = int(request.form.get('year'))

        # --- Cast Star Power from 3 actor names ---
        actor1 = request.form.get('actor1', '').strip()
        actor2 = request.form.get('actor2', '').strip()
        actor3 = request.form.get('actor3', '').strip()
        actors = [a for a in [actor1, actor2, actor3] if a]

        if actors:
            cast_power, _ = calculate_cast_power(actors, actor_score_map, global_mean_score)
        else:
            cast_power = global_mean_score

        input_vector[FEATURES.index('Cast_Star_Power_Total')] = cast_power
        
        # Checkbox handling (returns 'on' if checked, None otherwise)
        is_sequel = 1 if request.form.get('is_sequel') else 0
        input_vector[FEATURES.index('is_sequel')] = is_sequel

        # --- 4. MAP CATEGORICAL VALUES (ONE-HOT) ---
        
        # Country
        country = request.form.get('country')
        country_feature = f"country_{country}"
        if country_feature in FEATURES:
            input_vector[FEATURES.index(country_feature)] = 1
        else:
            input_vector[FEATURES.index('country_other')] = 1

        # Genre - Handle multiple selections
        genres = request.form.getlist('genre')
        if genres:
            for genre in genres:
                genre_feature = f"genre_{genre}"
                if genre_feature in FEATURES:
                    input_vector[FEATURES.index(genre_feature)] = 1
            
        # Language
        lang = request.form.get('language')
        lang_feature = f"orig_lang_{lang}"
        if lang_feature in FEATURES:
            input_vector[FEATURES.index(lang_feature)] = 1
        else:
            input_vector[FEATURES.index('orig_lang_Other')] = 1

        # --- 5. COMPUTE INTERACTION & SQUARED FEATURES ---
        # Grab the base values we need for the engineered features
        budget    = input_vector[FEATURES.index('budget_x')]
        csp       = input_vector[FEATURES.index('Cast_Star_Power_Total')]
        month     = input_vector[FEATURES.index('release-month')]
        year      = input_vector[FEATURES.index('release-year')]
        g_anim    = input_vector[FEATURES.index('genre_Animation')]
        g_doc     = input_vector[FEATURES.index('genre_Documentary')]

        # Interaction features (A x B)
        input_vector[FEATURES.index('budget_x_x_Cast_Star_Power_Total')]          = budget * csp
        input_vector[FEATURES.index('budget_x_x_release-year')]                   = budget * year
        input_vector[FEATURES.index('budget_x_x_genre_Animation')]                = budget * g_anim
        input_vector[FEATURES.index('budget_x_x_release-month')]                  = budget * month
        input_vector[FEATURES.index('budget_x_x_genre_Documentary')]              = budget * g_doc
        input_vector[FEATURES.index('Cast_Star_Power_Total_x_release-year')]      = csp * year
        input_vector[FEATURES.index('Cast_Star_Power_Total_x_genre_Animation')]   = csp * g_anim
        input_vector[FEATURES.index('Cast_Star_Power_Total_x_release-month')]     = csp * month
        input_vector[FEATURES.index('Cast_Star_Power_Total_x_genre_Documentary')] = csp * g_doc
        input_vector[FEATURES.index('release-year_x_genre_Animation')]            = year * g_anim
        input_vector[FEATURES.index('release-year_x_release-month')]              = year * month
        input_vector[FEATURES.index('release-year_x_genre_Documentary')]          = year * g_doc
        input_vector[FEATURES.index('genre_Animation_x_release-month')]           = g_anim * month
        input_vector[FEATURES.index('genre_Animation_x_genre_Documentary')]       = g_anim * g_doc
        input_vector[FEATURES.index('release-month_x_genre_Documentary')]         = month * g_doc

        # Squared features
        input_vector[FEATURES.index('budget_x_squared')]                = budget ** 2
        input_vector[FEATURES.index('Cast_Star_Power_Total_squared')]   = csp ** 2
        input_vector[FEATURES.index('release-year_squared')]            = year ** 2
        input_vector[FEATURES.index('genre_Animation_squared')]         = g_anim ** 2
        input_vector[FEATURES.index('release-month_squared')]           = month ** 2
        input_vector[FEATURES.index('genre_Documentary_squared')]       = g_doc ** 2

        # --- 6. PREDICT WITH SELECTED MODELS ---
        selected_type = request.form.get('model_type')
        models_to_run = []

        # Filter models based on user selection
        if selected_type == 'all':
            models_to_run = list(MODELS.values())
        elif selected_type in MODELS:
            models_to_run = [MODELS[selected_type]]
        else:
            models_to_run = list(MODELS.values())

        predictions = []
        for model_name, model in models_to_run:
            # Build a DataFrame with feature names so scalers work correctly
            input_df = pd.DataFrame([input_vector], columns=FEATURES)
            prediction = model.predict(input_df)[0]
            predictions.append({'name': model_name, 'value': f"${prediction:,.2f}"})

        # Format genres text
        genres_text = ", ".join(genres) if genres else "None"

        return render_template('index.html', 
                               predictions=predictions,
                               cast_power_text=f"Cast Star Power Used: {cast_power:,.0f}",
                               genres_selected=f"Genres: {genres_text}",
                               scroll='result')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

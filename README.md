# Movie Revenue Prediction App

A Flask application to predict movie revenue based on various features such as budget, actors, genre, etc.

## Structure

```
movie-revenue-app/
│
├── app.py              # Main Flask application
├── models/             # Trained models and data
│   ├── decision_tree_regressor_latest.pkl
│   ├── knn_regressor_latest.pkl
│   └── actor-star-power.csv
│
├── static/             # Static resources
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/          # HTML templates
│   ├── index.html
│   └── result.html
│
├── utils.py            # Helper functions
└── requirements.txt    # Project dependencies
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

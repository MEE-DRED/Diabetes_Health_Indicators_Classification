# Diabetes health indicators - ML vs DL
Notebook outputs. Best traditional model: E4-HGB-lr0.1-l31; best neural net: DL-E3-reg. Seed 42.
Load: `joblib.load("models/best_traditional_pipeline.joblib").predict_proba(df[features])` (features in results/run_configuration.json).
Neural net: standardise with `models/nn_preprocessor.joblib`, then `keras.saving.load_model("models/best_neural_network.keras")`.
Association study on self-reported survey data - not a diagnostic tool. Raw data are NOT included; see the notebook for the Kaggle source.

# Predicting Self-Reported Diabetes Status from Health Indicators
### Traditional Machine Learning vs Deep Learning on the CDC BRFSS 2015 Survey

A comparison of scikit-learn models and a Keras neural network for classifying **self-reported diabetes** from 21 survey indicators, with logged experiments, a held-out test set and error analysis.

> **This is an association study of self-reported survey data. It is not a diagnostic tool, does not predict future diabetes, and does not show that any feature causes diabetes.**

| Resource | Link |
|---|---|
| Written report | Mildred Ebomah_Intro to ML[SUMMATIVE] |
| Google Drive (raw data zip, saved models) | https://drive.google.com/drive/folders/1qVu8UPkFtYcAQ5IzjQ7eqBuDwZKQv2xf?usp=sharing |


---

## Key results

Test set: 38,052 respondents (5,302 positive), evaluated once after all models and thresholds were frozen on validation data.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|---|
| Majority-class baseline | 0.50 | 0.861 | 0.000 | 0.000 | 0.000 | 0.5000 | 0.1393 |
| Histogram Gradient Boosting (`E4-HGB-lr0.1-l31`) | 0.27 | 0.826 | 0.406 | 0.533 | 0.461 | 0.8293 | 0.4270 |
| Keras MLP (`DL-E3-reg`) | 0.23 | 0.807 | 0.381 | 0.617 | 0.471 | 0.8299 | 0.4275 |

Precision, recall and F1 are for the positive class (self-reported diabetes). Thresholds maximise F1 on the validation set; they are not clinical cut-offs.

- The two final models rank respondents almost identically. Differences of 0.0006 (ROC-AUC) and 0.0005 (average precision) come from one seed and one split and were **not** tested for significance.
- Class weighting and lower thresholds trade precision for recall; they barely change ranking metrics.
- Hyperparameter search and feature selection did not beat the default gradient-boosting configuration on validation.

## The target variable

`Diabetes_binary` was checked against the three-class `Diabetes_012` file: **1 = reported diabetes** (`Diabetes_012 == 2`, n = 35,346) and **0 = no diabetes or prediabetes** (`Diabetes_012` in {0, 1}, n = 218,334). Respondents reporting prediabetes (n = 4,631) are in the negative class.

## Dataset 

Source: [Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), a cleaned subset of the CDC [2015 BRFSS](https://www.cdc.gov/brfss/annual_data/annual_2015.html) prepared by the dataset curator.

| File | Rows | Used for |
|---|---|---|
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 | All modelling (21 predictors + target) |
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253,680 | Verifying the target definition only |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70,692 | Ignored (artificially balanced) |

**The raw data is zipped in the Google Drive folder above.** 

## How to run

### Google Colab 
1. Put the CSV files in a Drive folder named `SUMMATIVE 2026` (or edit `DRIVE_FOLDER` in the data-loading cell).
2. Open `Diabetes_Health_Indicators_Classification.ipynb` in Colab and choose **Runtime > Restart session and run all**.
3. Allow Drive access when prompted.

### Locally
1. Create an environment (Python 3.11+ recommended; the recorded run used 3.13) and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. In the data-loading cell, set `MOUNT_DRIVE = False` and point `DRIVE_FOLDER` to the folder holding the CSVs.
3. Run all cells in order, top to bottom.

A GPU is not required. Set `DEBUG_SUBSAMPLE` to an integer only for a quick dry run; those results must not be reported. Weights & Biases runs in **offline mode**: no account or API key is needed and nothing is uploaded.

## Method summary

| Stage | What the notebook does |
|---|---|
| Audit | Shape, missing values (0), range checks, duplicates (24,206 exact; 67 groups with conflicting labels), class balance (13.93% positive), correlations, IQR flags |
| Split | Stratified 70/15/15, seed 42: 177,576 / 38,052 / 38,052 rows |
| Preprocessing | Imputation and standardisation inside scikit-learn pipelines, fitted on training data only; trees use unscaled features; duplicates and IQR-flagged rows retained |
| Traditional ML | Majority baseline; Logistic Regression (C, class weight); Random Forest (trees, depth, class weight); Histogram Gradient Boosting (learning rate, leaves, L2, class weight) |
| Optimisation | `RandomizedSearchCV` (12 candidates, 3-fold, average precision) on a 50,000-row training subset; mutual-information feature selection (top 8, 12, 16) |
| Deep learning | Keras Sequential MLP with `tf.data`; 8 experiments varying capacity, dropout/L2, learning rate, class weights, batch size; early stopping on validation PR-AUC |
| Selection | Highest validation average precision per family; decision threshold = F1-maximising value on validation; test set used once |
| Analysis | Learning curves, confusion matrices, ROC and precision-recall curves, subgroup error rates, permutation importance |

25 unique validation experiments are registered (17 traditional including the baseline, 8 neural).

## Outputs

Running the notebook creates `outputs/` and `diabetes_outputs.zip`:

| Folder | Contents |
|---|---|
| `figures/` | EDA plots, learning curves, ROC/PR curves, confusion matrices, permutation importance |
| `results/` | Experiment tables (CSV), final test metrics (CSV/JSON), test predictions, run configuration, environment versions |
| `models/` | `best_traditional_pipeline.joblib`, `best_neural_network.keras`, `nn_preprocessor.joblib` |
| `logs/` | `experiment_log.jsonl` and Weights & Biases offline run folders (optional `wandb sync` later) |

### Loading the saved models
```python
import joblib
from tensorflow import keras

gb = joblib.load("outputs/models/best_traditional_pipeline.joblib")
p_gb = gb.predict_proba(df[features])[:, 1]          # classify with threshold 0.27

prep = joblib.load("outputs/models/nn_preprocessor.joblib")
nn = keras.saving.load_model("outputs/models/best_neural_network.keras")
p_nn = nn.predict(prep.transform(df[features]).astype("float32")).ravel()   # threshold 0.23
```
`features` is the list of 21 predictor names saved in `outputs/results/run_configuration.json`.

## Environment of the recorded run

Python 3.13.15, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1, SciPy 1.16.3, TensorFlow 2.21.0, Keras 3.13.2, matplotlib 3.10.0, seaborn 0.13.2, wandb 0.28.1, joblib 1.6.0. Results can differ slightly across hardware and library versions.

## Limitations

- Self-reported outcome; prediabetes is grouped with "no diabetes".
- US adults in 2015; no external validation, so results may not transfer to other populations.
- About 12.6-12.8% of validation and test rows share an exact predictor vector with a training row, so test scores may be optimistic for unseen predictor patterns.
- Single seed and split; no confidence intervals or significance tests.
- Error rates differ by age and income group; any real use would need subgroup evaluation.

## Repository structure
```
.
├── Diabetes_Health_Indicators_Classification.ipynb
├── requirements.txt
├── README.md
```

## References and acknowledgements

Dataset: CDC BRFSS 2015 and the Kaggle dataset linked above (curated by A. Teboul). Full literature review and citations are in the written report.

## Licence

The dataset remains subject to its original CDC/Kaggle terms.

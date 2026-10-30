# AttritionIQ – Explainable Employee Attrition Prediction

AttritionIQ is a FastAPI-based backend for employee attrition analysis. It combines structured Human Resources (HR) features with three prediction outputs:

1. **Random Forest** – a scikit-learn tree-ensemble model.
2. **Artificial Neural Network (ANN)** – a feed-forward PyTorch neural network.
3. **Hybrid Probability Fusion** – a weighted combination of the Random Forest and ANN probabilities.

The backend also provides dataset inspection and evaluation, single-employee prediction, CSV batch prediction, and SHAP/LIME explanation endpoints.

> **Scope:** The supplied Python package uses structured/tabular HR data. It does not implement TF-IDF, BERT, or employee-text processing.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Hybrid Probability Fusion](#hybrid-probability-fusion)
- [Dataset Features](#dataset-features)
- [Installation](#installation)
- [Required Model and Dataset Files](#required-model-and-dataset-files)
- [Run the API](#run-the-api)
- [API Endpoints](#api-endpoints)
- [Dataset Evaluation Workflow](#dataset-evaluation-workflow)
- [Evaluation Metrics and Graphs](#evaluation-metrics-and-graphs)
- [Prediction Workflow](#prediction-workflow)
- [Explainable AI](#explainable-ai)
- [Generated Output Files](#generated-output-files)
- [Important Implementation Notes](#important-implementation-notes)
- [Responsible Use](#responsible-use)

---

## Project Structure

```text
project-root/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   │
│   └── services/
│       ├── dataset_evaluation.py
│       ├── explainability.py
│       ├── model_utils.py
│       └── prediction.py
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   ├── rf_model.joblib
│   ├── ann_model.pth
│   ├── preprocessor.joblib
│   ├── feature_names.json
│   ├── fusion_config.json
│   └── train_features_sample.csv
│
├── reports/
│   ├── metrics/
│   │   ├── evaluation_summary.json
│   │   └── current_evaluation.json
│   └── figures/
│
├── requirements.txt
└── README.md
```

The ZIP supplied for this README contains the `app/` Python package. The `data/`, `models/`, `reports/`, frontend files, training scripts, and `requirements.txt` are expected project assets and are not included in that ZIP.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend and machine-learning implementation |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| pandas / NumPy | Dataset processing and numerical operations |
| scikit-learn | Random Forest, preprocessing, data splitting, and evaluation metrics |
| PyTorch | Artificial Neural Network |
| SHAP | Feature contribution explanations |
| LIME | Local interpretable prediction explanations |
| Matplotlib / Seaborn | Evaluation graph generation |
| Joblib | Loading saved scikit-learn models and preprocessors |
| Pydantic | API input and output validation |

Install compatible versions of these packages in the project's Python environment. The exact pinned package versions are not specified in the supplied source files.

---

## Model Architecture

The prediction pipeline follows this sequence:

```text
Structured Employee HR Data
            |
            v
    Input Validation
            |
            v
    Data Preprocessing
 (Numerical + Categorical)
            |
            v
   Transformed HR Features
            |
       +----+----+
       |         |
       v         v
 Random Forest  PyTorch ANN
       |         |
       v         v
   P_RF        P_ANN
       |         |
       +----+----+
            |
            v
  Hybrid Probability Fusion
            |
            v
   Decision Threshold
            |
            v
 Attrition / Retention Result
            |
      +-----+------+
      |            |
      v            v
  Risk Tier    SHAP / LIME
```

### ANN Network

The `AttritionANN` class in `app/services/model_utils.py` defines the feed-forward neural network:

```text
Input Layer (D features)
        |
Linear(D, 64)
        |
Batch Normalization
        |
ReLU + Dropout(0.25)
        |
Linear(64, 32)
        |
Batch Normalization
        |
ReLU + Dropout(0.15)
        |
Linear(32, 16)
        |
ReLU
        |
Linear(16, 1)
        |
Sigmoid
        |
Attrition Probability
```

The ANN returns a probability for the positive class, **Attrition = Yes**.

---

## Hybrid Probability Fusion

The hybrid model combines the positive-class probabilities from Random Forest and ANN.

\[
P_{hybrid}=w_{RF}P_{RF}+(1-w_{RF})P_{ANN}
\]

The inference service reads the fusion weights and decision threshold from:

```text
models/fusion_config.json
```

Example configuration format:

```json
{
  "fusion_weight_rf": 0.18,
  "fusion_weight_ann": 0.82,
  "decision_threshold": 0.70
}
```

With this example configuration:

\[
P_{hybrid}=0.18P_{RF}+0.82P_{ANN}
\]

The final class is determined by comparing the hybrid probability with the configured threshold.

- If `P_hybrid >= decision_threshold`: **Attrition (Yes)**
- Otherwise: **Retained (No)**

The values above are an example matching the dashboard configuration. Use the actual `fusion_config.json` from the trained project when running the application.

---

## Dataset Features

The model utility module defines **30 structured input features**.

### Numerical Features (23)

```text
Age
DailyRate
DistanceFromHome
Education
EnvironmentSatisfaction
HourlyRate
JobInvolvement
JobLevel
JobSatisfaction
MonthlyIncome
MonthlyRate
NumCompaniesWorked
PercentSalaryHike
PerformanceRating
RelationshipSatisfaction
StockOptionLevel
TotalWorkingYears
TrainingTimesLastYear
WorkLifeBalance
YearsAtCompany
YearsInCurrentRole
YearsSinceLastPromotion
YearsWithCurrManager
```

### Categorical Features (7)

```text
BusinessTravel
Department
EducationField
Gender
JobRole
MaritalStatus
OverTime
```

### Target Variable

```text
Attrition
```

The target represents whether the employee left the organization:

- `Yes` – Attrition
- `No` – Retained

The model utility module also identifies these columns as non-predictive/drop columns where applicable:

```text
EmployeeCount
StandardHours
Over18
EmployeeNumber
```

---

## Installation

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Create a `requirements.txt` containing the packages used by the project, then run:

```bash
pip install -r requirements.txt
```

The source code imports the following main packages:

```text
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
torch
joblib
python-multipart
matplotlib
seaborn
shap
lime
```

If the frontend is served by this backend, also ensure its built/static files are placed in the location configured in `app/main.py`.

---

## Required Model and Dataset Files

The prediction service loads trained artifacts from the `models/` directory:

| File | Purpose |
|---|---|
| `rf_model.joblib` | Trained Random Forest model |
| `ann_model.pth` | Trained PyTorch ANN state dictionary |
| `preprocessor.joblib` | Fitted feature preprocessor |
| `feature_names.json` | ANN input feature names and input dimension |
| `fusion_config.json` | Fusion weights and decision threshold |
| `train_features_sample.csv` | Background sample used to initialize explainability |

The default benchmark dataset path used by some endpoints is:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

**Important:** These artifacts must be generated by the project's training workflow and must correspond to the same feature order and preprocessing pipeline. The supplied ZIP does not contain a model-training script or the trained model files.

---

## Run the API

From the project root directory:

```bash
uvicorn app.main:app --reload
```

The local API will normally be available at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Checks whether the API is running |
| GET | `/api/status` | Returns dataset, model, and evaluation readiness |
| GET | `/api/metrics` | Returns the latest available evaluation result |
| POST | `/api/dataset/inspect` | Inspects an uploaded CSV and identifies target-column candidates |
| POST | `/api/dataset/evaluate` | Trains and evaluates RF, ANN, and hybrid models on one uploaded CSV |
| GET | `/api/metrics/confusion-matrix` | Retrieves a model's confusion matrix |
| POST | `/api/predict` | Predicts attrition for one employee |
| POST | `/api/predict/csv` | Performs batch prediction from an uploaded CSV |
| GET | `/api/dataset/records` | Retrieves benchmark dataset records |
| POST | `/api/predict/dataset-all` | Runs prediction for the full benchmark dataset |
| POST | `/api/predict/demo-batch` | Runs batch prediction for a selected benchmark sample count |
| GET | `/api/sample-csv` | Downloads a sample CSV template |
| POST | `/api/explain/shap` | Generates SHAP feature explanations |
| POST | `/api/explain/lime` | Generates LIME local explanation rules |
| GET | `/api/sample-employees` | Returns sample employee profiles for UI testing |

### Example: Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

### Example: Inspect a Dataset

```bash
curl -X POST "http://127.0.0.1:8000/api/dataset/inspect" \
  -F "file=@employee_data.csv"
```

### Example: Evaluate One Uploaded Dataset

```bash
curl -X POST "http://127.0.0.1:8000/api/dataset/evaluate?target_column=Attrition" \
  -F "file=@employee_data.csv"
```

The target column must exist in the uploaded CSV. If it is not supplied, the API responds with candidate target columns so the user can select the correct one.

### Example: Single Employee Prediction

Send a JSON object containing the employee feature fields to:

```text
POST /api/predict
```

Example request body:

```json
{
  "Age": 35,
  "BusinessTravel": "Travel_Rarely",
  "DailyRate": 800,
  "Department": "Research & Development",
  "DistanceFromHome": 10,
  "Education": 3,
  "EducationField": "Life Sciences",
  "EnvironmentSatisfaction": 3,
  "Gender": "Male",
  "HourlyRate": 65,
  "JobInvolvement": 3,
  "JobLevel": 2,
  "JobRole": "Research Scientist",
  "JobSatisfaction": 3,
  "MaritalStatus": "Single",
  "MonthlyIncome": 5000,
  "MonthlyRate": 14000,
  "NumCompaniesWorked": 2,
  "OverTime": "No",
  "PercentSalaryHike": 14,
  "PerformanceRating": 3,
  "RelationshipSatisfaction": 3,
  "StockOptionLevel": 1,
  "TotalWorkingYears": 10,
  "TrainingTimesLastYear": 2,
  "WorkLifeBalance": 3,
  "YearsAtCompany": 5,
  "YearsInCurrentRole": 3,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 3
}
```

This request uses the default values and validation constraints defined in `app/schemas.py`; replace them with the employee's actual structured HR values.

---

## Dataset Evaluation Workflow

The `/api/dataset/evaluate` endpoint evaluates **one uploaded dataset per request**. It does not create a dataset-to-dataset comparison.

1. Upload a CSV file.
2. Inspect the dataset using `/api/dataset/inspect`.
3. Select the correct binary target column (for example, `Attrition`).
4. Submit the CSV and target-column name to `/api/dataset/evaluate`.
5. The service prepares the features and target labels.
6. It creates a stratified train/test split.
7. The preprocessor is fitted on the training partition and applied to the test partition.
8. Random Forest and ANN are trained.
9. Their test-set probabilities are combined by the hybrid fusion function.
10. Metrics and graph files are generated for that uploaded dataset.

The evaluation response includes the dataset name, target mapping, row counts, model metrics, ANN training history, threshold analysis, feature importance, graph paths, and test-record predictions.

---

## Evaluation Metrics and Graphs

The evaluation service calculates the following metrics separately for Random Forest, ANN, and Hybrid Fusion on the same held-out test partition:

| Metric | Meaning |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Precision | Fraction of predicted attrition cases that are actual attrition cases |
| Recall / Sensitivity | Fraction of actual attrition cases detected |
| Specificity | Fraction of retained employees correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ranking/discrimination performance across thresholds |
| Average Precision | Summary measure derived from the precision–recall curve |
| Confusion Matrix | True negatives, false positives, false negatives, and true positives |

### Graphs generated for the currently evaluated dataset

- Model metrics comparison chart.
- ROC curves for Random Forest, ANN, and Hybrid Fusion.
- Precision–Recall curves for all three models.
- Individual confusion-matrix heatmaps.
- ANN training and validation loss/accuracy history.
- Hybrid decision-threshold analysis.
- Random Forest feature-importance chart.

Graph output URLs are returned in the `graphs` field of the evaluation response. The service writes graph files under:

```text
reports/figures/
```

The current evaluation JSON is written to:

```text
reports/metrics/current_evaluation.json
```

Only models evaluated on the **current uploaded dataset** are compared in that evaluation response. Results from different datasets should be treated as separate runs and must not be combined into a dataset leaderboard or cross-dataset comparison.

---

## Prediction Workflow

### Single Employee

The `/api/predict` endpoint:

1. Validates the employee attributes using the Pydantic `EmployeeInput` schema.
2. Applies the saved preprocessing object.
3. Obtains Random Forest probability.
4. Obtains ANN probability.
5. Computes hybrid probability using the configured fusion weights.
6. Applies the hybrid decision threshold.
7. Returns all three model outputs and a risk-tier assessment.

### CSV Batch Prediction

The `/api/predict/csv` endpoint accepts a CSV file and returns employee-level model probabilities and risk information, together with aggregate batch statistics.

The batch prediction service maps common column-name aliases to expected IBM HR feature names. If expected columns are missing, the current implementation fills them with predefined default values. For reliable analysis, use a CSV containing the actual required HR attributes and inspect the returned output carefully.

---

## Explainable AI

AttritionIQ provides two explanation endpoints for the hybrid probability-fusion prediction workflow.

### SHAP

```text
POST /api/explain/shap
```

SHAP estimates how individual input features contribute to the model output. The implementation uses a Kernel SHAP explainer initialized with a background sample from `models/train_features_sample.csv`.

### LIME

```text
POST /api/explain/lime
```

LIME produces local decision rules that approximate the model's behavior around a selected employee record.

Both endpoints require the trained model artifacts and explainability background data to be available. Explanations describe model behavior; they do not prove that a feature causes an employee to leave.

---

## Generated Output Files

After a dataset evaluation, the service creates the following files:

```text
reports/
├── metrics/
│   └── current_evaluation.json
│
└── figures/
    ├── model_metrics.png
    ├── roc_curves_current.png
    ├── precision_recall_curves_current.png
    ├── ann_training_history_current.png
    ├── threshold_analysis_current.png
    ├── feature_importance_current.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_artificial_neural_network.png
    └── confusion_matrix_hybrid_fusion.png
```

The exact output is created when the corresponding evaluation endpoint runs. Do not manually enter or hardcode evaluation values in the report; use the JSON result and generated plots from the actual run.

---

## Important Implementation Notes

- **Single-dataset evaluation:** `/api/dataset/evaluate` handles the uploaded dataset in the current request. It does not compare separate datasets.
- **Split:** The evaluation service uses a stratified 80/20 train/test split with `random_state=42`.
- **Preprocessing:** The `ColumnTransformer` scales numerical columns and one-hot encodes categorical columns. It is fitted on the training partition and then applied to the test partition.
- **Missing values:** Numeric columns are filled using their median and categorical columns with `"Unknown"` before the split in the current implementation. For a strict leakage-free workflow, calculate imputation values from training data only.
- **SMOTE:** The supplied `evaluate_dataset` implementation does not call SMOTE. Do not claim that SMOTE is used in this evaluation unless the code is updated and verified.
- **Fusion-weight consistency:** The uploaded-dataset evaluation function currently sets `weight_rf = 0.5` (therefore 50% RF + 50% ANN). The saved-model inference service reads weights from `models/fusion_config.json` (the dashboard configuration may be 18% RF + 82% ANN). Align these settings before claiming that uploaded-dataset evaluation and deployed prediction use the same fusion weights.
- **Threshold consistency:** The uploaded-dataset evaluation function currently evaluates at a 0.5 threshold. The saved-model inference service reads the threshold from `fusion_config.json`. Keep the paper, dashboard, and code consistent.
- **Training history:** The current ANN evaluation loop records training and validation loss/accuracy for 40 epochs. Do not describe a different epoch count unless the code is changed.
- **Model artifacts:** The ZIP does not include the training script or saved model files. The API's saved-model prediction and SHAP/LIME services cannot fully initialize until the expected artifacts are provided.
- **No fabricated results:** Metrics must come from an actual execution of the evaluation pipeline.

---

## Responsible Use

This project is intended as a decision-support and workforce-analysis tool. Attrition probabilities are estimates, not certainty about an employee's intentions or future actions. Model explanations are not causal evidence. HR professionals should review predictions carefully, assess fairness and privacy risks, and never use an automated model score as the sole basis for hiring, firing, promotion, or other consequential employment decisions.

---

## License and Dataset Attribution

Add the dataset's original source, licence, and any required attribution here before public distribution. The supplied Python source does not specify a project software licence.

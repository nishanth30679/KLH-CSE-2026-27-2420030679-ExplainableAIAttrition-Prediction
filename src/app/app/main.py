"""
AttritionIQ - FastAPI Enterprise Backend Application
Exposes RESTful endpoints for health checks, platform status, evaluation metrics,
multi-model predictions (RF, ANN, Hybrid), and Explainable AI (SHAP, LIME).
Serves the modern frontend application and static visual assets.
"""

import json
import os
import sys
import io
import pandas as pd
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import (
    EmployeeInput,
    PredictionResponse,
    SHAPExplanationResponse,
    LIMEExplanationResponse,
    PlatformStatusResponse,
    ConfusionMatrixResponse
)
from app.services.prediction import model_manager
from app.services.explainability import explainability_engine
from app.services.dataset_evaluation import inspect_dataset, evaluate_dataset

app = FastAPI(
    title="AttritionIQ API",
    description="Explainable Workforce Intelligence Platform powered by Random Forest, PyTorch ANN, Probability Fusion, SHAP, and LIME.",
    version="1.0.0"
)

# Enable CORS for local cross-origin interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports/figures as static files so frontend can display generated charts
reports_fig_dir = os.path.join(os.getcwd(), "reports", "figures")
if os.path.exists(reports_fig_dir):
    app.mount("/reports/figures", StaticFiles(directory=reports_fig_dir), name="figures")

# Mount frontend directory for assets
frontend_dir = os.path.join(os.getcwd(), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/api/health")
def get_health() -> Dict[str, str]:
    """Health check endpoint to verify backend operational readiness."""
    return {
        "status": "healthy",
        "platform": "AttritionIQ",
        "tagline": "Explainable Workforce Intelligence",
        "version": "1.0.0"
    }


@app.get("/api/status", response_model=PlatformStatusResponse)
def get_platform_status() -> PlatformStatusResponse:
    """Returns comprehensive readiness status of dataset, models, and evaluation results."""
    current_eval_path = os.path.join("reports", "metrics", "current_evaluation.json")
    if os.path.exists(current_eval_path):
        try:
            with open(current_eval_path, "r", encoding="utf-8") as f:
                current = json.load(f)
            hybrid = current.get("models", {}).get("hybrid_fusion", {})
            return PlatformStatusResponse(
                dataset_ready=True,
                dataset_path=f"uploaded:{current.get('dataset_name', 'current dataset')}",
                dataset_records=int(current.get("row_count", 0)),
                models_trained=True,
                evaluation_completed=True,
                rf_available=True,
                ann_available=True,
                fusion_available=True,
                fusion_weight_rf=0.5,
                fusion_weight_ann=0.5,
                decision_threshold=hybrid.get("threshold", 0.5),
                engineered_features=None,
                accuracy=hybrid.get("accuracy"),
                f1_score=hybrid.get("f1_score"),
                precision=hybrid.get("precision"),
                recall=hybrid.get("recall_sensitivity"),
                specificity=hybrid.get("specificity"),
                roc_auc=hybrid.get("roc_auc"),
                confusion_matrix=hybrid.get("confusion_matrix"),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Unable to read current dataset status: {e}")
    dataset_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    dataset_ready = os.path.exists(dataset_path)
    records_count = 0
    if dataset_ready:
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            records_count = len(df)
        except Exception:
            pass

    rf_path = os.path.join("models", "rf_model.joblib")
    ann_path = os.path.join("models", "ann_model.pth")
    cfg_path = os.path.join("models", "fusion_config.json")
    eval_path = os.path.join("reports", "metrics", "evaluation_summary.json")

    rf_available = os.path.exists(rf_path)
    ann_available = os.path.exists(ann_path)
    fusion_available = os.path.exists(cfg_path)
    models_trained = rf_available and ann_available and fusion_available
    evaluation_completed = os.path.exists(eval_path)

    w_rf, w_ann, threshold, feat_count = None, None, None, None
    if fusion_available:
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            w_rf = cfg.get("fusion_weight_rf")
            w_ann = cfg.get("fusion_weight_ann")
            threshold = cfg.get("decision_threshold")
            feat_count = cfg.get("engineered_feature_count")
        except Exception:
            pass

    # Read genuine evaluation metrics if completed
    accuracy, f1_score_val, precision, recall, specificity, roc_auc, cm_val = (
        None, None, None, None, None, None, None
    )
    if evaluation_completed:
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            hyb = eval_data.get("models", {}).get("hybrid_fusion", {})
            accuracy = eval_data.get("accuracy", hyb.get("accuracy"))
            f1_score_val = eval_data.get("f1_score", hyb.get("f1_score"))
            precision = eval_data.get("precision", hyb.get("precision"))
            recall = eval_data.get("recall_sensitivity", hyb.get("recall_sensitivity"))
            specificity = eval_data.get("specificity", hyb.get("specificity"))
            roc_auc = eval_data.get("roc_auc", hyb.get("roc_auc"))
            cm_val = eval_data.get("confusion_matrix", hyb.get("confusion_matrix"))
        except Exception:
            pass

    return PlatformStatusResponse(
        dataset_ready=dataset_ready,
        dataset_path=dataset_path,
        dataset_records=records_count,
        models_trained=models_trained,
        evaluation_completed=evaluation_completed,
        rf_available=rf_available,
        ann_available=ann_available,
        fusion_available=fusion_available,
        fusion_weight_rf=w_rf,
        fusion_weight_ann=w_ann,
        decision_threshold=threshold,
        engineered_features=feat_count,
        accuracy=accuracy,
        f1_score=f1_score_val,
        precision=precision,
        recall=recall,
        specificity=specificity,
        roc_auc=roc_auc,
        confusion_matrix=cm_val
    )


@app.get("/api/metrics")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Returns the genuine evaluation metrics on the untouched test set."""
    current_path = os.path.join("reports", "metrics", "current_evaluation.json")
    if os.path.exists(current_path):
        try:
            with open(current_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Unable to read current dataset evaluation: {e}")
    eval_path = os.path.join("reports", "metrics", "evaluation_summary.json")
    if not os.path.exists(eval_path):
        raise HTTPException(
            status_code=404,
            detail="Evaluation results pending. Run evaluate.py to generate genuine metrics."
        )

    try:
        with open(eval_path, "r") as f:
            metrics_data = json.load(f)
        return metrics_data
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Evaluation metrics summary file is corrupted. Re-run evaluate.py."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read evaluation metrics: {str(e)}"
        )


@app.post("/api/dataset/inspect")
async def inspect_uploaded_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Inspect an upload and return binary target candidates before training."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        if df.empty:
            raise ValueError("Uploaded CSV file is empty.")
        return inspect_dataset(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to inspect dataset: {e}")


@app.post("/api/dataset/evaluate")
async def evaluate_uploaded_dataset(
    file: UploadFile = File(...),
    target_column: str | None = None,
) -> Dict[str, Any]:
    """Train and evaluate RF, ANN, and hybrid using only this uploaded dataset."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        if df.empty:
            raise ValueError("Uploaded CSV file is empty.")
        if not target_column:
            candidates = inspect_dataset(df)["target_candidates"]
            raise HTTPException(
                status_code=422,
                detail={"message": "Select the attrition target column before evaluation.", "target_candidates": candidates},
            )
        return evaluate_dataset(
            df,
            target_column=target_column,
            output_dir=os.path.join("reports", "metrics"),
            dataset_name=file.filename,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset evaluation failed: {e}")


@app.get("/api/metrics/confusion-matrix", response_model=ConfusionMatrixResponse)
def get_confusion_matrix(model: str = "hybrid_fusion") -> ConfusionMatrixResponse:
    """
    Returns the genuine confusion matrix (TN, FP, FN, TP) and associated performance metrics
    (Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC) for the specified model
    (default: hybrid_fusion) evaluated on the untouched test set.
    """
    eval_path = os.path.join("reports", "metrics", "evaluation_summary.json")
    if not os.path.exists(eval_path):
        raise HTTPException(
            status_code=404,
            detail="Evaluation results pending. Run evaluate.py to generate genuine metrics."
        )

    try:
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading evaluation data: {str(e)}"
        )

    models_data = eval_data.get("models", {})
    target_model = models_data.get(model, models_data.get("hybrid_fusion"))
    if not target_model or "confusion_matrix" not in target_model:
        raise HTTPException(
            status_code=404,
            detail=f"Confusion matrix for model '{model}' not found in evaluation summary."
        )

    cm_info = target_model["confusion_matrix"]
    tn = int(cm_info.get("tn", cm_info.get("true_negative", 0)))
    fp = int(cm_info.get("fp", cm_info.get("false_positive", 0)))
    fn = int(cm_info.get("fn", cm_info.get("false_negative", 0)))
    tp = int(cm_info.get("tp", cm_info.get("true_positive", 0)))

    return ConfusionMatrixResponse(
        status="success",
        model=model,
        labels=cm_info.get("labels", ["Stay", "Leave"]),
        target_encoding={"Stay": 0, "Leave": 1},
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        matrix=cm_info.get("matrix", [[tn, fp], [fn, tp]]),
        accuracy=float(target_model.get("accuracy", 0.0)),
        precision=float(target_model.get("precision", 0.0)),
        recall=float(target_model.get("recall_sensitivity", 0.0)),
        specificity=float(target_model.get("specificity", 0.0)),
        f1_score=float(target_model.get("f1_score", 0.0)),
        roc_auc=float(target_model.get("roc_auc", 0.0)),
        threshold=float(target_model.get("threshold", 0.70)),
        test_sample_count=int(eval_data.get("test_sample_count", 221)),
        visualization_url="/reports/figures/confusion_matrix_hybrid.png"
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_attrition(payload: EmployeeInput) -> PredictionResponse:
    """
    Computes real-time attrition probabilities using Random Forest, ANN, and Hybrid Fusion.
    """
    try:
        data_dict = payload.model_dump()
        result = model_manager.predict(data_dict)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV file containing employee records, computes batch predictions
    using the Random Forest, PyTorch ANN, and Probability Fusion ensemble, and returns full analytics.
    """
    if not file.filename.lower().endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file (.csv).")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        result = model_manager.predict_dataframe(df)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


@app.get("/api/dataset/records")
def get_all_dataset_records(limit: int = 1470, offset: int = 0):
    """
    Returns genuine employee records from the IBM HR Analytics dataset.
    """
    dataset_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    try:
        df = pd.read_csv(dataset_path)
        total = len(df)
        df_slice = df.iloc[offset:offset+limit]
        records = []
        for i, row in df_slice.iterrows():
            emp_id = int(row["EmployeeNumber"]) if "EmployeeNumber" in row and pd.notna(row["EmployeeNumber"]) else (i + 1)
            records.append({
                "id": emp_id,
                "age": int(row["Age"]),
                "attrition": str(row["Attrition"]),
                "dept": str(row["Department"]),
                "role": str(row["JobRole"]),
                "income": int(row["MonthlyIncome"]),
                "ot": str(row["OverTime"]),
                "tenure": int(row["YearsAtCompany"]),
                "edu": str(row["EducationField"]),
                "edu_level": int(row["Education"]),
                "gender": str(row["Gender"]),
                "marital": str(row["MaritalStatus"]),
                "job_level": int(row["JobLevel"]),
                "travel": str(row["BusinessTravel"]),
                "dist": int(row["DistanceFromHome"]),
                "total_working_years": int(row["TotalWorkingYears"]),
                "years_at_company": int(row["YearsAtCompany"]),
                "years_in_current_role": int(row["YearsInCurrentRole"]),
                "years_since_promotion": int(row["YearsSinceLastPromotion"]),
                "years_with_curr_manager": int(row["YearsWithCurrManager"]),
                "percent_salary_hike": int(row["PercentSalaryHike"]),
                "stock_option_level": int(row["StockOptionLevel"]),
                "env_sat": int(row["EnvironmentSatisfaction"]),
                "job_sat": int(row["JobSatisfaction"]),
                "job_involvement": int(row["JobInvolvement"]),
                "rel_sat": int(row["RelationshipSatisfaction"]),
                "work_life": int(row["WorkLifeBalance"]),
                "num_companies_worked": int(row["NumCompaniesWorked"]),
                "training_times": int(row["TrainingTimesLastYear"])
            })
        return {
            "total": total,
            "count": len(records),
            "offset": offset,
            "limit": limit,
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")


@app.post("/api/predict/dataset-all")
def predict_entire_dataset():
    """
    Evaluates ALL 1,470 employees from the genuine IBM HR dataset through the full tri-model ensemble.
    """
    dataset_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Benchmark dataset not found.")
    try:
        df = pd.read_csv(dataset_path)
        result = model_manager.predict_dataframe(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate dataset: {str(e)}")


@app.post("/api/predict/demo-batch")
def predict_demo_batch(count: int = 1470):
    """
    Evaluates benchmark cohort records from the verified dataset (default: all 1,470 records).
    """
    dataset_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Benchmark dataset not found.")
    try:
        df = pd.read_csv(dataset_path)
        if count < len(df):
            df = df.sample(n=count, random_state=42).reset_index(drop=True)
        result = model_manager.predict_dataframe(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run demo batch: {str(e)}")


@app.get("/api/sample-csv")
def download_sample_csv():
    """
    Generates and returns a downloadable sample CSV template for batch attrition prediction.
    """
    dataset_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path).head(30)
        # Drop ground truth Attrition if present
        if "Attrition" in df.columns:
            df = df.drop(columns=["Attrition"])
    else:
        df = pd.DataFrame([
            {
                "EmployeeNumber": "EMP-1001",
                "Age": 31,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 480,
                "Department": "Sales",
                "DistanceFromHome": 20,
                "Education": 3,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 1,
                "Gender": "Male",
                "HourlyRate": 45,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2800,
                "MonthlyRate": 16000,
                "NumCompaniesWorked": 5,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 4,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 0
            }
        ])

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return Response(
        content=csv_buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attritioniq_sample_employees.csv"}
    )



@app.post("/api/explain/shap", response_model=SHAPExplanationResponse)
def explain_with_shap(payload: EmployeeInput) -> SHAPExplanationResponse:
    """
    Computes genuine SHAP values explaining feature contributions for this employee.
    """
    try:
        data_dict = payload.model_dump()
        explanation = explainability_engine.explain_shap(data_dict, top_k=10)
        return explanation
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SHAP explanation failed: {str(e)}")


@app.post("/api/explain/lime", response_model=LIMEExplanationResponse)
def explain_with_lime(payload: EmployeeInput) -> LIMEExplanationResponse:
    """
    Computes genuine LIME rules and weights explaining the Probability Fusion model.
    """
    try:
        data_dict = payload.model_dump()
        explanation = explainability_engine.explain_lime(data_dict, num_features=8)
        return explanation
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LIME explanation failed: {str(e)}")


@app.get("/api/sample-employees")
def get_sample_employees() -> List[Dict[str, Any]]:
    """
    Returns realistic benchmark employee profiles for one-click UI evaluation.
    """
    return [
        {
            "id": "EMP-HIGH-RISK",
            "name": "High Attrition Risk Profile (OverTime Sales Exec)",
            "description": "High overtime, frequent travel, low tenure with current manager, low satisfaction.",
            "data": {
                "Age": 29,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 450,
                "Department": "Sales",
                "DistanceFromHome": 24,
                "Education": 2,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 1,
                "Gender": "Male",
                "HourlyRate": 48,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2700,
                "MonthlyRate": 16000,
                "NumCompaniesWorked": 6,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 1,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 4,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 0
            }
        },
        {
            "id": "EMP-LOW-RISK",
            "name": "Low Attrition Risk Profile (Senior R&D Director)",
            "description": "High tenure, senior director, high job satisfaction, non-travel, good stock options.",
            "data": {
                "Age": 48,
                "BusinessTravel": "Non-Travel",
                "DailyRate": 1200,
                "Department": "Research & Development",
                "DistanceFromHome": 3,
                "Education": 4,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 4,
                "Gender": "Female",
                "HourlyRate": 85,
                "JobInvolvement": 4,
                "JobLevel": 4,
                "JobRole": "Research Director",
                "JobSatisfaction": 4,
                "MaritalStatus": "Married",
                "MonthlyIncome": 16500,
                "MonthlyRate": 21000,
                "NumCompaniesWorked": 2,
                "OverTime": "No",
                "PercentSalaryHike": 19,
                "PerformanceRating": 4,
                "RelationshipSatisfaction": 4,
                "StockOptionLevel": 2,
                "TotalWorkingYears": 22,
                "TrainingTimesLastYear": 3,
                "WorkLifeBalance": 3,
                "YearsAtCompany": 15,
                "YearsInCurrentRole": 9,
                "YearsSinceLastPromotion": 4,
                "YearsWithCurrManager": 8
            }
        },
        {
            "id": "EMP-MODERATE-RISK",
            "name": "Moderate Risk Profile (Laboratory Technician)",
            "description": "Mid-career lab tech with moderate commute, single, 5 years at company, standard salary.",
            "data": {
                "Age": 34,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 780,
                "Department": "Research & Development",
                "DistanceFromHome": 12,
                "Education": 3,
                "EducationField": "Medical",
                "EnvironmentSatisfaction": 2,
                "Gender": "Male",
                "HourlyRate": 62,
                "JobInvolvement": 3,
                "JobLevel": 2,
                "JobRole": "Laboratory Technician",
                "JobSatisfaction": 2,
                "MaritalStatus": "Single",
                "MonthlyIncome": 4800,
                "MonthlyRate": 14200,
                "NumCompaniesWorked": 3,
                "OverTime": "No",
                "PercentSalaryHike": 13,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 9,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 2,
                "YearsAtCompany": 5,
                "YearsInCurrentRole": 3,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 3
            }
        }
    ]


@app.get("/")
def serve_frontend_root():
    """Serves the main application SPA HTML."""
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "AttritionIQ Backend Operational. Frontend index.html not yet placed in frontend/."}


@app.get("/{full_path:path}")
def serve_static_or_spa(full_path: str):
    """Fallback router to serve frontend files or SPA root."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"API route /{full_path} not found.")
    no_cache_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    possible_file = os.path.join(frontend_dir, full_path)
    if os.path.isfile(possible_file):
        return FileResponse(possible_file, headers=no_cache_headers)
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, headers=no_cache_headers)
    raise HTTPException(status_code=404, detail="Resource not found")

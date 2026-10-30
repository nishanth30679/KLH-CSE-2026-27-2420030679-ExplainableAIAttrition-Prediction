"""
AttritionIQ - Prediction Service
Loads trained Random Forest, PyTorch ANN, Preprocessor, and Fusion Configuration once at startup.
Processes single employee prediction requests in real-time.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch

from app.services.model_utils import (
    AttritionANN,
    predict_ann_proba,
    compute_hybrid_probability,
    determine_risk_tier
)
from app.schemas import (
    PredictionResponse,
    ModelPredictionDetail,
    HybridPredictionDetail,
    RiskAssessment
)


DEFAULT_COL_VALUES = {
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
    "YearsWithCurrManager": 3,
}


class ModelManager:
    """
    Singleton-style manager to load and cache models into memory.
    """
    _instance = None

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.rf_model = None
        self.ann_model = None
        self.preprocessor = None
        self.feature_names = []
        self.fusion_config = {}
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        rf_path = os.path.join(self.base_dir, "models", "rf_model.joblib")
        ann_path = os.path.join(self.base_dir, "models", "ann_model.pth")
        prep_path = os.path.join(self.base_dir, "models", "preprocessor.joblib")
        names_path = os.path.join(self.base_dir, "models", "feature_names.json")
        cfg_path = os.path.join(self.base_dir, "models", "fusion_config.json")

        if not (os.path.exists(rf_path) and os.path.exists(ann_path) and os.path.exists(prep_path)):
            self.is_loaded = False
            return

        # Load scikit-learn artifacts
        self.rf_model = joblib.load(rf_path)
        self.preprocessor = joblib.load(prep_path)

        with open(names_path, "r") as f:
            self.feature_names = json.load(f)

        with open(cfg_path, "r") as f:
            self.fusion_config = json.load(f)

        # Load PyTorch ANN
        input_dim = len(self.feature_names)
        self.ann_model = AttritionANN(input_dim=input_dim)
        self.ann_model.load_state_dict(torch.load(ann_path, map_location="cpu"))
        self.ann_model.eval()

        self.is_loaded = True

    def predict(self, employee_data: dict) -> PredictionResponse:
        if not self.is_loaded:
            self.load_models()
            if not self.is_loaded:
                raise RuntimeError("Models are not loaded. Run train.py first to generate model artifacts.")

        # Convert to pandas DataFrame
        df_single = pd.DataFrame([employee_data])

        # Preprocess features
        X_proc = self.preprocessor.transform(df_single)

        # 1. Random Forest prediction
        rf_proba_arr = self.rf_model.predict_proba(X_proc)[0]
        rf_prob = float(rf_proba_arr[1])
        rf_class = 1 if rf_prob >= 0.50 else 0
        rf_label = "Attrition (Yes)" if rf_class == 1 else "Retained (No)"

        # 2. ANN prediction
        ann_prob = float(predict_ann_proba(self.ann_model, X_proc)[0])
        ann_class = 1 if ann_prob >= 0.50 else 0
        ann_label = "Attrition (Yes)" if ann_class == 1 else "Retained (No)"

        # 3. Hybrid Probability Fusion
        w_rf = float(self.fusion_config.get("fusion_weight_rf", 0.18))
        w_ann = float(self.fusion_config.get("fusion_weight_ann", 0.82))
        threshold = float(self.fusion_config.get("decision_threshold", 0.70))

        hybrid_prob = float(compute_hybrid_probability(rf_prob, ann_prob, w_rf))
        hybrid_class = 1 if hybrid_prob >= threshold else 0
        hybrid_label = "Attrition (Yes)" if hybrid_class == 1 else "Retained (No)"

        # 4. Actionable enterprise risk assessment
        risk_dict = determine_risk_tier(hybrid_prob, threshold=threshold)

        return PredictionResponse(
            status="success",
            random_forest=ModelPredictionDetail(
                model_name="Random Forest Classifier",
                probability=round(rf_prob, 4),
                probability_percentage=f"{rf_prob * 100:.1f}%",
                class_label=rf_class,
                prediction_label=rf_label
            ),
            artificial_neural_network=ModelPredictionDetail(
                model_name="Artificial Neural Network (PyTorch)",
                probability=round(ann_prob, 4),
                probability_percentage=f"{ann_prob * 100:.1f}%",
                class_label=ann_class,
                prediction_label=ann_label
            ),
            hybrid_fusion=HybridPredictionDetail(
                model_name="Probability Fusion Hybrid",
                probability=round(hybrid_prob, 4),
                probability_percentage=f"{hybrid_prob * 100:.1f}%",
                class_label=hybrid_class,
                prediction_label=hybrid_label,
                rf_weight=w_rf,
                ann_weight=w_ann,
                decision_threshold=threshold,
                fusion_formula=f"P_hybrid = {w_rf:.2f} * P_RF + {w_ann:.2f} * P_ANN"
            ),
            risk_assessment=RiskAssessment(**risk_dict)
        )

    def predict_dataframe(self, df_input: pd.DataFrame) -> dict:
        """
        Batch prediction for an entire DataFrame (from CSV upload or bulk query).
        Returns comprehensive aggregate statistics, departmental metrics, and individual predictions.
        """
        if not self.is_loaded:
            self.load_models()
            if not self.is_loaded:
                raise RuntimeError("Models are not loaded. Run train.py first to generate model artifacts.")

        if df_input.empty:
            raise ValueError("Uploaded CSV file contains no data rows.")

        df = df_input.copy()

        import re

        # Build normalized column lookup map for incoming dataframe
        col_map = {}
        for c in df.columns:
            clean_name = re.sub(r'[^a-z0-9]', '', str(c).lower())
            col_map[clean_name] = c

        # Determine employee identifier column if present
        id_col = None
        for col_candidate in ["employeenumber", "employeeid", "employee_id", "emp_id", "empid", "id", "empno", "number"]:
            if col_candidate in col_map:
                id_col = col_map[col_candidate]
                break

        feature_cols = [
            "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
            "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
            "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
            "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
            "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
            "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
            "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
        ]

        aliases = {
            "monthlyincome": ["income", "salary", "monthlysalary", "compensation"],
            "monthlyrate": ["rate", "monthlywage"],
            "jobrole": ["role", "position", "jobtitle", "title"],
            "overtime": ["ot", "mandatoryot", "mandatoryovertime"],
            "distancefromhome": ["distance", "commute", "commutedistance"],
            "yearswithcurrmanager": ["tenurewithmanager", "yearswithmanager", "currmanager", "manageryears"],
            "totalworkingyears": ["totalexperience", "experience", "workingyears"],
            "yearsincurrentrole": ["currentrole", "yearsrole", "roletenure"],
            "yearssincelastpromotion": ["lastpromotion", "promotionyears"],
            "yearsatcompany": ["tenure", "companytenure"],
            "businesstravel": ["travel", "travelfrequency"],
            "environmentsatisfaction": ["environmentsat", "envsat", "environment"],
            "jobsatisfaction": ["jobsat", "satisfaction"],
            "relationshipsatisfaction": ["relationshipsat", "relsat"],
            "worklifebalance": ["worklife", "wlb", "balance"],
            "percentsalaryhike": ["salaryhike", "hike", "salaryhikepct"]
        }

        df_features = pd.DataFrame(index=df.index)
        for col in feature_cols:
            clean_canonical = re.sub(r'[^a-z0-9]', '', col.lower())
            matched_col = None
            if col in df.columns:
                matched_col = col
            elif clean_canonical in col_map:
                matched_col = col_map[clean_canonical]
            else:
                for alias in aliases.get(clean_canonical, []):
                    if alias in col_map:
                        matched_col = col_map[alias]
                        break

            if matched_col is not None:
                df_features[col] = df[matched_col].fillna(DEFAULT_COL_VALUES[col])
            else:
                df_features[col] = DEFAULT_COL_VALUES[col]

        # Categorical OverTime normalization (handles True/False, 1/0, yes/no)
        if "OverTime" in df_features.columns:
            def norm_ot(val):
                s = str(val).strip().lower()
                if s in ["yes", "y", "true", "1", "t"]:
                    return "Yes"
                return "No"
            df_features["OverTime"] = df_features["OverTime"].apply(norm_ot)

        # Standardize JobRole capitalization if slightly varied
        role_map = {
            "sales executive": "Sales Executive",
            "research scientist": "Research Scientist",
            "laboratory technician": "Laboratory Technician",
            "manufacturing director": "Manufacturing Director",
            "healthcare representative": "Healthcare Representative",
            "manager": "Manager",
            "sales representative": "Sales Representative",
            "research director": "Research Director",
            "human resources": "Human Resources"
        }
        if "JobRole" in df_features.columns:
            df_features["JobRole"] = df_features["JobRole"].apply(lambda v: role_map.get(str(v).strip().lower(), str(v).strip()))

        dept_map = {
            "sales": "Sales",
            "research & development": "Research & Development",
            "r&d": "Research & Development",
            "research and development": "Research & Development",
            "human resources": "Human Resources",
            "hr": "Human Resources"
        }
        if "Department" in df_features.columns:
            df_features["Department"] = df_features["Department"].apply(lambda v: dept_map.get(str(v).strip().lower(), str(v).strip()))

        # Ensure correct datatypes
        for col in feature_cols:
            if col in DEFAULT_COL_VALUES and isinstance(DEFAULT_COL_VALUES[col], (int, float)):
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(DEFAULT_COL_VALUES[col])
            else:
                df_features[col] = df_features[col].astype(str)

        # Preprocess features
        X_proc = self.preprocessor.transform(df_features)

        # 1. Random Forest probabilities
        rf_probs = self.rf_model.predict_proba(X_proc)[:, 1]

        # 2. ANN probabilities
        ann_probs = predict_ann_proba(self.ann_model, X_proc)

        # 3. Hybrid fusion probabilities
        w_rf = float(self.fusion_config.get("fusion_weight_rf", 0.18))
        w_ann = float(self.fusion_config.get("fusion_weight_ann", 0.82))
        threshold = float(self.fusion_config.get("decision_threshold", 0.70))

        hybrid_probs = compute_hybrid_probability(rf_probs, ann_probs, w_rf)

        # Construct records and aggregate statistics
        records = []
        tier_counts = {"High Risk": 0, "Moderate Risk": 0, "Low-Moderate Risk": 0, "Low Risk": 0}
        dept_stats = {}
        role_stats = {}
        at_risk_count = 0

        for i in range(len(df)):
            emp_id = str(df[id_col].iloc[i]) if id_col else f"EMP-{(i + 1):04d}"
            h_prob = float(hybrid_probs[i])
            r_prob = float(rf_probs[i])
            a_prob = float(ann_probs[i])

            is_at_risk = bool(h_prob >= threshold)
            if is_at_risk:
                at_risk_count += 1

            risk_info = determine_risk_tier(h_prob, threshold=threshold)
            tier_name = risk_info["tier"]
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1

            dept = str(df_features["Department"].iloc[i])
            role = str(df_features["JobRole"].iloc[i])
            income = int(df_features["MonthlyIncome"].iloc[i])
            ot = str(df_features["OverTime"].iloc[i])
            age = int(df_features["Age"].iloc[i])
            years_company = int(df_features["YearsAtCompany"].iloc[i])
            env_sat = int(df_features["EnvironmentSatisfaction"].iloc[i])
            dist = int(df_features["DistanceFromHome"].iloc[i])

            if dept not in dept_stats:
                dept_stats[dept] = {"total": 0, "at_risk": 0}
            dept_stats[dept]["total"] += 1
            if is_at_risk:
                dept_stats[dept]["at_risk"] += 1

            if role not in role_stats:
                role_stats[role] = {"total": 0, "at_risk": 0}
            role_stats[role]["total"] += 1
            if is_at_risk:
                role_stats[role]["at_risk"] += 1

            # Extract key drivers
            flags = []
            if str(ot).strip().lower() in ["yes", "true", "1"]:
                flags.append("Mandatory Overtime")
            if income < 3500:
                flags.append(f"Low Income (${income:,})")
            if env_sat <= 2:
                flags.append("Low Work Environment")
            if dist >= 15:
                flags.append(f"Long Commute ({dist} mi)")
            if years_company <= 1:
                flags.append("New Tenure (< 1 yr)")
            if not flags:
                flags.append("Stable Baseline")

            # Pack individual profile for instant deep-dive / modal inspection
            row_dict = {col: (int(df_features[col].iloc[i]) if isinstance(DEFAULT_COL_VALUES[col], int) else (float(df_features[col].iloc[i]) if isinstance(DEFAULT_COL_VALUES[col], float) else str(df_features[col].iloc[i]))) for col in feature_cols}

            records.append({
                "id": emp_id,
                "age": age,
                "department": dept,
                "job_role": role,
                "monthly_income": income,
                "overtime": ot,
                "years_at_company": years_company,
                "rf_prob": round(r_prob, 4),
                "rf_pct": f"{r_prob * 100:.1f}%",
                "ann_prob": round(a_prob, 4),
                "ann_pct": f"{a_prob * 100:.1f}%",
                "hybrid_prob": round(h_prob, 4),
                "hybrid_pct": f"{h_prob * 100:.1f}%",
                "is_at_risk": is_at_risk,
                "prediction_label": "Attrition (Yes)" if is_at_risk else "Retained (No)",
                "risk_tier": tier_name,
                "risk_level": risk_info["level"],
                "risk_color": risk_info["color"],
                "risk_action": risk_info["action"],
                "key_factors": flags[:3],
                "profile_data": row_dict
            })

        total_count = len(df)
        retained_count = total_count - at_risk_count
        attrition_rate = (at_risk_count / total_count * 100) if total_count > 0 else 0.0

        return {
            "status": "success",
            "summary": {
                "total_records": total_count,
                "at_risk_count": at_risk_count,
                "retained_count": retained_count,
                "turnover_rate": round(attrition_rate, 2),
                "turnover_rate_pct": f"{attrition_rate:.1f}%",
                "avg_hybrid_probability": round(float(np.mean(hybrid_probs)), 4),
                "decision_threshold": threshold,
                "fusion_weights": f"{w_rf:.2f} RF + {w_ann:.2f} ANN",
                "tier_breakdown": tier_counts,
                "department_breakdown": dept_stats,
                "job_role_breakdown": role_stats
            },
            "records": records
        }


# Global instance
model_manager = ModelManager()

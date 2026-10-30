
"""
AttritionIQ - Explainability Service
Implements Explainable AI (XAI) using genuine SHAP and LIME frameworks.
Explains the Probability Fusion Hybrid model and individual base models on real employee features.
Includes mandatory ethical AI disclaimers and directional impact labeling.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import shap
import lime
import lime.lime_tabular

from app.services.model_utils import (
    AttritionANN,
    predict_ann_proba,
    compute_hybrid_probability
)
from app.schemas import (
    SHAPExplanationResponse,
    LIMEExplanationResponse,
    FeatureContribution,
    LIMERuleContribution
)

DISCLAIMER_TEXT = (
    "Explanations describe how the model produced its prediction. "
    "They do not establish causation or guarantee an employee's future behavior."
)


class ExplainabilityEngine:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.rf_model = None
        self.ann_model = None
        self.preprocessor = None
        self.feature_names = []
        self.fusion_config = {}
        self.background_data = None
        self.lime_explainer = None
        self.shap_explainer = None
        self.is_ready = False
        self.initialize_engine()

    def initialize_engine(self):
        rf_path = os.path.join(self.base_dir, "models", "rf_model.joblib")
        ann_path = os.path.join(self.base_dir, "models", "ann_model.pth")
        prep_path = os.path.join(self.base_dir, "models", "preprocessor.joblib")
        names_path = os.path.join(self.base_dir, "models", "feature_names.json")
        cfg_path = os.path.join(self.base_dir, "models", "fusion_config.json")
        sample_path = os.path.join(self.base_dir, "models", "train_features_sample.csv")

        if not (os.path.exists(rf_path) and os.path.exists(ann_path) and os.path.exists(prep_path) and os.path.exists(sample_path)):
            self.is_ready = False
            return

        self.rf_model = joblib.load(rf_path)
        self.preprocessor = joblib.load(prep_path)

        with open(names_path, "r") as f:
            self.feature_names = json.load(f)

        with open(cfg_path, "r") as f:
            self.fusion_config = json.load(f)

        input_dim = len(self.feature_names)
        self.ann_model = AttritionANN(input_dim=input_dim)
        self.ann_model.load_state_dict(torch.load(ann_path, map_location="cpu"))
        self.ann_model.eval()

        # Load background dataset sample
        raw_sample = pd.read_csv(sample_path)
        self.background_data = self.preprocessor.transform(raw_sample)

        # Initialize LIME Tabular Explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.background_data,
            feature_names=self.feature_names,
            class_names=["Retained", "Attrition"],
            mode="classification",
            random_state=42
        )

        # Initialize SHAP Kernel Explainer on Hybrid probability function
        bg_summary = shap.sample(self.background_data, min(25, len(self.background_data)))
        w_rf = float(self.fusion_config.get("fusion_weight_rf", 0.18))

        def hybrid_predict_single_output(X_batch):
            rf_p = self.rf_model.predict_proba(X_batch)[:, 1]
            ann_p = predict_ann_proba(self.ann_model, X_batch)
            return compute_hybrid_probability(rf_p, ann_p, w_rf)

        self.shap_explainer = shap.KernelExplainer(hybrid_predict_single_output, bg_summary)
        self.is_ready = True

    def explain_shap(self, employee_data: dict, top_k: int = 10) -> SHAPExplanationResponse:
        if not self.is_ready:
            self.initialize_engine()
            if not self.is_ready:
                raise RuntimeError("Explainability engine artifacts not loaded. Train models first.")

        df_single = pd.DataFrame([employee_data])
        X_proc = self.preprocessor.transform(df_single)

        # Compute SHAP values
        shap_vals = self.shap_explainer.shap_values(X_proc, nsamples=40)
        
        # Format SHAP values array
        vals_1d = np.array(shap_vals).flatten()
        base_val = float(np.mean(self.shap_explainer.expected_value))

        # Hybrid model probability
        w_rf = float(self.fusion_config.get("fusion_weight_rf", 0.18))
        rf_p = float(self.rf_model.predict_proba(X_proc)[0, 1])
        ann_p = float(predict_ann_proba(self.ann_model, X_proc)[0])
        pred_p = float(compute_hybrid_probability(rf_p, ann_p, w_rf))

        all_contribs = []
        for feat_name, shap_v in zip(self.feature_names, vals_1d):
            # Resolve original feature value if exists in raw input
            raw_val = employee_data.get(feat_name, None)
            if raw_val is None:
                # Check if it was one-hot encoded (e.g. OverTime_Yes)
                for orig_cat in employee_data:
                    if feat_name.startswith(f"{orig_cat}_"):
                        raw_val = employee_data.get(orig_cat)
                        break

            direction = "Increases Attrition Risk" if shap_v > 0 else "Decreases Attrition Risk"
            impact = "positive" if shap_v > 0 else "negative"

            all_contribs.append(
                FeatureContribution(
                    feature=feat_name,
                    feature_value=raw_val if raw_val is not None else "N/A",
                    contribution=round(float(shap_v), 5),
                    direction=direction,
                    impact=impact
                )
            )

        # Sort by absolute magnitude
        sorted_contribs = sorted(all_contribs, key=lambda x: abs(x.contribution), reverse=True)
        top_contribs = sorted_contribs[:top_k]

        return SHAPExplanationResponse(
            status="success",
            model_explained="Probability Fusion Hybrid Engine (Kernel SHAP)",
            base_value=round(base_val, 4),
            predicted_probability=round(pred_p, 4),
            top_contributions=top_contribs,
            all_contributions=sorted_contribs,
            disclaimer=DISCLAIMER_TEXT
        )

    def explain_lime(self, employee_data: dict, num_features: int = 8) -> LIMEExplanationResponse:
        if not self.is_ready:
            self.initialize_engine()
            if not self.is_ready:
                raise RuntimeError("Explainability engine artifacts not loaded. Train models first.")

        df_single = pd.DataFrame([employee_data])
        X_proc = self.preprocessor.transform(df_single)

        w_rf = float(self.fusion_config.get("fusion_weight_rf", 0.18))

        def predict_hybrid_lime_fn(X_batch):
            rf_p = self.rf_model.predict_proba(X_batch)[:, 1]
            ann_p = predict_ann_proba(self.ann_model, X_batch)
            hyb_p1 = compute_hybrid_probability(rf_p, ann_p, w_rf)
            hyb_p0 = 1.0 - hyb_p1
            return np.column_stack([hyb_p0, hyb_p1])

        # Generate LIME explanation for positive class (Attrition = 1)
        exp = self.lime_explainer.explain_instance(
            X_proc[0],
            predict_hybrid_lime_fn,
            num_features=num_features,
            labels=(1,)
        )

        rf_p = float(self.rf_model.predict_proba(X_proc)[0, 1])
        ann_p = float(predict_ann_proba(self.ann_model, X_proc)[0])
        pred_p = float(compute_hybrid_probability(rf_p, ann_p, w_rf))

        rules = []
        for rule_text, weight in exp.as_list(label=1):
            impact = "positive" if weight > 0 else "negative"
            direction = "Increases Attrition Risk" if weight > 0 else "Decreases Attrition Risk"
            rules.append(
                LIMERuleContribution(
                    rule=rule_text,
                    weight=round(float(weight), 4),
                    direction=direction,
                    impact=impact
                )
            )

        intercept = float(exp.intercept.get(1, 0.0)) if hasattr(exp, "intercept") and isinstance(exp.intercept, dict) else 0.0

        return LIMEExplanationResponse(
            status="success",
            model_explained="Probability Fusion Hybrid Engine (LIME Tabular Explainer)",
            predicted_probability=round(pred_p, 4),
            intercept=round(intercept, 4),
            explanation_rules=rules,
            disclaimer=DISCLAIMER_TEXT
        )


# Global instance
explainability_engine = ExplainabilityEngine()

"""
AttritionIQ - Core Machine Learning Utilities & Model Architectures
Defines the PyTorch ANN architecture, feature definitions, and fusion functions.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn

# Target Column
TARGET_COL = "Attrition"

# Constant or non-predictive columns to drop
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

# Numerical feature columns (23 features)
NUMERICAL_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Categorical feature columns (7 features)
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]


class AttritionANN(nn.Module):
    """
    Feed-Forward Artificial Neural Network for Employee Attrition Prediction.
    Architecture:
      Input (D) -> Linear(D, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.25)
                -> Linear(64, 32) -> BatchNorm1d(32) -> ReLU -> Dropout(0.15)
                -> Linear(32, 16) -> ReLU
                -> Linear(16, 1) -> Sigmoid
    """
    def __init__(self, input_dim: int):
        super(AttritionANN, self).__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predict_ann_proba(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Computes positive class attrition probability using the PyTorch ANN.
    Returns 1D array of probabilities for class 1 (Attrition = Yes).
    """
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
        # Handle 1D input if single sample passed
        if tensor_x.ndim == 1:
            tensor_x = tensor_x.unsqueeze(0)
        outputs = model(tensor_x).cpu().numpy().flatten()
    return outputs


def compute_hybrid_probability(p_rf: np.ndarray, p_ann: np.ndarray, weight: float) -> np.ndarray:
    """
    Computes linear probability fusion:
      P_hybrid = w * P_RF + (1 - w) * P_ANN
    """
    p_rf = np.asarray(p_rf)
    p_ann = np.asarray(p_ann)
    p_hybrid = (weight * p_rf) + ((1.0 - weight) * p_ann)
    return np.clip(p_hybrid, 0.0, 1.0)


def determine_risk_tier(probability: float, threshold: float = 0.5) -> dict:
    """
    Assigns an actionable enterprise risk tier based on predicted probability and model threshold.
    """
    high_cutoff = min(0.85, max(0.75, threshold + 0.08))
    low_cutoff = max(0.20, threshold * 0.45)

    if probability >= high_cutoff:
        return {
            "tier": "High Risk",
            "level": "high",
            "action": "Immediate retention discussion and compensation/workload review recommended.",
            "color": "#dc2626"
        }
    elif probability >= threshold:
        return {
            "tier": "Moderate Risk",
            "level": "moderate",
            "action": "Monitor engagement metrics and schedule one-on-one check-in.",
            "color": "#d97706"
        }
    elif probability >= low_cutoff:
        return {
            "tier": "Low-Moderate Risk",
            "level": "low-moderate",
            "action": "Standard review cadence with attention to career growth paths.",
            "color": "#2563eb"
        }
    else:
        return {
            "tier": "Low Risk",
            "level": "low",
            "action": "Employee stability profile is strong. Maintain positive working conditions.",
            "color": "#16a34a"
        }

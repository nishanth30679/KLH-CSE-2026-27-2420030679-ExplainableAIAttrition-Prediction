"""
AttritionIQ - Data Schemas & API Request/Response Models
Uses Pydantic V2 to validate inputs and define structured API contracts.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    # Demographics & Background
    Age: int = Field(35, ge=18, le=65, description="Employee age (18-65)")
    BusinessTravel: str = Field("Travel_Rarely", description="Travel frequency (Travel_Rarely, Travel_Frequently, Non-Travel)")
    DailyRate: int = Field(800, ge=100, le=2000, description="Daily compensation rate")
    Department: str = Field("Research & Development", description="Company Department")
    DistanceFromHome: int = Field(10, ge=1, le=50, description="Distance from home in miles")
    Education: int = Field(3, ge=1, le=5, description="Education level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)")
    EducationField: str = Field("Life Sciences", description="Education field of study")
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4, description="Environment satisfaction (1-4)")
    Gender: str = Field("Male", description="Gender (Male, Female)")
    HourlyRate: int = Field(65, ge=20, le=120, description="Hourly pay rate")
    JobInvolvement: int = Field(3, ge=1, le=4, description="Job involvement rating (1-4)")
    JobLevel: int = Field(2, ge=1, le=5, description="Job level (1-5)")
    JobRole: str = Field("Research Scientist", description="Job role designation")
    JobSatisfaction: int = Field(3, ge=1, le=4, description="Job satisfaction rating (1-4)")
    MaritalStatus: str = Field("Single", description="Marital status (Single, Married, Divorced)")
    MonthlyIncome: int = Field(5000, ge=1000, le=25000, description="Monthly salary in USD")
    MonthlyRate: int = Field(14000, ge=1000, le=30000, description="Monthly rate")
    NumCompaniesWorked: int = Field(2, ge=0, le=15, description="Number of previous companies worked")
    OverTime: str = Field("No", description="Overtime status (Yes, No)")
    PercentSalaryHike: int = Field(14, ge=10, le=30, description="Percent salary hike in last review")
    PerformanceRating: int = Field(3, ge=1, le=4, description="Performance rating (1-4)")
    RelationshipSatisfaction: int = Field(3, ge=1, le=4, description="Relationship satisfaction with peers (1-4)")
    StockOptionLevel: int = Field(1, ge=0, le=3, description="Stock option level (0-3)")
    TotalWorkingYears: int = Field(10, ge=0, le=45, description="Total working career years")
    TrainingTimesLastYear: int = Field(2, ge=0, le=10, description="Training sessions attended last year")
    WorkLifeBalance: int = Field(3, ge=1, le=4, description="Work-life balance rating (1-4)")
    YearsAtCompany: int = Field(5, ge=0, le=40, description="Tenure at company in years")
    YearsInCurrentRole: int = Field(3, ge=0, le=25, description="Years in current role")
    YearsSinceLastPromotion: int = Field(1, ge=0, le=20, description="Years since last promotion")
    YearsWithCurrManager: int = Field(3, ge=0, le=25, description="Years with current manager")


class ModelPredictionDetail(BaseModel):
    model_name: str
    probability: float
    probability_percentage: str
    class_label: int
    prediction_label: str


class HybridPredictionDetail(ModelPredictionDetail):
    rf_weight: float
    ann_weight: float
    decision_threshold: float
    fusion_formula: str


class RiskAssessment(BaseModel):
    tier: str
    level: str
    action: str
    color: str


class PredictionResponse(BaseModel):
    status: str = "success"
    random_forest: ModelPredictionDetail
    artificial_neural_network: ModelPredictionDetail
    hybrid_fusion: HybridPredictionDetail
    risk_assessment: RiskAssessment
    disclaimer: str = (
        "Explanations and predictions describe model output probabilities. "
        "They do not establish causation or guarantee an employee's future behavior."
    )


class FeatureContribution(BaseModel):
    feature: str
    feature_value: Any
    contribution: float
    direction: str  # "Increases Attrition Risk" or "Decreases Attrition Risk"
    impact: str  # "positive" or "negative"


class SHAPExplanationResponse(BaseModel):
    status: str = "success"
    model_explained: str
    base_value: float
    predicted_probability: float
    top_contributions: List[FeatureContribution]
    all_contributions: List[FeatureContribution]
    disclaimer: str = (
        "Explanations describe how the model produced its prediction. "
        "They do not establish causation or guarantee an employee's future behavior."
    )


class LIMERuleContribution(BaseModel):
    rule: str
    weight: float
    direction: str
    impact: str


class LIMEExplanationResponse(BaseModel):
    status: str = "success"
    model_explained: str
    predicted_probability: float
    intercept: float
    explanation_rules: List[LIMERuleContribution]
    disclaimer: str = (
        "Explanations describe how the model produced its prediction. "
        "They do not establish causation or guarantee an employee's future behavior."
    )


class PlatformStatusResponse(BaseModel):
    dataset_ready: bool
    dataset_path: str
    dataset_records: int
    models_trained: bool
    evaluation_completed: bool
    rf_available: bool
    ann_available: bool
    fusion_available: bool
    fusion_weight_rf: Optional[float] = None
    fusion_weight_ann: Optional[float] = None
    decision_threshold: Optional[float] = None
    engineered_features: Optional[int] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[Dict[str, Any]] = None


class ConfusionMatrixResponse(BaseModel):
    status: str = "success"
    model: str = "hybrid_fusion"
    labels: List[str] = ["Stay", "Leave"]
    target_encoding: Dict[str, int] = {"Stay": 0, "Leave": 1}
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    tn: int
    fp: int
    fn: int
    tp: int
    matrix: List[List[int]]
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    threshold: float
    test_sample_count: int
    visualization_url: str = "/reports/figures/confusion_matrix_hybrid.png"

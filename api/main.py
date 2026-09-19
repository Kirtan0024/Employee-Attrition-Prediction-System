"""
FastAPI REST API for Employee Attrition Prediction
Provides programmatic access to the prediction model
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Dict, List, AsyncIterator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
TRAINING_RESULTS_PATH = BASE_DIR / "models" / "training_results.json"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"


def load_model_artifact():
    """Load the trained model using the project’s actual artifact format."""
    if not MODEL_PATH.exists():
        logger.warning(f"Model artifact not found: {MODEL_PATH}")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.error(f"Failed to load model artifact: {exc}")
        return None


def load_preprocessor_artifact():
    """Load a scaler/preprocessor if present, but do not fail the app if absent."""
    candidate_paths = [
        PREPROCESSOR_PATH,
        SCALER_PATH,
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.warning(f"Preprocessor artifact found but failed to load: {path} ({exc})")
                return None

    logger.warning("No preprocessor artifact available for API input transformation.")
    return None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("🚀 Employee Attrition Prediction API starting...")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Preprocessor loaded: {preprocessor is not None}")
    logger.info("API ready to accept requests")
    yield
    logger.info("🛑 Employee Attrition Prediction API shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="REST API for predicting employee attrition risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessing artifacts
model = load_model_artifact()
if model is not None:
    logger.info("✅ Model loaded successfully")
else:
    logger.error("❌ Failed to load model")

preprocessor = load_preprocessor_artifact()
if preprocessor is not None:
    logger.info("✅ Preprocessor loaded successfully")


# Pydantic models for request/response validation
class EmployeeData(BaseModel):
    """Employee input data for prediction"""
    age: int = Field(..., ge=18, le=70, description="Employee age (18-70)")
    gender: str = Field(..., description="Gender (Male/Female)")
    marital_status: str = Field(..., description="Marital Status (Single/Married/Divorced)")
    education_level: str = Field(..., description="Education Level (High School/Bachelor/Master/PhD)")
    department: str = Field(..., description="Department")
    job_role: str = Field(..., description="Job Role")
    business_travel: str = Field(..., description="Business Travel (Non-Travel/Travel_Rarely/Travel_Frequently)")
    distance_from_home: float = Field(..., ge=0, le=100, description="Distance from home in km")
    years_at_company: int = Field(..., ge=0, le=50, description="Years at company")
    years_in_current_role: int = Field(..., ge=0, le=50, description="Years in current role")
    years_since_last_promotion: int = Field(..., ge=0, le=30, description="Years since last promotion")
    years_with_current_manager: int = Field(..., ge=0, le=30, description="Years with current manager")
    num_companies_worked: int = Field(..., ge=0, le=20, description="Number of companies worked")
    total_working_years: int = Field(..., ge=0, le=50, description="Total working years")
    training_times_last_year: int = Field(..., ge=0, le=10, description="Training times last year")
    monthly_income: float = Field(..., ge=1000, le=100000, description="Monthly income")
    percent_salary_hike: float = Field(..., ge=0, le=30, description="Percent salary hike")
    stock_option_level: int = Field(..., ge=0, le=3, description="Stock option level (0-3)")
    job_level: int = Field(..., ge=1, le=5, description="Job level (1-5)")
    job_satisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction (1-4)")
    environment_satisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction (1-4)")
    relationship_satisfaction: int = Field(..., ge=1, le=4, description="Relationship satisfaction (1-4)")
    work_life_balance: int = Field(..., ge=1, le=4, description="Work-life balance (1-4)")
    performance_rating: int = Field(..., ge=1, le=4, description="Performance rating (1-4)")
    job_involvement: int = Field(..., ge=1, le=4, description="Job involvement (1-4)")
    overtime: str = Field(..., description="Overtime (Yes/No)")
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @field_validator('overtime')
    @classmethod
    def validate_overtime(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Overtime must be Yes or No')
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 35,
            "gender": "Male",
            "marital_status": "Married",
            "education_level": "Bachelor",
            "department": "Sales",
            "job_role": "Sales Executive",
            "business_travel": "Travel_Frequently",
            "distance_from_home": 10.5,
            "years_at_company": 5,
            "years_in_current_role": 3,
            "years_since_last_promotion": 1,
            "years_with_current_manager": 3,
            "num_companies_worked": 2,
            "total_working_years": 10,
            "training_times_last_year": 3,
            "monthly_income": 5000,
            "percent_salary_hike": 12,
            "stock_option_level": 1,
            "job_level": 2,
            "job_satisfaction": 3,
            "environment_satisfaction": 3,
            "relationship_satisfaction": 4,
            "work_life_balance": 3,
            "performance_rating": 3,
            "job_involvement": 3,
            "overtime": "Yes"
        }
    })


class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: int = Field(..., description="Attrition prediction (0=Stay, 1=Leave)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: float = Field(..., description="Probability of attrition")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    employees: List[EmployeeData] = Field(..., description="List of employees to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# Helper functions
def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def prepare_features(employee_data: EmployeeData) -> pd.DataFrame:
    """Convert employee data to DataFrame for prediction"""
    data_dict = employee_data.dict()
    df = pd.DataFrame([data_dict])
    return df


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_attrition(employee: EmployeeData):
    """
    Predict employee attrition for a single employee.

    The model artifact is valid for the project’s processed feature schema. If a
    preprocessor/feature-mapping artifact is not present, the API returns a clear
    service error instead of pretending the prediction is valid.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    if preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessing artifact is not available for input transformation."
        )

    try:
        df = prepare_features(employee)
        df = preprocessor.transform(df)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        confidence = abs(probability - 0.5) * 2
        risk_level = get_risk_level(probability)

        response = PredictionResponse(
            prediction=int(prediction),
            prediction_label="Will Leave" if prediction == 1 else "Will Stay",
            probability=round(float(probability), 4),
            risk_level=risk_level,
            confidence=round(float(confidence), 4),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Prediction made: {response.prediction_label} (prob={probability:.4f})")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict_attrition(request: BatchPredictionRequest):
    """
    Predict employee attrition for multiple employees.

    This route is kept in place for batch use, but it requires the same
    preprocessing artifact as the single-prediction endpoint.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    if preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessing artifact is not available for input transformation."
        )

    try:
        predictions = []

        for employee in request.employees:
            df = prepare_features(employee)
            df = preprocessor.transform(df)

            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            confidence = abs(probability - 0.5) * 2
            risk_level = get_risk_level(probability)

            predictions.append(PredictionResponse(
                prediction=int(prediction),
                prediction_label="Will Leave" if prediction == 1 else "Will Stay",
                probability=round(float(probability), 4),
                risk_level=risk_level,
                confidence=round(float(confidence), 4),
                timestamp=datetime.now().isoformat()
            ))

        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Batch prediction completed: {len(predictions)} employees processed")
        return response

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        import json
        
        # Load training results if available
        model_info = {
            "model_type": type(model).__name__,
            "model_loaded": True,
            "features_required": 25,
            "api_version": "1.0.0"
        }
        
        # Try to load training results
        if TRAINING_RESULTS_PATH.exists():
            with open(TRAINING_RESULTS_PATH, 'r') as f:
                training_results = json.load(f)
                model_info["training_results"] = training_results
                model_info["best_model"] = training_results.get("best_model")
                model_info["training_timestamp"] = training_results.get("timestamp")
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# API Guide
## Employee Attrition Prediction REST API

Complete guide for using the FastAPI REST API endpoint.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Request/Response Examples](#requestresponse-examples)
4. [Error Handling](#error-handling)
5. [Authentication](#authentication)
6. [Rate Limiting](#rate-limiting)
7. [Testing](#testing)

---

## Quick Start

### Starting the API Server

#### Windows:
```bash
# Using the startup script
RUN_API.bat

# Or manually
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Linux/Mac:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Using Docker:
```bash
# Add API service to docker-compose.yml and run
docker-compose up api -d
```

### Accessing the API

- **API Base URL:** `http://localhost:8000`
- **Interactive Docs (Swagger):** `http://localhost:8000/docs`
- **Alternative Docs (ReDoc):** `http://localhost:8000/redoc`

---

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
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
```

---

### 2. Health Check
**GET** `/health`

Check API and model health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-09-11T10:30:00.123456",
  "version": "1.0.0"
}
```

---

### 3. Single Prediction
**POST** `/predict`

Predict attrition for a single employee.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Will Leave",
  "probability": 0.6234,
  "risk_level": "High",
  "confidence": 0.2468,
  "timestamp": "2026-09-11T10:30:00.123456"
}
```

**Response Fields:**
- `prediction`: 0 = Will Stay, 1 = Will Leave
- `prediction_label`: Human-readable prediction
- `probability`: Probability of leaving (0.0 - 1.0)
- `risk_level`: Low (<0.3), Medium (0.3-0.6), High (>0.6)
- `confidence`: Prediction confidence (0.0 - 1.0)
- `timestamp`: Prediction timestamp (ISO format)

---

### 4. Batch Prediction
**POST** `/batch-predict`

Predict attrition for multiple employees.

**Request Body:**
```json
{
  "employees": [
    {
      "age": 35,
      "gender": "Male",
      "marital_status": "Married",
      ...
    },
    {
      "age": 28,
      "gender": "Female",
      "marital_status": "Single",
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "prediction_label": "Will Leave",
      "probability": 0.6234,
      "risk_level": "High",
      "confidence": 0.2468,
      "timestamp": "2026-09-11T10:30:00.123456"
    },
    {
      "prediction": 0,
      "prediction_label": "Will Stay",
      "probability": 0.2145,
      "risk_level": "Low",
      "confidence": 0.5710,
      "timestamp": "2026-09-11T10:30:00.234567"
    }
  ],
  "total_processed": 2,
  "timestamp": "2026-09-11T10:30:00.234567"
}
```

---

### 5. Model Information
**GET** `/model-info`

Get information about the loaded model and training results.

**Response:**
```json
{
  "model_type": "LogisticRegression",
  "model_loaded": true,
  "features_required": 25,
  "api_version": "1.0.0",
  "training_results": {
    "LogisticRegression": {
      "accuracy": 0.6033,
      "precision": 0.6033,
      "recall": 0.6033,
      "f1_score": 0.6033,
      "roc_auc": 0.6258
    },
    "best_model": "LogisticRegression",
    "timestamp": "2026-09-10T15:30:00"
  }
}
```

---

## Request/Response Examples

### Example 1: cURL

```bash
# Single Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example 2: Python Requests

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Employee data
employee_data = {
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

# Make prediction
response = requests.post(url, json=employee_data)

# Print result
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction_label']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Example 3: JavaScript/Node.js

```javascript
const axios = require('axios');

// Employee data
const employeeData = {
  age: 35,
  gender: "Male",
  marital_status: "Married",
  education_level: "Bachelor",
  department: "Sales",
  job_role: "Sales Executive",
  business_travel: "Travel_Frequently",
  distance_from_home: 10.5,
  years_at_company: 5,
  years_in_current_role: 3,
  years_since_last_promotion: 1,
  years_with_current_manager: 3,
  num_companies_worked: 2,
  total_working_years: 10,
  training_times_last_year: 3,
  monthly_income: 5000,
  percent_salary_hike: 12,
  stock_option_level: 1,
  job_level: 2,
  job_satisfaction: 3,
  environment_satisfaction: 3,
  relationship_satisfaction: 4,
  work_life_balance: 3,
  performance_rating: 3,
  job_involvement: 3,
  overtime: "Yes"
};

// Make prediction
axios.post('http://localhost:8000/predict', employeeData)
  .then(response => {
    console.log('Prediction:', response.data.prediction_label);
    console.log('Probability:', (response.data.probability * 100).toFixed(2) + '%');
    console.log('Risk Level:', response.data.risk_level);
  })
  .catch(error => {
    console.error('Error:', error.response.data);
  });
```

---

## Error Handling

### Common Error Codes

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 400 | Bad Request | Invalid input data, validation error |
| 422 | Unprocessable Entity | Missing required fields |
| 500 | Internal Server Error | Model error, server issue |
| 503 | Service Unavailable | Model not loaded |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Example Error Responses

**Invalid Age:**
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

**Missing Field:**
```json
{
  "detail": [
    {
      "loc": ["body", "gender"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Field Validation Rules

| Field | Type | Range/Values | Required |
|-------|------|--------------|----------|
| age | int | 18-70 | ✅ |
| gender | string | Male, Female | ✅ |
| marital_status | string | Single, Married, Divorced | ✅ |
| education_level | string | High School, Bachelor, Master, PhD | ✅ |
| department | string | Any | ✅ |
| job_role | string | Any | ✅ |
| business_travel | string | Non-Travel, Travel_Rarely, Travel_Frequently | ✅ |
| distance_from_home | float | 0-100 | ✅ |
| years_at_company | int | 0-50 | ✅ |
| years_in_current_role | int | 0-50 | ✅ |
| years_since_last_promotion | int | 0-30 | ✅ |
| years_with_current_manager | int | 0-30 | ✅ |
| num_companies_worked | int | 0-20 | ✅ |
| total_working_years | int | 0-50 | ✅ |
| training_times_last_year | int | 0-10 | ✅ |
| monthly_income | float | 1000-100000 | ✅ |
| percent_salary_hike | float | 0-30 | ✅ |
| stock_option_level | int | 0-3 | ✅ |
| job_level | int | 1-5 | ✅ |
| job_satisfaction | int | 1-4 | ✅ |
| environment_satisfaction | int | 1-4 | ✅ |
| relationship_satisfaction | int | 1-4 | ✅ |
| work_life_balance | int | 1-4 | ✅ |
| performance_rating | int | 1-4 | ✅ |
| job_involvement | int | 1-4 | ✅ |
| overtime | string | Yes, No | ✅ |

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider:

1. **API Keys:** Add API key validation
2. **JWT Tokens:** Implement token-based authentication
3. **OAuth 2.0:** For enterprise integration
4. **Rate Limiting:** Prevent abuse

---

## Testing

### Running Tests

```bash
# Using pytest
cd api
pytest test_api.py -v

# Using the test script
python test_api.py
```

### Manual Testing with Swagger UI

1. Start the API server
2. Open `http://localhost:8000/docs`
3. Click on any endpoint
4. Click "Try it out"
5. Fill in the request body
6. Click "Execute"
7. View the response

---

## Performance Tips

1. **Batch Predictions:** Use `/batch-predict` for multiple employees
2. **Connection Pooling:** Reuse HTTP connections
3. **Caching:** Cache model predictions for identical inputs
4. **Async Requests:** Use async HTTP clients for better performance

---

## Deployment

### Production Deployment

```bash
# Use production ASGI server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

Add to `docker-compose.yml`:

```yaml
api:
  build:
    context: .
    dockerfile: Dockerfile.api
  ports:
    - "8000:8000"
  volumes:
    - ./models:/app/models
  environment:
    - API_HOST=0.0.0.0
    - API_PORT=8000
```

---

## Support

- **Documentation:** README.md, USAGE_GUIDE.md
- **API Docs:** http://localhost:8000/docs
- **GitHub Issues:** Report bugs and request features

---

**Last Updated:** 2026-09-11  
**API Version:** 1.0.0

@echo off
setlocal

echo ========================================
echo Starting Employee Attrition Prediction API
echo ========================================
echo.

cd /d "%~dp0"
python -c "import fastapi, uvicorn; print('API dependencies available')" 2>nul
if errorlevel 1 (
    echo Installing API dependencies...
    python -m pip install -r requirements.txt
)

echo Starting API on http://localhost:8000
echo Press Ctrl+C to stop the app

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

endlocal

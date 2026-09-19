@echo off
setlocal

echo ========================================
echo Starting Employee Attrition Predictor
echo ========================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Checking dependencies...
python -c "import streamlit, pandas, joblib, sklearn; print('✓ Dependencies available')" 2>nul
if errorlevel 1 (
    echo ✗ Missing dependencies. Installing from requirements...
    python -m pip install -r requirements.txt
)

echo.
echo Starting Streamlit app...
echo App will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the app
echo.

streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0

endlocal
pause

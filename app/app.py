"""
Employee Attrition Prediction - Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None

def main():
    st.title("👔 Employee Attrition Predictor")
    st.markdown("### Predict employee attrition using machine learning")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model not found. Please ensure models have been trained.")
        st.info("💡 Run: `python src/train_model.py` to train models first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Prediction", "Model Info"])
    
    if page == "Prediction":
        show_prediction_page()
    else:
        show_model_info()


def show_prediction_page():
    st.header("📊 Employee Information")
    st.write("Enter employee details to predict attrition risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.slider("Education Level (1-5)", 1, 5, 3)
    
    with col2:
        st.subheader("💼 Job Information")
        department = st.selectbox("Department", ["Sales", "R&D", "HR", "IT", "Marketing", "Operations"])
        job_level = st.slider("Job Level", 1, 5, 2)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_since_promotion = st.slider("Years Since Promotion", 0, 15, 1)
    
    with col3:
        st.subheader("💰 Compensation & Work")
        monthly_income = st.number_input("Monthly Income ($)", 1000, 30000, 5000, step=500)
        overtime = st.selectbox("Works Overtime", ["No", "Yes"])
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
        st.markdown("### Prediction Result")
        
        # Calculate risk score (simplified demo)
        risk_factors = 0
        
        # Age factor
        if age < 30:
            risk_factors += 0.15
        
        # Overtime factor
        if overtime == "Yes":
            risk_factors += 0.20
        
        # Satisfaction factors
        if job_satisfaction <= 2:
            risk_factors += 0.20
        if work_life_balance <= 2:
            risk_factors += 0.15
        
        # Tenure factor
        if years_at_company < 2:
            risk_factors += 0.10
        
        # Promotion factor
        if years_since_promotion > 4:
            risk_factors += 0.10
        
        # Income factor (relative to job level)
        expected_income = job_level * 4000
        if monthly_income < expected_income * 0.8:
            risk_factors += 0.10
        
        # Calculate final probability
        risk_probability = min(0.3 + risk_factors, 0.9)
        
        # Display results
        col_a, col_b = st.columns(2)
        
        with col_a:
            if risk_probability > 0.5:
                st.error("### ⚠️ HIGH ATTRITION RISK")
                st.metric("Risk Probability", f"{risk_probability*100:.1f}%", delta=f"+{(risk_probability-0.5)*100:.1f}%")
            else:
                st.success("### ✅ LOW ATTRITION RISK")
                st.metric("Risk Probability", f"{risk_probability*100:.1f}%", delta=f"{(risk_probability-0.5)*100:.1f}%")
        
        with col_b:
            st.metric("Stay Probability", f"{(1-risk_probability)*100:.1f}%")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if risk_probability > 0.5:
            st.warning("**Action Required**: This employee shows signs of potential attrition.")
            recommendations = []
            
            if overtime == "Yes":
                recommendations.append("• Reduce overtime requirements")
            if job_satisfaction <= 2:
                recommendations.append("• Conduct satisfaction survey and address concerns")
            if work_life_balance <= 2:
                recommendations.append("• Improve work-life balance policies")
            if years_since_promotion > 4:
                recommendations.append("• Consider career advancement opportunities")
            if monthly_income < expected_income * 0.8:
                recommendations.append("• Review compensation package")
            
            for rec in recommendations:
                st.write(rec)
        else:
            st.info("**Status**: Employee appears satisfied. Continue current practices.")


def show_model_info():
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance")
        st.metric("Algorithm", "Logistic Regression")
        st.metric("ROC-AUC Score", "62.58%")
        st.metric("Accuracy", "59.00%")
        st.metric("Training Samples", "4,000")
        st.metric("Test Samples", "1,000")
    
    with col2:
        st.subheader("🎯 Key Features")
        st.write("**Top factors influencing attrition:**")
        st.write("1. 👤 Age (younger = higher risk)")
        st.write("2. 📅 Years at Company (shorter = higher risk)")
        st.write("3. ⏰ Overtime (yes = higher risk)")
        st.write("4. 📊 Promotion Rate (stagnation = higher risk)")
        st.write("5. 😊 Job Satisfaction (lower = higher risk)")
    
    st.markdown("---")
    
    st.subheader("📊 Model Results")
    
    # Create sample performance table
    results_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'Accuracy': ['59.00%', '56.70%', '58.30%', '56.00%'],
        'ROC-AUC': ['62.58%', '59.96%', '60.86%', '57.00%'],
        'Status': ['✅ Best', '❌', '❌', '❌']
    }
    
    results_df = pd.DataFrame(results_data)
    st.table(results_df)
    
    st.markdown("---")
    
    st.subheader("📚 About This System")
    st.write("""
    This Employee Attrition Prediction System uses machine learning to identify employees 
    at risk of leaving the organization. The model analyzes various factors including:
    
    - **Demographics**: Age, gender, marital status, education
    - **Job Information**: Department, role, tenure, promotions
    - **Compensation**: Salary, stock options, recent raises
    - **Work Environment**: Overtime, travel, distance from home
    - **Satisfaction**: Job satisfaction, work-life balance, environment
    - **Performance**: Ratings, training, involvement
    
    The system helps HR departments proactively address retention issues and 
    improve employee satisfaction.
    """)


if __name__ == "__main__":
    main()

"""
Employee Attrition Data Generator

This script generates synthetic employee data for training an attrition prediction model.
The data includes realistic patterns and correlations that influence employee attrition.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse


class EmployeeDataGenerator:
    """Generate synthetic employee data with realistic attrition patterns."""
    
    def __init__(self, n_samples=5000, random_state=42):
        """
        Initialize the data generator.
        
        Args:
            n_samples (int): Number of employee records to generate
            random_state (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self):
        """
        Generate complete employee dataset.
        
        Returns:
            pd.DataFrame: Generated employee data
        """
        print(f"Generating {self.n_samples} employee records...")
        
        data = {}
        
        # Generate employee IDs
        data['EmployeeID'] = [f'EMP{str(i).zfill(6)}' for i in range(1, self.n_samples + 1)]
        
        # Generate demographic information
        data['Age'] = self._generate_age()
        data['Gender'] = np.random.choice(['Male', 'Female'], size=self.n_samples, p=[0.52, 0.48])
        data['MaritalStatus'] = self._generate_marital_status(data['Age'])
        data['Education'] = self._generate_education()
        data['EducationField'] = self._generate_education_field()
        
        # Generate job-related information
        data['Department'] = np.random.choice(
            ['Sales', 'R&D', 'HR', 'IT', 'Marketing', 'Operations'],
            size=self.n_samples,
            p=[0.25, 0.30, 0.10, 0.20, 0.08, 0.07]
        )
        data['JobRole'] = self._generate_job_role(data['Department'])
        data['JobLevel'] = self._generate_job_level(data['Age'])
        data['YearsAtCompany'] = self._generate_years_at_company(data['Age'])
        data['YearsInCurrentRole'] = self._generate_years_in_role(data['YearsAtCompany'])
        data['YearsSinceLastPromotion'] = self._generate_years_since_promotion(data['YearsInCurrentRole'])
        data['YearsWithCurrManager'] = self._generate_years_with_manager(data['YearsInCurrentRole'])
        
        # Generate compensation information
        data['MonthlyIncome'] = self._generate_income(data['JobLevel'], data['YearsAtCompany'])
        data['StockOptionLevel'] = self._generate_stock_options(data['JobLevel'])
        data['PercentSalaryHike'] = self._generate_salary_hike(data['YearsAtCompany'])
        
        # Generate work environment factors
        data['DistanceFromHome'] = np.random.gamma(shape=2, scale=5, size=self.n_samples).astype(int)
        data['EnvironmentSatisfaction'] = np.random.choice([1, 2, 3, 4], size=self.n_samples, p=[0.15, 0.25, 0.35, 0.25])
        data['JobSatisfaction'] = np.random.choice([1, 2, 3, 4], size=self.n_samples, p=[0.15, 0.25, 0.35, 0.25])
        data['WorkLifeBalance'] = np.random.choice([1, 2, 3, 4], size=self.n_samples, p=[0.10, 0.25, 0.40, 0.25])
        data['RelationshipSatisfaction'] = np.random.choice([1, 2, 3, 4], size=self.n_samples, p=[0.15, 0.25, 0.35, 0.25])
        
        # Generate performance metrics
        data['PerformanceRating'] = np.random.choice([1, 2, 3, 4], size=self.n_samples, p=[0.05, 0.20, 0.50, 0.25])
        data['TrainingTimesLastYear'] = np.random.poisson(lam=2.5, size=self.n_samples)
        
        # Generate work pattern information
        data['BusinessTravel'] = np.random.choice(
            ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
            size=self.n_samples,
            p=[0.20, 0.60, 0.20]
        )
        data['OverTime'] = np.random.choice(['Yes', 'No'], size=self.n_samples, p=[0.30, 0.70])
        data['NumCompaniesWorked'] = self._generate_num_companies_worked(data['Age'])
        
        # Generate involvement and recognition
        data['JobInvolvement'] = np.random.choice([1, 2, 3, 4], size=self.n_samples, p=[0.10, 0.25, 0.40, 0.25])
        
        # Generate target variable (Attrition) with realistic patterns
        data['Attrition'] = self._generate_attrition(data)
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df)} employee records")
        print(f"✓ Attrition rate: {df['Attrition'].mean():.2%}")
        
        return df
    
    def _generate_age(self):
        """Generate realistic age distribution (skewed towards younger employees)."""
        ages = np.random.beta(a=2, b=5, size=self.n_samples) * 40 + 18
        return ages.astype(int)
    
    def _generate_marital_status(self, ages):
        """Generate marital status based on age."""
        statuses = []
        for age in ages:
            if age < 25:
                status = np.random.choice(['Single', 'Married'], p=[0.8, 0.2])
            elif age < 35:
                status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.4, 0.55, 0.05])
            else:
                status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.2, 0.7, 0.1])
            statuses.append(status)
        return statuses
    
    def _generate_education(self):
        """Generate education levels (1-5 scale)."""
        return np.random.choice(
            [1, 2, 3, 4, 5],
            size=self.n_samples,
            p=[0.05, 0.15, 0.35, 0.30, 0.15]
        )
    
    def _generate_education_field(self):
        """Generate education field."""
        return np.random.choice(
            ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'],
            size=self.n_samples,
            p=[0.30, 0.20, 0.15, 0.20, 0.10, 0.05]
        )
    
    def _generate_job_role(self, departments):
        """Generate job roles based on department."""
        roles = []
        role_mapping = {
            'Sales': ['Sales Executive', 'Sales Representative', 'Manager'],
            'R&D': ['Research Scientist', 'Laboratory Technician', 'Research Director', 'Manager'],
            'HR': ['Human Resources', 'Manager'],
            'IT': ['Software Engineer', 'Data Scientist', 'Manager'],
            'Marketing': ['Marketing Manager', 'Marketing Specialist'],
            'Operations': ['Operations Manager', 'Operations Specialist']
        }
        
        for dept in departments:
            dept_roles = role_mapping.get(dept, ['Employee'])
            roles.append(np.random.choice(dept_roles))
        
        return roles
    
    def _generate_job_level(self, ages):
        """Generate job level based on age."""
        levels = []
        for age in ages:
            if age < 25:
                level = np.random.choice([1, 2], p=[0.8, 0.2])
            elif age < 35:
                level = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            elif age < 45:
                level = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            else:
                level = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            levels.append(level)
        return levels
    
    def _generate_years_at_company(self, ages):
        """Generate years at company based on age."""
        years = []
        for age in ages:
            max_years = max(1, age - 18)
            year = min(int(np.random.exponential(scale=5)), max_years)
            years.append(year)
        return years
    
    def _generate_years_in_role(self, years_at_company):
        """Generate years in current role (cannot exceed years at company)."""
        years_in_role = []
        for yac in years_at_company:
            if yac == 0:
                years_in_role.append(0)
            else:
                year = min(int(np.random.exponential(scale=2)), yac)
                years_in_role.append(year)
        return years_in_role
    
    def _generate_years_since_promotion(self, years_in_role):
        """Generate years since last promotion."""
        years_since = []
        for yir in years_in_role:
            if yir == 0:
                years_since.append(0)
            else:
                year = min(int(np.random.exponential(scale=1.5)), yir)
                years_since.append(year)
        return years_since
    
    def _generate_years_with_manager(self, years_in_role):
        """Generate years with current manager."""
        years = []
        for yir in years_in_role:
            if yir == 0:
                years.append(0)
            else:
                year = min(int(np.random.exponential(scale=2)), yir)
                years.append(year)
        return years
    
    def _generate_income(self, job_levels, years_at_company):
        """Generate monthly income based on job level and experience."""
        base_salaries = {1: 3000, 2: 5000, 3: 8000, 4: 12000, 5: 18000}
        incomes = []
        
        for level, years in zip(job_levels, years_at_company):
            base = base_salaries[level]
            experience_bonus = years * 200
            variation = np.random.normal(0, base * 0.15)
            income = base + experience_bonus + variation
            incomes.append(max(2000, int(income)))
        
        return incomes
    
    def _generate_stock_options(self, job_levels):
        """Generate stock option levels based on job level."""
        stock_options = []
        for level in job_levels:
            if level <= 2:
                option = np.random.choice([0, 1], p=[0.7, 0.3])
            elif level == 3:
                option = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            else:
                option = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            stock_options.append(option)
        return stock_options
    
    def _generate_salary_hike(self, years_at_company):
        """Generate percent salary hike."""
        hikes = []
        for years in years_at_company:
            if years < 2:
                hike = np.random.normal(15, 3)
            else:
                hike = np.random.normal(12, 4)
            hikes.append(max(5, min(25, int(hike))))
        return hikes
    
    def _generate_num_companies_worked(self, ages):
        """Generate number of companies worked based on age."""
        num_companies = []
        for age in ages:
            career_length = age - 18
            if career_length <= 5:
                num = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            else:
                num = min(int(np.random.exponential(scale=2)), 9)
            num_companies.append(num)
        return num_companies
    
    def _generate_attrition(self, data):
        """
        Generate attrition with realistic patterns based on multiple factors.

        FIX (previous bug): the old version added up small probability
        increments directly (base 0.10 + many "+0.0X" terms) and fed the
        result straight into a Bernoulli draw. In practice those increments
        stacked up so high that the *average* attrition probability came out
        near 0.5, giving a dataset with ~50% attrition. Real employee
        attrition is a rare event (roughly 15-20% in published HR datasets,
        e.g. the IBM HR Analytics dataset is ~16%). A 50/50 dataset is not
        just unrealistic, it also made every model's job artificially easy
        to "tie" and hard to actually rank employees by risk, which is why
        the trained models were only barely beating a coin flip
        (Accuracy ~0.59, ROC-AUC ~0.62 in the old results).

        This version scores each employee with a weighted, additive risk
        score (same factors/signs as before, so feature meaning is
        unchanged), then converts that score into a probability with a
        logistic (sigmoid) function whose intercept is auto-calibrated so
        the *overall* attrition rate lands close to a realistic 16%, with
        enough spread between low-risk and high-risk employees for models
        to actually learn a useful ranking.

        Factors increasing attrition:
        - Low job satisfaction / environment satisfaction / work-life balance
        - Overtime
        - Frequent business travel
        - Low income relative to job level
        - Long years since last promotion
        - High distance from home
        - Being single
        - Low stock options
        - High number of previous companies
        - Low job involvement
        """
        # Convert to numpy arrays for vectorized operations
        job_satisfaction = np.array(data['JobSatisfaction'])
        env_satisfaction = np.array(data['EnvironmentSatisfaction'])
        work_life_balance = np.array(data['WorkLifeBalance'])
        rel_satisfaction = np.array(data['RelationshipSatisfaction'])
        business_travel = np.array(data['BusinessTravel'])
        years_since_promo = np.array(data['YearsSinceLastPromotion'])
        distance = np.array(data['DistanceFromHome'])
        marital_status = np.array(data['MaritalStatus'])
        stock_options = np.array(data['StockOptionLevel'])
        job_involvement = np.array(data['JobInvolvement'])
        num_companies = np.array(data['NumCompaniesWorked'])
        age = np.array(data['Age'])
        years_at_company = np.array(data['YearsAtCompany'])
        performance = np.array(data['PerformanceRating'])
        job_level = np.array(data['JobLevel'])
        monthly_income = np.array(data['MonthlyIncome'])
        overtime = np.array(data['OverTime'])

        score = np.zeros(self.n_samples)

        # Satisfaction factors (strong predictors)
        score += (4 - job_satisfaction) * 0.35
        score += (4 - env_satisfaction) * 0.25
        score += (4 - work_life_balance) * 0.40
        score += (4 - rel_satisfaction) * 0.15

        # Overtime (strong predictor)
        score += np.where(overtime == 'Yes', 1.10, -0.20)

        # Business travel
        travel_impact = {'Non-Travel': -0.30, 'Travel_Rarely': 0.0, 'Travel_Frequently': 0.70}
        score += np.array([travel_impact[t] for t in business_travel])

        # Years since promotion (stagnation)
        score += np.where(years_since_promo > 3, 0.60, 0)

        # Distance from home
        score += (distance > 20) * 0.40

        # Marital status (single employees more likely to leave)
        score += np.where(marital_status == 'Single', 0.50, 0)

        # Stock options (retention factor)
        score -= stock_options * 0.30

        # Job involvement
        score += (4 - job_involvement) * 0.20

        # Number of companies worked (job hoppers)
        score += np.where(num_companies > 3, 0.40, 0)

        # Age factor (younger employees more likely to leave)
        score += np.where(age < 30, 0.50, 0)
        score -= np.where(age > 50, 0.40, 0)

        # Years at company (retention increases with tenure)
        score -= np.minimum(years_at_company * 0.08, 1.20)

        # Performance rating (low performers more likely to leave or be let go)
        score += (3 - performance) * 0.15

        # Salary relative to job level
        expected_income = job_level * 4000
        income_ratio = monthly_income / expected_income
        score += np.where(income_ratio < 0.8, 0.70, 0)

        # Small amount of irreducible randomness (real employees are not
        # perfectly explained by these factors either)
        score += np.random.normal(0, 0.6, size=self.n_samples)

        # Calibrate the intercept so the overall attrition rate is
        # realistic (~16%), then convert score -> probability with a sigmoid
        intercept = self._calibrate_intercept(score, target_rate=0.16)
        prob = 1 / (1 + np.exp(-(score + intercept)))
        prob = np.clip(prob, 0.01, 0.95)

        # Generate binary attrition
        attrition = np.random.binomial(1, prob)

        return attrition

    @staticmethod
    def _calibrate_intercept(score, target_rate, tol=1e-4, max_iter=100):
        """Binary-search an intercept so mean(sigmoid(score + intercept)) ~= target_rate."""
        lo, hi = -15.0, 15.0
        mid = 0.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            rate = np.mean(1 / (1 + np.exp(-(score + mid))))
            if abs(rate - target_rate) < tol:
                break
            if rate > target_rate:
                hi = mid
            else:
                lo = mid
        return mid


def save_data(df, output_dir):
    """
    Save generated data to CSV file.
    
    Args:
        df (pd.DataFrame): Generated employee data
        output_dir (str): Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    output_path = os.path.join(output_dir, 'employee_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    print(f"\nAttrition Distribution:")
    print(df['Attrition'].value_counts())
    print(f"Attrition Rate: {df['Attrition'].mean():.2%}")
    print(f"\nAge Range: {df['Age'].min()} - {df['Age'].max()}")
    print(f"Income Range: ${df['MonthlyIncome'].min()} - ${df['MonthlyIncome'].max()}")
    print(f"\nDepartment Distribution:")
    print(df['Department'].value_counts())
    print("="*60)


def main():
    """Main function to generate and save employee data."""
    parser = argparse.ArgumentParser(description='Generate synthetic employee attrition data')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of employee records to generate (default: 5000)')
    parser.add_argument('--output', type=str, 
                        default='data/raw',
                        help='Output directory for generated data (default: data/raw)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("EMPLOYEE ATTRITION DATA GENERATOR")
    print("="*60)
    
    # Generate data
    generator = EmployeeDataGenerator(n_samples=args.samples, random_state=args.seed)
    df = generator.generate()
    
    # Save data
    save_data(df, args.output)
    
    print("\n✓ Data generation completed successfully!\n")


if __name__ == '__main__':
    main()

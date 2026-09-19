"""
Unit Tests for Data Preprocessing Module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import DataPreprocessor


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_path = os.path.join(cls.temp_dir, 'test_data.csv')
        
        # Create sample test data
        cls.sample_data = pd.DataFrame({
            'EmployeeID': [f'EMP{i:06d}' for i in range(1, 101)],
            'Age': np.random.randint(20, 60, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], 100),
            'Education': np.random.randint(1, 6, 100),
            'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing'], 100),
            'Department': np.random.choice(['Sales', 'R&D', 'HR'], 100),
            'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Manager'], 100),
            'JobLevel': np.random.randint(1, 6, 100),
            'YearsAtCompany': np.random.randint(0, 20, 100),
            'YearsInCurrentRole': np.random.randint(0, 15, 100),
            'YearsSinceLastPromotion': np.random.randint(0, 10, 100),
            'YearsWithCurrManager': np.random.randint(0, 15, 100),
            'MonthlyIncome': np.random.randint(2000, 20000, 100),
            'StockOptionLevel': np.random.randint(0, 4, 100),
            'PercentSalaryHike': np.random.randint(10, 25, 100),
            'OverTime': np.random.choice(['Yes', 'No'], 100),
            'BusinessTravel': np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], 100),
            'DistanceFromHome': np.random.randint(1, 50, 100),
            'EnvironmentSatisfaction': np.random.randint(1, 5, 100),
            'JobSatisfaction': np.random.randint(1, 5, 100),
            'RelationshipSatisfaction': np.random.randint(1, 5, 100),
            'WorkLifeBalance': np.random.randint(1, 5, 100),
            'PerformanceRating': np.random.randint(1, 5, 100),
            'TrainingTimesLastYear': np.random.randint(0, 7, 100),
            'JobInvolvement': np.random.randint(1, 5, 100),
            'NumCompaniesWorked': np.random.randint(0, 10, 100),
            'Attrition': np.random.choice([0, 1], 100)
        })
        
        cls.sample_data.to_csv(cls.test_data_path, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.output_dir = tempfile.mkdtemp()
        self.preprocessor = DataPreprocessor(
            input_path=self.test_data_path,
            output_dir=self.output_dir,
            test_size=0.2,
            random_state=42
        )
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_load_data(self):
        """Test data loading."""
        self.preprocessor.load_data()
        self.assertIsNotNone(self.preprocessor.df)
        self.assertEqual(len(self.preprocessor.df), 100)
    
    def test_clean_data_removes_employee_id(self):
        """Test that EmployeeID is dropped."""
        self.preprocessor.load_data().clean_data()
        self.assertNotIn('EmployeeID', self.preprocessor.df.columns)
    
    def test_engineer_features_creates_new_features(self):
        """Test that feature engineering creates new columns."""
        self.preprocessor.load_data().clean_data().engineer_features()
        
        new_features = ['AgeGroup', 'IncomeGroup', 'TenureGroup', 
                       'AvgSatisfaction', 'PromotionRate', 'IncomePerLevel',
                       'ExperienceRatio', 'PoorWorkLife', 'CareerStagnation']
        
        for feature in new_features:
            self.assertIn(feature, self.preprocessor.df.columns)
    
    def test_encode_categorical_creates_dummy_variables(self):
        """Test that categorical encoding creates dummy variables."""
        initial_cols = len(self.preprocessor.load_data().df.columns)
        self.preprocessor.clean_data().engineer_features().encode_categorical()
        final_cols = len(self.preprocessor.df.columns)
        
        # Should have more columns after encoding
        self.assertGreater(final_cols, initial_cols)
    
    def test_split_data_creates_train_test_sets(self):
        """Test train-test split."""
        self.preprocessor.load_data().clean_data().engineer_features()
        self.preprocessor.encode_categorical().split_data()
        
        self.assertIsNotNone(self.preprocessor.X_train)
        self.assertIsNotNone(self.preprocessor.X_test)
        self.assertIsNotNone(self.preprocessor.y_train)
        self.assertIsNotNone(self.preprocessor.y_test)
        
        # Check split ratio
        total_samples = len(self.preprocessor.X_train) + len(self.preprocessor.X_test)
        test_ratio = len(self.preprocessor.X_test) / total_samples
        self.assertAlmostEqual(test_ratio, 0.2, places=1)
    
    def test_scale_features(self):
        """Test feature scaling."""
        self.preprocessor.load_data().clean_data().engineer_features()
        self.preprocessor.encode_categorical().split_data().scale_features()
        
        self.assertIsNotNone(self.preprocessor.X_train_scaled)
        self.assertIsNotNone(self.preprocessor.X_test_scaled)
    
    def test_save_data_creates_files(self):
        """Test that preprocessing saves output files."""
        self.preprocessor.process()
        
        train_path = os.path.join(self.output_dir, 'train_data.csv')
        test_path = os.path.join(self.output_dir, 'test_data.csv')
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))
        self.assertTrue(os.path.exists(scaler_path))
    
    def test_no_data_leakage(self):
        """Test that there's no data leakage between train and test sets."""
        self.preprocessor.load_data().clean_data().engineer_features()
        self.preprocessor.encode_categorical().split_data()
        
        train_indices = set(self.preprocessor.X_train.index)
        test_indices = set(self.preprocessor.X_test.index)
        
        # No overlap between train and test indices
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)


if __name__ == '__main__':
    unittest.main()

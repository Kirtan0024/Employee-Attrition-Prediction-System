"""
Data Preprocessing Pipeline for Employee Attrition Prediction

This script handles:
- Loading raw data
- Data cleaning and validation
- Feature engineering
- Encoding categorical variables
- Feature scaling
- Train-test split
- Saving processed data
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataPreprocessor:
    """Preprocess employee attrition data for machine learning."""
    
    def __init__(self, input_path, output_dir='data/processed', test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            input_path (str): Path to raw data CSV file
            output_dir (str): Directory to save processed data
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load raw data from CSV file."""
        print(f"Loading data from: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        print(f"✓ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self
    
    def explore_data(self):
        """Display basic information about the dataset."""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumn Types:\n{self.df.dtypes.value_counts()}")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠ Missing Values:\n{missing[missing > 0]}")
        else:
            print(f"\n✓ No missing values found")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"✓ Duplicate records: {duplicates}")
        
        # Target variable distribution
        print(f"\n✓ Target Variable (Attrition):")
        print(self.df['Attrition'].value_counts())
        print(f"  Attrition Rate: {self.df['Attrition'].mean():.2%}")
        
        return self
    
    def clean_data(self):
        """Clean and validate data."""
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        initial_rows = len(self.df)
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        print(f"✓ Removed {initial_rows - len(self.df)} duplicate records")
        
        # Drop EmployeeID (not useful for prediction)
        if 'EmployeeID' in self.df.columns:
            self.df = self.df.drop('EmployeeID', axis=1)
            print(f"✓ Dropped EmployeeID column")
        
        # Handle missing values (if any)
        if self.df.isnull().sum().sum() > 0:
            print(f"✓ Handling missing values...")
            # Fill numeric columns with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print(f"✓ Final dataset: {len(self.df)} records")
        
        return self
    
    def engineer_features(self):
        """Create new features from existing ones."""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Age groups
        self.df['AgeGroup'] = pd.cut(self.df['Age'], 
                                      bins=[0, 25, 35, 45, 100], 
                                      labels=['18-25', '26-35', '36-45', '46+'])
        print("✓ Created AgeGroup feature")
        
        # Income groups
        self.df['IncomeGroup'] = pd.cut(self.df['MonthlyIncome'], 
                                         bins=[0, 5000, 10000, 15000, 100000], 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
        print("✓ Created IncomeGroup feature")
        
        # Tenure groups
        self.df['TenureGroup'] = pd.cut(self.df['YearsAtCompany'], 
                                         bins=[-1, 2, 5, 10, 100], 
                                         labels=['0-2', '3-5', '6-10', '10+'])
        print("✓ Created TenureGroup feature")
        
        # Satisfaction score (average of all satisfaction metrics)
        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                            'RelationshipSatisfaction', 'WorkLifeBalance']
        self.df['AvgSatisfaction'] = self.df[satisfaction_cols].mean(axis=1)
        print("✓ Created AvgSatisfaction feature")
        
        # Promotion rate (years at company / years since last promotion)
        self.df['PromotionRate'] = self.df['YearsAtCompany'] / (self.df['YearsSinceLastPromotion'] + 1)
        print("✓ Created PromotionRate feature")
        
        # Income to job level ratio
        self.df['IncomePerLevel'] = self.df['MonthlyIncome'] / self.df['JobLevel']
        print("✓ Created IncomePerLevel feature")
        
        # Experience to age ratio
        self.df['ExperienceRatio'] = self.df['YearsAtCompany'] / self.df['Age']
        print("✓ Created ExperienceRatio feature")
        
        # Work-life balance indicator
        self.df['PoorWorkLife'] = ((self.df['OverTime'] == 'Yes') & 
                                    (self.df['WorkLifeBalance'] <= 2)).astype(int)
        print("✓ Created PoorWorkLife feature")
        
        # Career stagnation indicator
        self.df['CareerStagnation'] = (self.df['YearsSinceLastPromotion'] >= 4).astype(int)
        print("✓ Created CareerStagnation feature")
        
        print(f"\n✓ Total features after engineering: {len(self.df.columns)}")
        
        return self
    
    def encode_categorical(self):
        """Encode categorical variables."""
        print("\n" + "="*60)
        print("ENCODING CATEGORICAL VARIABLES")
        print("="*60)
        
        # Binary encoding for binary variables
        binary_mappings = {
            'Gender': {'Male': 1, 'Female': 0},
            'OverTime': {'Yes': 1, 'No': 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
                print(f"✓ Binary encoded: {col}")
        
        # One-hot encoding for nominal variables with few categories
        nominal_cols = ['Department', 'JobRole', 'EducationField', 'MaritalStatus', 
                       'BusinessTravel', 'AgeGroup', 'IncomeGroup', 'TenureGroup']
        
        existing_nominal = [col for col in nominal_cols if col in self.df.columns]
        
        if existing_nominal:
            self.df = pd.get_dummies(self.df, columns=existing_nominal, drop_first=True)
            print(f"✓ One-hot encoded {len(existing_nominal)} categorical features")
        
        print(f"\n✓ Total features after encoding: {len(self.df.columns)}")
        
        return self
    
    def split_data(self):
        """Split data into training and testing sets."""
        print("\n" + "="*60)
        print("SPLITTING DATA")
        print("="*60)
        
        # Separate features and target
        X = self.df.drop('Attrition', axis=1)
        y = self.df['Attrition']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Training set: {len(self.X_train)} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"✓ Test set: {len(self.X_test)} samples ({self.test_size*100:.0f}%)")
        print(f"\n✓ Training set attrition rate: {self.y_train.mean():.2%}")
        print(f"✓ Test set attrition rate: {self.y_test.mean():.2%}")
        
        return self
    
    def scale_features(self):
        """Scale numerical features."""
        print("\n" + "="*60)
        print("SCALING FEATURES")
        print("="*60)
        
        # Fit scaler on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"✓ Scaled {len(self.X_train.columns)} features using StandardScaler")
        
        return self
    
    def save_data(self):
        """Save processed data and preprocessing objects."""
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save train and test sets
        train_data = pd.concat([self.X_train_scaled, self.y_train], axis=1)
        test_data = pd.concat([self.X_test_scaled, self.y_test], axis=1)
        
        train_path = os.path.join(self.output_dir, 'train_data.csv')
        test_path = os.path.join(self.output_dir, 'test_data.csv')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"✓ Saved training data: {train_path}")
        print(f"✓ Saved test data: {test_path}")
        
        # Save feature names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.X_train.columns.tolist()))
        print(f"✓ Saved feature names: {feature_names_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Saved scaler: {scaler_path}")
        
        # Save preprocessing summary
        summary_path = os.path.join(self.output_dir, 'preprocessing_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("PREPROCESSING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Input file: {self.input_path}\n")
            f.write(f"Total records: {len(self.df)}\n")
            f.write(f"Total features: {len(self.X_train.columns)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(f"Test size: {self.test_size*100:.0f}%\n")
            f.write(f"Random state: {self.random_state}\n")
            f.write(f"\nFeature list:\n")
            for i, feature in enumerate(self.X_train.columns, 1):
                f.write(f"{i}. {feature}\n")
        
        print(f"✓ Saved summary: {summary_path}")
        
        return self
    
    def process(self):
        """Run the complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("EMPLOYEE ATTRITION DATA PREPROCESSING")
        print("="*60)
        
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.engineer_features()
        self.encode_categorical()
        self.split_data()
        self.scale_features()
        self.save_data()
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nProcessed data saved to: {self.output_dir}")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Total features: {len(self.X_train.columns)}")
        print("\n")


def main():
    """Main function to run preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess employee attrition data')
    parser.add_argument('--input', type=str, 
                        default='data/raw/employee_data.csv',
                        help='Path to raw data CSV file')
    parser.add_argument('--output', type=str, 
                        default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = DataPreprocessor(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    preprocessor.process()


if __name__ == '__main__':
    main()

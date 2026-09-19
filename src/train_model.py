"""
Model Training Pipeline for Employee Attrition Prediction

This script handles:
- Loading preprocessed data
- Training multiple ML models
- Handling class imbalance with SMOTE
- Hyperparameter tuning
- Cross-validation
- Model saving and evaluation
"""

import pandas as pd
import numpy as np
import os
import argparse
import joblib
import json
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and evaluate multiple ML models for attrition prediction."""
    
    def __init__(self, data_dir='data/processed', models_dir='models', random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            data_dir (str): Directory containing preprocessed data
            models_dir (str): Directory to save trained models
            random_state (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
    def load_data(self):
        """Load preprocessed training and test data."""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        train_path = os.path.join(self.data_dir, 'train_data.csv')
        test_path = os.path.join(self.data_dir, 'test_data.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        self.X_train = train_df.drop('Attrition', axis=1)
        self.y_train = train_df['Attrition']
        self.X_test = test_df.drop('Attrition', axis=1)
        self.y_test = test_df['Attrition']
        
        print(f"✓ Training set: {len(self.X_train)} samples, {len(self.X_train.columns)} features")
        print(f"✓ Test set: {len(self.X_test)} samples")
        print(f"✓ Class distribution in training set:")
        print(f"  - Class 0 (Stayed): {(self.y_train == 0).sum()} ({(self.y_train == 0).mean()*100:.2f}%)")
        print(f"  - Class 1 (Left): {(self.y_train == 1).sum()} ({(self.y_train == 1).mean()*100:.2f}%)")
        
        return self
    
    def apply_smote(self):
        """Apply SMOTE to balance the training data."""
        print("\n" + "="*60)
        print("APPLYING SMOTE FOR CLASS BALANCING")
        print("="*60)
        
        print(f"Before SMOTE:")
        print(f"  - Class 0: {(self.y_train == 0).sum()}")
        print(f"  - Class 1: {(self.y_train == 1).sum()}")
        
        smote = SMOTE(random_state=self.random_state)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"\nAfter SMOTE:")
        print(f"  - Class 0: {(self.y_train_balanced == 0).sum()}")
        print(f"  - Class 1: {(self.y_train_balanced == 1).sum()}")
        print(f"✓ Training set balanced: {len(self.X_train_balanced)} total samples")
        
        return self
    
    def define_models(self):
        """Define the models to train."""
        print("\n" + "="*60)
        print("DEFINING MODELS")
        print("="*60)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'SVM': SVC(
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            )
        }
        
        print(f"✓ Defined {len(self.models)} models:")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        
        return self
    
    def train_models(self, use_smote=True):
        """Train all models."""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Choose training data
        if use_smote:
            X_train = self.X_train_balanced
            y_train = self.y_train_balanced
            print("Using SMOTE-balanced data for training")
        else:
            X_train = self.X_train
            y_train = self.y_train
            print("Using original imbalanced data for training")
        
        self.trained_models = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")
            
            try:
                # Train model
                print(f"Fitting {name}...")
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Cross-validation on training data
                print(f"Running 5-fold cross-validation...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                           scoring='roc_auc', n_jobs=-1)
                
                print(f"✓ {name} trained successfully")
                print(f"  Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
                continue
        
        print(f"\n✓ Successfully trained {len(self.trained_models)} models")
        return self
    
    def evaluate_models(self):
        """Evaluate all trained models on test set."""
        print("\n" + "="*60)
        print("EVALUATING MODELS ON TEST SET")
        print("="*60)
        
        self.results = {}
        
        for name, model in self.trained_models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print metrics
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
            print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
        
        return self
    
    def print_comparison(self):
        """Print comparison of all models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        best_roc_auc = self.results[best_model_name]['roc_auc']
        
        print(f"\n{'='*60}")
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   ROC-AUC Score: {best_roc_auc:.4f}")
        print(f"{'='*60}")
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        return self
    
    def save_models(self):
        """Save all trained models and results."""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        for name, model in self.trained_models.items():
            model_filename = f"{name.replace(' ', '_').lower()}_{timestamp}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"✓ Saved: {model_filename}")
        
        # Save best model separately
        best_model_path = os.path.join(self.models_dir, 'best_model.pkl')
        joblib.dump(self.best_model, best_model_path)
        print(f"✓ Saved best model: best_model.pkl ({self.best_model_name})")
        
        # Save results
        results_summary = {}
        for name, metrics in self.results.items():
            results_summary[name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc'])
            }
        
        results_summary['best_model'] = self.best_model_name
        results_summary['timestamp'] = timestamp
        
        results_path = os.path.join(self.models_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"✓ Saved results: training_results.json")
        
        # Save feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.X_train.columns.tolist()))
        print(f"✓ Saved feature names: feature_names.txt")
        
        print(f"\n✓ All models and results saved to: {self.models_dir}")
        
        return self
    
    def train_pipeline(self, use_smote=True):
        """Run the complete training pipeline."""
        print("\n" + "="*60)
        print("EMPLOYEE ATTRITION MODEL TRAINING PIPELINE")
        print("="*60)
        
        self.load_data()
        
        if use_smote:
            self.apply_smote()
        
        self.define_models()
        self.train_models(use_smote=use_smote)
        self.evaluate_models()
        self.print_comparison()
        self.save_models()
        
        print("\n" + "="*60)
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nBest Model: {self.best_model_name}")
        print(f"ROC-AUC Score: {self.results[self.best_model_name]['roc_auc']:.4f}")
        print(f"Models saved to: {self.models_dir}")
        print("\n")


def main():
    """Main function to run model training."""
    parser = argparse.ArgumentParser(description='Train employee attrition prediction models')
    parser.add_argument('--data-dir', type=str, 
                        default='data/processed',
                        help='Directory containing preprocessed data')
    parser.add_argument('--models-dir', type=str, 
                        default='models',
                        help='Directory to save trained models')
    parser.add_argument('--no-smote', action='store_true',
                        help='Disable SMOTE (use imbalanced data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run training
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        random_state=args.seed
    )
    
    trainer.train_pipeline(use_smote=not args.no_smote)


if __name__ == '__main__':
    main()

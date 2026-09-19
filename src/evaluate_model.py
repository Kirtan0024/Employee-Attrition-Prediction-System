"""
Model Evaluation Script for Employee Attrition Prediction

This script handles:
- Loading trained models
- Detailed evaluation metrics
- Confusion matrix visualization
- ROC curves
- Feature importance analysis
- Classification reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import argparse

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate trained models with comprehensive metrics and visualizations."""
    
    def __init__(self, models_dir='models', data_dir='data/processed', output_dir='results'):
        """
        Initialize the evaluator.
        
        Args:
            models_dir (str): Directory containing trained models
            data_dir (str): Directory containing test data
            output_dir (str): Directory to save evaluation results
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_test_data(self):
        """Load test data."""
        print("="*60)
        print("LOADING TEST DATA")
        print("="*60)
        
        test_path = os.path.join(self.data_dir, 'test_data.csv')
        test_df = pd.read_csv(test_path)
        
        self.X_test = test_df.drop('Attrition', axis=1)
        self.y_test = test_df['Attrition']
        
        print(f"✓ Test set loaded: {len(self.X_test)} samples")
        print(f"✓ Features: {len(self.X_test.columns)}")
        print(f"✓ Class distribution: {self.y_test.value_counts().to_dict()}")
        
        return self
    
    def load_best_model(self):
        """Load the best trained model."""
        print("\n" + "="*60)
        print("LOADING BEST MODEL")
        print("="*60)
        
        model_path = os.path.join(self.models_dir, 'best_model.pkl')
        self.model = joblib.load(model_path)
        
        # Load training results
        results_path = os.path.join(self.models_dir, 'training_results.json')
        with open(results_path, 'r') as f:
            self.training_results = json.load(f)
        
        self.best_model_name = self.training_results['best_model']
        
        print(f"✓ Loaded model: {self.best_model_name}")
        print(f"✓ Training ROC-AUC: {self.training_results[self.best_model_name]['roc_auc']:.4f}")
        
        return self
    
    def make_predictions(self):
        """Make predictions on test set."""
        print("\n" + "="*60)
        print("MAKING PREDICTIONS")
        print("="*60)
        
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print(f"✓ Predictions generated for {len(self.y_pred)} samples")
        
        return self
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1_score': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        print(f"\n✓ Model: {self.best_model_name}")
        print(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall:    {self.metrics['recall']:.4f}")
        print(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        return self
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Stayed', 'Left'],
                   yticklabels=['Stayed', 'Left'])
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')
        
        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved: {output_path}")
        
        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  True Negatives (TN):  {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP):  {tp}")
        
        return self
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        print("\n" + "="*60)
        print("ROC CURVE")
        print("="*60)
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', linewidth=2,
                label=f'ROC Curve (AUC = {self.metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'roc_curve.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curve saved: {output_path}")
        
        return self
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for tree-based models."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        # Check if model has feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.X_test.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}',
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            output_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Feature importance plot saved: {output_path}")
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, 'feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            print(f"✓ Feature importance data saved: {csv_path}")
            
            # Print top 10
            print(f"\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:40s}: {row['importance']:.4f}")
        
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coefficients = np.abs(self.model.coef_[0])
            feature_names = self.X_test.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Absolute Coefficient Value')
            plt.title(f'Top {top_n} Feature Coefficients - {self.best_model_name}',
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            output_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Feature importance plot saved: {output_path}")
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, 'feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            print(f"✓ Feature importance data saved: {csv_path}")
            
            # Print top 10
            print(f"\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:40s}: {row['importance']:.4f}")
        else:
            print("✗ Model does not support feature importance")
        
        return self
    
    def generate_classification_report(self):
        """Generate detailed classification report."""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(self.y_test, self.y_pred,
                                      target_names=['Stayed', 'Left'],
                                      digits=4)
        print(report)
        
        # Save to file
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {self.best_model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f"\n✓ Classification report saved: {report_path}")
        
        return self
    
    def save_evaluation_summary(self):
        """Save comprehensive evaluation summary."""
        print("\n" + "="*60)
        print("SAVING EVALUATION SUMMARY")
        print("="*60)
        
        summary = {
            'model_name': self.best_model_name,
            'test_samples': len(self.y_test),
            'metrics': self.metrics,
            'confusion_matrix': confusion_matrix(self.y_test, self.y_pred).tolist(),
            'class_distribution': self.y_test.value_counts().to_dict()
        }
        
        summary_path = os.path.join(self.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"✓ Evaluation summary saved: {summary_path}")
        
        return self
    
    def evaluate(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*60)
        print("MODEL EVALUATION PIPELINE")
        print("="*60)
        
        self.load_test_data()
        self.load_best_model()
        self.make_predictions()
        self.calculate_metrics()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_feature_importance()
        self.generate_classification_report()
        self.save_evaluation_summary()
        
        print("\n" + "="*60)
        print("✓ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Test Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Test ROC-AUC: {self.metrics['roc_auc']:.4f}")
        print("\n")


def main():
    """Main function to run model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate employee attrition prediction model')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()

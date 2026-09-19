# Model Card: Employee Attrition Prediction

## Purpose
This model predicts whether an employee is likely to leave the organization based on HR and work-related features.

## Data
The project uses synthetic employee records generated for training and evaluation. This makes the dataset suitable for experimentation and demonstration, but it is not a substitute for real HR business data.

## Model
Current best-performing model: Logistic Regression

## Performance
Measured on the project evaluation dataset:
- Accuracy: 0.59
- Precision: 0.5905
- Recall: 0.5928
- F1: 0.5916
- ROC-AUC: 0.6258

## Known limitations
- Synthetic data may not reflect real-world HR patterns.
- Model performance is moderate and should not be used as a sole decision-maker in employee retention actions.
- This should be treated as a decision-support tool, not an autonomous HR policy system.

## Recommended use
- Use for exploratory analysis and internal retention insights.
- Combine with HR review, manager feedback, and employee surveys.
- Validate on real organizational data before operational deployment.

Measles Outbreak Detection Model
Overview

This project builds a classification model to identify U.S. counties at higher risk of measles outbreaks using vaccination and demographic data. The goal is to understand what factors are most associated with outbreak risk and see how well we can predict it.

Data

Data was pulled from a few public sources:

CDC vaccination coverage data
U.S. Census county-level population data
Johns Hopkins measles case data

All datasets were merged using county FIPS codes.

Approach
Data Preparation
Cleaned and standardized columns across datasets
Converted percentage fields to numeric
Merged datasets at the county level
Handled missing values with median imputation
Feature Engineering

Some of the main features used:

vaccination rates
vaccine hesitancy estimates
population size
age distribution
Modeling

Tested a few classification models:

Logistic Regression
Random Forest
XGBoost
Class Imbalance

Outbreaks are rare, so the dataset is very imbalanced.

To deal with this:

used class weighting
focused on recall and ROC-AUC instead of accuracy
adjusted decision thresholds
Results
ROC-AUC ≈ 0.75 on validation data
Models were able to pick up signal beyond random guessing
Increasing recall helped capture more potential outbreak counties, at the cost of more false positives
Takeaways
Vaccination rates and hesitancy were strong predictors
Population size also played a role
Model performance is limited by how few outbreak cases exist
Limitations
Very small number of positive cases
Public data may be incomplete or noisy
Results depend heavily on feature quality
Tech Stack
Python (pandas, numpy, scikit-learn, xgboost)

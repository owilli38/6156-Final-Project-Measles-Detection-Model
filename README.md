# 6156-Final-Project-Measles-Detection-Model
Using John Hopkins measles cases, US census, and CDC datasets, created model that detects for high-risk counties for proactive public health outreach and interventions to protect loss of life and public health funds.

Model Risk Management Documentation

Summary: This document describes the Model Risk Management (MRM) framework for the Proactive Measles Detector, a supervised machine learning model designed to identify U.S. counties at risk of measles outbreaks. The model leverages demographic, vaccination, and behavioral data to generate county-level risk scores and classifications. Given its potential influence on public health decision-making and resource allocation, the model is classified as high-risk and is subject to rigorous validation, monitoring, and governance standards. The framework prioritizes recall to minimize missed outbreaks and supports explainable, actionable insights for public health stakeholders. 
1. Model Overview
1.1 Model Name: Proactive Measles Detector 
1.2 Model ID / Version: M_Detector_v1
1.3 Date Created / Last Updated: 
Created on March 30th, 2026
Last updated on April 20th, 2026
1.4 Model Owner: Owen Williamson
1.5 Model Developer(s): Owen Williamson
1.6 Model Reviewer(s): Owen Williamson
1.7 Model Users / Stakeholders: 
	County officials
	State health departments 
1.8 Purpose & Business Use Case
This model identifies counties currently experiencing measles outbreaks and, more importantly, those at elevated risk of future outbreaks based on demographic, vaccination, and behavioral characteristics.
The primary objective is proactive intervention. By identifying high-risk counties before outbreaks occur, public health officials can deploy targeted strategies such as vaccination campaigns (MMR), localized awareness initiatives, and resource allocation.

Fortunately, there have been no reported deaths due to measles outbreaks in the United States. However, measles outbreaks can impose significant financial burdens on tax payers and health departments. According to Johns Hopkins estimates, the initial case in an outbreak can cost approximately $250,000, with additional cases costing around $15,000 each. Early detection and intervention therefore provide both public health and economic benefits.
This model supports decision-making for:
 		- County-level intervention prioritization
- Resource allocation planning
- Preventative healthcare strategy development
1.9 Model Category / Type
Primary Models:
Supervised classification models: Logistic Regression, Balanced Random Forest, and XGBoost.
Logistic Regression is used as a transparent baseline model
Balanced Random Forest is used to capture nonlinear relationships while addressing class imbalance.
XGBoost is the primary predictive model due to its strong performance and ability to model complex interactions.
The final modeling framework focuses only on supervised classification for county-level measles outbreak risk detection and transmission risk assessment.

1.10 Criticality / Risk Tiering
High: Rationale
Direct public health impact (potential loss of life)
Influences allocation of limited healthcare and public health resources
Risk of false negatives (missed potential outbreaks) has higher consequences
Model outputs may influence government decision making
Therefore, high standards for validation, monitoring and governance are required.

2. Model Scope & Boundaries
2.1 In-Scope Components
Describe components included in the model boundary:
Data pipeline
Integration of CDC, US Census and John Hopkins datasets
Feature engineering
Missing value imputation at the county/state level 
One-hot encoding for categorical variables
Correlation-based feature reduction
Model-based feature selection
Features created that mix two variables together to get interaction effects and greater interpretation from the models.
Algorithm(s)
Logistic regression (baseline, interpretable benchmark)
Random forest (nonlinear relationship with imbalance handling)
XGBoost (primary predictive model)
Post‑processing
Probability threshold tuning for classification
Comparison of alternative outbreak target definitions:
Target_At_Least_1: counties with one or more cases
Target_At_Least_2: counties with two or more cases
This indicates local transmission
SHAP-based global and local explainability
Identification of high-risk counties without current reported outbreaks
Monitoring dashboards
Streamlit dashboard for visualization and stakeholder use 
2.2 Out-of-Scope Components
Document excluded systems or processes.
Individual-level prediction of measles cases
Real-time outbreak tracking beyond available data refresh frequency
Vaccine distribution logistics optimization
Clinical diagnosis or medical decision-making
International measles outbreak prediction outside the United States
Integration with electronic health record (EHR) systems

2.3 Dependencies & Interfaces
Upstream data sources
Vaccine Hesitancy - US Census
Measles Cases by County - John Hopkins
CDC MMR Vaccine Coverage - CDC
County Demographics - US Census
Downstream consumers
State and county health officials via dashboard 
APIs, databases, platforms
Positron for Python coding 
Streamlit for dashboard 

3. Data Documentation
3.1 Data Sources
List each source with:
Description
Owner
Refresh frequency
Access controls
John Hopkins Measles Cases by County Tracker
Owner: John Hopkins, Refreshed ~ Once Weekly, Access Available via Github Repository
Vaccine Hesitancy for COVID 19
Owner: CDC, Not Regularly Refreshed, Available via CDC Gov page
US Census County Characteristics
Owner: US Census, Updated Every 5 Years, Available via US Census gov page 
CDC MMR Vaccine Coverage 
Owner: CDC, Estimated Every School Year, Available via data.cdc.gov

3.2 Data Quality Assessment
Completeness
County-level coverage is strong across the merged dataset, but some variables contain missing values, especially vaccination- and hesitancy-related fields. Missingness was most common in state-reported vaccination coverage and selected demographic or vulnerability variables. To preserve geographic structure, missing numeric fields were imputed using state-level medians, with fallback to the global median when needed.
Accuracy
Data sources were obtained from reputable public institutions, including the U.S. Census, CDC, and Johns Hopkins measles tracker. However, accuracy is still limited by the quality of the source systems. Measles case counts may be affected by underreporting, delayed reporting, or inconsistent county attribution. Vaccine hesitancy is estimated rather than directly observed, and COVID-19 vaccine hesitancy is used as a proxy for broader vaccine resistance behavior.
Timeliness
John Hopkins provides close to real-time data and provides a comprehensive count of reported measles cases
Vaccine hesitancy is from 2021 and is used as proxy; it would be more favorable if we had up-to-date metrics.
Overall, the dataset is appropriate for county-level risk modeling, but not for real-time operational outbreak surveillance.
Consistency	
The major consistency issue involved differences in county identifiers and formatting across sources. FIPS codes had to be standardized to a common 5-digit format before merging. State names and geography labels also required alignment across vaccination and census files. Percentage variables appeared in mixed formats and were standardized before modeling.
Known data issues / mitigations
FIPS codes required zero-padding and type conversion before merging.
County and state naming conventions varied across datasets and required standardization.
Percentage fields were stored as strings in some files and converted to numeric decimal form.
Missing numeric values were handled using state-level median imputation.
Target leakage risk from raw case-count fields was mitigated by removing total_measles_cases and related direct identifiers from modeling features.

3.3 Data Preprocessing
Cleaning steps	
Standardized FIPS codes to 5-digit strings across all source datasets.
Removed rows corresponding to state-level aggregates where county-level modeling was required.
Filtered vaccination data to MMR-specific records only.
Selected the most recent available vaccination record by geography.
Cleaned percentage fields by removing percent signs and converting them to numeric values.
Standardized rate features onto a 0–1 scale when appropriate.
Transformations
Converted vaccination and hesitancy percentages into numeric rates.
Created grouped demographic variables such as AGE_18_44, AGE_45_PLUS, and AGE_CHILD.
Created ratio features such as AGE_18_44_RATIO, AGE_45_PLUS_RATIO, and AGE_CHILD_RATIO.
Created engineered interaction terms, including:
unvaccinated_rate
low_vax_flag
hesitancy_vax_interaction
risk_interaction
risk_interaction_2
vax_hesitancy_interaction
not_covid_vaccinated
Filtering rules
Removed direct target-like variables such as total measles case counts from the model input.
Removed identifiers and administrative fields such as FIPS, county names, and state codes from the final modeling matrix.
Removed geometry and map-shape fields.
Removed race, ethnicity, and overly granular demographic proxy fields where they were not necessary for the final model.
Removed highly redundant or overlapping raw age fields after grouped age variables were constructed.
Handling of missing values
Missing numeric values were imputed using state-level medians to preserve geographic structure. If a state-level median could not be computed, the global median was used. After feature engineering, a second pass ensured that remaining numeric missingness and divide-by-zero artifacts were addressed.
Encoding / normalization
Low-cardinality categorical variables were one-hot encoded.
High-cardinality categorical variables were dropped rather than expanded.
Logistic Regression used scaled features.
Tree-based models used unscaled features, though the preprocessing pipeline maintained consistent numeric formatting across all models.
3.4 Training / Testing / Validation Samples
Time ranges: The merged dataset combines the most recent available county demographic estimates, historical vaccination coverage, COVID-19 hesitancy estimates, and current measles case tracking. Because source datasets come from different years, the modeling task is cross-sectional rather than a true time-series forecast.
Sample sizes: The final county-level dataset includes roughly all U.S. counties with available mergeable information. The dataset was split into:
70% training
15% validation
15% test
Exact sample counts vary slightly depending on the target definition and post-filtering preprocessing steps.
Methods for splitting: A stratified random split was used so that positive outbreak counties remained proportionally represented across training, validation, and test samples. This was especially important because both target definitions are imbalanced, and the Target_At_Least_2 formulation is even more sparse than Target_At_Least_1. 
Representativeness analysis: The dataset is geographically broad and covers counties across the United States. However, positive cases are rare, which means the training set contains relatively few counties with outbreaks. The Target_At_Least_1 definition is better for broad detection, while Target_At_Least_2 is more representative of local transmission but creates a harder, more imbalanced learning problem. Class imbalance mitigation was therefore necessary in model design.

4. Methodology & Model Design
4.1 Conceptual Design
A supervised classification framework was used to balance predictive accuracy and interpretability.
Logistic Regression, Balanced Random Forest, and XGBoost were used to estimate the probability of measles outbreak risk at the county level. Two target definitions were evaluated:
Target_At_Least_1: counties with one or more reported measles cases
Target_At_Least_2: counties with two or more reported measles cases
This two-target framework allows the model to distinguish between general exposure risk (Target_At_Least_1) and more meaningful local transmission risk (Target_At_Least_2). The first target is useful for early-warning surveillance, while the second target better reflects sustained outbreak conditions within a county.
The final framework supports proactive intervention by identifying counties with demographic, behavioral, and vaccination-related characteristics associated with elevated measles risk.
4.2 Algorithm Selection Rationale
Multiple supervised models were selected to balance interpretability, predictive performance, and robustness:
Logistic Regression: provides a transparent benchmark and interpretable baseline
Balanced Random Forest: captures nonlinear relationships and interactions while handling class imbalance
XGBoost: selected as the primary model due to strong performance on imbalanced classification tasks, threshold flexibility, and superior explainability when paired with SHAP
KNN and clustering-based approaches were not retained in the final project because they provided less actionable value than the supervised framework and reduced the conceptual clarity of the modeling objective.

4.3 Model Architecture
Include diagrams if applicable.
End-to-End Modeling Pipeline for Measles Risk Detection 
The final modeling architecture is a supervised county-level classification pipeline with shared preprocessing and multiple model classes.
Pipeline stages:
Data ingestion from census, measles, vaccination, and hesitancy sources
Identifier standardization and merging
Numeric cleaning and percentage normalization
State-level median imputation
Feature engineering and grouped demographic construction
Leakage and identifier removal
One-hot encoding of remaining low-cardinality categorical variables
Model-based feature selection using Random Forest importance
Training of benchmark and nonlinear classifiers
Validation-set threshold tuning for XGBoost
Test-set evaluation
SHAP-based explainability on the held-out test set
Final model stack:
Logistic Regression: benchmark and interpretable baseline
Balanced Random Forest: nonlinear benchmark with imbalance handline
XGBoost: final champion model
Two binary target definitions were evaluated:
Target_At_Least_1: counties with one or more measles cases
Target_At_Least_2: counties with two or more measles cases
This allows the architecture to support both early detection and sustained-transmission risk assessment.
4.4 Feature Engineering
List features
unvaccinated_rate
low_vax_flag
not_covid_vaccinated
hesitancy_vax_interaction
risk_interaction
risk_interaction_2
vax_hesitancy_interaction
AGE_18_44_RATIO
AGE_45_PLUS_RATIO
AGE_CHILD_RATIO
Definitions
unvaccinated_rate: 1 - Estimate (%); represents the county-level gap in MMR coverage.
low_vax_flag: binary indicator equal to 1 when MMR vaccination is below 90%; used as a herd-immunity warning signal.
not_covid_vaccinated: 1 - Percent adults fully vaccinated against COVID-19; used as a proxy for broader vaccine resistance or low adult vaccine uptake.
hesitancy_vax_interaction: interaction between estimated hesitancy and MMR unvaccinated rate.
risk_interaction: interaction between child population share and hesitancy, capturing heightened risk when susceptible age groups coexist with resistance to vaccination.
risk_interaction_2: interaction between social vulnerability and hesitancy, capturing access and trust barriers.
vax_hesitancy_interaction: interaction between adult COVID vaccination and hesitancy, intended to reflect generalized vaccine behavior patterns.
AGE_*_RATIO features: grouped age totals normalized by county population to better compare counties of different sizes and reduces redundancy in age brackets.
Importance
These help to add greater context for the model and for interpretability, and add needed information for the models. 
Feature engineering was used to improve both predictive performance and interpretability. 
The grouped age variables made the model easier to explain and reduced redundancy across raw demographic bins. 
Interaction terms were especially useful in tree-based models because they exposed nonlinear relationships between hesitancy, age structure, and vulnerability.
Transformations
Percent-based features were converted to numeric rates.
Highly overlapping raw age columns were collapsed into grouped age measures.
Ratio features were calculated using county population as the denominator.
Infinite values created by division were replaced and re-imputed.
Rationale for inclusion/exclusion: Included features were selected because they reflect plausible outbreak drivers: vaccination coverage, vaccine resistance, age structure, and social vulnerability. Excluded features generally fell into one of the following groups:
direct leakage fields
identifiers
geometry/map fields
very high-cardinality categoricals
redundant raw demographic detail that was replaced by grouped measures
4.5 Model Assumptions
Statistical assumptions
Counties are treated as independent observations.
The supervised classification problem is cross-sectional, not longitudinal.
Nonlinear relationships are expected and are handled by tree-based models.
Class imbalance is substantial and must be explicitly addressed in modeling and evaluation.
Validation data can be used to select classification thresholds, which are then fixed before test evaluation.
Business assumptions
For public health, we would rather over-predict potential counties for an outbreak than to miss a county that has an outbreak. Overall, we want Recall to be higher so that we get positive cases, and allow for false positives in classification models.
Missing a true outbreak is more costly than flagging a false alarm.
Public health users can act on ranked or thresholded county risk predictions.
County-level risk is a meaningful planning unit for intervention prioritization.
Two target definitions represent two practical use cases:
Target_At_Least_1 for early detection
Target_At_Least_2 for sustained transmission risk
Data assumptions
COVID-19 vaccination hesitancy is an acceptable proxy for broader vaccine resistance behavior.
County demographic structure is stable enough to be useful in a cross-sectional model.
The merged data sources, although temporally imperfect, still capture meaningful county-level risk structure.
Reported measles cases are directionally informative even if some underreporting exists.
4.6 Limitations
Technical:
Class imbalance remains a significant challenge, especially for the stricter Target_At_Least_2 definition
Performance depends on the selected threshold, particularly when prioritizing recall versus precision
Some engineered features are correlated and may split importance across related variables
Data:
Potential underreporting of measles cases
Temporal misalignment between datasets
Missing or noisy vaccination data
COVID-19 vaccine hesitancy used as proxy for MMR vaccine hesitancy 
Business:
Model predictions are probabilistic and should not be used as sole decision criteria
Ethical:
Risk of misallocation of resources if predictions are incorrect
Potential bias due to demographic proxies in data

5. Model Performance
5.1 Training Performance Metrics
e.g., accuracy, AUC, RMSE, precision/recall, stability.
Primary performance metric was Recall and ROC-AUC for good detection of outbreaks, while allowing for False Positives to indicate counties with higher risk of potential outbreak of measles.
Primary evaluation metrics included Recall, Precision, F1 Score, and ROC-AUC.
Recall was prioritized due to the high cost of false negatives (missed outbreaks).
Precision was monitored to ensure operational feasibility and avoid excessive false positives. 
F1 Score was used to balance precision-recall tradeoffs.
ROC-AUC was used to evaluate overall model discrimination ability across thresholds.
5.2 Validation Performance
Model performance was evaluated on a holdout validation dataset for both target definitions.
Logistic Regression provided an interpretable benchmark
 Balanced Random Forest provided strong nonlinear performance
XGBoost consistently performed best overall and was selected as the primary model
Two target definitions were compared:
Target_At_Least_1 produced better recall and is more suitable for early-warning detection
Target_At_Least_2 produced higher ROC-AUC and cleaner interpretability, but lower recall, making it more suitable for identifying sustained transmission risk
Special emphasis was placed on recall to minimize false negatives, while threshold tuning was used to manage the tradeoff between sensitivity and precision.
5.3 Backtesting (if relevant)
Method: A traditional backtest was not possible in the strict time-series sense because the final project dataset is assembled from cross-sectional public data sources collected at different times. Instead, out-of-sample validation and held-out test evaluation were used to simulate forward-looking performance.
Results
Both target definitions were evaluated on held-out data:
Target_At_Least_1 produced stronger recall and broader detection
Target_At_Least_2 produced better rank-order separation and cleaner interpretability but lower recall
Thresholds: XGBoost classification thresholds were optimized on the validation set using precision-recall tradeoffs and then fixed for test-set evaluation. This allowed operating points to better reflect public health priorities than the default 0.5 threshold.
5.4 Sensitivity Analysis
Stress tests
The model was stress-tested conceptually by comparing two target definitions:
one or more cases
two or more cases
This revealed meaningful changes in feature importance and operational behavior.
Adversarial tests: adversarial testing was not conducted. However, robustness was assessed by:
comparing multiple model classes
checking whether key explanatory variables remained stable across target definitions
reviewing SHAP results for consistency
Scenario analysis: Scenario analysis showed that the Target_At_Least_1 formulation behaves like an early-warning screen, while the Target_At_Least_2 formulation behaves like a sustained-transmission detector. This supports the use of the two targets as complementary modeling perspectives rather than as interchangeable problem statements. 
5.5 Benchmarking
Alternative models:
Logistic Regression
Balanced Random Forest
XGBoost
Champion/challenger comparisons:
Logistic Regression served as the transparent benchmark
Balanced Random Forest served as the nonlinear challenger
XGBoost was selected as the champion model
The project also benchmarked two target definitions:
Target_At_Least_1 for broad outbreak detection
Target_At_Least_2 for sustained, local transmission detection
The final project therefore benefits from presenting both as complementary views of outbreak risk.
5.6 Actionable Insights 
The supervised modeling framework identified vaccination behavior and demographic structure as the primary drivers of predicted measles risk.
Key actionable findings include:
Counties with higher proportions of unvaccinated individuals are consistently higher risk
Counties with larger working-age populations (18–44) and higher child population ratios show elevated transmission risk; this indicates demographics play a significant role in transmission, especially looking at results for modeling with Target_At_Least_2, which would suggest local transmission within the county. 
Vaccine hesitancy amplifies underlying risk, particularly through interaction effects with vaccination and social vulnerability measures
These outputs support targeted public health strategies such as localized MMR vaccination outreach, education campaigns, and resource prioritization.


6. Explainability & Interpretability
6.1 Global Explainability
Global importance was assessed using:
SHAP summary plots
SHAP bar plots
Across both target definitions, the dominant features were consistently tied to:
age structure
vaccination uptake
adult non-vaccination proxy behavior
vaccine hesitancy
social vulnerability interactions
For Target_At_Least_1, vaccination-related features were especially dominant. For Target_At_Least_2, age-structure variables such as AGE_18_44_RATIO became even more prominent, suggesting that transmission risk is more dependent on population structure than isolated-case occurrence.




Model reasoning
The model learns that counties are higher risk when they combine:
lower vaccination coverage
greater vaccine hesitancy
larger working-age and child population shares
higher social vulnerability in interaction with hesitancy
This pattern is epidemiologically reasonable because measles spread depends on both susceptibility and social contact structure.
Explainability tools (e.g., SHAP, LIME)
SHAP was used as the primary explainability framework because it provides:
global ranking of influential features
directional interpretation of feature effects
local explanations for individual county predictions
6.2 Local Explainability
Example predictions explained

This is for test sample using XGboost for general outbreak detection (Target_At_Least_1)
This shows a case that has been classified as 1, high probability of outbreak, where not_covid_vaccinated and age_18_44_ratio is driving the probability of measles outbreak higher. 

This is for test sample using XGBoost for disease transmission (Target_At_Least_2)
This shows a case that has been classified as 0, not at higher risk for local transmission, where age demographics are driving the probability down 
Edge‑case behavior
The two target definitions revealed useful edge-case differences:
Under Target_At_Least_1, counties could be flagged based on vaccination-risk structure even if they did not resemble sustained transmission environments.
Under Target_At_Least_2, the model required stronger demographic and interaction evidence before classifying a county as positive.
This suggests that local explanations are more conservative and transmission-focused under the stricter target definition.

6.3 Business Interpretability Assessment
Is the model understandable to stakeholders?
Model outputs are interpretable at a high level through:
Risk scores (predicted probabilities)
Binary risk classifications based on optimized thresholds
SHAP-based explanations for both global and county-level predictions
These outputs can be communicated to non-technical stakeholders as:
Higher-risk counties
Lower-risk counties
Counties at risk of sustained transmission 
This enables actionable decision-making without requiring deep technical understanding.

7. Fairness, Bias, and Ethical Assessment
7.1 Protected Attributes Reviewed
The model incorporates county-level demographic and socioeconomic data, which may indirectly reflect protected characteristics. While individual-level protected attributes are not explicitly included, the following proxies were reviewed for potential bias:
Race and ethnicity composition (from U.S. Census data)
Income levels and poverty rates (SVI)
Geographic location (urban vs. rural)
These variables may act as proxies for protected classes and were carefully evaluated to ensure the model does not disproportionately disadvantage specific populations.
7.2 Bias Metrics Evaluated
To assess fairness, model performance was evaluated across different population segments using the following metrics:
Equal Opportunity: Comparison of true positive rates (recall) across groups to ensure equal detection of outbreaks
Predictive Parity: Comparison of precision across groups to evaluate consistency in prediction accuracy
Subgroup review focused on potential proxy dimensions such as:
social vulnerability levels
urban versus rural structure where inferable
demographic composition proxies available at the county level
Because the project uses county-level rather than individual-level data, fairness interpretation is necessarily approximate. The review therefore focused less on formal compliance testing and more on whether the model’s error distribution and reliance on proxy variables suggested meaningful risk of systematic disadvantage.
No evidence of severe disparity was observed across reviewed proxy groups; however, continued monitoring is recommended due to the use of aggregated demographic features. 
7.3 Fairness Mitigation Approaches
Several strategies were implemented to mitigate potential bias:
Pre-processing:
 Careful feature selection to remove redundant or highly correlated variables that may amplify bias
 Handling of missing data to avoid systematic exclusion of certain populations
In-model:
 Use of ensemble methods (Random Forest, XGBoost) to reduce reliance on any single biased feature
Class imbalance handling (e.g., class weights, scale_pos_weight) to ensure minority outbreak cases are properly detected
Post-processing:
Threshold tuning to prioritize recall, ensuring outbreaks are not disproportionately missed in vulnerable populations
SHAP-based review of feature effects to confirm that predictions are driven by meaningful risk factors rather than administrative artifacts

7.4 Ethical Considerations
List potential harms and mitigations.
Potential Harms:
Misclassification Risk: False negatives may result in missed outbreaks, leading to delayed interventions and increased public health risk
Resource Misallocation: False positives may lead to inefficient allocation of limited healthcare resources
Bias Amplification: Use of demographic proxies may unintentionally reinforce existing inequalities in healthcare access
Mitigation Strategies:
Prioritization of recall to minimize missed outbreaks
Use of multiple supervised model classes and alternative target definitions to validate robustness of risk identification
Transparent communication of model limitations to stakeholders
Use of the model as a decision-support tool rather than a sole decision-maker
Ethical Use:
The model is intended to support equitable public health interventions by identifying at-risk populations early, with the goal of improving access to preventative care such as vaccinations and education campaigns.


8. Model Controls & Governance
8.1 Access & Permission Controls
Who can retrain, update, deploy, or view outputs.
Retrain: Model developer (Owen Williamson)
Update: Model developer via controlled updates to preprocessing, feature engineering, and modeling code
Deploy: Model developer through the Streamlit application
View outputs: County officials, state health departments, and approved project stakeholders via dashboard outputs and summary reports

8.2 Change Management
Versioning: First Version 
Code repositories
Github Repository: owilli38/6156-Final-Project-Measles-Detection-Model
Approval requirements: N/a
Versioning follows a controlled update process, where all changes to data, features, or model parameters are tracked through the GitHub repository.
Major updates require re-validation of model performance and threshold calibration prior to deployment. 
8.3 Model Inventory Registration
Model registration details for centralized governance.
Not Applicable to Project 


9. Model Monitoring Strategy
9.1 Performance Monitoring
Metrics tracked:
ROC-AUC
Recall
Precision
F1 score
Positive prediction rate
Thresholds: Performance should be reviewed if recall drops materially below historical validation levels or if precision changes significantly enough to reduce operational usefulness (i.e, more robust education campaigns are needed as county-specific identification becomes too minimal and cases expand exponentially across the county).
Frequency: ~ Weekly, aligned with Johns Hopkins measles data updates
Recall drop >10% from validation baseline
ROC-AUC drop >5%
Feature distribution shift beyond defined statistical thresholds
9.2 Data Drift & Stability Monitoring
Drift metrics:
Changes in feature distributions over time
Changes in vaccination coverage distributions
Changes in predicted positive rates
Changes in class prevalence for both target definitions
Feature distribution changes: Special monitoring should focus on
Not_covid_vaccinated
Estimated hesitant
AGE_18_44_RATIO
AGE_CHILD_RATIO
Social Vulnerability Index (SVI)
9.3 Alerts & Escalation Procedures
Alerts should be triggered when:
model recall or ROC-AUC deteriorates materially
data drift is observed in key explanatory variables
missingness materially increases in critical source data
Escalation Procedures:
initial review by model developer
retraining or recalibration if degradation persists
communication of material changes to stakeholders using dashboard outputs
9.4 Periodic Review Requirements
Quarterly / annual review expectations.
Quarterly review of:
model performance
threshold suitability
feature stability
source data quality
Annual review of:
target definition suitability
modeling assumptions
feature engineering relevance
explainability outputs



10. Validation Summary - NOT NECESSARY FOR PROJECT
(Completed by independent validator or internal MRM function.)
10.1 Validation Scope N/A
10.2 Validation Findings
Strengths N/A
Weaknesses N/A
Remediation items N/A
10.3 Model Risk Rating N/A
10.4 Validation Conclusion / Recommendation N/A

11. Compliance & Regulatory Considerations - NOT NECESSARY PROJECT
Relevant regulatory frameworks N/A
Compliance controls N/A
Data protection considerations (GDPR, DPDP, HIPAA, etc.) N/A

12. Documentation Appendix
Diagrams:
End-to-end data pipeline
Ingest sources → clean and normalize → feature engineering and selection (dropping features that are correlated, contain race, or geographic data) → model training, tuning, and parameter selection → explainability and results 
Threshold selection workflow

Technical specifications:
Final model hyperparameters
For Target_At_Least_1: 
Best XGBoost Params: {'colsample_bytree': np.float64(0.733015577358303), 'learning_rate': np.float64(0.03279351625419417), 'max_delta_step': 2, 'max_depth': 3, 'n_estimators': 315, 'reg_alpha': np.float64(0.352568856334169), 'reg_lambda': np.float64(0.30478125815802903), 'subsample': np.float64(0.7493967559428825)}
Optimal XGBoost Threshold: 0.4599
Best Validation F1 at Optimal Threshold: 0.3239
XGBoost Validation Metrics: {'Precision': 0.22330097087378642, 'Recall': 0.5897435897435898, 'F1': 0.323943661971831, 'ROC-AUC': 0.7511103215491207}
For Target_At_Least_2: 
Best XGBoost Params: {'colsample_bytree': np.float64(0.8348352022414609), 'learning_rate': np.float64(0.01954101164904113), 'max_delta_step': 3, 'max_depth': 7, 'n_estimators': 298, 'reg_alpha': np.float64(0.5912977877077271), 'reg_lambda': np.float64(0.27472179299006416), 'subsample': np.float64(0.8683730277543102)}
Optimal XGBoost Threshold: 0.6372
Best Validation F1 at Optimal Threshold: 0.3846
XGBoost Validation Metrics: {'Precision': 0.37037037037037035, 'Recall': 0.4, 'F1': 0.38461538461538464, 'ROC-AUC': 0.8168232662192394}

Feature definitions and transformations

Code snippets (non-sensitive):
FIPS merge logic

target construction logic

feature engineering examples

threshold optimization logic

Glossary:
ROC-AUC: ability for a model to distinguish between two classes  (1 would be perfect, .5 is random, less than 0.5 the model is flawed)
Precision: measures the ability to accurately classify a case that is positive as positive 
Recall: Proportion of actual positive cases correctly identified by the model
SHAP: provides local explainability for XGBoost model
SVI: measures a community’s susceptibility to negative impacts from disasters or crises ranging from 0 (lowest) to 1 (highest)
Acronyms:
CDC: Centers for Disease Control and Prevention 
SVI: Social Vulnerability Index 
MMR: Measles Mumps and Rubella Vaccine 
SHAP:SHapley Additive exPlanations 
Testing logs:
validation performance summaries
test set confusion matrices


precision-recall curves

Decision logs:
removal of clustering from final project scope
comparison of Target_At_Least_1 versus Target_At_Least_2
selection of XGBoost as primary model



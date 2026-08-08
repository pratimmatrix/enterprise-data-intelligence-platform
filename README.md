# 📊 Enterprise Data Intelligence Platform

> AI-powered customer campaign decision intelligence platform combining data quality, statistical analysis, machine learning, business rules, model explainability, and an interactive Streamlit dashboard.

---

## 🚀 Overview

The **Enterprise Data Intelligence Platform** is an end-to-end machine learning and business intelligence system designed to analyze customer campaign data and support marketing decision-making.

The platform takes raw customer banking data and processes it through a complete data intelligence pipeline:

**Data Ingestion → Validation → Profiling → Anomaly Detection → Feature Engineering → Feature Validation → Model Training → Model Comparison → Model Selection → Prediction → Business Rules → Insights → Explainability → Dashboard**

The system does not simply generate a machine learning prediction. It converts the prediction into a **business decision** by combining:

- Data quality checks
- Statistical anomaly detection
- Business-oriented feature engineering
- Multiple machine learning models
- Model evaluation
- Business rules
- Customer insights
- Model explainability
- Interactive visualization

---

# 🎯 Project Objective

The objective of the platform is to predict whether a customer is likely to respond positively to a marketing campaign.

The system uses historical banking campaign data to:

1. Validate the incoming dataset.
2. Identify potentially anomalous observations.
3. Create meaningful business features.
4. Train multiple machine learning models.
5. Compare model performance.
6. Select the best model using F1 Score.
7. Generate customer-level predictions.
8. Assign a business risk category.
9. Determine campaign priority.
10. Generate business insights.
11. Explain the model's prediction.
12. Present the complete decision through an interactive dashboard.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │   Raw Customer Data  │
                         │    bank-full.csv    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Data Ingestion     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Data Validation    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Data Profiling     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Anomaly Detection    │
                         │ IQR + Z-Score        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Validation   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Machine Learning Layer    │
                    │                               │
                    │ Logistic Regression           │
                    │ Random Forest                 │
                    │ Gradient Boosting             │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Model Comparison    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Model Selection    │
                         │      F1 Score        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Model Prediction   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             Business Rules     Insights       Explainability
                    │               │                │
                    └───────────────┼────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Dashboard  │
                         └──────────────────────┘
🔧 Main Components
1. Data Ingestion

Loads the banking campaign dataset and performs initial dataset checks.

The project uses the bank-full.csv dataset.

Dataset size:

45,211 rows
17 original columns
2. Data Validation

The validation layer checks:

Number of rows
Number of columns
Missing values
Duplicate rows
Data types
Memory usage

Example validation results:

Rows              : 45211
Columns           : 17
Duplicate Rows    : 0
Missing Values    : 0

This ensures that downstream machine learning components receive a structurally valid dataset.

3. Data Profiling

The profiling layer analyzes the structure of the dataset.

It identifies:

Numeric columns
Categorical columns
Dataset dimensions
Basic dataset characteristics

The original dataset contains:

7 numeric columns
10 categorical/text columns
🚨 4. Anomaly Detection

The platform performs statistical anomaly detection using:

IQR Method

The Interquartile Range method identifies observations outside the expected statistical range.

Z-Score Method

Z-score analysis identifies observations that are statistically far from the mean.

The system analyzes variables including:

Age
Balance
Duration
Campaign contacts
Days since previous contact
Previous contacts

Example anomaly analysis:

Column: balance
IQR Anomalies       : 4729
Z-Score Anomalies   : 745
Anomaly Percentage  : 10.46%

Column: duration
IQR Anomalies       : 3235
Z-Score Anomalies   : 963
Anomaly Percentage  : 7.16%

These values are treated as potential statistical anomalies rather than automatically being classified as data errors.

🧠 5. Feature Engineering

The platform creates additional business-oriented features.

Engineered Features
age_group
balance_log
campaign_log
previous_contact
previously_contacted
zero_balance
loan_burden
campaign_intensity
contact_unknown
previous_success
Feature descriptions
Feature	Description
age_group	Groups customers into age ranges
balance_log	Log-transformed account balance
campaign_log	Log-transformed campaign contact count
previous_contact	Indicates whether previous contact exists
previously_contacted	Binary indicator of previous campaign interaction
zero_balance	Indicates whether account balance is zero
loan_burden	Represents loan-related customer burden
campaign_intensity	Categorizes campaign contact intensity
contact_unknown	Indicates unknown contact channel
previous_success	Indicates previous campaign success

After feature engineering:

Original columns : 17
New columns      : 27
Features created : 10
✅ 6. Feature Validation

The feature validation layer automatically checks the engineered dataset.

Validation includes:

Feature existence
Row integrity
Binary feature values
Numeric NaN values
Numeric infinite values
Categorical values
Duplicate rows

Example:

Expected features : 10
Binary features   : 6
Numeric features  : 2
Categorical       : 2

FEATURE VALIDATION PASSED
🤖 Machine Learning

The platform evaluates three classification models:

Logistic Regression
Random Forest
Gradient Boosting

The models are evaluated using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC

Because campaign response is an imbalanced classification problem, F1 Score is used as the primary model-selection metric.

📊 Model Comparison
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Logistic Regression	76.55%	27.97%	63.80%	38.89%	77.58%
Random Forest	81.21%	33.32%	60.49%	42.97%	79.46%
Gradient Boosting	89.47%	67.91%	19.00%	29.69%	79.89%
🏆 Final Model

The platform selects the Random Forest Classifier using F1 Score.

Selected Model Performance
Model       : Random Forest
Accuracy    : 81.21%
Precision   : 33.32%
Recall      : 60.49%
F1 Score    : 42.97%
ROC-AUC     : 79.46%
Why Random Forest?

Gradient Boosting achieved higher accuracy and ROC-AUC, but its recall was only approximately 19%.

The Random Forest model achieved the highest F1 Score among the evaluated models.

Since the project aims to balance:

Identifying potential campaign responders
Avoiding excessive false positives

F1 Score provides a more useful selection criterion than accuracy alone.

🔮 Model Prediction

The trained Random Forest pipeline is saved as:

random_forest_pipeline.pkl

The prediction system generates:

Prediction
Probability
Risk Category

Example:

Prediction        : NO
Probability       : 45.39%
Risk Category     : MEDIUM
💼 Business Rules

Machine learning output is converted into a business decision using a dedicated business rule engine.

The system determines:

Priority
Recommended action
Campaign treatment

Example:

Priority : MEDIUM

Action:
Include customer in standard marketing follow-up.

This separates the machine learning prediction from the business decision layer.

💡 Business Insights

The Insight Engine converts model output into understandable business statements.

Example:

Customer shows a lower likelihood of responding to the campaign.

Prediction probability is moderate, indicating some potential
for customer response.

Risk category is classified as MEDIUM.

Customer may receive campaign attention after higher-priority customers.

This makes the ML output easier for business users to understand.

🔍 Model Explainability

The platform includes an Explainability Engine.

The engine explains the prediction using:

Prediction probability
Risk category
Customer attributes
Global feature importance

Example explanation:

The model predicts NO with a probability of 45.39%.

The resulting model risk category is MEDIUM.

The customer has no previous campaign contact history.

There is no known outcome from a previous campaign contact.

The current campaign contact count is 2,
indicating relatively low contact intensity.

The customer was contacted through a cellular communication channel.

The current contact duration is 300 seconds,
indicating relatively strong customer engagement.

The customer is 35 years old and belongs to the 31-40 age group.

The customer has a positive account balance of 1500.
📈 Global Feature Importance

The Random Forest model provides global feature importance values.

Top features from the trained model include:

Rank	Feature	Importance
1	categorical__poutcome_success	0.0675
2	numeric__previous_success	0.0617
3	numeric__age	0.0590
4	numeric__contact_unknown	0.0524
5	numeric__balance_log	0.0496
6	numeric__balance	0.0494
7	categorical__contact_unknown	0.0487
8	numeric__day	0.0456
9	numeric__pdays	0.0403
10	categorical__housing_no	0.0392
🖥️ Interactive Dashboard

The project includes a Streamlit dashboard.

The dashboard provides:

Customer Input

Users can provide:

Age
Job
Marital status
Education
Credit default
Account balance
Housing loan
Personal loan
Contact channel
Contact day
Month
Contact duration
Campaign contacts
Days since previous contact
Previous contacts
Previous campaign outcome
Dashboard Output

The dashboard displays:

Prediction
Probability
Risk category
Campaign priority
Recommended business action
Business insights
Customer profile
Model information
Feature importance
Model explanation
Final decision summary
🔄 End-to-End Workflow

The complete pipeline can be executed through:

python app.py

The pipeline performs:

1. Data Ingestion
2. Data Validation
3. Data Profiling
4. Anomaly Detection
5. Feature Engineering
6. Feature Validation
7. Baseline Model Training
8. Model Comparison
9. Model Selection
10. Final Summary
🌐 Running the Dashboard

The Streamlit dashboard can be started using:

python -m streamlit run .\dashboard\Dashboard.py

Streamlit will provide a local URL where the dashboard can be opened in the browser.

📁 Project Structure
enterprise-data-intelligence-platform/
│
├── README.md
├── app.py
├── bank-full.csv
│
├── dashboard/
│   └── Dashboard.py
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── validator.py
│   │
│   ├── profiling/
│   │   ├── profiler.py
│   │   └── anomalies.py
│   │
│   ├── feature_engineering/
│   │   └── feature_engineer.py
│   │
│   ├── feature_validation/
│   │   └── feature_validator.py
│   │
│   ├── modeling/
│   │   ├── model_trainer.py
│   │   ├── model_comparator.py
│   │   ├── model_selector.py
│   │   └── ModelPredictor.py
│   │
│   ├── business_rules/
│   │   ├── BusinessRuleEngine.py
│   │   └── DecisionEngine.py
│   │
│   ├── insights/
│   │   └── InsightEngine.py
│   │
│   └── explainability/
│       ├── __init__.py
│       └── ExplainabilityEngine.py
│
└── .gitignore
🛠️ Technology Stack
Programming
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
Logistic Regression
Random Forest
Gradient Boosting
Visualization
Matplotlib
Streamlit
Model Persistence
Joblib
Development
VS Code
Git
GitHub
Python Virtual Environment
📦 Installation

Clone the repository:

git clone https://github.com/pratimmatrix/enterprise-data-intelligence-platform.git

Navigate into the project:

cd enterprise-data-intelligence-platform

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install pandas numpy scikit-learn matplotlib streamlit joblib
▶️ Running the Project
Run the complete ML pipeline
python app.py
Run the dashboard
python -m streamlit run .\dashboard\Dashboard.py
Test the prediction engine
python .\src\modeling\ModelPredictor.py
Test the decision engine
python -m src.business_rules.DecisionEngine
Test explainability
python .\src\explainability\ExplainabilityEngine.py
📌 Example Business Scenario

Consider a customer with:

Age                  : 35
Job                  : Management
Education            : Tertiary
Balance              : 1500
Housing Loan         : Yes
Personal Loan        : No
Campaign Contacts    : 2
Previous Contacts    : 0
Previous Outcome     : Unknown
Contact Channel      : Cellular
Duration             : 300 seconds

The platform may generate:

Prediction        : NO
Probability       : 45.39%
Risk Category     : MEDIUM
Priority          : MEDIUM

Business decision:

Include customer in standard marketing follow-up.

The platform then provides business insights and an explanation of the prediction.

🧩 Design Principles

The project follows a modular architecture.

Each major responsibility is separated into its own component:

Data
 ↓
Validation
 ↓
Feature Engineering
 ↓
Machine Learning
 ↓
Business Rules
 ↓
Insights
 ↓
Explainability
 ↓
Dashboard

This makes the platform easier to:

Maintain
Test
Extend
Debug
Deploy
🔐 Model and Data Considerations

The model should be treated as a decision-support system rather than an autonomous decision-maker.

Important considerations include:

Model predictions are probabilistic.
Feature importance does not imply causality.
Statistical anomalies are not necessarily incorrect records.
Model performance depends on the training dataset.
Business rules should be reviewed according to campaign objectives.
Production deployment would require additional monitoring and validation.
🚀 Future Improvements

Potential future improvements include:

SHAP-based local explanations
Probability calibration
Hyperparameter optimization
Cross-validation
Automated model monitoring
Data drift detection
Model drift detection
Experiment tracking
Database integration
REST API deployment
Docker containerization
Cloud deployment
Role-based dashboard access
Automated retraining
Advanced campaign optimization
A/B testing integration
📊 Key Project Results
Dataset Size       : 45,211 customers
Original Features  : 17
Engineered Features: 10
Final Features     : 27

Models Evaluated   : 3

Selected Model     : Random Forest

Accuracy           : 81.21%
Precision          : 33.32%
Recall             : 60.49%
F1 Score           : 42.97%
ROC-AUC            : 79.46%
🎓 What This Project Demonstrates

This project demonstrates practical experience with:

End-to-end machine learning pipelines
Data ingestion
Data quality validation
Exploratory profiling
Statistical anomaly detection
Feature engineering
Feature validation
Classification
Imbalanced classification
Model comparison
Model selection
Model persistence
Business rule systems
Decision intelligence
Model explainability
Streamlit dashboards
Modular Python architecture
Git and GitHub
👨‍💻 Author

Pratim Mistry

Mechanical Engineering student transitioning into Data Science / AI & ML.

GitHub:

https://github.com/pratimmatrix

Project:

https://github.com/pratimmatrix/enterprise-data-intelligence-platform

⭐ Project Status

Status: Completed

The current version includes:

✅ Data ingestion
✅ Data validation
✅ Data profiling
✅ Anomaly detection
✅ Feature engineering
✅ Feature validation
✅ Model training
✅ Model comparison
✅ Model selection
✅ Model prediction
✅ Business rules
✅ Business insights
✅ Model explainability
✅ Streamlit dashboard
✅ GitHub repository
✅ Project documentation
📜 License

This project is intended for educational, portfolio, and demonstration purposes.


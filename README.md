🏠 House Price Prediction Using Machine Learning

A machine learning project for predicting residential house sale prices using structured housing data. The project implements a complete end-to-end regression workflow, including data upload, robust outlier removal, feature engineering/selection, preprocessing, hyperparameter optimization, model evaluation, visualization, and an interactive house price predictor.

Two regression algorithms are trained and optimized using cross-validated grid search:

Random Forest Regressor
Support Vector Regression (SVR) with an RBF kernel

The best-performing model is automatically selected based on its R² score and can then be used to estimate the market value of a house based on user-provided characteristics.

📌 Project Overview

Accurately estimating residential property prices is a common machine learning regression problem. House prices are influenced by numerous factors, including property size, quality, age, location, and construction characteristics.

This project focuses on building a practical and optimized prediction pipeline using selected numerical and categorical housing features.

The workflow is designed to:

Load a housing dataset interactively in Google Colab.
Remove records without a target sale price.
Detect and remove extreme outliers using the Interquartile Range (IQR) method.
Select important numerical and categorical features.
Split the data into training and testing sets.
Handle missing values automatically.
Scale numerical features.
Encode categorical features using one-hot encoding.
Optimize Random Forest and SVR models using GridSearchCV.
Evaluate both models using MAE, RMSE, and R².
Visualize actual vs. predicted prices.
Automatically identify the best-performing model.
Provide an interactive interface for predicting the price of a hypothetical house.
🎯 Objectives

The main objectives of this project are:

Develop a reliable house price regression model.
Improve model performance through preprocessing and hyperparameter tuning.
Reduce the influence of extreme observations through robust outlier removal.
Compare two different machine learning approaches.
Evaluate models using multiple regression metrics.
Provide an easy-to-use interactive prediction interface.
Demonstrate a complete machine learning pipeline suitable for real-world structured data.
📊 Dataset

The project expects a CSV file named:

HousePricePrediction.csv


The dataset should contain a target column:

SalePrice


The model uses the following features when they are available in the uploaded dataset.

Numerical Features
LotArea — Lot size of the property
OverallQual — Overall material and finish quality
OverallCond — Overall condition of the property
YearBuilt — Original construction year
YearRemodAdd — Year of remodeling
TotalBsmtSF — Total basement area
GrLivArea — Above-ground living area
Categorical Features
MSSubClass — Building class
MSZoning — General zoning classification
BldgType — Type of dwelling
Neighborhood — Physical location within the city

The code dynamically checks whether each feature exists in the uploaded dataset, allowing it to handle datasets with slightly different column availability.

🧹 Data Cleaning and Outlier Removal

Before model training, records without a SalePrice value are removed.

The project then applies an IQR-based outlier detection method to several important numerical variables:

SalePrice
GrLivArea
LotArea
TotalBsmtSF

For each feature, the first quartile (Q1) and third quartile (Q3) are calculated.

The Interquartile Range is:

IQR = Q3 - Q1


Observations outside the following range are removed:

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR


This helps reduce the influence of extreme observations that could negatively affect model performance.

⚙️ Data Preprocessing

The project uses a Scikit-learn ColumnTransformer to apply different preprocessing techniques to numerical and categorical variables.

Numerical Pipeline

Numerical features are processed using:

Median imputation for missing values.
StandardScaler for feature scaling.
Missing numerical values
        ↓
Median Imputation
        ↓
Standard Scaling

Categorical Pipeline

Categorical features are processed using:

Most-frequent-value imputation.
One-hot encoding.
Missing categorical values
        ↓
Most Frequent Imputation
        ↓
One-Hot Encoding


The encoder uses:

handle_unknown='ignore'


This allows the model to handle previously unseen categorical values during prediction without producing an error.

🧠 Machine Learning Models
1. Random Forest Regressor

The first model is a Random Forest Regressor, an ensemble learning algorithm that combines multiple decision trees to produce predictions.

The project searches across several hyperparameter combinations:

n_estimators: [100, 200]
max_depth: [10, 15, None]
min_samples_split: [2, 5]


The model is initialized with:

random_state=42


This provides reproducible results.

2. Support Vector Regression

The second model is Support Vector Regression using an RBF kernel.

The hyperparameter search includes:

C: [50000, 100000, 200000]
epsilon: [0.1, 1, 10]


The RBF kernel allows the model to capture nonlinear relationships between housing characteristics and sale prices.

🔍 Hyperparameter Optimization

Instead of relying on default model parameters, the project uses GridSearchCV to identify strong hyperparameter combinations.

Three-fold cross-validation is used:

GridSearchCV(..., cv=3, scoring='r2')


The model with the highest cross-validated R² score is selected as the best configuration.

This approach systematically evaluates different parameter combinations instead of manually selecting a single configuration.

📈 Model Evaluation

The trained models are evaluated on the held-out test set using three standard regression metrics.

Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted house prices.

MAE = average(|actual - predicted|)


A lower MAE indicates better performance.

Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

RMSE = √(average((actual - predicted)²))


RMSE gives greater weight to larger prediction errors.

A lower RMSE indicates better performance.

R² Score

R² measures how much of the variation in house prices is explained by the model.

R² = 1 - (SSres / SStot)


A value closer to 1.0 generally indicates stronger predictive performance.

📊 Visualization

The project generates an Actual vs. Predicted Price visualization for both models.

The plot compares:

X-axis: Actual Sale Price
Y-axis: Predicted Sale Price

A red dashed diagonal line represents perfect predictions.

Predicted Price
      │
      │        /
      │      /
      │    /   ← Ideal predictions
      │  /
      │/
      └──────────────── Actual Price


The visualization is saved automatically as:

Optimized_Model_Performance.png


at 300 DPI.

🏆 Best Model Selection

After evaluating both models, the project automatically identifies the model with the highest test-set R² score.

The selected model is then used by the interactive prediction component.

This means the user does not need to manually decide which algorithm to use.

🏡 Interactive House Price Predictor

After model evaluation, the program asks:

Would you like to predict a house price? (yes/no):


If the user chooses yes, the best-performing model is used for prediction.

For every numerical feature, the default value is the dataset median.

For categorical features, the default value is the dataset mode.

The user can either:

Press Enter to use the default value.
Enter a custom value.

The model then produces an estimated market value:

-----------------------------------
         PREDICTION RESULT
-----------------------------------
Estimated Market Value: $XXX,XXX.XX
-----------------------------------

🔄 Project Workflow
                 HousePricePrediction.csv
                           │
                           ▼
                    Upload Dataset
                           │
                           ▼
                    Data Validation
                           │
                           ▼
                  Remove Missing Target
                           │
                           ▼
                   Outlier Detection
                    (IQR Method)
                           │
                           ▼
                    Feature Selection
                           │
                           ▼
                    Train/Test Split
                           │
                           ▼
                  Preprocessing Pipeline
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Numerical Data       Categorical Data
          Median Imputation    Mode Imputation
          Standard Scaling     One-Hot Encoding
                 │                   │
                 └─────────┬─────────┘
                           ▼
                   Machine Learning
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
           Random Forest            SVR
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    GridSearchCV
                           │
                           ▼
                    Model Evaluation
                  MAE / RMSE / R²
                           │
                           ▼
                    Best Model Selected
                           │
                           ▼
                 Interactive Prediction
                           │
                           ▼
                    Estimated Price

🛠️ Technologies Used
Python
NumPy — Numerical computation
Pandas — Data manipulation and analysis
Matplotlib — Data visualization
Seaborn — Statistical visualization
Scikit-learn — Machine learning and preprocessing
Google Colab — Interactive development environment
Main Scikit-learn Components
train_test_split
GridSearchCV
ColumnTransformer
Pipeline
SimpleImputer
OneHotEncoder
StandardScaler
RandomForestRegressor
SVR
mean_absolute_error
mean_squared_error
r2_score
🚀 How to Run the Project
Option 1 — Google Colab
Open the notebook in Google Colab.
Run the code.
When prompted, upload:
HousePricePrediction.csv

Wait for preprocessing and hyperparameter optimization to complete.
Review the model evaluation results.
Examine the generated performance visualization.
Choose whether to use the interactive predictor.
Enter house characteristics when prompted.
Option 2 — Local Python Environment

Install the required dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn


Because the current implementation uses:

from google.colab import files


the interactive file-upload section is specifically designed for Google Colab. For local execution, the CSV-loading section should be adapted to use a local file path.

📁 Expected Project Structure

A recommended repository structure is:

House-Price-Prediction/
│
├── HousePricePrediction.csv
├── house_price_prediction.ipynb
├── Optimized_Model_Performance.png
├── README.md
└── requirements.txt


Example requirements.txt:

numpy
pandas
matplotlib
seaborn
scikit-learn

🔬 Key Features
✅ Interactive CSV file upload
✅ Automated data cleaning
✅ IQR-based multi-feature outlier removal
✅ Automatic feature availability checking
✅ Missing-value handling
✅ Numerical feature scaling
✅ Categorical feature encoding
✅ Machine learning pipelines
✅ Random Forest regression
✅ Support Vector Regression
✅ Automated hyperparameter tuning
✅ Three-fold cross-validation
✅ MAE, RMSE, and R² evaluation
✅ Actual vs. predicted visualization
✅ Automatic best-model selection
✅ Interactive house price prediction
⚠️ Limitations

Although the project provides a complete machine learning workflow, predictions should be interpreted as model estimates rather than guaranteed market values.

Some limitations include:

The model uses a relatively limited subset of potentially relevant housing features.
House prices can be influenced by factors not included in the dataset.
Removing outliers can improve model robustness but may also remove legitimate high-value properties.
Model performance depends heavily on the quality and representativeness of the uploaded dataset.
The train/test split uses a single random split, so reported test performance can vary with a different split.
The interactive predictor uses median/mode defaults when the user does not provide a value.
Real-world property valuation may require additional information such as property condition, amenities, local market trends, and economic conditions.
🔮 Future Improvements

Potential improvements include:

Add more relevant housing features.
Experiment with Gradient Boosting, XGBoost, LightGBM, or other advanced regression algorithms.
Use randomized or Bayesian hyperparameter optimization.
Apply log transformation to SalePrice to reduce skewness.
Perform feature importance analysis.
Add residual/error analysis.
Use repeated cross-validation for more robust evaluation.
Build an interactive web application using Streamlit or Flask.
Save the trained model using joblib or pickle.
Add automated data-quality checks.
Implement confidence or prediction intervals.
Deploy the final model as a web-based house valuation service.
📜 License

This project is intended for educational and research purposes. If you plan to use it commercially, ensure that the underlying dataset and all third-party components permit such use.

👤 Author

[Ishrat Mubarak]

Machine Learning / Data Science Project

⭐ Project Summary

This project demonstrates how machine learning can be applied to residential property data to build an end-to-end house price prediction system. By combining robust preprocessing, outlier handling, feature selection, hyperparameter optimization, model comparison, and interactive prediction, the project provides a practical framework for exploring automated property valuation.

Key takeaway: The system compares Random Forest and Support Vector Regression models, selects the stronger model based on R² performance, and uses that model to generate interactive house price estimates.

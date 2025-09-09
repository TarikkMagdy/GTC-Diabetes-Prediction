# GTC Project 2: Diabetes Prediction Model

### Project Overview

This project focuses on building and evaluating machine learning models to predict the onset of diabetes based on a set of diagnostic medical measurements. The primary goal is to compare the performance of at least two models—one of which must be a Support Vector Machine (SVM)—and select the best one based on relevant evaluation metrics. This project follows a standard data science pipeline from data exploration and cleaning to model training and final prediction.

---

### Dataset

The dataset used is the **Pima Indians Diabetes Database** (`diabetes.csv`). It contains 768 entries with 8 medical predictor variables and one target variable (`Outcome`). The predictor variables include the number of pregnancies, BMI, insulin level, age, etc.

---

### Project Pipeline

The project was structured into four main phases:

**1. Exploratory Data Analysis (EDA)**
* Loaded the dataset and performed an initial inspection.
* Identified "hidden" missing values (zeros in columns like `Glucose`, `BMI`, etc.).
* Visualized feature distributions using histograms, which revealed that many features were skewed.
* Created a correlation heatmap to analyze relationships between features.

**2. Data Preprocessing**
* Cleaned the data by replacing the invalid `0` values with the median of each respective column.
* Handled outliers using the IQR method by capping extreme values.
* Split the data into training (80%) and testing (20%) sets to prevent data leakage.
* Applied `StandardScaler` to standardize the feature values, fitting the scaler on the training data only.

**3. Modeling & Evaluation**
* Built and evaluated three different models:
    * **Support Vector Machine (SVM)** (Required Model)
    * **Random Forest**
    * **Logistic Regression** (Baseline)
* The default Random Forest and SVM models were the initial top performers.

**4. Model Optimization with SMOTE**
* To address the class imbalance in the dataset (fewer diabetic patients), we used the **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data.
* This created a balanced training set, which was then used to retrain our best model (Random Forest).

---

### Final Model Selection

The **Random Forest model trained on the SMOTE-balanced data** was chosen as the champion model. While its overall accuracy was slightly lower than the default models, it achieved a significantly higher **recall** for the 'Diabetic' class. In a medical context, correctly identifying positive cases (high recall) is often more critical than overall accuracy.

**Final Model Performance:**
```
              precision    recall  f1-score   support

Non-Diabetic       0.82      0.78      0.80        99
    Diabetic       0.62      0.71      0.66        55

    accuracy                           0.75       154
```

---

### How to Use

1.  Clone this repository.
2.  Ensure you have the required libraries (pandas, numpy, scikit-learn, imblearn, seaborn).
3.  Open the `.ipynb` notebook file in an environment like Google Colab or Jupyter Notebook.
4.  Run the cells sequentially to see the full analysis and prediction process. The final cell contains a prediction function that can be used to classify new patient data.

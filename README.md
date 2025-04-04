# Heart Attack Risk Prediction Project

## Overview

This project focuses on building a machine learning model to predict the risk of a heart attack based on several medical features. The analysis involves data exploration, preprocessing, feature selection, model training using Logistic Regression, and evaluation of the model's performance. Hyperparameter tuning using GridSearchCV was also performed to optimize the model.

## Libraries Used

* **NumPy:** For numerical computations and array manipulation.
* **Pandas:** For data manipulation and analysis using DataFrames.
* **Seaborn:** For creating informative and visually appealing statistical graphics.
* **Matplotlib:** For creating basic plots and visualizations.
* **Scikit-learn (sklearn):** For machine learning tasks, including:
    * `model_selection.train_test_split`: Splitting data into training and testing sets.
    * `linear_model.LogisticRegression`: Implementing the Logistic Regression algorithm.
    * `metrics.confusion_matrix`: Evaluating model performance using a confusion matrix.
    * `model_selection.GridSearchCV`: Performing hyperparameter tuning.

## Data Loading and Initial Exploration

* The dataset, `heart attack analysis and prediction.csv`, was loaded using pandas.
* The first 10 rows of the DataFrame were displayed to get an initial understanding of the data.
* The shape of the dataset (number of rows and columns) was checked.
* `df.info()` was used to get a concise summary of the DataFrame, including data types and non-null values.

## Data Preprocessing

* **Dropping Irrelevant Columns:** The columns `cp`, `fbs`, `restecg`, `exang`, `slp`, `caa`, and `thall` were dropped as they were deemed less relevant for this simplified model.
* **Checking for Duplicate Values:** The number of duplicate rows in the dataset was identified.
* **Removing Duplicate Values:** Duplicate rows were removed to ensure data integrity.
* **Summary Statistics:** Descriptive statistics for the numerical features were calculated and displayed.

## Exploratory Data Analysis (EDA)

* **Box Plots:** Box plots were generated to visualize the distribution and identify potential outliers in `trtbps` (resting blood pressure), `chol` (cholesterol levels), and `oldpeak`.
* **Histograms:** Histograms were plotted to show the distribution of `trtbps` and `chol` against age.
* **Heatmap:** A heatmap was created to visualize the correlation matrix between different variables in the dataset.
* **Scatter Plot:** A scatter plot was used to visualize the relationship between age, the chance of a heart attack (`output`), and sex.

## Feature Selection

* The following features were selected as independent variables (X) for the model: `age`, `sex`, `trtbps`, `thalachh` (maximum heart rate achieved), and `chol`.
* The `output` column was selected as the target variable (y).

## Data Splitting

* The dataset was split into training and testing sets using `train_test_split` with an 80% training and 20% testing ratio. `random_state` was set for reproducibility.

## Model Training (Logistic Regression)

* A Logistic Regression model was initialized with the `liblinear` solver and a `random_state` for reproducibility.
* The model was trained using the training data (`X_train`, `y_train`).

## Model Evaluation

* **Confusion Matrix:** The model's predictions on the test set (`y_pred`) were used to generate a confusion matrix, which was then printed and visualized using a seaborn heatmap. The True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) were extracted from the confusion matrix.
* **Precision:** The precision score was calculated to measure the proportion of correctly predicted positive cases out of all predicted positive cases.
* **Recall (Sensitivity):** The recall score was calculated to measure the proportion of correctly predicted positive cases out of all actual positive cases.
* **Accuracy:** The accuracy of the model on both the training and testing sets was calculated and printed.

## Hyperparameter Optimization using GridSearchCV

* **Parameter Grid:** A parameter grid was defined to explore different values for the `penalty` (`l1`, `l2`) and `C` (regularization strength) hyperparameters of the Logistic Regression model.
* **GridSearchCV:** The `GridSearchCV` function was used to perform an exhaustive search over the specified parameter grid, using 5-fold cross-validation to evaluate each combination of hyperparameters. The `accuracy` score was used as the scoring metric.
* **Best Model Examination:** The best score achieved during the grid search, the parameters that yielded the best results, and the best estimator chosen by GridSearchCV were printed.
* **GridSearchCV Score on Test Set:** The performance of the best model (found by GridSearchCV) was evaluated on the test set, and the accuracy score was printed.

## Conclusion

This project successfully implemented a Logistic Regression model to predict the risk of a heart attack. The model's performance was evaluated using a confusion matrix, precision, recall, and accuracy. Hyperparameter tuning using GridSearchCV helped to identify a potentially better-performing model. Further improvements could involve exploring other machine learning algorithms, more extensive feature engineering, and collecting more data.

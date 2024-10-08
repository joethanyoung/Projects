######################################## import required packages
# Standard library imports
from pathlib import Path
import pickle
import time

# Third-party imports
import featuretools as ft
import joblib
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve,make_scorer)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, Normalizer, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# Start timing
start_time = time.time()

######################################## load the data 
# load and preprocess the dataset
base_path = Path(__file__).resolve().parent / 'data'
    
# Define the path to the CSV file
data_path = base_path / 'diabetes.csv'
    
# Load the data from the CSV file
df = pd.read_csv(data_path)
print(df.shape)
print(df.head(5))


# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

######################################## EDA analysis
# Histograms for each feature
df.hist(bins=20, figsize=(14, 7), layout=(3, 3))  # Smaller figure size and 3x3 grid layout
plt.tight_layout()  # Adjusts subplot parameters for a neat fit
plt.savefig(base_path / 'histograms.png')
plt.close()

# boxplot
num_columns = len(df.columns)  # Number of columns 
# Increase the number of rows to reduce overall width
rows = int(num_columns ** 0.5) + 1  # ensures more vertical distribution
cols = (num_columns // rows) + (num_columns % rows > 0)  # Determining the number of columns for subplots
# Set fixed dimensions for each subplot (width, height)
subplot_width = 5 
subplot_height = 3  
# Calculate the figure dimensions based on the number of rows and columns
fig_width = cols * subplot_width  # Increase total width by increasing subplot width
fig_height = rows * subplot_height
plt.figure(figsize=(fig_width, fig_height))
# 
for i, column in enumerate(df.columns):
    ax = plt.subplot(rows, cols, i + 1)  # Create a subplot for each column
    sns.boxplot(x=df[column])
    plt.title(f'Box plot of {column}', fontsize=10)  # Reduced title font size

    # Set smaller font size for x and y axis labels and titles
    ax.set_xlabel(column, fontsize=8)  # Smaller font size for x-axis labels
    ax.set_ylabel('Values', fontsize=8)  # Smaller font size for y-axis labels
    ax.tick_params(axis='both', which='major', labelsize=8)  # Smaller ticks
# Adjust layout 
plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.6)  # Adjust horizontal and vertical spaces
plt.savefig(base_path / 'boxplots.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.savefig(base_path / 'correlation_matrix.png')
plt.close()

# Create a pair plot with adjusted size
plt.figure(dpi=300)
sns.pairplot(df, hue='Outcome', height=3, aspect=1)
plt.tight_layout()  # Adjust layout to make room for all elements
plt.savefig(base_path / 'pairplot.png')
plt.close()

# Scatter plots for specific variables
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df)
plt.title('Glucose vs BMI colored by Outcome')
plt.savefig(base_path / 'glucose_vs_bmi.png')
plt.close()

######################################## preprocessing
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

# Calculate percentage of missing values in each column
missing_percentage = df[columns_with_zeros].isnull().mean() * 100
print("Percentage of missing values in each column:")
print(missing_percentage)

# Matrix plot to visualize missing data
msno.matrix(df[columns_with_zeros])
plt.savefig(base_path / 'missing_data_matrix.png')
plt.close()

# Create binary indicators for missing data directly within the DataFrame
df['Insulin_missing'] = df['Insulin'].isnull().astype(int)
df['BMI_missing'] = df['BMI'].isnull().astype(int)

# Create a contingency table and perform the Chi-square test
contingency = pd.crosstab(df['Insulin_missing'], df['BMI_missing'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f'Chi-square test results for Insulin vs BMI Missingness: p={p}')

# Interpretation of the Chi-square test
if p < 0.05:
    print('''There is a statistically significant association between 
          the missingness of Insulin and BMI.''')
else:
    print('''There is no statistically significant association between 
          the missingness of Insulin and BMI.''')

# Perform Little's MCAR test and print the result
mcar_test = MCARTest()
result = mcar_test(df[columns_with_zeros])
print(f"Little's MCAR test result: p-value={result}")

# Interpretation of Little's MCAR test
if result < 0.05:
    print('''The data are not missing completely at random (MCAR).
           There is a pattern to the missingness.''')
else:
    print('''The data are missing completely at random (MCAR). 
          There is no apparent pattern to the missingness.''')
    

# Define columns
columns_with_less_missing = ['Glucose', 'BloodPressure', 'BMI']
columns_with_more_missing = ['Insulin', 'SkinThickness']
target = 'Outcome'

# Replace 0 with NaN in the specified columns
for column in columns_with_less_missing + columns_with_more_missing:
    df[column] = df[column].replace(0, np.nan)

# Prepare imputations for less and more missing data columns
preprocessor = ColumnTransformer(
    transformers=[
        ('less_missing', Pipeline([('imputer', SimpleImputer(strategy='median'))]), columns_with_less_missing),
        ('more_missing', Pipeline([('imputer', KNNImputer())]), columns_with_more_missing)
    ],
    remainder='passthrough'
)

# Pipeline with classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=1234))
])

# Parameter grid
param_grid = {
    'preprocessor__more_missing__imputer__n_neighbors': range(1, 11),
    'preprocessor__less_missing__imputer': [IterativeImputer(max_iter=iter, random_state=1234) for iter in [10, 20, 50, 100]],
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [5, 10, 20, 30, 50, None]
}

# Split dataset
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

# Extract the best estimator
best_pipeline = grid_search.best_estimator_

# Get the best preprocessor from the pipeline
best_preprocessor = best_pipeline.named_steps['preprocessor']

# Get the feature names in the correct order
feature_names = (columns_with_less_missing + 
                 columns_with_more_missing + 
                 [col for col in df.columns if col not in columns_with_less_missing + 
                  columns_with_more_missing + [target]])

# Transform both training and test data
X_train_transformed = best_preprocessor.transform(X_train)
X_test_transformed = best_preprocessor.transform(X_test)

# Create DataFrames for both transformed datasets
train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

# Add the target variable back
train_df[target] = y_train
test_df[target] = y_test

# Output y_train to Excel
y_train_df = pd.DataFrame(y_train)
y_train_df.to_excel("y_train.xlsx", index=False)  

# Output y_test to Excel
y_test_df = pd.DataFrame(y_test)
y_test_df.to_excel("y_test.xlsx", index=False)

# Combine the transformed train and test sets to get the full imputed dataset
full_imputed_df = pd.concat([train_df, test_df])
print(full_imputed_df.shape)

# Function to plot correlation matrix
def plot_correlation(df, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=1, 
                linecolor='white')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(base_path / f'{title.replace(" ", "_").lower()}_correlation_matrix.png')
    plt.close()

# Plotting correlation matrices
plot_correlation(full_imputed_df, "Correlation Matrix with Best Imputation")

# feature engineering
# Check if 'index' column exists, if not, reset index to create one
if 'index' not in full_imputed_df.columns:
    full_imputed_df.reset_index(inplace=True, drop=False)

# Define custom feature creation functions
def add_custom_features(df):
    df_copy = df.copy()
    
    # Interaction features
    df_copy['BMI_Age'] = df_copy['BMI'] * df_copy['Age']
    df_copy['Preg_Age'] = df_copy['Pregnancies'] * df_copy['Age']
    
    # Ratio features with conditional check to avoid division by zero
    df_copy['Insulin_Glucose_Ratio'] = np.where(df_copy['Glucose'] != 0, df_copy['Insulin'] / df_copy['Glucose'], np.nan)
    df_copy['Skin_BMI_Ratio'] = np.where(df_copy['BMI'] != 0, df_copy['SkinThickness'] / df_copy['BMI'], np.nan)
    
    # Polynomial features
    df_copy['Glucose_Squared'] = df_copy['Glucose'] ** 2
    df_copy['BMI_Squared'] = df_copy['BMI'] ** 2
    
    # More complex features based on initial code snippet provided
    df_copy['glucose_insulin_ratio'] = np.where(df_copy['Insulin'] != 0, df_copy['Glucose'] / df_copy['Insulin'], np.nan)
    df_copy['bmi_skinthickness_product'] = df_copy['BMI'] * df_copy['SkinThickness']
    df_copy['age_adjusted_risk'] = (df_copy['Age'] * df_copy['Glucose'] * df_copy['BMI']) / 1000
    df_copy['pregnancy_health_impact'] = (df_copy['Pregnancies'] + 1) * (df_copy['BMI'] / 25) * (df_copy['BloodPressure'] / 120)
    
    return df_copy

# Apply custom feature function on a copy of the DataFrame
enhanced_df = add_custom_features(full_imputed_df)

# Create an EntitySet
es = ft.EntitySet(id='Diabetes_Data')

# Define variable types including new custom features
es.add_dataframe(
    dataframe_name='diabetes',
    dataframe=enhanced_df,
    index='index',
    logical_types={
        'Glucose': 'Double',
        'BloodPressure': 'Double',
        'SkinThickness': 'Double',
        'Insulin': 'Double',
        'BMI': 'Double',
        'DiabetesPedigreeFunction': 'Double',
        'Age': 'Double',
        'Outcome': 'Boolean',
        'BMI_Age': 'Double',
        'Preg_Age': 'Double',
        'Insulin_Glucose_Ratio': 'Double',
        'Skin_BMI_Ratio': 'Double',
        'Glucose_Squared': 'Double',
        'BMI_Squared': 'Double',
        'glucose_insulin_ratio': 'Double',
        'bmi_skinthickness_product': 'Double',
        'age_adjusted_risk': 'Double',
        'pregnancy_health_impact': 'Double'
    }
)

# Generate new features using deep feature synthesis
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='diabetes',
    trans_primitives=['add_numeric', 'multiply_numeric', 'divide_numeric'],
    agg_primitives=[],
    max_depth=1
)

# Display the head of the generated feature matrix
print(feature_matrix.head())

X = feature_matrix.drop('Outcome', axis=1)
y = feature_matrix['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Replace infinite values with NaNs
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)



# If you're specifically working with a training set:
print("Class distribution in the training dataset (y_train):")
print(y_train.value_counts())
print("\nPercentage of each class in the training dataset (y_train):")
print(y_train.value_counts(normalize=True) * 100)


# Create and fit the estimator
# Setup Stratified K-Fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Parameter grid for Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                           scoring='roc_auc', n_jobs=-1, cv=kfold, verbose=1)
grid_search.fit(X_train, y_train)

# Best estimator after grid search
best_classifier = grid_search.best_estimator_



# Cross-validation results
cv_results = cross_val_score(best_classifier, X_train, y_train, cv=kfold, scoring='roc_auc')
print("Average ROC AUC after tuning:", np.mean(cv_results))

# Fit the best classifier to the training data
best_classifier.fit(X_train, y_train)

# Get feature importances directly from the tuned classifier
feature_importances = best_classifier.feature_importances_

# Additional outputs for verification
print("Best parameters found:", grid_search.best_params_)


# Rank features by importance
importance_indices = np.argsort(feature_importances)[::-1]
sorted_features = X_train.columns[importance_indices]
sorted_importances = feature_importances[importance_indices]

# Initialize list to store AUC scores
auc_scores = []
n_features_list = range(1, len(sorted_features) + 1)

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

for n in n_features_list:
    selected_features = sorted_features[:n]
    # Use stratified cross-validation to evaluate performance
    cv_scores = cross_val_score(best_classifier, X_train[selected_features], y_train, cv=stratified_kfold, scoring='roc_auc')
    mean_auc_score = np.mean(cv_scores)
    auc_scores.append(mean_auc_score)
    print(f"Mean ROC-AUC with top {n} features: {mean_auc_score}")

# select optimal features
max_auc_score = max(auc_scores)
print(max_auc_score)
optimal_features = n_features_list[auc_scores.index(max_auc_score)]
print(f"Optimal number of features: {optimal_features} with ROC-AUC: {max_auc_score}")


# Save the AUC scores and optimal features
results = {
    'auc_scores': auc_scores,
    'optimal_features': optimal_features,
    'max_auc_score': max_auc_score
}

# subset to get dataframe
optimal_feature_names = sorted_features[:optimal_features].tolist()
optimal_feature_names.append('Outcome')   
print(optimal_feature_names)

# Subsetting training and testing data with optimal features
if 'Outcome' in optimal_feature_names:
    optimal_feature_names.remove('Outcome')

# Subsetting training and testing data with optimal features
X_train_optimal = X_train[optimal_feature_names]
X_test_optimal = X_test[optimal_feature_names]

print("Number of optimal features:", len(optimal_feature_names))
print("Optimal feature names:", optimal_feature_names)

def visualize_distributions(data, title):
    num_cols = 3  # Number of columns in subplot
    num_rows = (len(data.columns) + num_cols - 1) // num_cols  # Calculate the necessary number of rows
    plt.figure(figsize=(num_cols * 5, num_rows * 4))  # Dynamically size the figure based on number of features
    for i, feature in enumerate(data.columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(data[feature], kde=True)
        plt.title(feature)
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.savefig(base_path / f'{title.replace(" ", "_").lower()}_distributions.png')
    plt.close()

# Calling the function for both train and test datasets
visualize_distributions(X_train_optimal, "Training Data Distributions")
visualize_distributions(X_test_optimal, "Testing Data Distributions")

# Checking if there are any missing values in X_train_optimal
missing_in_train = X_train_optimal.isna().any().any()
print(f"Are there missing values in X_train_optimal? {missing_in_train}")

# Checking if there are any missing values in X_test_optimal
missing_in_test = X_test_optimal.isna().any().any()
print(f"Are there missing values in X_test_optimal? {missing_in_test}")

# medical data equal to 0 didn't have biological meaning
X_train_optimal.replace(0, np.nan, inplace=True)
X_test_optimal.replace(0, np.nan, inplace=True)

# Define imputation function
def impute_data(train, test=None, strategy='mean', n_neighbors=5):
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    elif strategy == 'mice':
        imputer = IterativeImputer(random_state=1234)

    # Fit on the training data and transform 
    train_imputed = imputer.fit_transform(train)
    if test is not None:
        test_imputed = imputer.transform(test)
    else:
        test_imputed = None
    return train_imputed, test_imputed

# Define the stratified K-Fold and scoring
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Evaluate imputation methods
def evaluate_imputation_methods(X_train_optimal, y_train, strategies, cv_strategy, scorer):
    imputation_performance = {}
    classifier = RandomForestClassifier(random_state=1234)

    for strategy in strategies:
        X_train_imputed, _ = impute_data(X_train_optimal, strategy=strategy)
        scores = cross_val_score(classifier, X_train_imputed, y_train, cv=cv_strategy, scoring=scorer)
        imputation_performance[strategy] = np.mean(scores)
    return imputation_performance

# Main execution block
strategies = ['mean', 'median', 'most_frequent', 'knn', 'mice']
roc_auc_scorer = make_scorer(roc_auc_score, needs_threshold=True)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

performance_roc_auc = evaluate_imputation_methods(X_train_optimal, y_train, 
                                                  strategies, 
                                                  cv_strategy,
                                                  roc_auc_scorer)

# Display the performance results
print("\nROC-AUC Scores by Imputation Method:")
print(performance_roc_auc)

# Find the best method based on ROC-AUC score
best_roc_auc_method = max(performance_roc_auc, key=performance_roc_auc.get)

# Impute using the best ROC-AUC method
X_train_final, X_test_final = impute_data(X_train_optimal, X_test_optimal, 
                                          strategy=best_roc_auc_method)  

X_train_final_df = pd.DataFrame(X_train_final, columns=optimal_feature_names)
X_test_final_df = pd.DataFrame(X_test_final, columns=optimal_feature_names)

# Now you can use the to_excel method and the columns will have proper headers
X_train_final_df.to_excel('X_train_final.xlsx', index=False)
X_test_final_df.to_excel('X_test_final.xlsx', index=False)

print("Data has been successfully imputed and saved to Excel files.")

#############

# Define models dictionary
models = {
    'Naive_Bayes': GaussianNB(),
    'Logistic_Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'Random_Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Neural_Network': MLPClassifier()
}

# Define scalers dictionary
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'minmax': MinMaxScaler()
}

# Define hyperparameters for each model
param_grid = {
    'Naive_Bayes': {
        'classifier__var_smoothing': np.logspace(0, -9, num=100)
    },
    'Logistic_Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    },
    'Random_Forest': {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__max_depth': [None, 10, 20, 30, 50],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0],
        'classifier__algorithm': ['SAMME', 'SAMME.R']
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 6, 9, 12],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0]
    },
    'Neural_Network': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100)],
        'classifier__activation': ['relu', 'tanh', 'logistic'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive'],
        'classifier__early_stopping': [True],
        'classifier__validation_fraction': [0.1]
    }
}

# Setting up Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# Function to run grid search
def run_grid_search(model_name, model, param_grid, scaler_name='standard'):
    print(f"Running grid search for {model_name} with {scaler_name}")
    scaler = scalers[scaler_name]
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid.fit(X_train_final, y_train)
    joblib.dump(grid, f'{model_name}_{scaler_name}_grid.pkl')  # Save the fitted grid

# Function to load results and evaluate
def load_results_and_evaluate(model_name, scaler_name='standard'):
    grid = joblib.load(f'{model_name}_{scaler_name}_grid.pkl')
    y_pred = grid.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, grid.predict_proba(X_test_final)[:, 1])

    return {
        'Model': model_name,
        'Scaler': scaler_name,
        'Best Score (ROC-AUC)': grid.best_score_,
        'Test Accuracy': accuracy,
        'Test F1 Score': f1,
        'Test ROC-AUC Score': roc_auc,
        'Best Params': grid.best_params_
    }

# Run grid search for each model
for model_name, model in models.items():
    run_grid_search(model_name, model, param_grid[model_name])

# Load and compile results
results = []
for model_name in models.keys():
    result = load_results_and_evaluate(model_name)
    results.append(result)

# Display results
for result in results:
    print(result)

# Function to get the best estimator from grid results
def get_best_estimator(model_name, scaler_name='standard'):
    grid = joblib.load(f'{model_name}_{scaler_name}_grid.pkl')
    return grid.best_estimator_

# Collect best models
best_estimators = [
    get_best_estimator(model_name)
    for model_name in models.keys()
]

# Meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Stacked classifier
stacked_model = StackingClassifier(
    estimators=[(name, estimator) for name, estimator in zip(models.keys(), best_estimators)],
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=True
)

# Train the stacked model
stacked_model.fit(X_train_final, y_train)

# Evaluate the stacked model
y_pred_stack = stacked_model.predict(X_test_final)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack, average='binary')
roc_auc_stack = roc_auc_score(y_test, stacked_model.predict_proba(X_test_final)[:, 1])

# Display results
stack_results = {
    'Test Accuracy': accuracy_stack,
    'Test F1 Score': f1_stack,
    'Test ROC-AUC Score': roc_auc_stack
}

print("Stacked Model Performance:", stack_results)



# # End timing
# end_time = time.time()

# # Calculate the duration in seconds
# duration_seconds = end_time - start_time

# # Convert seconds to minutes
# duration_minutes = duration_seconds / 60

# # Print the duration
# print(f"The code ran for {duration_minutes:.2f} minutes")
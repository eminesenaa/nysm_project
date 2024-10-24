# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Set the style for seaborn
sns.set(style="whitegrid")

# Load the dataset
url = "titanic_dataset.csv"
data = pd.read_csv(url)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Examine the data
print(data.head())
print(data.info())
print(data.describe())

# Data exploration: Survival rate by gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Sex', data=data, palette='coolwarm')
plt.title('Survival Count by Gender')
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Age'].dropna(), bins=30, kde=True, color='blue')
plt.title('Age Distribution of Passengers')
plt.show()

# Clean missing data
data['Age'].fillna(data['Age'].mean(), inplace=True)  # Fill Age with mean
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Fill Embarked with mode
data.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)  # Drop Cabin due to many missing values, Name and Ticket are not meaningful for the model

# Convert categorical variables for gender and Embarked to numerical values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Separate features and target variable
X = data.drop(columns=['Survived'])  # Features
y = data['Survived']  # Target (Survival status)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use GridSearchCV for hyperparameter tuning of Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Test the model
y_pred = best_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# ROC-AUC score
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Visualize feature importances
feature_importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

# Load dataset
df = pd.read_csv('malicious_bot_twitter_data.csv')

# Data cleaning

# Check for missing values
print("Missing Values:\n", df.isnull().sum())
# Fill missing values with mean


# Drop irrelevant columns for EDA and modeling
df_cleaned = df.drop(['User ID', 'Username', 'Tweet', 'Location', 'Created At', 'Hashtags', 'Post url', 'Tweet Link'], axis=1)
#
# # Fill missing values (if any) or handle them based on your data
# df_cleaned['Follower Count'].fillna(df_cleaned['Follower Count'].mean(), inplace=True)

# EDA
# Summary statistics
print("Summary Statistics:\n", df_cleaned.describe())

# Correlation matrix
corr_matrix = df_cleaned.corr()
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlational Matrix")
plt.show()
#
# Pairplot for selected features
sns.pairplot(df_cleaned[['Retweet Count', 'Mention Count', 'Follower Count', 'Bot Label', 'Sentiment']], hue='Bot Label', markers=["o", "s"])
plt.title("Pairplot of Selected Features")
plt.show()

# Data Preprocessing
print(df_cleaned.columns)

# Split the data into features (X) and target variable (y)
X = df_cleaned.drop('Bot Label', axis=1)
y = df_cleaned['Bot Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Prediction
y_pred = rf_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Obtain predicted probabilities for the positive class (bot)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Optimal Threshold:", optimal_threshold)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the model
joblib.dump(rf_model, 'bot_detection_model_3.pkl')
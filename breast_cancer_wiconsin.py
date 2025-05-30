# Choose a binary classification dataset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, classification_report, roc_curve

df = pd.read_csv("data.csv")  
print(df.head())
print(df.columns)

df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])  # M = 1, B = 0

if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop(['Unnamed: 32'], axis=1, inplace=True)

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# rain/test split and standardize features.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Fit Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# .Evaluate with confusion matrix, precision, reca l, ROC-AUC
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Tune threshold and explain sigmoid function
threshold = 0.3
y_pred_thresh = (y_proba >= threshold).astype(int)

print(f"\nConfusion Matrix at Threshold {threshold}:\n", confusion_matrix(y_test, y_pred_thresh))

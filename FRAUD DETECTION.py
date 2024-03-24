import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("creditcard.csv")
print(df.info())
print(df.head())
print(df.isnull().sum())
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Time', 'Amount'], axis=1)
fraud_count = df['Class'].value_counts()[1]
genuine_count = df['Class'].value_counts()[0]
print("Fraudulent Transactions:", fraud_count)
print("Genuine Transactions:", genuine_count)
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(df[df['Class'] == 1]['NormalizedAmount'], bins=20, color='red', alpha=0.5, label='Fraudulent')
plt.hist(df[df['Class'] == 0]['NormalizedAmount'], bins=20, color='blue', alpha=0.5, label='Genuine')
plt.title('Transaction Amount Distribution')
plt.xlabel('Normalized Amount')
plt.ylabel('Frequency')
plt.legend()
plt.show()

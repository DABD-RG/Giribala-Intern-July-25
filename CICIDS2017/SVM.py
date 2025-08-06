# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 1: Load and combine CSV files
import pandas as pd
import numpy as np
import os
import glob

data_dir = '/content/drive/MyDrive/CICIDS2017'

all_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)

if not all_files:
    raise FileNotFoundError(f"No CSV files found in: {data_dir}")

print(f"Found {len(all_files)} CSV files.")
df_list = []
for file in all_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file, low_memory=False)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
print(f"Combined dataset shape: {full_df.shape}")

# STEP 2: Data Cleaning and Preprocessing
full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]
full_df.dropna(axis=1, how='all', inplace=True)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)
full_df.columns = full_df.columns.str.strip()

if 'Label' not in full_df.columns:
    raise ValueError("Label column not found!")

full_df['Label'] = full_df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)

non_numeric = full_df.select_dtypes(include=['object']).columns.tolist()
if 'Label' in non_numeric:
    non_numeric.remove('Label')
full_df.drop(columns=non_numeric, inplace=True)

print(f"Cleaned dataset shape: {full_df.shape}")

# STEP 3: Feature Scaling and Train/Test Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = full_df.drop('Label', axis=1)
y = full_df['Label']

print("Target class distribution:")
print(y.value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# STEP 4: Train and Calibrate LinearSVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import time

svm_model = LinearSVC(max_iter=10000)
svm_model.fit(X_train, y_train)

calibrated_svm = CalibratedClassifierCV(svm_model, method='sigmoid', cv='prefit')

start_time = time.time()
calibrated_svm.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Calibration time: {training_time:.2f} seconds")

y_pred = calibrated_svm.predict(X_test)
y_pred_proba = calibrated_svm.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# STEP 5: Plot Confusion Matrix & ROC Curve
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

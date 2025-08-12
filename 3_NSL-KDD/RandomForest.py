import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# STEP 1: Load and combine all the csv files
print("\nSTEP 1")

data_dir = '/content/drive/MyDrive/nsl-kdd'  # CHANGE THIS PATH if necessary
train_file = os.path.join(data_dir, 'KDDTrain+.txt')
test_file = os.path.join(data_dir, 'KDDTest+.txt')

column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_cat', 'difficulty_level'
]

print(f"Loading training data: {train_file}")
df_train = pd.read_csv(train_file, header=None, names=column_names, low_memory=False)

print(f"Loading testing data: {test_file}")
df_test = pd.read_csv(test_file, header=None, names=column_names, low_memory=False)

# Combine train and test for consistent preprocessing (important for categorical features)
full_df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Combined dataset shape: {full_df.shape}")
display(full_df.head())

# STEP 2: Data cleaning and pre-processing
print("\nSTEP 2")

# Drop 'difficulty_level' as it's not a feature for classification
if 'difficulty_level' in full_df.columns:
    full_df = full_df.drop('difficulty_level', axis=1)

# Handle potential NaNs or infs
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

# Label Encoding for categorical features
categorical_cols = full_df.select_dtypes(include=['object']).columns.tolist()
if 'attack_cat' in categorical_cols:
    categorical_cols.remove('attack_cat')

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col])
    label_encoders[col] = le

# Convert target column 'attack_cat' to binary (1 for attack, 0 for normal)
if 'attack_cat' not in full_df.columns:
    raise ValueError("Target column 'attack_cat' not found!")
full_df['attack_cat'] = full_df['attack_cat'].apply(lambda x: 0 if x.lower() == 'normal' else 1)

print(f"Shape after cleaning and encoding: {full_df.shape}")
print("\nTarget distribution:")
print(full_df['attack_cat'].value_counts())
display(full_df.head())

# STEP 3: Feature scaling & test-train split
print("\nSTEP 3")

# Split into features and target
X = full_df.drop('attack_cat', axis=1)
y = full_df['attack_cat']

# Scale features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

# Split back into training and testing sets based on original file sizes
X_train = X.iloc[:len(df_train)]
y_train = y.iloc[:len(df_train)]
X_test = X.iloc[len(df_train):]
y_test = y.iloc[len(df_train):]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# STEP 4: Training the Random Forest model
print("\nSTEP 4")

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict on the test set
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except ValueError:
    roc_auc = float('nan')

print(f"Training time: {training_time:.2f} seconds")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")

# STEP 5: Confusion matrix and specificity
print("\nSTEP 5")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Handle specificity calculation robustly
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
else:
    specificity = float('nan')
print(f"Specificity  : {specificity:.4f}")

# Optional: Plot ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

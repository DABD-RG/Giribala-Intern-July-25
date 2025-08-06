# training random forest model for CICIDS2017
from google.colab import drive
drive.mount('/content/drive')


# step 1 - load and combine all the csv files
print("\nSTEP 1")
import pandas as pd
import os
import glob

data_dir = '/content/drive/MyDrive/CICIDS2017'
os.listdir(data_dir)

all_files = glob.glob(os.path.join(data_dir, "*.csv"))
df_list = []
for file in all_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file, low_memory=False)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
# print(full_df)
full_df.head()


# step 2 - data cleaning and pre-processing
'''
1. first remove all the unnamed columns
2. drop all columns with NA values (NaN)
3. handle infinity and missing values - np.inf and -np.inf
4. strip the leading spaces from the column names (cols were ' Label')
5. check if target column exists (usually Label or Output - we are trying to predict)
6. convert label to binary
7. drop non-numeric columns
'''
print("\nSTEP 2")

import numpy as np

full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]
full_df.dropna(axis=1, how='all', inplace=True)

full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

print("Remaining columns:", full_df.columns.tolist())

full_df.columns = full_df.columns.str.strip()

if 'Label' not in full_df.columns:
    raise ValueError("Label column not found!")

# label is made binary
full_df['Label'] = full_df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

non_numeric = full_df.select_dtypes(include=['object']).columns.tolist()
if 'Label' in non_numeric:
    non_numeric.remove('Label')
full_df.drop(columns=non_numeric, inplace=True)

# final shape check
print(f"Shape after cleaning: {full_df.shape}")
full_df.head()


# step 3 - feature scalin & test-train split
'''
1. split into features and target
2. feature scaling
3. train test split
'''
print("\nSTEP 3")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = full_df.drop('Label', axis=1) # features
y = full_df['Label'] # target

print(y.value_counts()) # added to check if leaky or not

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# step 4 - training the model

print("\nSTEP 4")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# initialize and train
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

# predict
y_pred = rf_model.predict(X_test)

# metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Training time is {training_time:.2f} seconds")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1-Score  = {f1:.4f}")


# step 5 - confusion metrics

print("\nSTEP 5")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n=== ROC Metrics ===")
print(f"False Positive Rate (FPR): {fpr}")
print(f"True Positive Rate (TPR): {tpr}")
print(f"ROC AUC Score: {roc_auc:.4f}")

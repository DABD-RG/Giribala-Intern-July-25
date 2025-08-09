#-------------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

#-------------------------------------------------------------------------------
import pandas as pd

train_path = '/content/drive/MyDrive/UNSW/UNSW_NB15_training-set.parquet'
test_path = '/content/drive/MyDrive/UNSW/UNSW_NB15_testing-set.parquet'

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

#-------------------------------------------------------------------------------
full_df = pd.concat([df_train, df_test], ignore_index=True)
print("Combined shape:", full_df.shape)
print("Columns:", full_df.columns.tolist())
full_df.head()

#-------------------------------------------------------------------------------
import numpy as np

full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]

full_df.columns = full_df.columns.str.strip()
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

#print("Columns:", full_df.columns.tolist())

if 'label' in full_df.columns:
    full_df['label'] = full_df['label'].astype(int)
    target_col = 'label'
elif 'attack_cat' in full_df.columns:
    full_df['attack_cat'] = full_df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)
    target_col = 'attack_cat'
else:
    raise ValueError("No valid target column found.")

# Drop non-numeric columns explicitly, excluding the target column
non_numeric_cols = full_df.select_dtypes(include=['object', 'category']).columns.tolist()
if target_col in non_numeric_cols:
    non_numeric_cols.remove(target_col)
full_df.drop(columns=non_numeric_cols, inplace=True)


print("Final shape after cleaning:", full_df.shape)

#-------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# Features and target
X = full_df.drop(target_col, axis=1)
y = full_df[target_col]

print(y.value_counts())

# Re-create train and test sets after cleaning the full DataFrame
X_train = X.iloc[:len(df_train)]
y_train = y.iloc[:len(df_train)]

X_test = X.iloc[len(df_train):]
y_test = y.iloc[len(df_train):]


print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

#-------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Training time: {training_time:.2f} seconds")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Confusion matrix to calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# ROC AUC
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Specificity  : {specificity:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")

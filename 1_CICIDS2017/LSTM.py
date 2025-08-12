# STEP 0 -----------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# STEP 1 -----------------------------------------------------------------------
import pandas as pd
import os
import glob

data_dir = '/content/drive/MyDrive/CICIDS2017'
all_files = glob.glob(os.path.join(data_dir, "*.csv"))[:4 ]

df_list = []
for file in all_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file, low_memory=False)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)

# STEP 2 -----------------------------------------------------------------------
import numpy as np

full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]
full_df.dropna(axis=1, how='all', inplace=True)

full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

full_df.columns = full_df.columns.str.strip()

if 'Label' not in full_df.columns:
    raise ValueError("Label column not found!")

full_df['Label'] = full_df['Label'].apply(lambda x: 0 if 'BENIGN' in x.upper() else 1)

non_numeric = full_df.select_dtypes(include=['object']).columns.tolist()
if 'Label' in non_numeric:
    non_numeric.remove('Label')
full_df.drop(columns=non_numeric, inplace=True)

# STEP 3 -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = full_df.drop('Label', axis=1)
y = full_df['Label']

print("Class distribution:\n", y.value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 4 -----------------------------------------------------------------------
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# STEP 5 -----------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models

lstm_model = models.Sequential([
    layers.LSTM(64, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

# STEP 6 -----------------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred_prob = lstm_model.predict(X_test_lstm)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Specificity: {specificity:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Load UNSW data
train_path = '/content/drive/MyDrive/UNSW/UNSW_NB15_training-set.parquet'
test_path = '/content/drive/MyDrive/UNSW/UNSW_NB15_testing-set.parquet'

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

full_df = pd.concat([df_train, df_test], ignore_index=True)

# Data Cleaning and Preprocessing (from previous UNSW cells)
full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]
full_df.columns = full_df.columns.str.strip()
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

target_col = None
if 'label' in full_df.columns:
    full_df['label'] = full_df['label'].astype(int)
    target_col = 'label'
elif 'attack_cat' in full_df.columns:
    full_df['attack_cat'] = full_df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)
    target_col = 'attack_cat'
else:
    raise ValueError("No valid target column found.")

non_numeric_cols = full_df.select_dtypes(include=['object', 'category']).columns.tolist()
if target_col in non_numeric_cols:
    non_numeric_cols.remove(target_col)
full_df.drop(columns=non_numeric_cols, inplace=True)

# Split into features and target
X = full_df.drop(target_col, axis=1)
y = full_df[target_col]

# Re-create train and test sets after cleaning the full DataFrame
X_train = X.iloc[:len(df_train)]
y_train = y.iloc[:len(df_train)]
X_test = X.iloc[len(df_train):]
y_test = y.iloc[len(df_train):]

# Scaling (using MinMaxScaler as in the original CNN UNSW cell for consistency with NN models)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Reshape data for LSTM [samples, time steps, features]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = keras.Sequential([
    layers.LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(32),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train_lstm, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

from sklearn.metrics import confusion_matrix, roc_auc_score

y_pred_probs = model.predict(X_test_lstm)
y_pred_lstm = (y_pred_probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred_lstm)
precision = precision_score(y_test, y_pred_lstm)
recall = recall_score(y_test, y_pred_lstm)
f1 = f1_score(y_test, y_pred_lstm)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lstm).ravel()
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")

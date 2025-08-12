# Upgrade TensorFlow silently
!pip install --upgrade tensorflow > /dev/null 2>&1

# Mount Google Drive -----------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Load CSV files ---------------------------------------------------
import pandas as pd
import os
import glob
import numpy as np

data_dir = '/content/drive/MyDrive/CICIDS2017'
all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

df_list = []
for file in all_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file, low_memory=False)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)

# Clean and preprocess data ---------------------------------------
full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]
full_df.dropna(axis=1, how='all', inplace=True)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

full_df.columns = full_df.columns.str.strip()
if 'Label' not in full_df.columns:
    raise ValueError("Label column not found!")

# Encode 'Label' column -------------------------------------------
full_df['Label'] = full_df['Label'].apply(lambda x: 0 if 'BENIGN' in x.upper() else 1)

# Drop non-numeric columns except Label ---------------------------
non_numeric = full_df.select_dtypes(include=['object']).columns.tolist()
if 'Label' in non_numeric:
    non_numeric.remove('Label')
full_df.drop(columns=non_numeric, inplace=True)

# Convert to float32 to reduce memory
full_df = full_df.astype(np.float32)

print(f"Shape after cleaning: {full_df.shape}")
print(full_df['Label'].value_counts())

# Split and scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = full_df.drop('Label', axis=1)
y = full_df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Prepare Autoencoder inputs ----------------------------------------
from sklearn.utils import resample

X_train_auto = X_train[y_train == 0]

# Downsample to prevent OOM crash
if len(X_train_auto) > 100_000:
    X_train_auto = resample(X_train_auto, n_samples=100_000, random_state=42)

X_test_auto = X_test
y_test_auto = y_test

# Build Autoencoder -------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_dim = X_train_auto.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Train Autoencoder
from tensorflow.keras import backend as K
K.clear_session()

history = autoencoder.fit(
    X_train_auto, X_train_auto,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# Evaluate on test set 
reconstructions = autoencoder.predict(X_test_auto)
mse = np.mean(np.square(X_test_auto - reconstructions), axis=1)

# Set threshold at 95th percentile
threshold = np.percentile(mse, 95)
y_pred = (mse > threshold).astype(int)

# Print Metrics ------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

accuracy = accuracy_score(y_test_auto, y_pred)
precision = precision_score(y_test_auto, y_pred)
recall = recall_score(y_test_auto, y_pred)
f1 = f1_score(y_test_auto, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test_auto, y_pred).ravel()
specificity = tn / (tn + fp)

fpr, tpr, _ = roc_curve(y_test_auto, mse)
roc_auc = roc_auc_score(y_test_auto, mse)

print(f"Threshold (95th percentile): {threshold:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

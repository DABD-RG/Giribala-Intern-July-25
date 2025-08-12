# Install necessary packages silently
!pip install tensorflow keras --quiet

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------------------------------------
X_train_auto = X_train.copy()
X_test_auto = X_test.copy()

label_encoders = {}
for col in X_train_auto.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train_auto[col] = le.fit_transform(X_train_auto[col])
    X_test_auto[col] = le.transform(X_test_auto[col])
    label_encoders[col] = le

scaler = StandardScaler()
X_train_auto = scaler.fit_transform(X_train_auto)
X_test_auto = scaler.transform(X_test_auto)

X_train_normal = X_train_auto[y_train == 0]

# ------------------------------------------------------------------------------
input_dim = X_train_normal.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ------------------------------------------------------------------------------
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# ------------------------------------------------------------------------------
reconstructions = autoencoder.predict(X_test_auto)
mse = np.mean(np.square(X_test_auto - reconstructions), axis=1)

threshold = np.percentile(mse, 95)
y_pred_auto = (mse > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_auto)
precision = precision_score(y_test, y_pred_auto)
recall = recall_score(y_test, y_pred_auto)
f1 = f1_score(y_test, y_pred_auto)
roc_auc = roc_auc_score(y_test, mse)
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_auto).ravel()
specificity = tn / (tn + fp)

print(f"Threshold   : {threshold:.4f}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

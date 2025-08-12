import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

X_train_auto = X_train.copy()
X_test_auto = X_test.copy()

label_encoders = {}
for col in X_train_auto.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train_auto[col] = le.fit_transform(X_train_auto[col])
    X_test_auto[col] = le.transform(X_test_auto[col])
    label_encoders[col] = le

X_train_normal = X_train_auto[y_train == 0]

input_dim = X_train_normal.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

reconstructions = autoencoder.predict(X_test_auto)
mse = np.mean(np.square(X_test_auto - reconstructions), axis=1)

threshold = np.percentile(mse, 95)

y_pred_auto = (mse > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_auto)
precision = precision_score(y_test, y_pred_auto)
recall = recall_score(y_test, y_pred_auto)
f1 = f1_score(y_test, y_pred_auto)

print(f"Threshold   : {threshold:.4f}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-Score    : {f1:.4f}")

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

cm = confusion_matrix(y_test, y_pred_auto)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

roc_auc = roc_auc_score(y_test, mse)

print(f"Specificity : {specificity:.4f}")
print(f"ROC AUC     : {roc_auc:.4f}")

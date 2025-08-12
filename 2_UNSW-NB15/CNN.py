# Install necessary packages silently
!pip install tensorflow keras --quiet

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1-----------------------------------------------------------------------------
X_train_cnn = X_train.copy()
X_test_cnn = X_test.copy()

label_encoders = {}
for col in X_train_cnn.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train_cnn[col] = le.fit_transform(X_train_cnn[col])
    X_test_cnn[col] = le.transform(X_test_cnn[col])
    label_encoders[col] = le

# 2-----------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_cnn = scaler.fit_transform(X_train_cnn)
X_test_cnn = scaler.transform(X_test_cnn)

# 3-----------------------------------------------------------------------------
X_train_cnn = X_train_cnn.reshape((X_train_cnn.shape[0], X_train_cnn.shape[1], 1))
X_test_cnn = X_test_cnn.reshape((X_test_cnn.shape[0], X_test_cnn.shape[1], 1))

# 4-----------------------------------------------------------------------------
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5-----------------------------------------------------------------------------
history = model.fit(
    X_train_cnn, y_train,
    epochs=15,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

# 6-----------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, roc_auc_score

y_pred_probs = model.predict(X_test_cnn)
y_pred_cnn = (y_pred_probs > 0.5).astype(int)
y_pred_cnn = y_pred_cnn.flatten()

accuracy = accuracy_score(y_test, y_pred_cnn)
precision = precision_score(y_test, y_pred_cnn)
recall = recall_score(y_test, y_pred_cnn)
f1 = f1_score(y_test, y_pred_cnn)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_cnn).ravel()
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")

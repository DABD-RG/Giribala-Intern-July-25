# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 3: Load Data
train_path = '/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt'
test_path = '/content/drive/MyDrive/nsl-kdd/KDDTest+.txt'

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty_level"
]

df_train = pd.read_csv(train_path, names=columns)
df_test = pd.read_csv(test_path, names=columns)

# Drop difficulty_level column
df_train.drop('difficulty_level', axis=1, inplace=True)
df_test.drop('difficulty_level', axis=1, inplace=True)

# Convert 'class' to binary
df_train['class'] = df_train['class'].apply(lambda x: 0 if x.lower() == 'normal' else 1)
df_test['class'] = df_test['class'].apply(lambda x: 0 if x.lower() == 'normal' else 1)

# Combine for consistent encoding
X_train = df_train.drop('class', axis=1)
y_train = df_train['class']
X_test = df_test.drop('class', axis=1)
y_test = df_test['class']

combined = pd.concat([X_train, X_test])
cat_cols = combined.select_dtypes(include='object').columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le

X_train = combined.iloc[:len(X_train)]
X_test = combined.iloc[len(X_train):]

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# Step 4: CNN Model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Train
history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=1)

# Step 6: Predict & Evaluate
y_pred_prob = model.predict(X_test_cnn).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"ROC AUC     : {roc_auc:.4f}")

# Step 7: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CNN Confusion Matrix')
plt.show()

# Step 8: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN ROC Curve")
plt.legend()
plt.grid()
plt.show()

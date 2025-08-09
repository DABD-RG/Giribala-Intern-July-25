import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# Ensure your data is in DataFrame format
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Make copies for processing
X_train_svm = X_train.copy()
X_test_svm = X_test.copy()

# Label encoding for categorical (object) columns
label_encoders = {}
object_cols = X_train_svm.select_dtypes(include=['object']).columns

for col in object_cols:
    le = LabelEncoder()
    X_train_svm[col] = le.fit_transform(X_train_svm[col])
    X_test_svm[col] = le.transform(X_test_svm[col])
    label_encoders[col] = le

# Initialize and train the LinearSVC model
svm_model = LinearSVC(max_iter=10000)

start_time = time.time()
svm_model.fit(X_train_svm, y_train)
training_time = time.time() - start_time

# Make predictions
y_pred_svm = svm_model.predict(X_test_svm)

# Compute classification metrics
accuracy = accuracy_score(y_test, y_pred_svm)
precision = precision_score(y_test, y_pred_svm)
recall = recall_score(y_test, y_pred_svm)
f1 = f1_score(y_test, y_pred_svm)

# Compute specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()
specificity = tn / (tn + fp)

# Compute ROC AUC using decision function
try:
    y_scores = svm_model.decision_function(X_test_svm)
    roc_auc = roc_auc_score(y_test, y_scores)
except:
    roc_auc = None

# Print results
print(f"Training time: {training_time:.2f} seconds")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}" if roc_auc is not None else "ROC AUC      : N/A")

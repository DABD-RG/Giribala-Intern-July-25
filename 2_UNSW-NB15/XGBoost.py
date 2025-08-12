import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import time

X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

#-------------------------------------------------------------------------------
label_encoders = {}
for col in X_train_xgb.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train_xgb[col] = le.fit_transform(X_train_xgb[col])
    X_test_xgb[col] = le.transform(X_test_xgb[col])
    label_encoders[col] = le

if y_train.dtype == 'object' or y_train.dtype.name == 'category':
    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    y_test = le_y.transform(y_test)

dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
dtest = xgb.DMatrix(X_test_xgb, label=y_test)

#-------------------------------------------------------------------------------
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.3,
    'seed': 42
}

#-------------------------------------------------------------------------------
start_time = time.time()
xgb_model = xgb.train(params, dtrain, num_boost_round=100)
training_time = time.time() - start_time

y_pred_prob = xgb_model.predict(dtest)
y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Training time: {training_time:.2f} seconds")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")

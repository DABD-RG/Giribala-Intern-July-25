!pip install pandas scikit-learn transformers torch --quiet

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import os
from sklearn.model_selection import train_test_split

folder_path = '/content/drive/MyDrive/CICIDS2017'
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
# step is same as all other models
df = pd.concat((pd.read_csv(file, low_memory=False) for file in all_files), ignore_index=True)
df.dropna(inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

df['Label'] = df['Label'].str.strip()
df['target'] = df['Label'].apply(lambda x: 0 if x.lower() in ['benign', 'normal'] else 1)

text_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df['text'] = df[text_columns].astype(str).agg(' '.join, axis=1)

# final data will be this
texts = df['text'].tolist()
labels = df['target'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

from transformers import BertForSequenceClassification, AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(2):  # You can increase epochs
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

model.eval()
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]

        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp)

print(f"Accuracy    : {acc:.4f}")
print(f"Precision   : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"Specificity : {spec:.4f}")
print(f"ROC AUC     : {auc:.4f}")

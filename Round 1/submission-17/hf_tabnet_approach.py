import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, TrainingArguments, Trainer, EarlyStoppingCallback
import warnings
warnings.filterwarnings('ignore')
print('Loading data...')
df = pd.read_csv('C:\\Projects\\dataverse-nsut\\data\\main.csv')
test_df = pd.read_csv('C:\\Projects\\dataverse-nsut\\data\\test.csv')
print(f'Train shape: {df.shape}, Test shape: {test_df.shape}')
test_ids = test_df['sha256'].values
df = df.drop(columns=['sha256'])
print('Handling missing values...')
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
print('Clipping outliers...')
for col in df.select_dtypes(include=['float64']).columns:
    lower, upper = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lower, upper)
y = df['Label'].values
feature_columns = [col for col in df.columns if col != 'Label']
X = df[feature_columns]
print(f'Features: {len(feature_columns)}')
print(f'Target distribution: {np.bincount(y)}')
print('Preprocessing features...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape}, Val: {X_val.shape}')

class TabularDataset(Dataset):

    def __init__(self, features, labels=None):
        self.features = features.values if hasattr(features, 'values') else features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {'numerical_features': torch.tensor(self.features[idx], dtype=torch.float32)}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
train_dataset = TabularDataset(X_train, y_train)
val_dataset = TabularDataset(X_val, y_val)
print('Setting up custom tabular model...')

class TabularTransformer(nn.Module):

    def __init__(self, num_features, hidden_size=256, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(num_features, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size // 2, 2))

    def forward(self, numerical_features, labels=None):
        x = self.input_projection(numerical_features)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}
model = TabularTransformer(num_features=len(feature_columns), hidden_size=256, num_layers=6, num_heads=8, dropout=0.1)
print(f'Model created with {sum((p.numel() for p in model.parameters()))} parameters')
training_args = TrainingArguments(output_dir='submission-17/tabnet_output', num_train_epochs=20, per_device_train_batch_size=32, per_device_eval_batch_size=64, warmup_steps=100, weight_decay=0.01, learning_rate=0.0002, logging_dir='submission-17/logs', logging_steps=50, eval_strategy='epoch', save_strategy='epoch', save_total_limit=3, load_best_model_at_end=True, metric_for_best_model='eval_auc', greater_is_better=True, report_to=None, seed=42, dataloader_drop_last=False)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_binary = np.argmax(predictions, axis=1)
    pred_proba = torch.softmax(torch.tensor(predictions, dtype=torch.float32), dim=-1).numpy()
    accuracy = accuracy_score(labels, pred_binary)
    if pred_proba.shape[1] == 2:
        auc = roc_auc_score(labels, pred_proba[:, 1])
    else:
        auc = 0.5
    f1 = f1_score(labels, pred_binary, average='weighted')
    return {'accuracy': accuracy, 'auc': auc, 'f1': f1}

class TabularTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        loss = outputs.get('loss')
        return (loss, outputs) if return_outputs else loss
trainer = TabularTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
print('Starting training...')
trainer.train()
print('Evaluating on validation set...')
eval_results = trainer.evaluate()
print('Validation Results:')
for key, value in eval_results.items():
    print(f'  {key}: {value:.4f}')
print('Making test predictions...')
test_df_processed = test_df.drop(columns=['sha256']).copy()
for col in feature_columns:
    if col in test_df_processed.columns:
        if test_df_processed[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            test_df_processed[col] = test_df_processed[col].fillna(median_val)
for col in feature_columns:
    if col in test_df_processed.columns and col in df.columns:
        lower, upper = df[col].quantile([0.01, 0.99])
        test_df_processed[col] = test_df_processed[col].clip(lower, upper)
X_test_scaled = scaler.transform(test_df_processed[feature_columns])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
test_dataset = TabularDataset(X_test_scaled)
print('Generating test predictions...')
model.eval()
test_predictions = []
with torch.no_grad():
    for batch in test_dataset:
        numerical_features = batch['numerical_features'].unsqueeze(0)
        outputs = model(numerical_features)
        logits = outputs['logits']
        test_predictions.append(logits.cpu().numpy())
test_predictions = np.concatenate(test_predictions, axis=0)
test_probs = torch.softmax(torch.tensor(test_predictions), dim=-1).numpy()
test_preds_binary = np.argmax(test_probs, axis=1)
test_preds_proba = test_probs[:, 1]
print('Creating submission file...')
submission = pd.DataFrame({'sha256': test_ids, 'Label': test_preds_binary})
submission.to_csv('submission-17/B7JYI.csv', index=False)
print(f'Submission saved with {len(submission)} predictions')
print(f'Prediction distribution: {np.bincount(submission['Label'])}')
prob_submission = pd.DataFrame({'sha256': test_ids, 'Label': test_preds_proba})
prob_submission.to_csv('submission-17/probabilities.csv', index=False)
print('Probabilities saved for analysis')
print('Saving model and scaler...')
torch.save(model.state_dict(), 'submission-17/tabular_transformer.pt')
import joblib
joblib.dump(scaler, 'submission-17/scaler.pkl')
print('\nCustom Tabular Transformer training and prediction completed!')
print('Files saved:')
print('- submission-17/tabular_transformer.pt (model weights)')
print('- submission-17/scaler.pkl (feature scaler)')
print('- submission-17/B7JYI.csv (submission file)')
print('- submission-17/probabilities.csv (prediction probabilities)')
print('- submission-17/tabnet_output/ (training outputs)')
print('- submission-17/logs/ (training logs)')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib
import os
print('Loading data...')
df = pd.read_csv('data/main.csv')
test_df = pd.read_csv('data/test.csv')
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
X = df.drop(columns=['Label']).values
print('Scaling features...')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class TabDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=512, shuffle=True)
val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=1024, shuffle=False)

class TabTransformer(nn.Module):

    def __init__(self, num_features, hidden_dim=256, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return torch.sigmoid(self.head(x))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
model = TabTransformer(num_features=X_train.shape[1]).to(device)
print(f'Model created with {sum((p.numel() for p in model.parameters()))} parameters')
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-05)
epochs = 15
print('Starting training...')
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for Xb, yb in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        Xb, yb = (Xb.to(device), yb.to(device).unsqueeze(1))
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    preds_val = []
    y_true = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            preds = model(Xb).cpu().numpy()
            preds_val.extend(preds)
            y_true.extend(yb.numpy())
    preds_bin = (np.array(preds_val) > 0.5).astype(int)
    acc = accuracy_score(y_true, preds_bin)
    auc = roc_auc_score(y_true, preds_val)
    f1 = f1_score(y_true, preds_bin)
    print(f'Val Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}')
print('Saving model and scaler...')
torch.save(model.state_dict(), 'submission-16/tabtransformer_malware.pt')
joblib.dump(scaler, 'submission-16/scaler.pkl')
print('Making test predictions...')
test_df_processed = test_df.drop(columns=['sha256']).copy()
for col in test_df_processed.columns:
    if test_df_processed[col].dtype in ['float64', 'int64']:
        median_val = df[col].median()
        test_df_processed[col] = test_df_processed[col].fillna(median_val)
for col in test_df_processed.select_dtypes(include=['float64']).columns:
    if col in df.columns:
        lower, upper = df[col].quantile([0.01, 0.99])
        test_df_processed[col] = test_df_processed[col].clip(lower, upper)
X_test = scaler.transform(test_df_processed.values)
test_dataset = TabDataset(X_test, np.zeros(len(X_test)))
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
model.eval()
test_predictions = []
with torch.no_grad():
    for Xb, _ in tqdm(test_loader, desc='Predicting on test set'):
        Xb = Xb.to(device)
        preds = model(Xb).cpu().numpy()
        test_predictions.extend(preds)
test_predictions = np.array(test_predictions).flatten()
print('Creating submission file...')
submission = pd.DataFrame({'sha256': test_ids, 'Label': (test_predictions > 0.5).astype(int)})
submission.to_csv('submission-16/B7JYI.csv', index=False)
print(f'Submission saved with {len(submission)} predictions')
print(f'Prediction distribution: {np.bincount(submission['Label'])}')
prob_submission = pd.DataFrame({'sha256': test_ids, 'Label': test_predictions})
prob_submission.to_csv('submission-16/probabilities.csv', index=False)
print('Probabilities saved for analysis')
print('\nTabTransformer training and prediction completed!')
print('Files saved:')
print('- submission-16/tabtransformer_malware.pt (model weights)')
print('- submission-16/scaler.pkl (feature scaler)')
print('- submission-16/B7JYI.csv (submission file)')
print('- submission-16/probabilities.csv (prediction probabilities)')

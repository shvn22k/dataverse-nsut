import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
try:
    import rtdl
    print(' RTDL library available')
except ImportError:
    print(' RTDL library not found. Installing...')
    os.system('pip install rtdl')
    import rtdl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
print('=' * 80)
print('FT-TRANSFORMER APPROACH - DEEP LEARNING FOR TABULAR DATA')
print('=' * 80)
SEED = 42
N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
print('\n[STEP 1] Loading and preprocessing data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
df_proc = df.copy()
test_proc = test_df.copy()
numeric_cols = df_proc.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_proc[col].isnull().sum() > 0:
        median_val = df_proc[col].median()
        df_proc[col].fillna(median_val, inplace=True)
        if col in test_proc.columns:
            test_proc[col].fillna(median_val, inplace=True)
constant_features = [col for col in numeric_cols if col not in ['Label', 'sha256'] and df_proc[col].nunique() == 1]
if constant_features:
    df_proc.drop(columns=constant_features, inplace=True)
    test_proc.drop(columns=[c for c in constant_features if c in test_proc.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
FEATURES = [col for col in df_proc.columns if col not in ['Label', 'sha256']]
X = df_proc[FEATURES].values
y = df_proc['Label'].values
X_test = test_proc[FEATURES].values
print(f'Features: {X.shape[1]}')
print('\n[STEP 2] Scaling features...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print(f'Scaled features: {X_scaled.shape}')
print('\n[STEP 3] Cross-validation with FT-Transformer...')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_scores = []
oof_predictions = np.zeros(len(X_scaled))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f'\nFold {fold}/{N_FOLDS}')
    X_train, X_val = (X_scaled[train_idx], X_scaled[val_idx])
    y_train, y_val = (y[train_idx], y[val_idx])
    model = rtdl.FTTransformer.make_baseline(n_num_features=X_train.shape[1], cat_cardinalities=[], last_layer_query_idx=[-1], d_out=1, d_token=192, n_blocks=3, attention_dropout=0.1, ffn_d_hidden=192, ffn_dropout=0.1, residual_dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(EPOCHS):
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        preds = model(X_tensor, None).squeeze()
        loss = loss_fn(preds, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                val_preds = torch.sigmoid(model(X_val_tensor, None).squeeze())
                val_auc = roc_auc_score(y_val, val_preds.cpu().numpy())
                print(f'  Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}')
            model.train()
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        val_preds = torch.sigmoid(model(X_val_tensor, None).squeeze())
        val_auc = roc_auc_score(y_val, val_preds.cpu().numpy())
        oof_predictions[val_idx] = val_preds.cpu().numpy()
        cv_scores.append(val_auc)
        print(f'  Final Val AUC: {val_auc:.4f}')
mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
oof_auc = roc_auc_score(y, oof_predictions)
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 80)
print('FT-TRANSFORMER CROSS-VALIDATION RESULTS')
print('=' * 80)
print(f'Mean CV AUC: {mean_cv:.6f} (+/- {std_cv:.6f})')
print(f'OOF AUC: {oof_auc:.6f}')
print(f'OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'\nBaseline (99.57%): 0.995700')
print(f'Difference: {oof_auc - 0.9957:+.6f}')
print('=' * 80)
print('\n[STEP 4] Training final FT-Transformer on full data...')
final_model = rtdl.FTTransformer.make_baseline(n_num_features=X_scaled.shape[1], cat_cardinalities=[], last_layer_query_idx=[-1], d_out=1, d_token=192, n_blocks=3, attention_dropout=0.1, ffn_d_hidden=192, ffn_dropout=0.1, residual_dropout=0.1).to(DEVICE)
final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
final_loss_fn = nn.BCEWithLogitsLoss()
final_model.train()
for epoch in range(EPOCHS):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    preds = final_model(X_tensor, None).squeeze()
    loss = final_loss_fn(preds, y_tensor)
    final_optimizer.zero_grad()
    loss.backward()
    final_optimizer.step()
    if epoch % 5 == 0:
        print(f'  Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item():.4f}')
print(' Final model trained')
print('\n[STEP 5] Generating predictions...')
final_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    test_preds = torch.sigmoid(final_model(X_test_tensor, None).squeeze())
    test_preds = test_preds.cpu().numpy()
test_binary = (test_preds > 0.5).astype(int)
submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': test_binary})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_ft_transformer_{timestamp}.csv'
submission.to_csv(filename, index=False)
print('\n[STEP 6] Saving results...')
results = {'timestamp': timestamp, 'approach': 'ft_transformer', 'device': str(DEVICE), 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'cv_auc_mean': float(mean_cv), 'cv_auc_std': float(std_cv), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc)}
with open(f'ft_transformer_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
torch.save(final_model.state_dict(), f'ft_transformer_model_{timestamp}.pth')
torch.save(scaler, f'scaler_{timestamp}.pkl')
print('\n' + '=' * 80)
print('FT-TRANSFORMER APPROACH COMPLETE!')
print('=' * 80)
print(f'\nSubmission: {filename}')
print(f'Results: ft_transformer_results_{timestamp}.json')
print(f'Model: ft_transformer_model_{timestamp}.pth')
print(f'Scaler: scaler_{timestamp}.pkl')
print(f'\nSamples: {len(submission):,}')
print(f'  Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'  Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nFT-Transformer Performance:')
print(f'  CV AUC: {oof_auc:.6f}')
print(f'  Accuracy: {oof_acc * 100:.2f}%')
print(f'\nExpected leaderboard: ~{oof_auc * 100:.2f}%')
print('=' * 80)

import numpy as np
import pandas as pd
import warnings
import joblib
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import xgboost as xgb
import lightgbm as lgb
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    print(' PyTorch available')
    if torch.cuda.is_available():
        print(f'GPU available: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
    else:
        print(' No GPU available, using CPU')
except ImportError:
    print(' PyTorch not available. Install with: pip install torch')
    exit()
warnings.filterwarnings('ignore')
print('=' * 80)
print('NEURAL NETWORK EXPERIMENT')
print('=' * 80)
print('\n1. Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}')
print(f'Test: {test_df.shape}')
target_col = 'Label'
id_col = 'sha256'
print('\n2. Preprocessing...')
df_processed = df.copy()
test_processed = test_df.copy()
for col in df_processed.select_dtypes(include=[np.number]).columns:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(median_val, inplace=True)
constant_features = []
for col in df_processed.select_dtypes(include=[np.number]).columns:
    if col not in [target_col, id_col]:
        if df_processed[col].nunique() == 1:
            constant_features.append(col)
if constant_features:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
X_full = df_processed.drop(columns=[target_col, id_col] if id_col in df_processed.columns else [target_col])
y_full = df_processed[target_col]
print('\n3. Feature selection...')
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
n_top_features = 52
top_features = mi_df.head(n_top_features)['Feature'].tolist()
print(f'Using {len(top_features)} features')
X = X_full[top_features]
test_ids = test_processed[id_col]
X_test = test_processed[top_features]
print('\n4. Feature scaling...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print(f'Features scaled to mean=0, std=1')
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_full, test_size=0.2, random_state=42, stratify=y_full)
print(f'\n5. Data split:')
print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test_scaled.shape[0]}')
print(f'Features: {X_train.shape[1]}')
print('\n6. Building PyTorch Neural Network...')

class MalwareClassifier(nn.Module):

    def __init__(self, input_dim):
        super(MalwareClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc5(x))
        return x
print('Architecture: 256 -> 128 -> 64 -> 32 -> 1')
print('Regularization: BatchNorm + Dropout')
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values).unsqueeze(1))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val.values).unsqueeze(1))
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
model = MalwareClassifier(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
total_params = sum((p.numel() for p in model.parameters()))
print(f'Total parameters: {total_params:,}')
print('\nTraining neural network...')
t0 = time.time()
best_val_loss = float('inf')
best_val_acc = 0
patience = 25
patience_counter = 0
for epoch in range(200):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        train_correct += (predicted == y_batch).sum().item()
        train_total += y_batch.size(0)
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    scheduler.step(val_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/200 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'neural_network_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            print(f'Best validation accuracy: {best_val_acc:.5f}')
            break
train_time = time.time() - t0
print(f'\nTraining time: {train_time:.1f}s ({train_time / 60:.1f} minutes)')
model.load_state_dict(torch.load('neural_network_model.pt'))
model.eval()
with torch.no_grad():
    y_pred_proba_nn = model(torch.FloatTensor(X_val).to(device)).cpu().numpy().flatten()
y_pred_nn = (y_pred_proba_nn >= 0.5).astype(int)
nn_acc = accuracy_score(y_val, y_pred_nn)
nn_auc = roc_auc_score(y_val, y_pred_proba_nn)
print(f'\nNeural Network Final Performance:')
print(f'Accuracy: {nn_acc:.5f}')
print(f'ROC-AUC: {nn_auc:.5f}')
print(f'Precision: {precision_score(y_val, y_pred_nn):.5f}')
print(f'Recall: {recall_score(y_val, y_pred_nn):.5f}')
print(f'F1-Score: {f1_score(y_val, y_pred_nn):.5f}')
joblib.dump(scaler, 'scaler.pkl')
print('\n Saved: neural_network_model.pt, scaler.pkl')
print('\n7. Creating hybrid ensemble (Trees + Neural Network)...')
try:
    xgb_model = joblib.load('../submission-1/xgboost_model.pkl')
    lgb_model = joblib.load('../submission-1/lightgbm_model.pkl')
    y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_pred_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]
    xgb_acc = accuracy_score(y_val, (y_pred_proba_xgb >= 0.5).astype(int))
    lgb_acc = accuracy_score(y_val, (y_pred_proba_lgb >= 0.5).astype(int))
    print(f'XGBoost accuracy: {xgb_acc:.5f}')
    print(f'LightGBM accuracy: {lgb_acc:.5f}')
    print(f'Neural Net accuracy: {nn_acc:.5f}')
    total_acc = xgb_acc + lgb_acc + nn_acc
    w_xgb = xgb_acc / total_acc
    w_lgb = lgb_acc / total_acc
    w_nn = nn_acc / total_acc
    print(f'\nEnsemble weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}, NN={w_nn:.3f}')
    y_pred_proba_ensemble = w_xgb * y_pred_proba_xgb + w_lgb * y_pred_proba_lgb + w_nn * y_pred_proba_nn
    thresholds = np.arange(0.35, 0.65, 0.01)
    accuracies = []
    for threshold in thresholds:
        y_pred = (y_pred_proba_ensemble >= threshold).astype(int)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_threshold_idx]
    best_accuracy = accuracies[best_threshold_idx]
    y_pred_ensemble = (y_pred_proba_ensemble >= best_threshold).astype(int)
    print(f'\nHybrid Ensemble Performance:')
    print(f'Best threshold: {best_threshold:.2f}')
    print(f'Accuracy: {best_accuracy:.5f}')
    print(f'ROC-AUC: {roc_auc_score(y_val, y_pred_proba_ensemble):.5f}')
    print('\n' + '=' * 80)
    print('MODEL COMPARISON')
    print('=' * 80)
    results = pd.DataFrame([{'Model': 'XGBoost', 'Accuracy': xgb_acc, 'ROC-AUC': roc_auc_score(y_val, y_pred_proba_xgb)}, {'Model': 'LightGBM', 'Accuracy': lgb_acc, 'ROC-AUC': roc_auc_score(y_val, y_pred_proba_lgb)}, {'Model': 'Neural Network', 'Accuracy': nn_acc, 'ROC-AUC': nn_auc}, {'Model': f'Hybrid Ensemble (t={best_threshold:.2f})', 'Accuracy': best_accuracy, 'ROC-AUC': roc_auc_score(y_val, y_pred_proba_ensemble)}])
    print(results.to_string(index=False))
    print('=' * 80)
    print('\n8. Generating test predictions...')
    model.eval()
    with torch.no_grad():
        test_pred_proba_nn = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()
    test_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    test_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
    test_pred_proba_ensemble = w_xgb * test_pred_proba_xgb + w_lgb * test_pred_proba_lgb + w_nn * test_pred_proba_nn
    test_pred_ensemble = (test_pred_proba_ensemble >= best_threshold).astype(int)
    submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred_ensemble})
    print(f'\nSubmission: {submission.shape}')
    print(f'Malware: {(test_pred_ensemble == 1).sum()} ({(test_pred_ensemble == 1).sum() / len(test_pred_ensemble) * 100:.1f}%)')
    print(f'Benign: {(test_pred_ensemble == 0).sum()} ({(test_pred_ensemble == 0).sum() / len(test_pred_ensemble) * 100:.1f}%)')
    submission.to_csv('submission_hybrid.csv', index=False)
    print('\n Saved: submission_hybrid.csv')
    config = {'model_type': 'hybrid_ensemble', 'models': ['XGBoost', 'LightGBM', 'Neural Network'], 'validation_accuracy': float(best_accuracy), 'threshold': float(best_threshold), 'weights': {'xgb': float(w_xgb), 'lgb': float(w_lgb), 'nn': float(w_nn)}, 'nn_architecture': '256-128-64-32-1', 'results': results.to_dict('records')}
    with open('hybrid_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print(' Saved: hybrid_config.json')
except FileNotFoundError:
    print('\n Tree models not found in submission-1/')
    print('Generating predictions with neural network only...')
    model.eval()
    with torch.no_grad():
        test_pred_proba = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()
    test_pred = (test_pred_proba >= 0.5).astype(int)
    submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred})
    submission.to_csv('submission_nn_only.csv', index=False)
    print(' Saved: submission_nn_only.csv')
print('\n' + '=' * 80)
print('EXPERIMENT COMPLETE')
print('=' * 80)

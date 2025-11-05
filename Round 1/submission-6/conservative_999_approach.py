import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
print('=' * 80)
print('CONSERVATIVE 999-AWARE APPROACH')
print('=' * 80)
print('\n[1/5] Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}')
print(f'Test: {test_df.shape}')
behavior_features = ['Is App Taking Backup', 'Remote Command Execution', 'Rootkit Installation', 'Exploit Delivery', 'Data Exfiltration', 'Credential Theft', 'Screen Logging', 'Keylogging', 'Audio Surveillance', 'Social Engineering Attack', 'GPS Spoofing', 'Device Bricking', 'Call Interception', 'Network Traffic Interception', 'Device Lockout', 'Browser Hijacking', 'System Settings Modification', 'File System Manipulation', 'Camera Hijacking', 'App Installation without User Consent', 'Location Tracking', 'Contact Information Theft', 'Browser History Theft', 'Package Management Manipulation', 'Notification Manipulation', 'System Log Manipulation', 'Process Management Manipulation', 'Alarm Hijacking', 'Calendar Event Manipulation', 'Task Manipulation', 'Fake App Installation', 'Bluetooth Hijacking', 'WiFi Network Hijacking', 'USB Debugging Exploitation', 'Screen Overlay Attack', 'Sim Card Manipulation', 'Battery Drain Attack', 'SMS Spamming', 'Ad Fraud', 'Account Information Theft', 'Certificate Manipulation', 'Runtime Environment Manipulation', 'Call Log Manipulation']
print('\n[2/5] Processing features (keeping all training data)...')
no_999_mask = ~(df[behavior_features] == 999).any(axis=1)
clean_medians = {}
for col in behavior_features:
    clean_medians[col] = df[no_999_mask][col].median()
print(f'Clean samples for median calculation: {no_999_mask.sum():,}')
print(f'Samples with 999s: {(~no_999_mask).sum():,}')
df_processed = df.copy()
for col in behavior_features:
    df_processed[f'{col}_is_999'] = (df_processed[col] == 999).astype(int)
    df_processed[col] = df_processed[col].replace(999, clean_medians[col])
df_processed['total_999_count'] = df_processed[[col + '_is_999' for col in behavior_features]].sum(axis=1)
print(f'Created {len(behavior_features)} binary 999 flags')
print(f'Kept ALL {len(df_processed):,} training samples')
print('\n[3/5] Creating interaction features...')
interaction_pairs = [('Call Interception', 'Sim Card Manipulation'), ('Task Manipulation', 'WiFi Network Hijacking'), ('Remote Command Execution', 'Data Exfiltration'), ('Keylogging', 'Credential Theft'), ('Network Traffic Interception', 'Call Interception')]
for feat1, feat2 in interaction_pairs:
    df_processed[f'{feat1}_x_{feat2}'] = df_processed[feat1] * df_processed[feat2]
print(f'Created {len(interaction_pairs)} interaction features')
print('\n[4/5] Feature selection...')
X_full = df_processed.drop(['sha256', 'Label'], axis=1, errors='ignore')
y = df_processed['Label']
X_full = X_full.fillna(X_full.median())
mi_scores = mutual_info_classif(X_full, y, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': X_full.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
TOP_N = 60
top_features = mi_df.head(TOP_N)['feature'].tolist()
print(f'Selected {TOP_N} features (conservative)')
print(f'Top 10:')
for feat in top_features[:10]:
    score = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
    print(f'  {feat[:50]:<50} {score:.6f}')
X = X_full[top_features]
print('\n[5/5] Training with conservative hyperparameters...')
print('Goal: Generalization over perfect CV scores')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/5')
    X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
    y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0, random_state=42, eval_metric='auc')
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    lgb_model = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    cat_model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, l2_leaf_reg=5, random_state=42, verbose=False)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_pred)
    total_auc = xgb_auc + lgb_auc + cat_auc
    w_xgb = xgb_auc / total_auc
    w_lgb = lgb_auc / total_auc
    w_cat = cat_auc / total_auc
    ensemble_pred = w_xgb * xgb_pred + w_lgb * lgb_pred + w_cat * cat_pred
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    oof_predictions[val_idx] = ensemble_pred
    cv_scores.append(ensemble_auc)
    print(f'  XGB: {xgb_auc:.6f} (w={w_xgb:.3f})')
    print(f'  LGB: {lgb_auc:.6f} (w={w_lgb:.3f})')
    print(f'  CAT: {cat_auc:.6f} (w={w_cat:.3f})')
    print(f'  Ensemble: {ensemble_auc:.6f}')
mean_cv_auc = np.mean(cv_scores)
std_cv_auc = np.std(cv_scores)
oof_auc = roc_auc_score(y, oof_predictions)
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 80)
print('CROSS-VALIDATION RESULTS')
print('=' * 80)
print(f'Mean CV AUC: {mean_cv_auc:.6f} (+/- {std_cv_auc:.6f})')
print(f'Overall OOF AUC: {oof_auc:.6f}')
print(f'Overall OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'\nTarget: ~99.6% (realistic, not overfit)')
print('=' * 80)
import json
results = {'approach': 'conservative_999_aware', 'training_samples': len(X), 'features': TOP_N, 'cv_auc_mean': float(mean_cv_auc), 'cv_auc_std': float(std_cv_auc), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc), 'top_features': top_features[:20]}
with open('conservative_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to conservative_results.json')

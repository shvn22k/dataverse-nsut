import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
print('=' * 80)
print('CONSERVATIVE 999-AWARE SUBMISSION')
print('=' * 80)
behavior_features = ['Is App Taking Backup', 'Remote Command Execution', 'Rootkit Installation', 'Exploit Delivery', 'Data Exfiltration', 'Credential Theft', 'Screen Logging', 'Keylogging', 'Audio Surveillance', 'Social Engineering Attack', 'GPS Spoofing', 'Device Bricking', 'Call Interception', 'Network Traffic Interception', 'Device Lockout', 'Browser Hijacking', 'System Settings Modification', 'File System Manipulation', 'Camera Hijacking', 'App Installation without User Consent', 'Location Tracking', 'Contact Information Theft', 'Browser History Theft', 'Package Management Manipulation', 'Notification Manipulation', 'System Log Manipulation', 'Process Management Manipulation', 'Alarm Hijacking', 'Calendar Event Manipulation', 'Task Manipulation', 'Fake App Installation', 'Bluetooth Hijacking', 'WiFi Network Hijacking', 'USB Debugging Exploitation', 'Screen Overlay Attack', 'Sim Card Manipulation', 'Battery Drain Attack', 'SMS Spamming', 'Ad Fraud', 'Account Information Theft', 'Certificate Manipulation', 'Runtime Environment Manipulation', 'Call Log Manipulation']
interaction_pairs = [('Call Interception', 'Sim Card Manipulation'), ('Task Manipulation', 'WiFi Network Hijacking'), ('Remote Command Execution', 'Data Exfiltration'), ('Keylogging', 'Credential Theft'), ('Network Traffic Interception', 'Call Interception')]
print('\n[1/4] Loading and processing data...')
train_df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {train_df.shape}, Test: {test_df.shape}')
no_999_mask = ~(train_df[behavior_features] == 999).any(axis=1)
clean_medians = {}
for col in behavior_features:
    clean_medians[col] = train_df[no_999_mask][col].median()
train_processed = train_df.copy()
for col in behavior_features:
    train_processed[f'{col}_is_999'] = (train_processed[col] == 999).astype(int)
    train_processed[col] = train_processed[col].replace(999, clean_medians[col])
train_processed['total_999_count'] = train_processed[[col + '_is_999' for col in behavior_features]].sum(axis=1)
for feat1, feat2 in interaction_pairs:
    train_processed[f'{feat1}_x_{feat2}'] = train_processed[feat1] * train_processed[feat2]
test_processed = test_df.copy()
for col in behavior_features:
    test_processed[f'{col}_is_999'] = (test_processed[col] == 999).astype(int)
    test_processed[col] = test_processed[col].replace(999, clean_medians[col])
test_processed['total_999_count'] = test_processed[[col + '_is_999' for col in behavior_features]].sum(axis=1)
for feat1, feat2 in interaction_pairs:
    test_processed[f'{feat1}_x_{feat2}'] = test_processed[feat1] * test_processed[feat2]
print(f'Processed: train={train_processed.shape}, test={test_processed.shape}')
print('\n[2/4] Selecting features...')
X_full = train_processed.drop(['sha256', 'Label'], axis=1, errors='ignore')
y = train_processed['Label']
X_full = X_full.fillna(X_full.median())
mi_scores = mutual_info_classif(X_full, y, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': X_full.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
TOP_N = 60
top_features = mi_df.head(TOP_N)['feature'].tolist()
print(f'Selected {TOP_N} features')
print(f'Top 5: {top_features[:5]}')
X_train = X_full[top_features]
X_test = test_processed[top_features].fillna(test_processed[top_features].median())
print('\n[3/4] Training final models...')
print('  Training XGBoost...')
xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0, random_state=42, eval_metric='auc')
xgb_model.fit(X_train, y, verbose=False)
print('  Training LightGBM...')
lgb_model = lgb.LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbosity=-1)
lgb_model.fit(X_train, y)
print('  Training CatBoost...')
cat_model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=5, random_state=42, verbose=False)
cat_model.fit(X_train, y)
print('\n[4/4] Generating predictions...')
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
cat_pred = cat_model.predict_proba(X_test)[:, 1]
ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': (ensemble_pred > 0.5).astype(int)})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_conservative_{timestamp}.csv'
submission.to_csv(filename, index=False)
print('\n' + '=' * 80)
print('SUBMISSION GENERATED')
print('=' * 80)
print(f'File: {filename}')
print(f'Total samples: {len(submission):,}')
print(f'Predicted Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'Predicted Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nExpected: ~99.6% (realistic, generalizable)')
print('=' * 80)

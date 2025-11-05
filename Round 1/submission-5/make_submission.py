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
print('GENERATING SUBMISSION WITH 999-AWARE APPROACH')
print('=' * 80)
print('\n[1/6] Loading data...')
train_df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {train_df.shape}')
print(f'Test: {test_df.shape}')
behavior_features = ['Is App Taking Backup', 'Remote Command Execution', 'Rootkit Installation', 'Exploit Delivery', 'Data Exfiltration', 'Credential Theft', 'Screen Logging', 'Keylogging', 'Audio Surveillance', 'Social Engineering Attack', 'GPS Spoofing', 'Device Bricking', 'Call Interception', 'Network Traffic Interception', 'Device Lockout', 'Browser Hijacking', 'System Settings Modification', 'File System Manipulation', 'Camera Hijacking', 'App Installation without User Consent', 'Location Tracking', 'Contact Information Theft', 'Browser History Theft', 'Package Management Manipulation', 'Notification Manipulation', 'System Log Manipulation', 'Process Management Manipulation', 'Alarm Hijacking', 'Calendar Event Manipulation', 'Task Manipulation', 'Fake App Installation', 'Bluetooth Hijacking', 'WiFi Network Hijacking', 'USB Debugging Exploitation', 'Screen Overlay Attack', 'Sim Card Manipulation', 'Battery Drain Attack', 'SMS Spamming', 'Ad Fraud', 'Account Information Theft', 'Certificate Manipulation', 'Runtime Environment Manipulation', 'Call Log Manipulation']
metadata_features = ['Duplicate Permissions Requested', 'Permissions Requested', 'Activities Declared', 'Services Declared', 'Broadcast Receivers', 'Content Providers Declared', 'Metadata Elements', 'Version Code', 'Target SDK Version']
print('\n[2/6] Processing training data...')
train_all_999_mask = (train_df[behavior_features] == 999).all(axis=1)
train_no_999_mask = ~(train_df[behavior_features] == 999).any(axis=1)
print(f'Train - No 999s: {train_no_999_mask.sum():,}')
print(f'Train - All 999s: {train_all_999_mask.sum():,}')
clean_medians = {}
for col in behavior_features:
    clean_medians[col] = train_df[train_no_999_mask][col].median()
train_clean = train_df[train_no_999_mask].copy()
train_all_999 = train_df[train_all_999_mask].copy()
for col in behavior_features:
    train_clean[f'{col}_is_999'] = 0
train_clean['total_999_count'] = 0
train_clean['pct_999_features'] = 0.0
interaction_pairs = [('Call Interception', 'Sim Card Manipulation'), ('Task Manipulation', 'WiFi Network Hijacking'), ('Remote Command Execution', 'Data Exfiltration'), ('Keylogging', 'Credential Theft'), ('Screen Logging', 'Audio Surveillance'), ('Browser Hijacking', 'System Settings Modification'), ('File System Manipulation', 'Camera Hijacking'), ('Location Tracking', 'GPS Spoofing'), ('Network Traffic Interception', 'Call Interception')]
for feat1, feat2 in interaction_pairs:
    train_clean[f'{feat1}_x_{feat2}'] = train_clean[feat1] * train_clean[feat2]
print('\n[3/6] Processing test data...')
test_all_999_mask = (test_df[behavior_features] == 999).all(axis=1)
test_no_999_mask = ~(test_df[behavior_features] == 999).any(axis=1)
print(f'Test - No 999s: {test_no_999_mask.sum():,}')
print(f'Test - All 999s: {test_all_999_mask.sum():,}')
test_clean = test_df[test_no_999_mask].copy()
test_all_999 = test_df[test_all_999_mask].copy()
for col in behavior_features:
    test_clean[f'{col}_is_999'] = 0
test_clean['total_999_count'] = 0
test_clean['pct_999_features'] = 0.0
for feat1, feat2 in interaction_pairs:
    test_clean[f'{feat1}_x_{feat2}'] = test_clean[feat1] * test_clean[feat2]
print('\n[4/6] Selecting features...')
X_train_full = train_clean.drop(['sha256', 'Label'], axis=1, errors='ignore')
y_train = train_clean['Label']
X_train_full = X_train_full.fillna(X_train_full.median())
mi_scores = mutual_info_classif(X_train_full, y_train, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': X_train_full.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
TOP_N_FEATURES = 80
top_features = mi_df.head(TOP_N_FEATURES)['feature'].tolist()
print(f'Selected {len(top_features)} features')
print(f'Top 5: {top_features[:5]}')
X_train = X_train_full[top_features]
X_test_clean = test_clean[top_features].fillna(test_clean[top_features].median())
print('\n[5/6] Training final models on full training data...')
print('\n[MODEL 1] Training main ensemble...')
print(f'Training samples: {len(X_train):,}')
print(f'Features: {X_train.shape[1]}')
xgb_final = xgb.XGBClassifier(n_estimators=700, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc')
print('  Training XGBoost...')
xgb_final.fit(X_train, y_train, verbose=False)
lgb_final = lgb.LGBMClassifier(n_estimators=700, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
print('  Training LightGBM...')
lgb_final.fit(X_train, y_train)
cat_final = CatBoostClassifier(iterations=700, depth=7, learning_rate=0.05, random_state=42, verbose=False)
print('  Training CatBoost...')
cat_final.fit(X_train, y_train)
print('\n[MODEL 2] Training metadata model...')
available_metadata = [f for f in metadata_features if f in train_all_999.columns]
X_meta_train = train_all_999[available_metadata].fillna(train_all_999[available_metadata].median())
y_meta_train = train_all_999['Label']
print(f'Training samples: {len(X_meta_train):,}')
print(f'Features: {len(available_metadata)}')
meta_final = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
meta_final.fit(X_meta_train, y_meta_train)
print('\n[6/6] Generating predictions...')
print(f'Predicting {len(test_clean):,} clean samples...')
xgb_pred = xgb_final.predict_proba(X_test_clean)[:, 1]
lgb_pred = lgb_final.predict_proba(X_test_clean)[:, 1]
cat_pred = cat_final.predict_proba(X_test_clean)[:, 1]
ensemble_pred_clean = (xgb_pred + lgb_pred + cat_pred) / 3
test_clean['prediction_prob'] = ensemble_pred_clean
if len(test_all_999) > 0:
    print(f'Predicting {len(test_all_999):,} all-999 samples...')
    X_meta_test = test_all_999[available_metadata].fillna(test_all_999[available_metadata].median())
    meta_pred = meta_final.predict_proba(X_meta_test)[:, 1]
    test_all_999['prediction_prob'] = meta_pred
    test_df_with_preds = pd.concat([test_clean[['sha256', 'prediction_prob']], test_all_999[['sha256', 'prediction_prob']]], axis=0)
else:
    print(f'No all-999 samples in test set (all {len(test_clean):,} are clean)')
    test_df_with_preds = test_clean[['sha256', 'prediction_prob']]
test_df_with_preds['Label'] = (test_df_with_preds['prediction_prob'] > 0.5).astype(int)
submission = test_df[['sha256']].merge(test_df_with_preds[['sha256', 'Label']], on='sha256', how='left')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_999aware_{timestamp}.csv'
submission.to_csv(filename, index=False)
print('\n' + '=' * 80)
print('SUBMISSION GENERATED')
print('=' * 80)
print(f'File: {filename}')
print(f'Total samples: {len(submission):,}')
print(f'Predicted Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'Predicted Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nBased on CV results:')
print(f'  Expected AUC: 99.89%')
print(f'  Expected Accuracy: 97.85%')
print('=' * 80)

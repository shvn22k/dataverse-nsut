import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
print('=' * 80)
print('STEP 1: 999 PATTERN ANALYSIS')
print('=' * 80)
df = pd.read_csv('../data/main.csv')
print(f'Dataset shape: {df.shape}')
behavior_features = ['Is App Taking Backup', 'Remote Command Execution', 'Rootkit Installation', 'Exploit Delivery', 'Data Exfiltration', 'Credential Theft', 'Screen Logging', 'Keylogging', 'Audio Surveillance', 'Social Engineering Attack', 'GPS Spoofing', 'Device Bricking', 'Call Interception', 'Network Traffic Interception', 'Device Lockout', 'Browser Hijacking', 'System Settings Modification', 'File System Manipulation', 'Camera Hijacking', 'App Installation without User Consent', 'Location Tracking', 'Contact Information Theft', 'Browser History Theft', 'Package Management Manipulation', 'Notification Manipulation', 'System Log Manipulation', 'Process Management Manipulation', 'Alarm Hijacking', 'Calendar Event Manipulation', 'Task Manipulation', 'Fake App Installation', 'Bluetooth Hijacking', 'WiFi Network Hijacking', 'USB Debugging Exploitation', 'Screen Overlay Attack', 'Sim Card Manipulation', 'Battery Drain Attack', 'SMS Spamming', 'Ad Fraud', 'Account Information Theft', 'Certificate Manipulation', 'Runtime Environment Manipulation', 'Call Log Manipulation']
metadata_features = ['Duplicate Permissions Requested', 'Permissions Requested', 'Activities Declared', 'Services Declared', 'Broadcast Receivers', 'Content Providers Declared', 'Metadata Elements', 'Version Code', 'Target SDK Version']
all_999_mask = (df[behavior_features] == 999).all(axis=1)
partial_999_mask = (df[behavior_features] == 999).any(axis=1) & ~all_999_mask
no_999_mask = ~(df[behavior_features] == 999).any(axis=1)
print(f'\nDataset split:')
print(f'  No 999s: {no_999_mask.sum():,} rows ({no_999_mask.sum() / len(df) * 100:.2f}%)')
print(f'  Partial 999s: {partial_999_mask.sum():,} rows ({partial_999_mask.sum() / len(df) * 100:.2f}%)')
print(f'  All 999s: {all_999_mask.sum():,} rows ({all_999_mask.sum() / len(df) * 100:.2f}%)')
print(f'\nAll-999 rows label distribution:')
print(f'  Malware (1): {df[all_999_mask]['Label'].sum():,}')
print(f'  Benign (0): {(~df[all_999_mask]['Label'].astype(bool)).sum():,}')
print(f'  Malware %: {df[all_999_mask]['Label'].mean() * 100:.2f}%')
df['row_type'] = 'no_999'
df.loc[partial_999_mask, 'row_type'] = 'partial_999'
df.loc[all_999_mask, 'row_type'] = 'all_999'
print('\n' + '=' * 80)
print('STEP 2: DATASET PREPARATION')
print('=' * 80)
df_no_999 = df[no_999_mask].copy()
df_partial_999 = df[partial_999_mask].copy()
df_all_999 = df[all_999_mask].copy()
print(f'\nDataset sizes:')
print(f'  Clean: {len(df_no_999):,}')
print(f'  Partial: {len(df_partial_999):,}')
print(f'  All-999: {len(df_all_999):,}')
print(f'\nProcessing behavioral features...')
clean_medians = {}
for col in behavior_features:
    clean_medians[col] = df_no_999[col].median()
for col in behavior_features:
    df_partial_999[f'{col}_is_999'] = (df_partial_999[col] == 999).astype(int)
    df_partial_999[col] = df_partial_999[col].replace(999, clean_medians[col])
    df_no_999[f'{col}_is_999'] = 0
df_partial_999['total_999_count'] = (df[partial_999_mask][behavior_features] == 999).sum(axis=1).values
df_partial_999['pct_999_features'] = df_partial_999['total_999_count'] / len(behavior_features)
df_no_999['total_999_count'] = 0
df_no_999['pct_999_features'] = 0.0
df_main = pd.concat([df_no_999, df_partial_999], axis=0).reset_index(drop=True)
print(f'\nMain model dataset: {len(df_main):,} rows')
is_999_cols = [col for col in df_main.columns if '_is_999' in col]
print(f'  - 999 binary flags: {len(is_999_cols)}')
print(f'  - Aggregate 999 features: 2')
print('\n' + '=' * 80)
print('STEP 3: INTERACTION FEATURES (NO 999 CONTAMINATION)')
print('=' * 80)
interaction_pairs = [('Call Interception', 'Sim Card Manipulation'), ('Task Manipulation', 'WiFi Network Hijacking'), ('Remote Command Execution', 'Data Exfiltration'), ('Keylogging', 'Credential Theft'), ('Screen Logging', 'Audio Surveillance'), ('Browser Hijacking', 'System Settings Modification'), ('File System Manipulation', 'Camera Hijacking'), ('Location Tracking', 'GPS Spoofing'), ('Network Traffic Interception', 'Call Interception')]
print(f'\nCreating {len(interaction_pairs)} interaction features...')
for feat1, feat2 in interaction_pairs:
    if feat1 in df_main.columns and feat2 in df_main.columns:
        interaction_name = f'{feat1}_x_{feat2}'
        df_main[interaction_name] = df_main[feat1] * df_main[feat2]
        max_val = df_main[interaction_name].max()
        min_val = df_main[interaction_name].min()
        print(f'  {interaction_name[:50]:<50} range [{min_val:8.1f}, {max_val:8.1f}]')
        if max_val > 10000:
            print('    Warning: unexpectedly large value')
print(f'\nTotal features in main dataset: {len(df_main.columns)}')
print('\n' + '=' * 80)
print('STEP 4: FEATURE SELECTION')
print('=' * 80)
X_main = df_main.drop(['sha256', 'Label', 'row_type'], axis=1, errors='ignore')
y_main = df_main['Label']
X_main = X_main.fillna(X_main.median())
print(f'\nCalculating mutual information for {len(X_main.columns)} features...')
mi_scores = mutual_info_classif(X_main, y_main, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': X_main.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
TOP_N_FEATURES = 80
top_features = mi_df.head(TOP_N_FEATURES)['feature'].tolist()
print(f'\nTop 20 features by Mutual Information:')
for i, row in mi_df.head(20).iterrows():
    print(f'  {row['feature'][:50]:<50} {row['mi_score']:.6f}')
is_999_in_top = [f for f in top_features if '_is_999' in f]
interaction_in_top = [f for f in top_features if '_x_' in f]
metadata_in_top = [f for f in top_features if f in metadata_features]
print(f'\nFeature composition in top {TOP_N_FEATURES}:')
print(f'  999 binary flags: {len(is_999_in_top)}')
print(f'  Interaction features: {len(interaction_in_top)}')
print(f'  Metadata features: {len(metadata_in_top)}')
print(f'  Original behavioral: {TOP_N_FEATURES - len(is_999_in_top) - len(interaction_in_top) - len(metadata_in_top)}')
if len(is_999_in_top) > 0:
    print(f'\nTop 999-aware features:')
    for feat in is_999_in_top[:5]:
        mi_score = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
        print(f'  {feat[:50]:<50} {mi_score:.6f}')
X_main_selected = X_main[top_features]
print(f'\nMain model final feature count: {X_main_selected.shape[1]}')
print('\n' + '=' * 80)
print('STEP 5: TRAINING TWO-MODEL SYSTEM')
print('=' * 80)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print('\n[MODEL 1] Training main 3-model ensemble on clean+partial data...')
print(f'Training samples: {len(X_main_selected):,}')
print(f'Features: {X_main_selected.shape[1]}')
main_cv_scores = []
main_oof_predictions = np.zeros(len(X_main_selected))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_main_selected, y_main), 1):
    print(f'\n  Fold {fold}/5')
    X_train, X_val = (X_main_selected.iloc[train_idx], X_main_selected.iloc[val_idx])
    y_train, y_val = (y_main.iloc[train_idx], y_main.iloc[val_idx])
    xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc')
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    cat_model = CatBoostClassifier(iterations=500, depth=7, learning_rate=0.05, random_state=42, verbose=False)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_pred)
    total_auc = xgb_auc + lgb_auc + cat_auc
    w_xgb = xgb_auc / total_auc
    w_lgb = lgb_auc / total_auc
    w_cat = cat_auc / total_auc
    ensemble_pred = w_xgb * xgb_pred + w_lgb * lgb_pred + w_cat * cat_pred
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    main_oof_predictions[val_idx] = ensemble_pred
    main_cv_scores.append(ensemble_auc)
    print(f'    XGB: {xgb_auc:.6f} (weight: {w_xgb:.3f})')
    print(f'    LGB: {lgb_auc:.6f} (weight: {w_lgb:.3f})')
    print(f'    CAT: {cat_auc:.6f} (weight: {w_cat:.3f})')
    print(f'    ENSEMBLE: {ensemble_auc:.6f}')
main_model_auc = np.mean(main_cv_scores)
print(f'\n  Main Model CV AUC: {main_model_auc:.6f} (+/- {np.std(main_cv_scores):.6f})')
print(f'\n[MODEL 2] Training metadata-only model for all-999 rows...')
print(f'Training samples: {len(df_all_999):,}')
available_metadata = [f for f in metadata_features if f in df_all_999.columns]
X_metadata = df_all_999[available_metadata].fillna(df_all_999[available_metadata].median())
y_metadata = df_all_999['Label']
print(f'Features: {len(available_metadata)} metadata features')
print(f'Class balance: {y_metadata.mean() * 100:.2f}% malware')
metadata_cv_scores = []
metadata_oof_predictions = np.zeros(len(X_metadata))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_metadata, y_metadata), 1):
    X_train, X_val = (X_metadata.iloc[train_idx], X_metadata.iloc[val_idx])
    y_train, y_val = (y_metadata.iloc[train_idx], y_metadata.iloc[val_idx])
    meta_model = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
    meta_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    meta_pred = meta_model.predict_proba(X_val)[:, 1]
    meta_auc = roc_auc_score(y_val, meta_pred)
    metadata_oof_predictions[val_idx] = meta_pred
    metadata_cv_scores.append(meta_auc)
    print(f'  Fold {fold}: {meta_auc:.6f}')
metadata_model_auc = np.mean(metadata_cv_scores)
print(f'\n  Metadata Model CV AUC: {metadata_model_auc:.6f} (+/- {np.std(metadata_cv_scores):.6f})')
print('\n' + '=' * 80)
print('STEP 6: COMBINED SYSTEM EVALUATION')
print('=' * 80)
combined_predictions = np.zeros(len(df))
combined_labels = df['Label'].values
main_indices = df[df['row_type'].isin(['no_999', 'partial_999'])].index
combined_predictions[main_indices] = main_oof_predictions
all_999_indices = df[df['row_type'] == 'all_999'].index
combined_predictions[all_999_indices] = metadata_oof_predictions
combined_auc = roc_auc_score(combined_labels, combined_predictions)
combined_acc = accuracy_score(combined_labels, (combined_predictions > 0.5).astype(int))
print(f'\nRESULTS BREAKDOWN:')
print(f'─' * 80)
print(f'Main Model (clean + partial 999s):')
print(f'  Samples: {len(main_indices):,} ({len(main_indices) / len(df) * 100:.2f}%)')
print(f'  CV AUC: {main_model_auc:.6f}')
print(f'')
print(f'Metadata Model (all 999s):')
print(f'  Samples: {len(all_999_indices):,} ({len(all_999_indices) / len(df) * 100:.2f}%)')
print(f'  CV AUC: {metadata_model_auc:.6f}')
print(f'')
print(f'COMBINED SYSTEM:')
print(f'  Total Samples: {len(df):,}')
print(f'  Overall AUC: {combined_auc:.6f}')
print(f'  Overall Accuracy: {combined_acc:.6f} ({combined_acc * 100:.2f}%)')
print(f'')
print(f'COMPARISON:')
print(f'  Baseline (original 99.57%): 0.995700')
print(f'  New approach: {combined_auc:.6f}')
print(f'  Improvement: {(combined_auc - 0.9957) * 100:+.4f}%')
print(f'  Target (99.7%): {(' ACHIEVED' if combined_auc >= 0.997 else ' Not yet')}')
print(f'─' * 80)
print('\n' + '=' * 80)
print('FINAL SUMMARY')
print('=' * 80)
print(f'\nProcessed {len(df):,} samples')
print(f'Identified {all_999_mask.sum():,} all-999 rows (4.03%)')
print(f'Created {len(is_999_cols)} binary 999 flags')
print(f'Generated {len(interaction_pairs)} clean interaction features')
print(f'Selected {TOP_N_FEATURES} features using MI')
print(f'Trained 3-model ensemble on {len(main_indices):,} samples')
print(f'Trained metadata model on {len(all_999_indices):,} samples')
print(f'\n🎯 FINAL AUC: {combined_auc:.6f}')
print(f'🎯 FINAL ACCURACY: {combined_acc * 100:.2f}%')
print('=' * 80)

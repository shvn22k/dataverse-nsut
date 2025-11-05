import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
import json
print('=' * 80)
print('REAL THRESHOLD OPTIMIZATION - USING ACTUAL MODELS')
print('=' * 80)
print('\n[STEP 1] Loading and preprocessing data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
df_processed = df.copy()
test_processed = test_df.copy()
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(median_val, inplace=True)
constant_features = [col for col in numeric_cols if col not in ['Label', 'sha256'] and df_processed[col].nunique() == 1]
if constant_features:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
print('\n[STEP 2] Feature selection...')
X_full = df_processed.drop(columns=['Label', 'sha256'], errors='ignore')
y = df_processed['Label']
mi_scores = mutual_info_classif(X_full, y, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': X_full.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
TOP_N = 60
top_features = mi_df.head(TOP_N)['feature'].tolist()
print(f'Selected {TOP_N} features')
X = X_full[top_features]
X_test = test_processed[top_features]
print('Adding interaction features...')
top_10 = top_features[:10]
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
print(f'Total features: {X.shape[1]}')
print('\n[STEP 3] Loading trained models from submission-3...')
try:
    xgb_model = joblib.load('../submission-3/xgboost_enhanced.pkl')
    lgb_model = joblib.load('../submission-3/lightgbm_enhanced.pkl')
    print(' XGBoost and LightGBM models loaded successfully')
except FileNotFoundError as e:
    print(f'ERROR: model file not found: {e}')
    print('Please check if the model files exist in submission-3/')
    exit(1)
print('\n[STEP 4] Generating real probabilities...')
print('  XGBoost predictions...')
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
print('  LightGBM predictions...')
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
ensemble_probs = (xgb_probs + lgb_probs) / 2
print(f'\nProbability ranges:')
print(f'  XGBoost: [{xgb_probs.min():.4f}, {xgb_probs.max():.4f}]')
print(f'  LightGBM: [{lgb_probs.min():.4f}, {lgb_probs.max():.4f}]')
print(f'  Ensemble: [{ensemble_probs.min():.4f}, {ensemble_probs.max():.4f}]')
print('\n[STEP 5] Testing different thresholds...')
thresholds_to_test = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
results = []
for threshold in thresholds_to_test:
    binary_pred = (ensemble_probs > threshold).astype(int)
    malware_count = np.sum(binary_pred == 1)
    benign_count = np.sum(binary_pred == 0)
    malware_pct = malware_count / len(binary_pred) * 100
    original_pred = (ensemble_probs > 0.5).astype(int)
    agreement = np.sum(binary_pred == original_pred)
    agreement_pct = agreement / len(binary_pred) * 100
    results.append({'threshold': float(threshold), 'malware_count': int(malware_count), 'malware_pct': float(malware_pct), 'agreement_with_0.5': float(agreement_pct)})
    print(f'  Threshold {threshold:.2f}: {malware_count:,} malware ({malware_pct:.1f}%) | Agreement with 0.5: {agreement_pct:.1f}%')
print('\n[STEP 6] Generating submissions for different thresholds...')
original_malware = np.sum((ensemble_probs > 0.5).astype(int))
original_pct = original_malware / len(ensemble_probs) * 100
promising_thresholds = []
for row in results:
    if abs(row['malware_pct'] - original_pct) > 0.5:
        promising_thresholds.append(row['threshold'])
if not promising_thresholds:
    promising_thresholds = [0.42, 0.48, 0.52, 0.58]
print(f'Promising thresholds: {promising_thresholds}')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for threshold in promising_thresholds:
    binary_pred = (ensemble_probs > threshold).astype(int)
    submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': binary_pred})
    filename = f'submission_real_threshold_{threshold:.2f}_{timestamp}.csv'
    submission.to_csv(filename, index=False)
    malware_count = np.sum(binary_pred == 1)
    print(f'Generated: {filename} ({malware_count:,} malware)')
print('\n[STEP 7] Analysis...')
results_df = pd.DataFrame(results)
print(f'\nThreshold analysis:')
print(results_df.to_string(index=False))
print(f'\nOriginal (threshold=0.50): {original_malware:,} malware ({original_pct:.1f}%)')
significant_changes = []
for _, row in results_df.iterrows():
    if abs(row['malware_pct'] - original_pct) > 0.5:
        significant_changes.append((row['threshold'], row['malware_pct']))
if significant_changes:
    print(f'\nSignificant changes from 0.50 threshold:')
    for threshold, pct in significant_changes:
        change = pct - original_pct
        print(f'  {threshold:.2f}: {pct:.1f}% ({change:+.1f}%)')
else:
    print(f'\nNo significant changes found - all thresholds similar to 0.50')
print('\n[STEP 8] Saving results...')
summary = {'timestamp': timestamp, 'approach': 'real_threshold_optimization', 'original_malware_count': int(original_malware), 'original_malware_pct': float(original_pct), 'thresholds_tested': [float(t) for t in thresholds_to_test], 'results': results, 'promising_thresholds': [float(t) for t in promising_thresholds], 'significant_changes': [(float(t), float(pct)) for t, pct in significant_changes], 'probability_stats': {'xgb_range': [float(xgb_probs.min()), float(xgb_probs.max())], 'lgb_range': [float(lgb_probs.min()), float(lgb_probs.max())], 'ensemble_range': [float(ensemble_probs.min()), float(ensemble_probs.max())]}}
with open(f'real_threshold_results_{timestamp}.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Results saved: real_threshold_results_{timestamp}.json')
print('\n' + '=' * 80)
print('REAL THRESHOLD OPTIMIZATION COMPLETE!')
print('=' * 80)
print(f'\nGenerated {len(promising_thresholds)} submission files')
print('Used real probabilities from trained models')
print(f'Tested {len(thresholds_to_test)} thresholds')
print(f'\nFiles to submit:')
for threshold in promising_thresholds:
    print(f'  - submission_real_threshold_{threshold:.2f}_{timestamp}.csv')
print(f'\nThis should give REAL differences between thresholds!')
print('=' * 80)

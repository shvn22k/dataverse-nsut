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
import json
warnings.filterwarnings('ignore')
print('=' * 80)
print('PSEUDO-LABELING: TRAIN + SUBMIT')
print('=' * 80)
print('\n[STEP 1] Loading data...')
train_df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
predictions_df = pd.read_csv('../submission-3/B7JYI-II.csv')
print(f'Train: {train_df.shape}, Test: {test_df.shape}')
test_with_labels = test_df.merge(predictions_df[['sha256', 'Label']], on='sha256', how='left')
print(f'\nPseudo-label distribution:')
print(f'  Malware: {test_with_labels['Label'].sum():.0f} ({test_with_labels['Label'].mean() * 100:.2f}%)')
print(f'  Benign: {(1 - test_with_labels['Label']).sum():.0f} ({(1 - test_with_labels['Label']).mean() * 100:.2f}%)')
augmented_train = pd.concat([train_df, test_with_labels], axis=0, ignore_index=True)
print(f'\nAugmented training set: {augmented_train.shape}')
print(f'  Original: {len(train_df):,}')
print(f'  Pseudo: {len(test_with_labels):,}')
print(f'  Total: {len(augmented_train):,}')
print('\n[STEP 2] Preprocessing...')
train_processed = augmented_train.copy()
test_processed = test_df.copy()
numeric_cols = train_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if train_processed[col].isnull().sum() > 0:
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(median_val, inplace=True)
constant_features = [col for col in numeric_cols if col not in ['Label', 'sha256'] and train_processed[col].nunique() == 1]
if constant_features:
    train_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
print('\n[STEP 3] Feature selection...')
X_full = train_processed.drop(columns=['Label', 'sha256'], errors='ignore')
y = train_processed['Label']
mi_scores = mutual_info_classif(X_full, y, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': X_full.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
TOP_N = 60
top_features = mi_df.head(TOP_N)['feature'].tolist()
print(f'Selected {TOP_N} features')
print(f'Top 10:')
for feat in top_features[:10]:
    score = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
    print(f'  {feat[:50]:<50} {score:.6f}')
X = X_full[top_features]
X_test = test_processed[top_features]
print('\nAdding interactions...')
top_10 = top_features[:10]
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
print(f'Total features: {X.shape[1]}')
print('\n[STEP 4] Cross-validation on augmented data...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/5')
    X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
    y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
    xgb_model = xgb.XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.6, min_child_weight=3, gamma=0, reg_alpha=0.1, reg_lambda=2, random_state=42, eval_metric='auc')
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, max_depth=5, learning_rate=0.07, num_leaves=31, subsample=0.9, colsample_bytree=0.6, min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5, random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    cat_model = CatBoostClassifier(iterations=500, depth=7, learning_rate=0.1, l2_leaf_reg=1, border_count=32, bagging_temperature=1, random_strength=0.5, random_state=42, verbose=False)
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
    print(f'  XGB: {xgb_auc:.6f} | LGB: {lgb_auc:.6f} | CAT: {cat_auc:.6f}')
    print(f'  Ensemble: {ensemble_auc:.6f}')
mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
oof_auc = roc_auc_score(y, oof_predictions)
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 80)
print('CV RESULTS')
print('=' * 80)
print(f'Mean CV AUC: {mean_cv:.6f} (+/- {std_cv:.6f})')
print(f'OOF AUC: {oof_auc:.6f}')
print(f'OOF Acc: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'\nBaseline: 0.995700')
print(f'Difference: {oof_auc - 0.9957:+.6f}')
print('=' * 80)
print('\n[STEP 5] Training final models on full augmented data...')
print('  XGBoost...')
xgb_final = xgb.XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.6, min_child_weight=3, gamma=0, reg_alpha=0.1, reg_lambda=2, random_state=42, eval_metric='auc')
xgb_final.fit(X, y, verbose=False)
print('  LightGBM...')
lgb_final = lgb.LGBMClassifier(n_estimators=1000, max_depth=5, learning_rate=0.07, num_leaves=31, subsample=0.9, colsample_bytree=0.6, min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5, random_state=42, verbosity=-1)
lgb_final.fit(X, y)
print('  CatBoost...')
cat_final = CatBoostClassifier(iterations=500, depth=7, learning_rate=0.1, l2_leaf_reg=1, border_count=32, bagging_temperature=1, random_strength=0.5, random_state=42, verbose=False)
cat_final.fit(X, y)
print('\n[STEP 6] Generating submission...')
xgb_pred = xgb_final.predict_proba(X_test)[:, 1]
lgb_pred = lgb_final.predict_proba(X_test)[:, 1]
cat_pred = cat_final.predict_proba(X_test)[:, 1]
ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': (ensemble_pred > 0.5).astype(int)})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_pseudo_{timestamp}.csv'
submission.to_csv(filename, index=False)
results = {'timestamp': timestamp, 'approach': 'pseudo_labeling', 'train_samples': len(train_df), 'pseudo_samples': len(test_with_labels), 'total_samples': len(X), 'features': X.shape[1], 'cv_auc_mean': float(mean_cv), 'cv_auc_std': float(std_cv), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc)}
with open(f'results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n' + '=' * 80)
print('COMPLETE!')
print('=' * 80)
print(f'\nSubmission: {filename}')
print(f'Results: results_{timestamp}.json')
print(f'\nSamples: {len(submission):,}')
print(f'  Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'  Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nCV Performance:')
print(f'  AUC: {oof_auc:.6f}')
print(f'  Accuracy: {oof_acc * 100:.2f}%')
print(f'\nExpected leaderboard: ~{oof_auc * 100:.2f}%')
print('=' * 80)

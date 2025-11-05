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
print('BACK TO BASICS - REPLICATE 99.57% APPROACH')
print('=' * 80)
print('\n[1/4] Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
print('\n[2/4] Preprocessing...')
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
print('\n[3/4] Feature selection...')
X_full = df_processed.drop(columns=['Label', 'sha256'], errors='ignore')
y = df_processed['Label']
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
top_10 = top_features[:10]
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
print(f'Total features with interactions: {X.shape[1]}')
print('\n[4/4] Training with proven hyperparameters...')
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
print(f'\nTarget: Match 99.57% from submission-3')
print('=' * 80)
import json
results = {'approach': 'back_to_basics', 'training_samples': len(X), 'features': X.shape[1], 'cv_auc_mean': float(mean_cv_auc), 'cv_auc_std': float(std_cv_auc), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc)}
with open('back_to_basics_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to back_to_basics_results.json')

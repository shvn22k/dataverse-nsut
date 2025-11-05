import numpy as np
import pandas as pd
import warnings
import joblib
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score
import lightgbm as lgb
warnings.filterwarnings('ignore')
print('=' * 70)
print('LIGHTGBM FOCUSED - TARGETING > 99.57%')
print('=' * 70)
N_FEATURES = 60
TUNE_ITERATIONS = 50
CV_FOLDS = 5
USE_FEATURE_ENGINEERING = True
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except:
    pass
print(f'Using GPU: {USE_GPU}')
print('\n[1/7] Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
target_col = 'Label'
id_col = 'sha256'
print('\n[2/7] Preprocessing...')
df_processed = df.copy()
test_processed = test_df.copy()
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(median_val, inplace=True)
constant_features = []
for col in numeric_cols:
    if col not in [target_col, id_col]:
        if df_processed[col].nunique() == 1:
            constant_features.append(col)
if len(constant_features) > 0:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
print(f'\n[3/7] Selecting top {N_FEATURES} features...')
X_full = df_processed.drop(columns=[target_col, id_col] if id_col in df_processed.columns else [target_col])
y_full = df_processed[target_col]
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
print('Top 20 features by MI:')
print(mi_df.head(20))
n_features = min(N_FEATURES, len(mi_df))
top_features = mi_df.head(n_features)['Feature'].tolist()
print(f'\nSelected {len(top_features)} features')
X = X_full[top_features].copy()
y = y_full.copy()
if USE_FEATURE_ENGINEERING:
    print(f'\n[4/7] Engineering interaction features...')
    top_interaction = min(10, len(top_features))
    interaction_count = 0
    for i in range(top_interaction):
        for j in range(i + 1, min(i + 3, top_interaction)):
            feat1 = top_features[i]
            feat2 = top_features[j]
            new_feat = f'{feat1}_x_{feat2}'
            X[new_feat] = X[feat1] * X[feat2]
            interaction_count += 1
    print(f'Created {interaction_count} interaction features')
    print(f'Total features: {X.shape[1]}')
    all_features = X.columns.tolist()
else:
    print('\n[4/7] Skipping feature engineering...')
    all_features = top_features
X_test = test_processed[top_features].copy()
test_ids = test_processed[id_col].values
if USE_FEATURE_ENGINEERING:
    top_interaction = min(10, len(top_features))
    for i in range(top_interaction):
        for j in range(i + 1, min(i + 3, top_interaction)):
            feat1 = top_features[i]
            feat2 = top_features[j]
            new_feat = f'{feat1}_x_{feat2}'
            X_test[new_feat] = X_test[feat1] * X_test[feat2]
print(f'Test features: {X_test.shape}')
print(f'\n[5/7] Hyperparameter tuning LightGBM ({TUNE_ITERATIONS} iterations)...')
lgb_device = 'gpu' if USE_GPU else 'cpu'
lgb_param_grid = {'n_estimators': [300, 500, 700, 1000, 1200], 'max_depth': [4, 5, 6, 7, 8, 10, -1], 'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.07, 0.1], 'num_leaves': [31, 50, 70, 100, 127, 150], 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 'min_child_samples': [5, 10, 20, 30, 50], 'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1.0], 'reg_lambda': [0.5, 1, 1.5, 2, 3, 5]}
try:
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device=lgb_device), lgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1 if not USE_GPU else 1, verbose=1)
    lgb_random.fit(X, y)
    lgb_best_params = lgb_random.best_params_
    print(f'\n  Best CV AUC: {lgb_random.best_score_:.6f}')
    print(f'  Best params: {lgb_best_params}')
except Exception as e:
    print(f'  GPU failed: {e}')
    print('  Retrying with CPU...')
    lgb_device = 'cpu'
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device='cpu', force_col_wise=True), lgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1, verbose=1)
    lgb_random.fit(X, y)
    lgb_best_params = lgb_random.best_params_
    print(f'\n  Best CV AUC: {lgb_random.best_score_:.6f}')
    print(f'  Best params: {lgb_best_params}')
print(f'\n[6/7] {CV_FOLDS}-Fold Cross-Validation (LightGBM only)...')
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
cv_scores = []
cv_aucs = []
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/{CV_FOLDS}')
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    if lgb_device == 'gpu':
        lgb_fold = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, device=lgb_device)
    else:
        lgb_fold = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True, device='cpu')
    lgb_fold.fit(X_train_fold, y_train_fold)
    lgb_pred = lgb_fold.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = lgb_pred
    fold_pred_binary = (lgb_pred > 0.5).astype(int)
    fold_acc = accuracy_score(y_val_fold, fold_pred_binary)
    fold_auc = roc_auc_score(y_val_fold, lgb_pred)
    cv_scores.append(fold_acc)
    cv_aucs.append(fold_auc)
    print(f'  Accuracy: {fold_acc:.6f} | AUC: {fold_auc:.6f}')
    lgb_test = lgb_fold.predict_proba(X_test)[:, 1]
    test_predictions += lgb_test / CV_FOLDS
print('\n' + '=' * 70)
print('[7/7] RESULTS')
print('=' * 70)
oof_pred_binary = (oof_predictions > 0.5).astype(int)
oof_acc = accuracy_score(y, oof_pred_binary)
oof_auc = roc_auc_score(y, oof_predictions)
print(f'\nLightGBM Performance:')
print(f'  CV Accuracy: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})')
print(f'  CV AUC: {np.mean(cv_aucs):.6f} (+/- {np.std(cv_aucs):.6f})')
print(f'  Overall OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'  Overall OOF AUC: {oof_auc:.6f}')
print(f'\nDetailed Metrics:')
print(f'  Precision: {precision_score(y, oof_pred_binary):.6f}')
print(f'  Recall: {recall_score(y, oof_pred_binary):.6f}')
print(f'  F1-Score: {f1_score(y, oof_pred_binary):.6f}')
print(f'  MCC: {matthews_corrcoef(y, oof_pred_binary):.6f}')
print(f"  Cohen's Kappa: {cohen_kappa_score(y, oof_pred_binary):.6f}")
cm = confusion_matrix(y, oof_pred_binary)
print(f'\nConfusion Matrix:')
print(f'  Predicted:  Benign  Malware')
print(f'  Benign      {cm[0, 0]:6d}  {cm[0, 1]:6d}')
print(f'  Malware     {cm[1, 0]:6d}  {cm[1, 1]:6d}')
print(f'\nGenerating submission...')
test_pred_binary = (test_predictions > 0.5).astype(int)
submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred_binary})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_file = f'submission_lgbm_{timestamp}.csv'
submission.to_csv(submission_file, index=False)
print(f'\nSubmission saved: {submission_file}')
print(f'Malware: {(test_pred_binary == 1).sum()} ({(test_pred_binary == 1).sum() / len(test_pred_binary) * 100:.2f}%)')
print(f'Benign: {(test_pred_binary == 0).sum()} ({(test_pred_binary == 0).sum() / len(test_pred_binary) * 100:.2f}%)')
print('\nTraining final model on full data...')
if lgb_device == 'gpu':
    lgb_final = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, device=lgb_device)
else:
    lgb_final = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True, device='cpu')
lgb_final.fit(X, y)
joblib.dump(lgb_final, 'lgbm_final.pkl')
config = {'timestamp': timestamp, 'oof_accuracy': oof_acc, 'oof_auc': oof_auc, 'cv_accuracy_mean': np.mean(cv_scores), 'cv_accuracy_std': np.std(cv_scores), 'cv_auc_mean': np.mean(cv_aucs), 'cv_auc_std': np.std(cv_aucs), 'n_features': len(all_features), 'tune_iterations': TUNE_ITERATIONS, 'cv_folds': CV_FOLDS, 'features': all_features, 'best_params': lgb_best_params}
with open(f'config_lgbm_{timestamp}.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)
print('\n' + '=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Baseline (ensemble): 99.57%')
print(f'LightGBM-only OOF: {oof_acc * 100:.2f}%')
if oof_acc > 0.9957:
    print(f'Improved by {(oof_acc - 0.9957) * 100:.2f}%')
elif oof_acc >= 0.995:
    print(f'⚠ Close! ({(oof_acc - 0.9957) * 100:.2f}%)')
else:
    print(f'✗ Lower ({(oof_acc - 0.9957) * 100:.2f}%)')
print('=' * 70)
print('\nTop 20 Feature Importances:')
feature_importance = pd.DataFrame({'feature': all_features, 'importance': lgb_final.feature_importances_}).sort_values('importance', ascending=False)
print(feature_importance.head(20).to_string(index=False))

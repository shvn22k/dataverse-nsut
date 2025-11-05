import numpy as np
import pandas as pd
import warnings
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
print('=' * 70)
print('FINAL PUSH - TARGET: 99.68%')
print('=' * 70)
N_FEATURES = 60
TUNE_ITERATIONS = 100
CV_FOLDS = 5
OPTIMIZE_THRESHOLD = True
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except:
    pass
print(f'Using GPU: {USE_GPU}')
print('\n[1/9] Loading data...')
df = pd.read_csv('data/main.csv')
test_df = pd.read_csv('data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
target_col = 'Label'
id_col = 'sha256'
print('\n[2/9] Preprocessing...')
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
print(f'\n[3/9] Selecting top {N_FEATURES} features...')
X_full = df_processed.drop(columns=[target_col, id_col] if id_col in df_processed.columns else [target_col])
y_full = df_processed[target_col]
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
n_features = min(N_FEATURES, len(mi_df))
top_features = mi_df.head(n_features)['Feature'].tolist()
print(f'Selected {len(top_features)} features')
X = X_full[top_features].copy()
y = y_full.copy()
print(f'\n[4/9] Advanced feature engineering...')
top_10 = top_features[:10]
interaction_count = 0
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 4, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        interaction_count += 1
        X[f'{feat1}_+_{feat2}'] = X[feat1] + X[feat2]
        interaction_count += 1
        X[f'{feat1}_-_{feat2}'] = X[feat1] - X[feat2]
        interaction_count += 1
for i, feat in enumerate(top_features[:5]):
    X[f'{feat}_squared'] = X[feat] ** 2
    X[f'{feat}_sqrt'] = np.sqrt(np.abs(X[feat]))
    interaction_count += 2
print(f'Created {interaction_count} engineered features')
print(f'Total features: {X.shape[1]}')
all_features = X.columns.tolist()
X_test = test_processed[top_features].copy()
test_ids = test_processed[id_col].values
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 4, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
        X_test[f'{feat1}_+_{feat2}'] = X_test[feat1] + X_test[feat2]
        X_test[f'{feat1}_-_{feat2}'] = X_test[feat1] - X_test[feat2]
for i, feat in enumerate(top_features[:5]):
    X_test[f'{feat}_squared'] = X_test[feat] ** 2
    X_test[f'{feat}_sqrt'] = np.sqrt(np.abs(X_test[feat]))
print(f'Test features: {X_test.shape}')
print(f'\n[5/9] Hyperparameter tuning ({TUNE_ITERATIONS} iterations)...')
xgb_tree_method = 'gpu_hist' if USE_GPU else 'hist'
lgb_device = 'gpu' if USE_GPU else 'cpu'
xgb_param_grid = {'n_estimators': [500, 700, 1000, 1200, 1500], 'max_depth': [3, 4, 5, 6, 7, 8], 'learning_rate': [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'min_child_weight': [1, 2, 3, 5, 7, 10], 'gamma': [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3], 'reg_alpha': [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 'reg_lambda': [0.1, 0.5, 1, 1.5, 2, 3, 5, 10]}
lgb_param_grid = {'n_estimators': [500, 700, 1000, 1200, 1500], 'max_depth': [3, 4, 5, 6, 7, 8, 10, -1], 'learning_rate': [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1], 'num_leaves': [15, 31, 50, 70, 100, 127, 150, 200], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'min_child_samples': [5, 10, 15, 20, 30, 50], 'reg_alpha': [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 'reg_lambda': [0.1, 0.5, 1, 1.5, 2, 3, 5, 10]}
cat_param_grid = {'iterations': [500, 700, 1000, 1200, 1500], 'depth': [3, 4, 5, 6, 7, 8, 10], 'learning_rate': [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1], 'l2_leaf_reg': [1, 3, 5, 7, 9, 12], 'border_count': [32, 64, 128, 254], 'bagging_temperature': [0, 0.3, 0.5, 0.7, 1, 1.5, 2], 'random_strength': [0.3, 0.5, 0.7, 1, 1.5, 2]}
print('Tuning XGBoost (100 iterations)...')
xgb_random = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, tree_method=xgb_tree_method, eval_metric='logloss'), xgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1 if not USE_GPU else 1, verbose=0)
xgb_random.fit(X, y)
xgb_best_params = xgb_random.best_params_
print(f'  Best CV AUC: {xgb_random.best_score_:.6f}')
print('\nTuning LightGBM (100 iterations)...')
try:
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device=lgb_device), lgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1 if not USE_GPU else 1, verbose=0)
    lgb_random.fit(X, y)
    lgb_best_params = lgb_random.best_params_
    print(f'  Best CV AUC: {lgb_random.best_score_:.6f}')
except:
    lgb_device = 'cpu'
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device='cpu', force_col_wise=True), lgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1, verbose=0)
    lgb_random.fit(X, y)
    lgb_best_params = lgb_random.best_params_
    print(f'  Best CV AUC: {lgb_random.best_score_:.6f}')
print('\nTuning CatBoost (100 iterations)...')
try:
    cat_random = RandomizedSearchCV(CatBoostClassifier(random_state=42, verbose=0, task_type='GPU' if USE_GPU else 'CPU'), cat_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1, verbose=0)
    cat_random.fit(X, y)
    cat_best_params = cat_random.best_params_
    print(f'  Best CV AUC: {cat_random.best_score_:.6f}')
except:
    cat_random = RandomizedSearchCV(CatBoostClassifier(random_state=42, verbose=0, task_type='CPU'), cat_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1, verbose=0)
    cat_random.fit(X, y)
    cat_best_params = cat_random.best_params_
    print(f'  Best CV AUC: {cat_random.best_score_:.6f}')
print(f'\n[6/9] {CV_FOLDS}-Fold CV with Stacking...')
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X))
oof_proba_xgb = np.zeros(len(X))
oof_proba_lgb = np.zeros(len(X))
oof_proba_cat = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
test_proba_xgb = np.zeros(len(X_test))
test_proba_lgb = np.zeros(len(X_test))
test_proba_cat = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/{CV_FOLDS}')
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    xgb_fold = xgb.XGBClassifier(**xgb_best_params, random_state=42, tree_method=xgb_tree_method, eval_metric='logloss')
    if lgb_device == 'gpu':
        lgb_fold = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, device=lgb_device)
    else:
        lgb_fold = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True, device='cpu')
    cat_fold = CatBoostClassifier(**cat_best_params, random_state=42, verbose=0)
    xgb_fold.fit(X_train_fold, y_train_fold)
    lgb_fold.fit(X_train_fold, y_train_fold)
    cat_fold.fit(X_train_fold, y_train_fold)
    xgb_pred = xgb_fold.predict_proba(X_val_fold)[:, 1]
    lgb_pred = lgb_fold.predict_proba(X_val_fold)[:, 1]
    cat_pred = cat_fold.predict_proba(X_val_fold)[:, 1]
    oof_proba_xgb[val_idx] = xgb_pred
    oof_proba_lgb[val_idx] = lgb_pred
    oof_proba_cat[val_idx] = cat_pred
    xgb_auc = roc_auc_score(y_val_fold, xgb_pred)
    lgb_auc = roc_auc_score(y_val_fold, lgb_pred)
    cat_auc = roc_auc_score(y_val_fold, cat_pred)
    total_auc = xgb_auc + lgb_auc + cat_auc
    w_xgb = xgb_auc / total_auc
    w_lgb = lgb_auc / total_auc
    w_cat = cat_auc / total_auc
    fold_pred = w_xgb * xgb_pred + w_lgb * lgb_pred + w_cat * cat_pred
    oof_predictions[val_idx] = fold_pred
    fold_auc = roc_auc_score(y_val_fold, fold_pred)
    print(f'  Fold AUC: {fold_auc:.6f}')
    xgb_test = xgb_fold.predict_proba(X_test)[:, 1]
    lgb_test = lgb_fold.predict_proba(X_test)[:, 1]
    cat_test = cat_fold.predict_proba(X_test)[:, 1]
    test_proba_xgb += xgb_test / CV_FOLDS
    test_proba_lgb += lgb_test / CV_FOLDS
    test_proba_cat += cat_test / CV_FOLDS
    test_predictions += (w_xgb * xgb_test + w_lgb * lgb_test + w_cat * cat_test) / CV_FOLDS
print(f'\n[7/9] Training stacking meta-learner...')
meta_features = np.column_stack([oof_proba_xgb, oof_proba_lgb, oof_proba_cat])
meta_test = np.column_stack([test_proba_xgb, test_proba_lgb, test_proba_cat])
meta_model = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
meta_model.fit(meta_features, y)
stacked_predictions = meta_model.predict_proba(meta_features)[:, 1]
stacked_test = meta_model.predict_proba(meta_test)[:, 1]
print(f'Meta-learner coefficients: {meta_model.coef_[0]}')
if OPTIMIZE_THRESHOLD:
    print(f'\n[8/9] Optimizing decision threshold...')
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_acc = 0
    for threshold in thresholds:
        pred_binary = (stacked_predictions > threshold).astype(int)
        acc = accuracy_score(y, pred_binary)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    print(f'  Optimal threshold: {best_threshold:.3f}')
    print(f'  Accuracy at optimal threshold: {best_acc:.6f}')
    print(f'  Improvement over 0.5: {(best_acc - accuracy_score(y, (stacked_predictions > 0.5).astype(int))) * 100:.3f}%')
else:
    best_threshold = 0.5
    best_acc = accuracy_score(y, (stacked_predictions > 0.5).astype(int))
    print(f'\n[8/9] Using default threshold: 0.5')
print('\n' + '=' * 70)
print('[9/9] FINAL RESULTS')
print('=' * 70)
oof_pred_binary = (stacked_predictions > best_threshold).astype(int)
oof_acc = accuracy_score(y, oof_pred_binary)
oof_auc = roc_auc_score(y, stacked_predictions)
print(f'\nStacked Ensemble Performance:')
print(f'  OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'  OOF AUC: {oof_auc:.6f}')
print(f'  Precision: {precision_score(y, oof_pred_binary):.6f}')
print(f'  Recall: {recall_score(y, oof_pred_binary):.6f}')
print(f'  F1-Score: {f1_score(y, oof_pred_binary):.6f}')
print(f'  MCC: {matthews_corrcoef(y, oof_pred_binary):.6f}')
cm = confusion_matrix(y, oof_pred_binary)
print(f'\nConfusion Matrix:')
print(f'  Predicted:  Benign  Malware')
print(f'  Benign      {cm[0, 0]:6d}  {cm[0, 1]:6d}')
print(f'  Malware     {cm[1, 0]:6d}  {cm[1, 1]:6d}')
print(f'\nGenerating submission...')
test_pred_binary = (stacked_test > best_threshold).astype(int)
submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred_binary})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_file = f'submission-3/submission_final_{timestamp}.csv'
submission.to_csv(submission_file, index=False)
print(f'\nSubmission saved: {submission_file}')
print(f'Malware: {(test_pred_binary == 1).sum()} ({(test_pred_binary == 1).sum() / len(test_pred_binary) * 100:.2f}%)')
print(f'Benign: {(test_pred_binary == 0).sum()} ({(test_pred_binary == 0).sum() / len(test_pred_binary) * 100:.2f}%)')
print('\nSaving models...')
xgb_final = xgb.XGBClassifier(**xgb_best_params, random_state=42, tree_method=xgb_tree_method, eval_metric='logloss')
xgb_final.fit(X, y)
joblib.dump(xgb_final, 'submission-3/xgb_final.pkl')
if lgb_device == 'gpu':
    lgb_final = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, device=lgb_device)
else:
    lgb_final = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True, device='cpu')
lgb_final.fit(X, y)
joblib.dump(lgb_final, 'submission-3/lgb_final.pkl')
cat_final = CatBoostClassifier(**cat_best_params, random_state=42, verbose=0)
cat_final.fit(X, y)
joblib.dump(cat_final, 'submission-3/cat_final.pkl')
joblib.dump(meta_model, 'submission-3/meta_model.pkl')
config = {'timestamp': timestamp, 'oof_accuracy': oof_acc, 'oof_auc': oof_auc, 'optimal_threshold': best_threshold, 'n_features': len(all_features), 'tune_iterations': TUNE_ITERATIONS, 'cv_folds': CV_FOLDS, 'best_params': {'xgb': xgb_best_params, 'lgb': lgb_best_params, 'cat': cat_best_params}, 'meta_coef': meta_model.coef_[0].tolist()}
with open(f'submission-3/config_final_{timestamp}.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)
print('\n' + '=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Current best: 99.57%')
print(f'Target: 99.68%')
print(f'OOF Result: {oof_acc * 100:.2f}%')
if oof_acc > 0.9957:
    print(f'Improved by {(oof_acc - 0.9957) * 100:.2f}%')
else:
    print(f'Not improved ({(oof_acc - 0.9957) * 100:.2f}%)')
print('=' * 70)

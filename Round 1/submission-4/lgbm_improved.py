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
print('LIGHTGBM IMPROVED - TARGETING 99.68%')
print('=' * 70)
N_FEATURES = 60
TUNE_ITERATIONS = 50
CV_FOLDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
OPTIMIZE_THRESHOLD = True
CLASS_WEIGHT_TUNING = True
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except:
    pass
print(f'Using GPU: {USE_GPU}')
print(f'Multi-seed bagging: {len(RANDOM_SEEDS)} seeds')
print(f'Threshold optimization: {OPTIMIZE_THRESHOLD}')
print('\n[1/8] Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
target_col = 'Label'
id_col = 'sha256'
print('\n[2/8] Preprocessing...')
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
print(f'\n[3/8] Selecting top {N_FEATURES} features...')
X_full = df_processed.drop(columns=[target_col, id_col] if id_col in df_processed.columns else [target_col])
y_full = df_processed[target_col]
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
n_features = min(N_FEATURES, len(mi_df))
top_features = mi_df.head(n_features)['Feature'].tolist()
print(f'Selected {len(top_features)} features')
X = X_full[top_features].copy()
y = y_full.copy()
print(f'\n[4/8] Advanced feature engineering...')
top_10 = top_features[:10]
interaction_count = 0
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        interaction_count += 1
for i in range(min(5, len(top_10))):
    for j in range(i + 1, min(i + 2, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-05)
        interaction_count += 1
for feat in top_features[:5]:
    X[f'{feat}_sq'] = X[feat] ** 2
    X[f'{feat}_sqrt'] = np.sqrt(np.abs(X[feat]))
    X[f'{feat}_log'] = np.log1p(np.abs(X[feat]))
    interaction_count += 3
print(f'Created {interaction_count} engineered features')
print(f'Total features before selection: {X.shape[1]}')
all_features = X.columns.tolist()
X_test = test_processed[top_features].copy()
test_ids = test_processed[id_col].values
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
for i in range(min(5, len(top_10))):
    for j in range(i + 1, min(i + 2, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X_test[f'{feat1}_div_{feat2}'] = X_test[feat1] / (X_test[feat2] + 1e-05)
for feat in top_features[:5]:
    X_test[f'{feat}_sq'] = X_test[feat] ** 2
    X_test[f'{feat}_sqrt'] = np.sqrt(np.abs(X_test[feat]))
    X_test[f'{feat}_log'] = np.log1p(np.abs(X_test[feat]))
best_class_weight = None
if CLASS_WEIGHT_TUNING:
    print(f'\n[5/8] Tuning class weights...')
    class_weight_options = [None, 'balanced', {0: 1, 1: 1.05}, {0: 1, 1: 1.1}, {0: 1, 1: 0.95}, {0: 1, 1: 0.9}]
    best_score = 0
    for cw in class_weight_options:
        model = lgb.LGBMClassifier(n_estimators=500, random_state=42, verbose=-1, class_weight=cw)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
        avg_score = scores.mean()
        if avg_score > best_score:
            best_score = avg_score
            best_class_weight = cw
    print(f'  Best class weight: {best_class_weight}')
    print(f'  CV Accuracy: {best_score:.6f}')
else:
    print(f'\n[5/8] Skipping class weight tuning...')
    best_class_weight = None
print(f'\n[6/8] Hyperparameter tuning ({TUNE_ITERATIONS} iterations)...')
lgb_device = 'gpu' if USE_GPU else 'cpu'
lgb_param_grid = {'n_estimators': [500, 700, 1000, 1200, 1500], 'max_depth': [4, 5, 6, 7, 8, 10, -1], 'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1], 'num_leaves': [31, 50, 70, 100, 127, 150], 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 'min_child_samples': [5, 10, 20, 30, 50], 'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1.0], 'reg_lambda': [0.5, 1, 1.5, 2, 3, 5]}
try:
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device=lgb_device, class_weight=best_class_weight), lgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1 if not USE_GPU else 1, verbose=0)
    lgb_random.fit(X, y)
    lgb_best_params = lgb_random.best_params_
    print(f'  Best CV AUC: {lgb_random.best_score_:.6f}')
except:
    lgb_device = 'cpu'
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device='cpu', force_col_wise=True, class_weight=best_class_weight), lgb_param_grid, n_iter=TUNE_ITERATIONS, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), random_state=42, n_jobs=-1, verbose=0)
    lgb_random.fit(X, y)
    lgb_best_params = lgb_random.best_params_
    print(f'  Best CV AUC: {lgb_random.best_score_:.6f}')
print(f'\n[7/8] Multi-seed bagging with {CV_FOLDS}-Fold CV...')
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
    print(f'\n  Seed {seed_idx}/{len(RANDOM_SEEDS)} (seed={seed})')
    seed_oof = np.zeros(len(X))
    seed_test = np.zeros(len(X_test))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        if lgb_device == 'gpu':
            lgb_fold = lgb.LGBMClassifier(**lgb_best_params, random_state=seed, verbose=-1, device=lgb_device, class_weight=best_class_weight)
        else:
            lgb_fold = lgb.LGBMClassifier(**lgb_best_params, random_state=seed, verbose=-1, force_col_wise=True, device='cpu', class_weight=best_class_weight)
        lgb_fold.fit(X_train_fold, y_train_fold)
        lgb_pred = lgb_fold.predict_proba(X_val_fold)[:, 1]
        seed_oof[val_idx] = lgb_pred
        lgb_test = lgb_fold.predict_proba(X_test)[:, 1]
        seed_test += lgb_test / CV_FOLDS
    oof_predictions += seed_oof / len(RANDOM_SEEDS)
    test_predictions += seed_test / len(RANDOM_SEEDS)
    seed_acc = accuracy_score(y, (seed_oof > 0.5).astype(int))
    seed_auc = roc_auc_score(y, seed_oof)
    print(f'    Seed OOF - Acc: {seed_acc:.6f}, AUC: {seed_auc:.6f}')
if OPTIMIZE_THRESHOLD:
    print(f'\n[8/8] Optimizing decision threshold...')
    thresholds = np.arange(0.3, 0.7, 0.005)
    threshold_results = []
    for threshold in thresholds:
        pred_binary = (oof_predictions > threshold).astype(int)
        acc = accuracy_score(y, pred_binary)
        threshold_results.append((threshold, acc))
    best_threshold, best_acc = max(threshold_results, key=lambda x: x[1])
    default_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
    print(f'  Optimal threshold: {best_threshold:.3f}')
    print(f'  Accuracy at optimal threshold: {best_acc:.6f}')
    print(f'  Accuracy at default (0.5): {default_acc:.6f}')
    print(f'  Improvement: {(best_acc - default_acc) * 100:.3f}%')
else:
    best_threshold = 0.5
    best_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 70)
print('FINAL RESULTS')
print('=' * 70)
oof_pred_binary = (oof_predictions > best_threshold).astype(int)
oof_acc = accuracy_score(y, oof_pred_binary)
oof_auc = roc_auc_score(y, oof_predictions)
print(f'\nMulti-Seed Bagged LightGBM:')
print(f'  OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'  OOF AUC: {oof_auc:.6f}')
print(f'  Optimal Threshold: {best_threshold:.3f}')
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
test_pred_binary = (test_predictions > best_threshold).astype(int)
submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred_binary})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_file = f'submission_lgbm_improved_{timestamp}.csv'
submission.to_csv(submission_file, index=False)
print(f'\nSubmission saved: {submission_file}')
print(f'Malware: {(test_pred_binary == 1).sum()} ({(test_pred_binary == 1).sum() / len(test_pred_binary) * 100:.2f}%)')
print(f'Benign: {(test_pred_binary == 0).sum()} ({(test_pred_binary == 0).sum() / len(test_pred_binary) * 100:.2f}%)')
print('\nTraining final models (one per seed)...')
models = []
for seed in RANDOM_SEEDS:
    if lgb_device == 'gpu':
        lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=seed, verbose=-1, device=lgb_device, class_weight=best_class_weight)
    else:
        lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=seed, verbose=-1, force_col_wise=True, device='cpu', class_weight=best_class_weight)
    lgb_model.fit(X, y)
    models.append(lgb_model)
joblib.dump(models, f'lgbm_models_bag_{timestamp}.pkl')
config = {'timestamp': timestamp, 'oof_accuracy': oof_acc, 'oof_auc': oof_auc, 'optimal_threshold': best_threshold, 'threshold_improvement': (best_acc - default_acc) * 100 if OPTIMIZE_THRESHOLD else 0, 'n_features': len(all_features), 'n_seeds': len(RANDOM_SEEDS), 'seeds': RANDOM_SEEDS, 'class_weight': str(best_class_weight), 'best_params': lgb_best_params}
with open(f'config_lgbm_improved_{timestamp}.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)
print('\n' + '=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Target: 99.68%')
print(f'Baseline: 99.57%')
print(f'Result: {oof_acc * 100:.2f}%')
if oof_acc >= 0.9968:
    print(f'🎯 TARGET ACHIEVED! (+{(oof_acc - 0.9957) * 100:.2f}%)')
elif oof_acc > 0.9957:
    print(f'Improved by {(oof_acc - 0.9957) * 100:.2f}%')
elif oof_acc >= 0.995:
    print(f'⚠ Close! ({(oof_acc - 0.9957) * 100:.2f}%)')
else:
    print(f'✗ Lower ({(oof_acc - 0.9957) * 100:.2f}%)')
print('=' * 70)

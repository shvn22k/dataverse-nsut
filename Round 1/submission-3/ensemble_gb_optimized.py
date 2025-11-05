import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
from datetime import datetime
warnings.filterwarnings('ignore')
try:
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.feature_selection import mutual_info_classif as cu_mutual_info
    USE_CUML = True
    print(' cuML loaded - GPU acceleration enabled')
except ImportError:
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_classif
    USE_CUML = False
    print(' cuML not available - using sklearn (CPU)')
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import joblib
DATA_DIR = Path('../data')
OUTPUT_DIR = Path('.')
TRAIN_FILE = DATA_DIR / 'main.csv'
TEST_FILE = DATA_DIR / 'test.csv'
SAMPLE_SUB = DATA_DIR / 'sample_submission.csv'
TOP_FEATURES = 80
INTERACTION_PAIRS = 15
N_SPLITS = 5
RANDOM_STATE = 42
N_ITER = 30
CV_FOLDS = 3
ENSEMBLE_WEIGHTS = {'xgb': 0.33, 'lgb': 0.33, 'cat': 0.34}
print('=' * 70)
print('GPU-ACCELERATED GRADIENT BOOSTING ENSEMBLE')
print('=' * 70)
print(f'Train file: {TRAIN_FILE}')
print(f'Test file: {TEST_FILE}')
print(f'Output dir: {OUTPUT_DIR}')
print(f'Using cuML GPU acceleration: {USE_CUML}')
print('=' * 70)
print('\n[1/9] Loading data...')
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
X = train_df.drop(['sha256', 'is_malware'], axis=1)
y = train_df['is_malware']
X_test = test_df.drop(['sha256'], axis=1)
test_ids = test_df['sha256'].values
print(f'Features: {X.shape[1]}')
print(f'Target distribution: {y.value_counts().to_dict()}')
print('\n[2/9] Selecting top features by mutual information...')
if USE_CUML:
    try:
        mi_scores = cu_mutual_info(X.values, y.values, random_state=RANDOM_STATE)
        mi_scores = mi_scores.get() if hasattr(mi_scores, 'get') else mi_scores
    except:
        print('  cuML MI failed, falling back to sklearn')
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
else:
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
feature_scores = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
top_features = feature_scores.head(TOP_FEATURES)['feature'].tolist()
print(f'Selected {len(top_features)} features')
print(f'Top 5 features: {top_features[:5]}')
X_selected = X[top_features].copy()
X_test_selected = X_test[top_features].copy()
print('\n[3/9] Engineering interaction features...')
top_interaction_features = top_features[:INTERACTION_PAIRS]
interaction_count = 0
for i in range(len(top_interaction_features)):
    for j in range(i + 1, min(i + 4, len(top_interaction_features))):
        feat1 = top_interaction_features[i]
        feat2 = top_interaction_features[j]
        new_feat_name = f'{feat1}_x_{feat2}'
        X_selected[new_feat_name] = X_selected[feat1] * X_selected[feat2]
        X_test_selected[new_feat_name] = X_test_selected[feat1] * X_test_selected[feat2]
        interaction_count += 1
print(f'Created {interaction_count} interaction features')
print(f'Final feature count: {X_selected.shape[1]}')
print('\n[4/9] Scaling features...')
if USE_CUML:
    try:
        scaler = cuStandardScaler()
        X_scaled = scaler.fit_transform(X_selected.values)
        X_test_scaled = scaler.transform(X_test_selected.values)
        X_scaled = X_scaled.get() if hasattr(X_scaled, 'get') else X_scaled
        X_test_scaled = X_test_scaled.get() if hasattr(X_test_scaled, 'get') else X_test_scaled
        print('  Using cuML GPU scaler')
    except Exception as e:
        print(f'  cuML scaling failed: {e}')
        print('  Falling back to sklearn')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected.values)
        X_test_scaled = scaler.transform(X_test_selected.values)
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected.values)
    X_test_scaled = scaler.transform(X_test_selected.values)
X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_selected.columns, index=X_test_selected.index)
print('Scaling complete')
print('\n[5/9] Tuning XGBoost...')
xgb_param_dist = {'max_depth': [4, 5, 6, 7, 8], 'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1], 'n_estimators': [200, 300, 500, 700], 'subsample': [0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0], 'min_child_weight': [1, 3, 5, 7], 'gamma': [0, 0.1, 0.2, 0.3], 'reg_alpha': [0, 0.01, 0.1, 1], 'reg_lambda': [0.1, 1, 5, 10]}
try:
    xgb_base = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
    print('  XGBoost GPU mode enabled')
except:
    xgb_base = xgb.XGBClassifier(tree_method='hist', random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
    print('  XGBoost CPU mode')
xgb_search = RandomizedSearchCV(xgb_base, xgb_param_dist, n_iter=N_ITER, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
xgb_search.fit(X_scaled, y)
xgb_best = xgb_search.best_estimator_
print(f'  Best XGBoost score: {xgb_search.best_score_:.4f}')
print(f'  Best params: {xgb_search.best_params_}')
print('\n[6/9] Tuning LightGBM...')
lgb_param_dist = {'max_depth': [4, 5, 6, 7, 8, -1], 'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1], 'n_estimators': [200, 300, 500, 700], 'num_leaves': [31, 50, 70, 100, 127], 'subsample': [0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0], 'min_child_samples': [5, 10, 20, 30], 'reg_alpha': [0, 0.01, 0.1, 1], 'reg_lambda': [0.1, 1, 5, 10]}
try:
    lgb_base = lgb.LGBMClassifier(device='gpu', random_state=RANDOM_STATE, verbose=-1)
    lgb_base.fit(X_scaled.head(100), y.head(100))
    print('  LightGBM GPU mode enabled')
except:
    lgb_base = lgb.LGBMClassifier(device='cpu', random_state=RANDOM_STATE, verbose=-1)
    print('  LightGBM CPU mode')
lgb_search = RandomizedSearchCV(lgb_base, lgb_param_dist, n_iter=N_ITER, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
lgb_search.fit(X_scaled, y)
lgb_best = lgb_search.best_estimator_
print(f'  Best LightGBM score: {lgb_search.best_score_:.4f}')
print(f'  Best params: {lgb_search.best_params_}')
print('\n[7/9] Tuning CatBoost...')
cat_param_dist = {'depth': [4, 5, 6, 7, 8], 'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1], 'iterations': [200, 300, 500, 700], 'l2_leaf_reg': [1, 3, 5, 7, 9], 'border_count': [32, 64, 128, 254], 'bagging_temperature': [0, 0.5, 1, 2]}
try:
    cat_base = CatBoostClassifier(task_type='GPU', random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)
    cat_base.fit(X_scaled.head(100), y.head(100))
    print('  CatBoost GPU mode enabled')
except:
    cat_base = CatBoostClassifier(task_type='CPU', random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)
    print('  CatBoost CPU mode')
cat_search = RandomizedSearchCV(cat_base, cat_param_dist, n_iter=N_ITER, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
cat_search.fit(X_scaled, y)
cat_best = cat_search.best_estimator_
print(f'  Best CatBoost score: {cat_search.best_score_:.4f}')
print(f'  Best params: {cat_search.best_params_}')
print(f'\n[8/9] Cross-validating ensemble ({N_SPLITS}-fold)...')
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
oof_predictions = np.zeros(len(X_scaled))
test_predictions = np.zeros(len(X_test_scaled))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f'\n  Fold {fold}/{N_SPLITS}')
    X_train_fold = X_scaled.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X_scaled.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    xgb_fold = xgb.XGBClassifier(**xgb_best.get_params())
    lgb_fold = lgb.LGBMClassifier(**lgb_best.get_params())
    cat_fold = CatBoostClassifier(**cat_best.get_params())
    xgb_fold.fit(X_train_fold, y_train_fold)
    lgb_fold.fit(X_train_fold, y_train_fold)
    cat_fold.fit(X_train_fold, y_train_fold)
    xgb_pred = xgb_fold.predict_proba(X_val_fold)[:, 1]
    lgb_pred = lgb_fold.predict_proba(X_val_fold)[:, 1]
    cat_pred = cat_fold.predict_proba(X_val_fold)[:, 1]
    fold_pred = ENSEMBLE_WEIGHTS['xgb'] * xgb_pred + ENSEMBLE_WEIGHTS['lgb'] * lgb_pred + ENSEMBLE_WEIGHTS['cat'] * cat_pred
    oof_predictions[val_idx] = fold_pred
    fold_pred_binary = (fold_pred > 0.5).astype(int)
    fold_acc = accuracy_score(y_val_fold, fold_pred_binary)
    fold_auc = roc_auc_score(y_val_fold, fold_pred)
    cv_scores.append(fold_acc)
    print(f'    Accuracy: {fold_acc:.4f} | AUC: {fold_auc:.4f}')
    xgb_test_pred = xgb_fold.predict_proba(X_test_scaled)[:, 1]
    lgb_test_pred = lgb_fold.predict_proba(X_test_scaled)[:, 1]
    cat_test_pred = cat_fold.predict_proba(X_test_scaled)[:, 1]
    test_predictions += ENSEMBLE_WEIGHTS['xgb'] * xgb_test_pred + ENSEMBLE_WEIGHTS['lgb'] * lgb_test_pred + ENSEMBLE_WEIGHTS['cat'] * cat_test_pred
test_predictions /= N_SPLITS
oof_pred_binary = (oof_predictions > 0.5).astype(int)
cv_acc = accuracy_score(y, oof_pred_binary)
cv_auc = roc_auc_score(y, oof_predictions)
print('\n' + '=' * 70)
print('CROSS-VALIDATION RESULTS')
print('=' * 70)
print(f'Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})')
print(f'Overall Accuracy: {cv_acc:.4f}')
print(f'Overall AUC: {cv_auc:.4f}')
precision = precision_score(y, oof_pred_binary)
recall = recall_score(y, oof_pred_binary)
f1 = f1_score(y, oof_pred_binary)
mcc = matthews_corrcoef(y, oof_pred_binary)
kappa = cohen_kappa_score(y, oof_pred_binary)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print(f"Cohen's Kappa: {kappa:.4f}")
cm = confusion_matrix(y, oof_pred_binary)
print(f'\nConfusion Matrix:\n{cm}')
print('\n[9/9] Generating submission file...')
test_pred_binary = (test_predictions > 0.5).astype(int)
submission = pd.DataFrame({'sha256': test_ids, 'is_malware': test_pred_binary})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_file = OUTPUT_DIR / f'submission_ensemble_{timestamp}.csv'
submission.to_csv(submission_file, index=False)
print(f'Submission saved: {submission_file}')
print(f'Predictions: {pd.Series(test_pred_binary).value_counts().to_dict()}')
print('\nSaving models and configuration...')
joblib.dump(xgb_best, OUTPUT_DIR / 'xgb_best.pkl')
joblib.dump(lgb_best, OUTPUT_DIR / 'lgb_best.pkl')
joblib.dump(cat_best, OUTPUT_DIR / 'cat_best.pkl')
joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
config = {'timestamp': timestamp, 'top_features': top_features, 'interaction_count': interaction_count, 'final_feature_count': X_scaled.shape[1], 'cv_accuracy': cv_acc, 'cv_auc': cv_auc, 'xgb_best_params': xgb_best.get_params(), 'lgb_best_params': lgb_best.get_params(), 'cat_best_params': cat_best.get_params(), 'ensemble_weights': ENSEMBLE_WEIGHTS, 'cv_scores': [float(s) for s in cv_scores]}
with open(OUTPUT_DIR / f'config_ensemble_{timestamp}.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)
print('\n' + '=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Models saved in: {OUTPUT_DIR}')
print(f'Submission file: {submission_file}')
print(f'Final CV Accuracy: {cv_acc:.4f}')
print(f'Final CV AUC: {cv_auc:.4f}')
print('=' * 70)

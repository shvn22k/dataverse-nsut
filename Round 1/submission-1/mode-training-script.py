import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import joblib
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, average_precision_score, matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
print('Training XGBoost + LightGBM Ensemble')
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f'GPU available: {torch.cuda.get_device_name(0)}')
    else:
        print('Running on CPU')
except:
    print('Running on CPU')
print('\nLoading data...')
df = pd.read_csv('data/main.csv')
test_df = pd.read_csv('data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
target_col = 'Label'
id_col = 'sha256'
df_processed = df.copy()
test_processed = test_df.copy()
for col in df_processed.select_dtypes(include=[np.number]).columns:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(median_val, inplace=True)
constant_features = []
for col in df_processed.select_dtypes(include=[np.number]).columns:
    if col not in [target_col, id_col]:
        if df_processed[col].nunique() == 1:
            constant_features.append(col)
if len(constant_features) > 0:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
print('\nFeature selection...')
X_full = df_processed.drop(columns=[target_col, id_col] if id_col in df_processed.columns else [target_col])
y_full = df_processed[target_col]
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
print('Top 20 features:')
print(mi_df.head(20))
n_features = min(40, len(mi_df))
top_features = mi_df.head(n_features)['Feature'].tolist()
print(f'Selected {len(top_features)} features')
X = X_full[top_features]
y = y_full
X_test = test_processed[top_features] if id_col in test_processed.columns else test_processed[top_features]
test_ids = test_processed[id_col] if id_col in test_processed.columns else None
print('\nSplitting data...')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')
print('\nHyperparameter tuning...')
xgb_tree_method = 'gpu_hist' if USE_GPU else 'hist'
lgb_device = 'gpu' if USE_GPU else 'cpu'
xgb_param_grid = {'n_estimators': [300, 500, 700], 'max_depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5], 'gamma': [0, 0.1, 0.2], 'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [1, 1.5, 2]}
lgb_param_grid = {'n_estimators': [300, 500, 700], 'max_depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_samples': [10, 20, 30], 'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [1, 1.5, 2]}
print('Tuning XGBoost...')
xgb_random = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, tree_method=xgb_tree_method, eval_metric='logloss'), xgb_param_grid, n_iter=15, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1 if not USE_GPU else 1, verbose=0)
xgb_random.fit(X_train, y_train)
xgb_best_params = xgb_random.best_params_
print(f'Best params: {xgb_best_params}')
print(f'CV score: {xgb_random.best_score_:.4f}')
print('\nTuning LightGBM...')
try:
    lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device=lgb_device), lgb_param_grid, n_iter=15, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1 if not USE_GPU else 1, verbose=0)
    lgb_random.fit(X_train, y_train)
    lgb_best_params = lgb_random.best_params_
    print(f'Best params: {lgb_best_params}')
    print(f'CV score: {lgb_random.best_score_:.4f}')
except Exception as e:
    if USE_GPU:
        print('GPU failed, using CPU')
        lgb_device = 'cpu'
        lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, device='cpu', force_col_wise=True), lgb_param_grid, n_iter=15, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=0)
        lgb_random.fit(X_train, y_train)
        lgb_best_params = lgb_random.best_params_
        print(f'Best params: {lgb_best_params}')
        print(f'CV score: {lgb_random.best_score_:.4f}')
print('\nTraining models...')
print('XGBoost...')
t0 = time.time()
xgb_model = xgb.XGBClassifier(**xgb_best_params, random_state=42, tree_method=xgb_tree_method, eval_metric='logloss', early_stopping_rounds=50)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_time = time.time() - t0
print(f'Done ({xgb_time:.1f}s)')
print('LightGBM...')
t0 = time.time()
if lgb_device == 'gpu':
    lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, device=lgb_device)
else:
    lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True, device='cpu')
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
lgb_time = time.time() - t0
print(f'Done ({lgb_time:.1f}s)')
print(f'\nTotal time: {xgb_time + lgb_time:.1f}s')
print('\nEvaluating...')
y_pred_xgb = xgb_model.predict(X_val)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_pred_lgb = lgb_model.predict(X_val)
y_pred_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]

def calculate_all_metrics(y_true, y_pred, y_pred_proba, model_name):
    cm = confusion_matrix(y_true, y_pred)
    metrics = {'Model': model_name, 'Accuracy': accuracy_score(y_true, y_pred), 'Precision': precision_score(y_true, y_pred), 'Recall': recall_score(y_true, y_pred), 'F1-Score': f1_score(y_true, y_pred), 'ROC-AUC': roc_auc_score(y_true, y_pred_proba), 'Average Precision': average_precision_score(y_true, y_pred_proba), 'Matthews Corr': matthews_corrcoef(y_true, y_pred), 'Cohen Kappa': cohen_kappa_score(y_true, y_pred), 'Specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]), 'NPV': cm[0, 0] / (cm[0, 0] + cm[1, 0])}
    return metrics
xgb_metrics = calculate_all_metrics(y_val, y_pred_xgb, y_pred_proba_xgb, 'XGBoost')
lgb_metrics = calculate_all_metrics(y_val, y_pred_lgb, y_pred_proba_lgb, 'LightGBM')
total_auc = xgb_metrics['ROC-AUC'] + lgb_metrics['ROC-AUC']
w_xgb = xgb_metrics['ROC-AUC'] / total_auc
w_lgb = lgb_metrics['ROC-AUC'] / total_auc
print(f'\nEnsemble weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}')
y_pred_proba_ensemble = w_xgb * y_pred_proba_xgb + w_lgb * y_pred_proba_lgb
y_pred_ensemble = (y_pred_proba_ensemble >= 0.5).astype(int)
ensemble_metrics = calculate_all_metrics(y_val, y_pred_ensemble, y_pred_proba_ensemble, 'Ensemble')
all_metrics = pd.DataFrame([xgb_metrics, lgb_metrics, ensemble_metrics])
print('\nModel Performance:')
print(all_metrics.to_string(index=False))
print('\nClassification Report:')
print(classification_report(y_val, y_pred_ensemble, target_names=['Benign', 'Malware']))
cm = confusion_matrix(y_val, y_pred_ensemble)
print(f'\nConfusion Matrix:')
print(f'Predicted:  Benign  Malware')
print(f'Benign      {cm[0, 0]:5d}   {cm[0, 1]:5d}')
print(f'Malware     {cm[1, 0]:5d}   {cm[1, 1]:5d}')
print('\nSaving models...')
joblib.dump(xgb_model, 'submission-1/xgboost_model.pkl')
joblib.dump(lgb_model, 'submission-1/lightgbm_model.pkl')
ensemble_config = {'weights': {'xgb': w_xgb, 'lgb': w_lgb}, 'features': top_features, 'metrics': all_metrics.to_dict('records'), 'best_params': {'xgb': xgb_best_params, 'lgb': lgb_best_params}}
with open('submission-1/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=4)
print('Models saved')
print('\nGenerating predictions...')
test_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
test_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
test_pred_proba_ensemble = w_xgb * test_pred_proba_xgb + w_lgb * test_pred_proba_lgb
test_pred_ensemble = (test_pred_proba_ensemble >= 0.5).astype(int)
submission = pd.DataFrame({id_col: test_ids if test_ids is not None else range(len(test_pred_ensemble)), 'Label': test_pred_ensemble})
submission.to_csv('submission-1/submission.csv', index=False)
print(f'\nSubmission created: {submission.shape}')
print(f'Malware: {(test_pred_ensemble == 1).sum()} ({(test_pred_ensemble == 1).sum() / len(test_pred_ensemble) * 100:.1f}%)')
print(f'Benign: {(test_pred_ensemble == 0).sum()} ({(test_pred_ensemble == 0).sum() / len(test_pred_ensemble) * 100:.1f}%)')
print('\nDone!')
print(f'Val ROC-AUC: {ensemble_metrics['ROC-AUC']:.4f}')
print(f'Val Accuracy: {ensemble_metrics['Accuracy']:.4f}')
print(f'Val F1: {ensemble_metrics['F1-Score']:.4f}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, average_precision_score, matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
print('=' * 80)
print('EXPLORING MODEL IMPROVEMENTS')
print('=' * 80)
print('\n1. Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}')
print(f'Test: {test_df.shape}')
print(f'Target distribution: {df['Label'].value_counts(normalize=True).to_dict()}')
target_col = 'Label'
id_col = 'sha256'
print('\n2. Preprocessing...')
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
if constant_features:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
X_full = df_processed.drop(columns=[target_col, id_col] if id_col in df_processed.columns else [target_col])
y_full = df_processed[target_col]
print(f'Features: {X_full.shape[1]}, Samples: {X_full.shape[0]}')
print('\n3. Feature selection...')
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
print('Top 20 features:')
print(mi_df.head(20).to_string(index=False))
n_top_features = 50
top_features = mi_df.head(n_top_features)['Feature'].tolist()
print(f'\nSelected {len(top_features)} top features')
print('\n4. Feature engineering...')
X_engineered = X_full[top_features].copy()
interaction_pairs = [('Call Interception', 'Sim Card Manipulation'), ('WiFi Network Hijacking', 'Network Traffic Interception'), ('Keylogging', 'Credential Theft'), ('Data Exfiltration', 'Remote Command Execution'), ('Social Engineering Attack', 'Fake App Installation')]
for feat1, feat2 in interaction_pairs:
    if feat1 in X_engineered.columns and feat2 in X_engineered.columns:
        new_feat = f'{feat1}_x_{feat2}'
        X_engineered[new_feat] = X_engineered[feat1] * X_engineered[feat2]
        print(f'Created: {new_feat}')
sum_pairs = [('Audio Surveillance', 'Screen Logging'), ('Location Tracking', 'Contact Information Theft')]
for feat1, feat2 in sum_pairs:
    if feat1 in X_engineered.columns and feat2 in X_engineered.columns:
        new_feat = f'{feat1}_sum_{feat2}'
        X_engineered[new_feat] = X_engineered[feat1] + X_engineered[feat2]
        print(f'Created: {new_feat}')
print(f'\nTotal features: {X_engineered.shape[1]}')
X_test_engineered = test_processed[top_features].copy()
for feat1, feat2 in interaction_pairs:
    if feat1 in X_test_engineered.columns and feat2 in X_test_engineered.columns:
        X_test_engineered[f'{feat1}_x_{feat2}'] = X_test_engineered[feat1] * X_test_engineered[feat2]
for feat1, feat2 in sum_pairs:
    if feat1 in X_test_engineered.columns and feat2 in X_test_engineered.columns:
        X_test_engineered[f'{feat1}_sum_{feat2}'] = X_test_engineered[feat1] + X_test_engineered[feat2]
test_ids = test_processed[id_col]
print('\n5. Splitting data...')
X_train, X_val, y_train, y_val = train_test_split(X_engineered, y_full, test_size=0.2, random_state=42, stratify=y_full)
print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test_engineered.shape[0]}')
print('\n6. Hyperparameter tuning...')
xgb_param_grid = {'n_estimators': [500, 700, 1000], 'max_depth': [4, 6, 8, 10], 'learning_rate': [0.01, 0.03, 0.05, 0.07], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5], 'gamma': [0, 0.1, 0.2, 0.3], 'reg_alpha': [0, 0.1, 0.5, 1.0], 'reg_lambda': [1, 1.5, 2, 3]}
print('Tuning XGBoost (50 iters, 5-fold CV)...')
t0 = time.time()
xgb_random = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, tree_method='hist', eval_metric='logloss'), xgb_param_grid, n_iter=50, scoring='accuracy', cv=5, random_state=42, n_jobs=-1, verbose=0)
xgb_random.fit(X_train, y_train)
xgb_best_params = xgb_random.best_params_
print(f'Time: {time.time() - t0:.1f}s')
print(f'Best CV accuracy: {xgb_random.best_score_:.5f}')
print(f'Best params: {xgb_best_params}')
lgb_param_grid = {'n_estimators': [500, 700, 1000], 'max_depth': [6, 8, 10, 12], 'learning_rate': [0.01, 0.03, 0.05, 0.07], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_samples': [10, 20, 30, 50], 'reg_alpha': [0, 0.1, 0.5, 1.0], 'reg_lambda': [1, 1.5, 2, 3], 'num_leaves': [31, 50, 70, 100]}
print('\nTuning LightGBM (50 iters, 5-fold CV)...')
t0 = time.time()
lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True), lgb_param_grid, n_iter=50, scoring='accuracy', cv=5, random_state=42, n_jobs=-1, verbose=0)
lgb_random.fit(X_train, y_train)
lgb_best_params = lgb_random.best_params_
print(f'Time: {time.time() - t0:.1f}s')
print(f'Best CV accuracy: {lgb_random.best_score_:.5f}')
print(f'Best params: {lgb_best_params}')
print('\n7. Training optimized models...')
print('XGBoost...')
xgb_model = xgb.XGBClassifier(**xgb_best_params, random_state=42, tree_method='hist', eval_metric='logloss', early_stopping_rounds=50)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb_model.predict(X_val)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
xgb_acc = accuracy_score(y_val, y_pred_xgb)
print(f'Val Accuracy: {xgb_acc:.5f}')
print('LightGBM...')
lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
y_pred_lgb = lgb_model.predict(X_val)
y_pred_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]
lgb_acc = accuracy_score(y_val, y_pred_lgb)
print(f'Val Accuracy: {lgb_acc:.5f}')
print('\n8. Threshold optimization...')
w_xgb = xgb_acc / (xgb_acc + lgb_acc)
w_lgb = lgb_acc / (xgb_acc + lgb_acc)
y_pred_proba_ensemble = w_xgb * y_pred_proba_xgb + w_lgb * y_pred_proba_lgb
thresholds = np.arange(0.35, 0.65, 0.01)
accuracies = []
for threshold in thresholds:
    y_pred = (y_pred_proba_ensemble >= threshold).astype(int)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
best_threshold_idx = np.argmax(accuracies)
best_threshold = thresholds[best_threshold_idx]
best_accuracy = accuracies[best_threshold_idx]
print(f'Best threshold: {best_threshold:.2f}')
print(f'Best accuracy: {best_accuracy:.5f}')
print(f'Improvement over 0.5: {best_accuracy - accuracies[15]:.5f}')
print('\n9. Stacking ensemble...')
base_models = [('xgb', xgb.XGBClassifier(**xgb_best_params, random_state=42, tree_method='hist', eval_metric='logloss')), ('lgb', lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, force_col_wise=True))]
meta_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
print('Training stacking model...')
t0 = time.time()
stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_val)
y_pred_proba_stack = stacking_model.predict_proba(X_val)[:, 1]
stack_acc = accuracy_score(y_val, y_pred_stack)
print(f'Time: {time.time() - t0:.1f}s')
print(f'Stacking Accuracy: {stack_acc:.5f}')
print(f'Improvement over weighted ensemble: {stack_acc - best_accuracy:.5f}')
print('\n10. Model comparison...')

def evaluate_model(y_true, y_pred, y_pred_proba, name):
    return {'Model': name, 'Accuracy': accuracy_score(y_true, y_pred), 'Precision': precision_score(y_true, y_pred), 'Recall': recall_score(y_true, y_pred), 'F1-Score': f1_score(y_true, y_pred), 'ROC-AUC': roc_auc_score(y_true, y_pred_proba), 'MCC': matthews_corrcoef(y_true, y_pred)}
y_pred_ensemble_opt = (y_pred_proba_ensemble >= best_threshold).astype(int)
results = [evaluate_model(y_val, y_pred_xgb, y_pred_proba_xgb, 'XGBoost'), evaluate_model(y_val, y_pred_lgb, y_pred_proba_lgb, 'LightGBM'), evaluate_model(y_val, y_pred_ensemble_opt, y_pred_proba_ensemble, f'Weighted Ensemble (t={best_threshold:.2f})'), evaluate_model(y_val, y_pred_stack, y_pred_proba_stack, 'Stacking Ensemble')]
results_df = pd.DataFrame(results)
print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)
print(results_df.to_string(index=False))
print('=' * 80)
print('\n11. Generating test predictions...')
best_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model_acc = results_df.loc[best_idx, 'Accuracy']
print(f'Best model: {best_model_name}')
print(f'Val Accuracy: {best_model_acc:.5f} ({best_model_acc * 100:.2f}%)')
if 'Stacking' in best_model_name:
    print('Using stacking ensemble...')
    test_pred = stacking_model.predict(X_test_engineered)
    model_to_save = stacking_model
    model_type = 'stacking'
else:
    print('Using weighted ensemble with optimized threshold...')
    test_pred_proba_xgb = xgb_model.predict_proba(X_test_engineered)[:, 1]
    test_pred_proba_lgb = lgb_model.predict_proba(X_test_engineered)[:, 1]
    test_pred_proba = w_xgb * test_pred_proba_xgb + w_lgb * test_pred_proba_lgb
    test_pred = (test_pred_proba >= best_threshold).astype(int)
    model_type = 'weighted_ensemble'
submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred})
print(f'\nSubmission: {submission.shape}')
print(f'Malware: {(test_pred == 1).sum()} ({(test_pred == 1).sum() / len(test_pred) * 100:.1f}%)')
print(f'Benign: {(test_pred == 0).sum()} ({(test_pred == 0).sum() / len(test_pred) * 100:.1f}%)')
submission.to_csv('submission_v2.csv', index=False)
print('\n Saved: submission_v2.csv')
print('\n12. Saving models...')
joblib.dump(xgb_model, 'xgboost_v2.pkl')
joblib.dump(lgb_model, 'lightgbm_v2.pkl')
if model_type == 'stacking':
    joblib.dump(stacking_model, 'stacking_model.pkl')
config = {'best_model': best_model_name, 'model_type': model_type, 'validation_accuracy': float(best_model_acc), 'threshold': float(best_threshold), 'weights': {'xgb': float(w_xgb), 'lgb': float(w_lgb)}, 'n_features': X_engineered.shape[1], 'xgb_params': {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v for k, v in xgb_best_params.items()}, 'lgb_params': {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v for k, v in lgb_best_params.items()}, 'results': results_df.to_dict('records')}
with open('config_v2.json', 'w') as f:
    json.dump(config, f, indent=4)
print(' Saved models and config')
print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'Baseline (submission-1): 99.41%')
print(f'Target: 99.68%')
print(f'Gap: 0.27%')
print(f'\nBest model: {best_model_name}')
print(f'Expected score: ~{best_model_acc * 100:.2f}%')
print(f'\nImprovements:')
print(f'  {len(interaction_pairs) + len(sum_pairs)} interaction features')
print('  50 iterations hyperparameter tuning (was 15)')
print('  5-fold CV (was 3)')
print(f'  Optimized threshold: {best_threshold:.2f} (was 0.5)')
print('  Stacking ensemble')
print('=' * 80)

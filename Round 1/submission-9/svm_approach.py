import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import json
warnings.filterwarnings('ignore')
print('=' * 80)
print('SVM APPROACH - SAME FEATURES AS 99.57%')
print('=' * 80)
print('\n[STEP 1] Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
print('\n[STEP 2] Preprocessing...')
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
print('\n[STEP 3] Feature selection...')
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
X_test = test_processed[top_features]
print('\nAdding interaction features...')
top_10 = top_features[:10]
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
print(f'Total features: {X.shape[1]}')
print('\n[STEP 4] Scaling features...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print(f'Scaled features: {X_scaled.shape}')
print('\n[STEP 5] Hyperparameter tuning...')
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'kernel': ['rbf', 'linear', 'poly'], 'class_weight': [None, 'balanced']}
print('Searching for best SVM parameters...')
print('This may take a few minutes...')
tune_size = min(10000, len(X_scaled))
tune_indices = np.random.choice(len(X_scaled), tune_size, replace=False)
X_tune = X_scaled[tune_indices]
y_tune = y.iloc[tune_indices]
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
svm_grid.fit(X_tune, y_tune)
print(f'\nBest parameters: {svm_grid.best_params_}')
print(f'Best CV score: {svm_grid.best_score_:.6f}')
print('\n[STEP 6] Cross-validation with best SVM...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X_scaled))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f'\nFold {fold}/5')
    X_train, X_val = (X_scaled[train_idx], X_scaled[val_idx])
    y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
    svm_model = SVC(**svm_grid.best_params_, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict_proba(X_val)[:, 1]
    svm_auc = roc_auc_score(y_val, svm_pred)
    oof_predictions[val_idx] = svm_pred
    cv_scores.append(svm_auc)
    print(f'  SVM AUC: {svm_auc:.6f}')
mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
oof_auc = roc_auc_score(y, oof_predictions)
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 80)
print('SVM CROSS-VALIDATION RESULTS')
print('=' * 80)
print(f'Mean CV AUC: {mean_cv:.6f} (+/- {std_cv:.6f})')
print(f'OOF AUC: {oof_auc:.6f}')
print(f'OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'\nBaseline (99.57%): 0.995700')
print(f'Difference: {oof_auc - 0.9957:+.6f}')
print('=' * 80)
print('\n[STEP 7] Training final SVM on full data...')
final_svm = SVC(**svm_grid.best_params_, probability=True, random_state=42)
final_svm.fit(X_scaled, y)
print(' Final SVM trained')
print('\n[STEP 8] Generating predictions...')
test_pred = final_svm.predict_proba(X_test_scaled)[:, 1]
submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': (test_pred > 0.5).astype(int)})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_svm_{timestamp}.csv'
submission.to_csv(filename, index=False)
results = {'timestamp': timestamp, 'approach': 'svm_same_features', 'features': X.shape[1], 'best_params': svm_grid.best_params_, 'cv_auc_mean': float(mean_cv), 'cv_auc_std': float(std_cv), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc)}
with open(f'svm_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n' + '=' * 80)
print('SVM APPROACH COMPLETE!')
print('=' * 80)
print(f'\nSubmission: {filename}')
print(f'Results: svm_results_{timestamp}.json')
print(f'\nSamples: {len(submission):,}')
print(f'  Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'  Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nSVM Performance:')
print(f'  CV AUC: {oof_auc:.6f}')
print(f'  Accuracy: {oof_acc * 100:.2f}%')
print(f'\nBest SVM parameters:')
for param, value in svm_grid.best_params_.items():
    print(f'  {param}: {value}')
print(f'\nExpected leaderboard: ~{oof_auc * 100:.2f}%')
print('=' * 80)

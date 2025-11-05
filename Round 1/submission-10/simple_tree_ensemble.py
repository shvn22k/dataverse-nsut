import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import json
warnings.filterwarnings('ignore')
print('=' * 80)
print('SIMPLE TREE ENSEMBLE - BACK TO BASICS')
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
print('\n[STEP 4] Cross-validation with simple tree models...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/5')
    X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
    y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred)
    et_model = ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict_proba(X_val)[:, 1]
    et_auc = roc_auc_score(y_val, et_pred)
    ab_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1, random_state=42)
    ab_model.fit(X_train, y_train)
    ab_pred = ab_model.predict_proba(X_val)[:, 1]
    ab_auc = roc_auc_score(y_val, ab_pred)
    ensemble_pred = (rf_pred + et_pred + ab_pred) / 3
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    oof_predictions[val_idx] = ensemble_pred
    cv_scores.append(ensemble_auc)
    print(f'  RF: {rf_auc:.6f} | ET: {et_auc:.6f} | AB: {ab_auc:.6f}')
    print(f'  Ensemble: {ensemble_auc:.6f}')
mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
oof_auc = roc_auc_score(y, oof_predictions)
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 80)
print('SIMPLE TREE ENSEMBLE RESULTS')
print('=' * 80)
print(f'Mean CV AUC: {mean_cv:.6f} (+/- {std_cv:.6f})')
print(f'OOF AUC: {oof_auc:.6f}')
print(f'OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'\nBaseline (99.57%): 0.995700')
print(f'Difference: {oof_auc - 0.9957:+.6f}')
print('=' * 80)
print('\n[STEP 5] Training final models on full data...')
print('  Random Forest...')
rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_final.fit(X, y)
print('  Extra Trees...')
et_final = ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
et_final.fit(X, y)
print('  AdaBoost...')
ab_final = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1, random_state=42)
ab_final.fit(X, y)
print('\n[STEP 6] Generating predictions...')
rf_pred = rf_final.predict_proba(X_test)[:, 1]
et_pred = et_final.predict_proba(X_test)[:, 1]
ab_pred = ab_final.predict_proba(X_test)[:, 1]
ensemble_pred = (rf_pred + et_pred + ab_pred) / 3
submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': (ensemble_pred > 0.5).astype(int)})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_simple_trees_{timestamp}.csv'
submission.to_csv(filename, index=False)
results = {'timestamp': timestamp, 'approach': 'simple_tree_ensemble', 'models': ['RandomForest', 'ExtraTrees', 'AdaBoost'], 'features': X.shape[1], 'cv_auc_mean': float(mean_cv), 'cv_auc_std': float(std_cv), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc)}
with open(f'simple_trees_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n' + '=' * 80)
print('SIMPLE TREE ENSEMBLE COMPLETE!')
print('=' * 80)
print(f'\nSubmission: {filename}')
print(f'Results: simple_trees_results_{timestamp}.json')
print(f'\nSamples: {len(submission):,}')
print(f'  Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'  Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nModels used:')
print(f'  - Random Forest (bagging)')
print(f'  - Extra Trees (more randomization)')
print(f'  - AdaBoost (classic boosting)')
print(f'\nPerformance:')
print(f'  CV AUC: {oof_auc:.6f}')
print(f'  Accuracy: {oof_acc * 100:.2f}%')
print(f'\nExpected leaderboard: ~{oof_auc * 100:.2f}%')
print('=' * 80)

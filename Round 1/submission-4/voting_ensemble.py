import numpy as np
import pandas as pd
import warnings
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
print('=' * 70)
print('HARD VOTING ENSEMBLE')
print('=' * 70)
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
target_col = 'Label'
id_col = 'sha256'
df_processed = df.copy()
test_processed = test_df.copy()
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(median_val, inplace=True)
constant_features = [col for col in numeric_cols if col not in [target_col, id_col] and df_processed[col].nunique() == 1]
if constant_features:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
X_full = df_processed.drop(columns=[target_col, id_col])
y_full = df_processed[target_col]
mi_scores = mutual_info_classif(X_full, y_full, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'Feature': X_full.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
top_features = mi_df.head(60)['Feature'].tolist()
X = X_full[top_features].copy()
y = y_full.copy()
top_10 = top_features[:10]
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        X[f'{top_10[i]}_x_{top_10[j]}'] = X[top_10[i]] * X[top_10[j]]
print(f'Features: {X.shape[1]}')
X_test = test_processed[top_features].copy()
test_ids = test_processed[id_col].values
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        X_test[f'{top_10[i]}_x_{top_10[j]}'] = X_test[top_10[i]] * X_test[top_10[j]]
print('\n5-Fold CV with Hard Voting...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_votes_xgb = np.zeros(len(X), dtype=int)
oof_votes_lgb = np.zeros(len(X), dtype=int)
oof_votes_cat = np.zeros(len(X), dtype=int)
test_votes_xgb = np.zeros(len(X_test), dtype=int)
test_votes_lgb = np.zeros(len(X_test), dtype=int)
test_votes_cat = np.zeros(len(X_test), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/5')
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    xgb_model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    oof_votes_xgb[val_idx] = xgb_model.predict(X_val)
    test_votes_xgb += xgb_model.predict(X_test)
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=70, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    oof_votes_lgb[val_idx] = lgb_model.predict(X_val)
    test_votes_lgb += lgb_model.predict(X_test)
    cat_model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=7, random_state=42, verbose=0)
    cat_model.fit(X_train, y_train)
    oof_votes_cat[val_idx] = cat_model.predict(X_val).astype(int).ravel()
    test_votes_cat += cat_model.predict(X_test).astype(int).ravel()
    fold_vote = (oof_votes_xgb[val_idx] + oof_votes_lgb[val_idx] + oof_votes_cat[val_idx] >= 2).astype(int)
    acc = accuracy_score(y_val, fold_vote)
    print(f'  Voting Acc: {acc:.6f}')
oof_prediction = (oof_votes_xgb + oof_votes_lgb + oof_votes_cat >= 2).astype(int)
test_prediction = (test_votes_xgb + test_votes_lgb + test_votes_cat >= 8).astype(int)
oof_acc = accuracy_score(y, oof_prediction)
print('\n' + '=' * 70)
print(f'OOF Accuracy (Hard Voting): {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print('=' * 70)
submission = pd.DataFrame({id_col: test_ids, 'Label': test_prediction})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission.to_csv(f'submission_voting_{timestamp}.csv', index=False)
print(f'\nSubmission saved: submission_voting_{timestamp}.csv')
print(f'\nOOF Vote distribution:')
print(f'  XGB=1: {np.sum(oof_votes_xgb)}')
print(f'  LGB=1: {np.sum(oof_votes_lgb)}')
print(f'  CAT=1: {np.sum(oof_votes_cat)}')
print(f'  Final=1: {np.sum(oof_prediction)}')

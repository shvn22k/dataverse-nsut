import numpy as np
import pandas as pd
import warnings
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
warnings.filterwarnings('ignore')
print('=' * 70)
print('CORRELATION-BASED FEATURE SELECTION')
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
if len(constant_features) > 0:
    df_processed.drop(columns=constant_features, inplace=True)
    test_processed.drop(columns=[c for c in constant_features if c in test_processed.columns], inplace=True)
print('\nSelecting features by correlation with target...')
X_full = df_processed.drop(columns=[target_col, id_col])
y_full = df_processed[target_col]
correlations = []
for col in X_full.columns:
    corr = X_full[col].corr(y_full)
    correlations.append({'Feature': col, 'Correlation': corr, 'Abs_Correlation': abs(corr)})
corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
print('Top 20 features by correlation:')
print(corr_df.head(20))
top_features = corr_df.head(60)['Feature'].tolist()
print(f'\nSelected {len(top_features)} features')
X = X_full[top_features].copy()
y = y_full.copy()
print('\nAdding interaction features...')
top_10 = top_features[:10]
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
print(f'Total features: {X.shape[1]}')
X_test = test_processed[top_features].copy()
test_ids = test_processed[id_col].values
for i in range(len(top_10)):
    for j in range(i + 1, min(i + 3, len(top_10))):
        feat1 = top_10[i]
        feat2 = top_10[j]
        X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
print('\n5-Fold Cross-Validation...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/5')
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=70, max_depth=7, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_predictions[val_idx] = val_pred
    acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    auc = roc_auc_score(y_val, val_pred)
    print(f'  Acc: {acc:.6f}, AUC: {auc:.6f}')
    test_predictions += model.predict_proba(X_test)[:, 1] / 5
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
oof_auc = roc_auc_score(y, oof_predictions)
print('\n' + '=' * 70)
print(f'OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'OOF AUC: {oof_auc:.6f}')
print('=' * 70)
test_pred_binary = (test_predictions > 0.5).astype(int)
submission = pd.DataFrame({id_col: test_ids, 'Label': test_pred_binary})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission.to_csv(f'submission_corr_{timestamp}.csv', index=False)
print(f'\nSubmission saved: submission_corr_{timestamp}.csv')

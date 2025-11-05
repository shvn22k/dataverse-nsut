import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
SEED = 42
N_FOLDS = 5
USE_GPU = False
ID_COL = 'sha256'
TARGET = 'Label'
DATA_PATH = '../data'
OUT_DIR = 'submission-enhanced'
os.makedirs(OUT_DIR, exist_ok=True)
BEST_PARAMS_FILE = os.path.join(OUT_DIR, 'best_params.json')
XGB_DEFAULT = {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': SEED, 'n_jobs': -1}
LGB_DEFAULT = {'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 70, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'random_state': SEED, 'n_jobs': -1}
CAT_DEFAULT = {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6, 'random_seed': SEED, 'verbose': 0}
print('Loading data...')
df = pd.read_csv(os.path.join(DATA_PATH, 'main.csv'))
test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
print(f'Train shape: {df.shape}, Test shape: {test_df.shape}')
print('Preprocessing: filling numeric NaNs, dropping constants...')
df_proc = df.copy()
test_proc = test_df.copy()
assert TARGET in df_proc.columns, f'{TARGET} missing!'
assert ID_COL in df_proc.columns.tolist() + test_proc.columns.tolist(), f'{ID_COL} missing!'
num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [TARGET]]
for c in num_cols:
    med = df_proc[c].median()
    df_proc[c].fillna(med, inplace=True)
    if c in test_proc.columns:
        test_proc[c].fillna(med, inplace=True)
const_cols = [c for c in num_cols if df_proc[c].nunique() <= 1]
if const_cols:
    print('Dropping constant cols:', len(const_cols))
    df_proc.drop(columns=const_cols, inplace=True)
    test_proc.drop(columns=[c for c in const_cols if c in test_proc.columns], inplace=True)
FEATS = [c for c in df_proc.columns if c not in [TARGET, ID_COL]]
print(f'Feature count before selection: {len(FEATS)}')
print('Creating light interaction features (top numeric correlations)...')
numeric_for_interact = df_proc[FEATS].select_dtypes(include=[np.number]).columns.tolist()
top_var = sorted(numeric_for_interact, key=lambda x: df_proc[x].var() if df_proc[x].nunique() > 1 else 0, reverse=True)[:8]
inter_count = 0
for i in range(len(top_var)):
    for j in range(i + 1, min(i + 3, len(top_var))):
        a, b = (top_var[i], top_var[j])
        new_col = f'{a}_x_{b}'
        df_proc[new_col] = df_proc[a] * df_proc[b]
        if b in test_proc.columns:
            test_proc[new_col] = test_proc[a] * test_proc[b]
        else:
            test_proc[new_col] = 0.0
        FEATS.append(new_col)
        inter_count += 1
print(f'Created {inter_count} interaction features. Total features now: {len(FEATS)}')
xgb_params = XGB_DEFAULT.copy()
lgb_params = LGB_DEFAULT.copy()
cat_params = CAT_DEFAULT.copy()
if os.path.exists(BEST_PARAMS_FILE):
    print('Loading best params from JSON...')
    try:
        with open(BEST_PARAMS_FILE, 'r') as f:
            bestp = json.load(f)
            xgb_params.update(bestp.get('xgb', {}))
            lgb_params.update(bestp.get('lgb', {}))
            cat_params.update(bestp.get('cat', {}))
            print('Loaded best params.')
    except Exception as e:
        print('Could not load best params:', e)
if USE_GPU:
    xgb_params['tree_method'] = 'gpu_hist'
    lgb_params['device'] = 'gpu'
    cat_params['task_type'] = 'GPU'
print('Starting cross-validated training (collect OOF preds per model)...')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
n_train = len(df_proc)
n_test = len(test_proc)
oof_xgb = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
test_preds_xgb = np.zeros(n_test)
test_preds_lgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(df_proc, df_proc[TARGET]), 1):
    print(f'\nFold {fold}/{N_FOLDS}')
    X_tr, X_val = (df_proc.iloc[tr_idx][FEATS], df_proc.iloc[val_idx][FEATS])
    y_tr, y_val = (df_proc.iloc[tr_idx][TARGET], df_proc.iloc[val_idx][TARGET])
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    pred_val_xgb = xgb_clf.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = pred_val_xgb
    test_preds_xgb += xgb_clf.predict_proba(test_proc[FEATS])[:, 1] / N_FOLDS
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
    pred_val_lgb = lgb_clf.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx] = pred_val_lgb
    test_preds_lgb += lgb_clf.predict_proba(test_proc[FEATS])[:, 1] / N_FOLDS
    cat_clf = CatBoostClassifier(**cat_params)
    cat_clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    pred_val_cat = cat_clf.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = pred_val_cat
    test_preds_cat += cat_clf.predict_proba(test_proc[FEATS])[:, 1] / N_FOLDS
    fold_auc_xgb = roc_auc_score(y_val, pred_val_xgb)
    fold_auc_lgb = roc_auc_score(y_val, pred_val_lgb)
    fold_auc_cat = roc_auc_score(y_val, pred_val_cat)
    fold_comb = (pred_val_xgb + pred_val_lgb + pred_val_cat) / 3.0
    fold_auc_comb = roc_auc_score(y_val, fold_comb)
    fold_scores.append(fold_auc_comb)
    print(f'  Fold AUCs -> XGB: {fold_auc_xgb:.4f}, LGB: {fold_auc_lgb:.4f}, CAT: {fold_auc_cat:.4f}, AVG: {fold_auc_comb:.4f}')
    joblib.dump(xgb_clf, os.path.join(OUT_DIR, f'xgb_fold{fold}.pkl'))
    joblib.dump(lgb_clf, os.path.join(OUT_DIR, f'lgb_fold{fold}.pkl'))
    cat_clf.save_model(os.path.join(OUT_DIR, f'cat_fold{fold}.cbm'))
print('\nCV done. Mean fold AUC:', np.mean(fold_scores))
print('Preparing meta features from OOF predictions...')
X_meta = np.vstack([oof_xgb, oof_lgb, oof_cat]).T
X_test_meta = np.vstack([test_preds_xgb, test_preds_lgb, test_preds_cat]).T
print('Optimizing ensemble weights on OOF (maximize AUC)...')

def neg_auc_loss(w, preds_matrix, y_true):
    w = np.clip(w, 0, 1)
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    w = w / w.sum()
    combined = np.dot(preds_matrix, w)
    return -roc_auc_score(y_true, combined)
init_w = np.array([1.0, 1.0, 1.0])
res = minimize(neg_auc_loss, init_w, args=(X_meta, df_proc[TARGET].values), method='SLSQP', bounds=[(0, 1)] * 3, options={'maxiter': 200})
w_opt = np.clip(res.x, 0, 1)
w_opt = w_opt / (w_opt.sum() + 1e-12)
print('Optimized weights:', w_opt)
oof_combined = np.dot(X_meta, w_opt)
test_combined = np.dot(X_test_meta, w_opt)
print('OOF combined AUC (pre-meta):', roc_auc_score(df_proc[TARGET].values, oof_combined))
print('Training meta-learner (LogisticRegressionCV) with OOF preds...')
meta_clf = LogisticRegressionCV(cv=5, Cs=10, penalty='l2', scoring='roc_auc', max_iter=5000, n_jobs=-1)
meta_clf.fit(X_meta, df_proc[TARGET].values)
meta_oof_pred = meta_clf.predict_proba(X_meta)[:, 1]
meta_test_pred = meta_clf.predict_proba(X_test_meta)[:, 1]
print('Meta OOF AUC:', roc_auc_score(df_proc[TARGET].values, meta_oof_pred))
print('Calibrating meta predictions with IsotonicRegression (fit on OOF preds)...')
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(meta_oof_pred, df_proc[TARGET].values)
meta_oof_cal = iso.transform(meta_oof_pred)
meta_test_cal = iso.transform(meta_test_pred)
print('Calibrated meta OOF AUC:', roc_auc_score(df_proc[TARGET].values, meta_oof_cal))
print('Tuning threshold for F1 using OOF calibrated preds...')
best_t, best_f1 = (0.5, 0.0)
for t in np.linspace(0.01, 0.99, 99):
    f1 = f1_score(df_proc[TARGET].values, (meta_oof_cal > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
print(f'Best OOF F1: {best_f1:.4f} at threshold {best_t:.3f}')
final_test_binary = (meta_test_cal > best_t).astype(int)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_file = os.path.join(OUT_DIR, f'submission_meta_{timestamp}.csv')
submission = pd.DataFrame({ID_COL: test_proc[ID_COL].values, TARGET: final_test_binary})
submission.to_csv(submission_file, index=False)
print('Submission saved:', submission_file)
np.save(os.path.join(OUT_DIR, 'oof_xgb.npy'), oof_xgb)
np.save(os.path.join(OUT_DIR, 'oof_lgb.npy'), oof_lgb)
np.save(os.path.join(OUT_DIR, 'oof_cat.npy'), oof_cat)
np.save(os.path.join(OUT_DIR, 'oof_combined.npy'), oof_combined)
joblib.dump(meta_clf, os.path.join(OUT_DIR, 'meta_logistic.pkl'))
joblib.dump(iso, os.path.join(OUT_DIR, 'isotonic_calibrator.pkl'))
with open(os.path.join(OUT_DIR, 'ensemble_weights.json'), 'w') as f:
    json.dump({'weights': w_opt.tolist(), 'threshold': float(best_t)}, f, indent=2)
oof_bin = (meta_oof_cal > best_t).astype(int)
print('\nFinal OOF metrics (after stacking+calibration+threshold):')
print('Accuracy:', accuracy_score(df_proc[TARGET].values, oof_bin))
print('Precision:', precision_score(df_proc[TARGET].values, oof_bin))
print('Recall:', recall_score(df_proc[TARGET].values, oof_bin))
print('F1:', f1_score(df_proc[TARGET].values, oof_bin))
print('AUC:', roc_auc_score(df_proc[TARGET].values, meta_oof_cal))
print('\nAll done. Models + artifacts are in:', OUT_DIR)

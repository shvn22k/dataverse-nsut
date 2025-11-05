import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TRAIN_PATH = 'C:\\Projects\\dataverse-nsut\\Round 2\\data\\main.csv'
TEST_PATH = 'C:\\Projects\\dataverse-nsut\\Round 2\\data\\test.csv'
SUBMISSION_OUT = 'submission.csv'
AP_HI_RANGE = (70, 250)
AP_LO_RANGE = (40, 150)
HEIGHT_RANGE = (130, 220)
WEIGHT_RANGE = (30, 200)
BMI_RANGE = (15, 50)

def load_data():
    print('Loading data...')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
    return (train, test)

def col_ok(df, cols):
    return all((c in df.columns for c in cols))

def clean_outliers(df):
    initial_len = len(df)
    keep = pd.Series(True, index=df.index)
    if 'ap_hi' in df.columns:
        keep &= (df['ap_hi'] >= AP_HI_RANGE[0]) & (df['ap_hi'] <= AP_HI_RANGE[1])
    if 'ap_lo' in df.columns:
        keep &= (df['ap_lo'] >= AP_LO_RANGE[0]) & (df['ap_lo'] <= AP_LO_RANGE[1])
    if col_ok(df, ['ap_hi', 'ap_lo']):
        keep &= df['ap_hi'] > df['ap_lo']
    if 'height' in df.columns:
        keep &= (df['height'] >= HEIGHT_RANGE[0]) & (df['height'] <= HEIGHT_RANGE[1])
    if 'weight' in df.columns:
        keep &= (df['weight'] >= WEIGHT_RANGE[0]) & (df['weight'] <= WEIGHT_RANGE[1])
    if col_ok(df, ['height', 'weight']):
        bmi = df['weight'] / (df['height'] / 100) ** 2
        keep &= (bmi >= BMI_RANGE[0]) & (bmi <= BMI_RANGE[1])
    df_clean = df[keep].copy()
    print(f'Outlier cleaning: dropped {initial_len - len(df_clean)} / {initial_len} rows')
    return df_clean

def engineer_features(df, chol_mean=None):
    if 'age' in df.columns:
        df['age_years'] = (df['age'] // 365).astype(int)
    if col_ok(df, ['height', 'weight']):
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    if col_ok(df, ['ap_hi', 'ap_lo']):
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['bp_high'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)
        df['ap_hi_to_lo'] = df['ap_hi'] / (df['ap_lo'] + 1e-06)
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    if col_ok(df, ['age_years', 'bmi']):
        df['age_bmi_interact'] = df['age_years'] * df['bmi']
    if col_ok(df, ['bmi', 'cholesterol']):
        df['bmi_chol_interact'] = df['bmi'] * df['cholesterol']
    if 'cholesterol' in df.columns:
        if chol_mean is None:
            chol_mean = df['cholesterol'].mean()
        df['chol_m_diff'] = df['cholesterol'] - chol_mean
    if 'ap_hi' in df.columns:
        df['ap_hi_sq'] = df['ap_hi'] ** 2
    if 'bp_high' in df.columns and 'cholesterol' in df.columns and ('gluc' in df.columns):
        df['risk_score'] = (df['bp_high'] == 1).astype(int) + (df['cholesterol'] == 3).astype(int) + (df['gluc'] == 3).astype(int)
    return (df, chol_mean)

def get_feature_names():
    return ['ap_hi', 'bp_high', 'ap_hi_to_lo', 'ap_lo', 'pulse_pressure', 'age_bmi_interact', 'bmi_chol_interact', 'age_years', 'cholesterol', 'bmi', 'weight', 'gluc', 'active', 'gender', 'alco', 'smoke', 'height', 'map', 'chol_m_diff', 'ap_hi_sq', 'risk_score']

def get_base_feature_candidates():
    return ['age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender', 'height', 'weight']

def stratified_split(X, y, test_size=0.2, seed=RANDOM_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def eval_metrics(y_true, y_pred, y_prob, name):
    print(f'\n{name} classification report:\n{classification_report(y_true, y_pred, digits=4)}')
    print(f'{name} Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'{name} Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'{name} Recall: {recall_score(y_true, y_pred):.4f}')
    print(f'{name} F1: {f1_score(y_true, y_pred):.4f}')
    print(f'{name} ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}')

def train_logreg(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED)
    model.fit(X_train_s, y_train)
    y_tr_pred = model.predict(X_train_s)
    y_val_pred = model.predict(X_val_s)
    y_tr_prob = model.predict_proba(X_train_s)[:, 1]
    y_val_prob = model.predict_proba(X_val_s)[:, 1]
    print('\n=== Logistic Regression ===')
    eval_metrics(y_train, y_tr_pred, y_tr_prob, 'Train')
    eval_metrics(y_val, y_val_pred, y_val_prob, 'Validation')
    return {'name': 'logreg', 'model': model, 'scaler': scaler, 'val_prob': y_val_prob, 'val_pred': y_val_pred}

def train_rf(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=20, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_tr_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    print('\n=== Random Forest ===')
    eval_metrics(y_train, y_tr_pred, y_tr_prob, 'Train')
    eval_metrics(y_val, y_val_pred, y_val_prob, 'Validation')
    return {'name': 'rf', 'model': model, 'scaler': None, 'val_prob': y_val_prob, 'val_pred': y_val_pred}

def train_xgb(X_train, y_train, X_val, y_val):
    if not HAS_XGB:
        return None
    try:
        model = XGBClassifier(n_estimators=400, learning_rate=0.07, max_depth=6, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        print('Warning: XGBoost early stopping not supported in this version, training without it.')
        model = XGBClassifier(n_estimators=400, learning_rate=0.07, max_depth=6, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    print('\n=== XGBoost ===')
    eval_metrics(y_val, y_val_pred, y_val_prob, 'Validation')
    return {'name': 'xgb', 'model': model, 'scaler': None, 'val_prob': y_val_prob, 'val_pred': y_val_pred}

def train_lgbm(X_train, y_train, X_val, y_val):
    if not HAS_LGBM:
        return None
    try:
        model = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_SEED)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        print('Warning: LightGBM early stopping not supported in this version, training without it.')
        model = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    print('\n=== LightGBM ===')
    eval_metrics(y_val, y_val_pred, y_val_prob, 'Validation')
    return {'name': 'lgbm', 'model': model, 'scaler': None, 'val_prob': y_val_prob, 'val_pred': y_val_pred}

def choose_best(results, y_val):
    best_name, best_score, best_obj = (None, -1.0, None)
    for res in results:
        if res is None:
            continue
        score = roc_auc_score(y_val, res['val_prob'])
        print(f'Model {res['name']} Validation ROC-AUC: {score:.4f}')
        if score > best_score:
            best_name, best_score, best_obj = (res['name'], score, res)
    print(f'Best model: {best_name} (ROC-AUC {best_score:.4f})')
    return best_obj

def find_best_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if thr is None or len(thr) == 0:
        return 0.5
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    best_t = float(thr[idx])
    best_t = float(np.clip(best_t, 0.05, 0.95))
    print(f'Chosen threshold on validation (Youden): {best_t:.3f}')
    return best_t

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_test_probabilities(model, X_test):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        d = model.decision_function(X_test)
        return sigmoid(d)
    else:
        return model.predict(X_test).astype(float)

def main():
    train, test = load_data()
    train_ids = train['id'] if 'id' in train.columns else None
    test_ids = test['id'] if 'id' in test.columns else None
    train = clean_outliers(train)
    test = clean_outliers(test)
    train_fe, chol_mean = engineer_features(train)
    test_fe, _ = engineer_features(test, chol_mean)
    if train_fe.isnull().any().any():
        print('Warning: missing values in train. Filling with medians.')
        train_fe = train_fe.fillna(train_fe.median(numeric_only=True))
    if test_fe.isnull().any().any():
        print('Missing values in test. Filling with train medians (numeric).')
        test_fe = test_fe.fillna(train_fe.median(numeric_only=True))
    eng_feats = get_feature_names()
    base_feats = get_base_feature_candidates()
    candidates = list(dict.fromkeys(eng_feats + base_feats))
    features = [f for f in candidates if f in train_fe.columns and f in test_fe.columns]
    print('Using features:', features)
    X = train_fe[features]
    y = train_fe['cardio']
    X_test = test_fe[features]
    X_train, X_val, y_train, y_val = stratified_split(X, y, test_size=0.2, seed=RANDOM_SEED)
    res = []
    res.append(train_logreg(X_train, y_train, X_val, y_val))
    res.append(train_rf(X_train, y_train, X_val, y_val))
    res.append(train_xgb(X_train, y_train, X_val, y_val))
    res.append(train_lgbm(X_train, y_train, X_val, y_val))
    best = choose_best(res, y_val)
    val_prob = best['val_prob']
    thr = find_best_threshold(y_val, val_prob)
    if best['scaler'] is not None:
        X_test_proc = best['scaler'].transform(X_test)
    else:
        X_test_proc = X_test
    probs = get_test_probabilities(best['model'], X_test_proc)
    print(f'Test prob summary -> min:{probs.min():.4f} mean:{probs.mean():.4f} max:{probs.max():.4f} thr:{thr:.3f}')
    y_test_pred = (probs >= thr).astype(int)
    if y_test_pred.sum() == 0 or y_test_pred.sum() == len(y_test_pred):
        prevalence = float(y.mean()) if 'y' in locals() else 0.5
        q = max(0.05, min(0.95, 1 - prevalence))
        low = float(probs.min())
        high = float(probs.max())
        thr2 = float(np.quantile(probs, q))
        thr2 = float(np.clip(thr2, low + 1e-06, high - 1e-06)) if high > low else 0.5
        y_test_pred = (probs >= thr2).astype(int)
        print(f'Adjusted threshold by prevalence ({prevalence:.3f}) -> thr:{thr2:.3f}, positives:{y_test_pred.sum()}')
    if y_test_pred.sum() == 0 or y_test_pred.sum() == len(y_test_pred):
        prevalence = float(y.mean()) if 'y' in locals() else 0.5
        k = int(round(prevalence * len(probs)))
        k = max(1, min(len(probs) - 1, k))
        order = np.argsort(probs)
        pred = np.zeros_like(probs, dtype=int)
        pred[order[-k:]] = 1
        y_test_pred = pred
        print(f'Applied rank-based prevalence safeguard: k={k}, positives={y_test_pred.sum()}')
    sub = pd.DataFrame({'id': test_fe['id'] if 'id' in test_fe.columns else np.arange(len(X_test)), 'cardio': y_test_pred.astype(int)})
    print(f'Positives in submission: {(sub['cardio'] == 1).sum()} / {len(sub)}')
    sub.to_csv(SUBMISSION_OUT, index=False)
    print(f'Submission saved to: {SUBMISSION_OUT}  shape={sub.shape}')
if __name__ == '__main__':
    main()

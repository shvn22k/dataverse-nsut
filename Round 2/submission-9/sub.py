import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print('Warning: XGBoost not available')
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    print('Warning: LightGBM not available')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TRAIN_PATH = 'C:\\Projects\\dataverse-nsut\\Round 2\\data\\main.csv'
TEST_PATH = 'C:\\Projects\\dataverse-nsut\\Round 2\\data\\test.csv'
SUBMISSION_OUT = 'submission.csv'

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f'Train {train.shape}, Test {test.shape}')
    return (train, test)

def clean_train_aggressively(df):
    d = df.copy()
    initial = len(d)
    d['bmi_tmp'] = d['weight'] / (d['height'] / 100.0) ** 2
    d['pp_tmp'] = d['ap_hi'] - d['ap_lo']
    keep = (d['ap_hi'] >= 70) & (d['ap_hi'] <= 200) & (d['ap_lo'] >= 40) & (d['ap_lo'] <= 130) & (d['ap_hi'] > d['ap_lo']) & (d['pp_tmp'] >= 20) & (d['pp_tmp'] <= 100) & (d['height'] >= 140) & (d['height'] <= 210) & (d['weight'] >= 40) & (d['weight'] <= 160) & (d['bmi_tmp'] >= 16) & (d['bmi_tmp'] <= 45) & (d['age'] >= 10000) & (d['age'] <= 25000)
    cleaned = d.loc[keep].drop(columns=['bmi_tmp', 'pp_tmp'])
    print(f'Cleaned: removed {initial - len(cleaned)} rows ({100 * (initial - len(cleaned)) / initial:.1f}%)')
    return cleaned

def engineer_features(df, source='train'):
    d = df.copy()
    if source == 'train':
        age_years = (d['age'] / 365.25).clip(lower=20, upper=90)
    else:
        age_years = pd.to_numeric(d.get('age', 50), errors='coerce').fillna(50).clip(lower=20, upper=90)
    if 'sex' in d.columns:
        sex = pd.to_numeric(d['sex'], errors='coerce').fillna(0).astype(int)
    else:
        sex = (d['gender'] - 1).clip(lower=0, upper=1).astype(int)
    ap_hi = d.get('ap_hi', pd.Series(120, index=d.index))
    ap_lo = d.get('ap_lo', pd.Series(80, index=d.index))
    if 'trestbps' in d.columns:
        trestbps = pd.to_numeric(d['trestbps'], errors='coerce').fillna(120)
    else:
        trestbps = ap_hi.fillna(120)
    if 'chol' in d.columns:
        chol = pd.to_numeric(d['chol'], errors='coerce').fillna(200)
    else:
        chol = d['cholesterol'].map({1: 200, 2: 240, 3: 280}).fillna(220)
    if 'fbs' in d.columns:
        fbs = pd.to_numeric(d['fbs'], errors='coerce').fillna(0).astype(int)
    else:
        fbs = (d.get('gluc', 1) >= 2).astype(int)
    pulse_pressure = (ap_hi - ap_lo).clip(lower=20, upper=100).fillna(40)
    if 'cp' in d.columns:
        cp = pd.to_numeric(d['cp'], errors='coerce').fillna(0).astype(int)
    else:
        cp = pd.Series(0, index=d.index)
        ap_hi_safe = ap_hi.fillna(120)
        ap_lo_safe = ap_lo.fillna(80)
        cp[(ap_hi_safe >= 120) & (ap_hi_safe < 140)] = 1
        cp[(ap_hi_safe >= 140) & (ap_hi_safe < 160)] = 2
        cp[ap_hi_safe >= 160] = 3
        cp = cp.astype(int)
    if 'thalach' in d.columns:
        thalach = pd.to_numeric(d['thalach'], errors='coerce').fillna(150)
    else:
        base_hr = (220 - age_years).clip(100, 200)
        activity_mult = d.get('active', 1).map({0: 0.85, 1: 0.95}).fillna(0.9)
        thalach = (base_hr * activity_mult).clip(70, 200)
    if 'exang' in d.columns:
        exang = pd.to_numeric(d['exang'], errors='coerce').fillna(0).astype(int)
    else:
        bp_high = ((ap_hi >= 140) | (ap_lo >= 90)).fillna(False).astype(int)
        inactive = (1 - d.get('active', 1)).astype(int)
        exang = (bp_high & inactive).astype(int)
    if 'oldpeak' in d.columns:
        oldpeak = pd.to_numeric(d['oldpeak'], errors='coerce').fillna(0)
    else:
        oldpeak = ((pulse_pressure - 40) / 20.0).clip(0, 3)
    if 'restecg' in d.columns:
        restecg = pd.to_numeric(d['restecg'], errors='coerce').fillna(1).astype(int)
        has_restecg = 1
    else:
        restecg = 1
        has_restecg = 0
    if 'slope' in d.columns:
        slope = pd.to_numeric(d['slope'], errors='coerce').fillna(2).astype(int)
        has_slope = 1
    else:
        slope = 2
        has_slope = 0
    if 'ca' in d.columns:
        ca = pd.to_numeric(d['ca'], errors='coerce').fillna(0).astype(int)
        has_ca = 1
    else:
        if 'weight' in d.columns and 'height' in d.columns:
            bmi = (d['weight'] / (d['height'] / 100) ** 2).fillna(25)
            ca_val = (d.get('cholesterol', 1) == 3).astype(int) + (d.get('gluc', 1) == 3).astype(int) + (bmi > 35).astype(int)
            ca = ca_val.clip(0, 3).astype(int)
        else:
            ca = 0
        has_ca = 0
    if 'thal' in d.columns:
        thal = pd.to_numeric(d['thal'], errors='coerce').fillna(2).astype(int)
        has_thal = 1
    else:
        thal = 2
        has_thal = 0
    age_risk_low = (age_years < 45).astype(int)
    age_risk_mid = ((age_years >= 45) & (age_years <= 60)).astype(int)
    age_risk_high = (age_years > 60).astype(int)
    bp_risk_0 = (trestbps < 120).astype(int)
    bp_risk_1 = ((trestbps >= 120) & (trestbps < 140)).astype(int)
    bp_risk_2 = (trestbps >= 140).astype(int)
    chol_risk = (chol > 240).astype(int)
    metabolic_risk = (chol_risk + fbs).clip(0, 2).astype(int)
    age_sex_interact = age_years * sex
    age_chol_interact = age_years * chol_risk
    bp_age_interact = trestbps * age_years / 100.0
    features = pd.DataFrame({'age': age_years, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal, 'has_restecg': has_restecg, 'has_slope': has_slope, 'has_ca': has_ca, 'has_thal': has_thal, 'age_risk_low': age_risk_low, 'age_risk_mid': age_risk_mid, 'age_risk_high': age_risk_high, 'bp_risk_0': bp_risk_0, 'bp_risk_1': bp_risk_1, 'bp_risk_2': bp_risk_2, 'chol_risk': chol_risk, 'metabolic_risk': metabolic_risk, 'age_sex_interact': age_sex_interact, 'age_chol_interact': age_chol_interact, 'bp_age_interact': bp_age_interact})
    for col in features.columns:
        if features[col].dtype in ['float64', 'float32']:
            features[col] = features[col].replace([np.inf, -np.inf], np.nan)
            features[col] = features[col].fillna(features[col].median())
    return features

def train_models(X_train, y_train, X_val, y_val):
    models = []
    if HAS_XGB:
        pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        xgb1 = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=5, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, gamma=2.0, reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=float(pos_weight), random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
        try:
            xgb1.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        except:
            xgb1.fit(X_train, y_train)
        models.append(('xgb_conservative', xgb1))
        xgb2 = XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, gamma=1.0, reg_alpha=0.3, reg_lambda=0.3, scale_pos_weight=float(pos_weight), random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
        try:
            xgb2.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        except:
            xgb2.fit(X_train, y_train)
        models.append(('xgb_deep', xgb2))
    if HAS_LGBM:
        lgbm1 = LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6, num_leaves=31, min_child_samples=40, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.6, reg_lambda=0.6, min_split_gain=0.1, random_state=RANDOM_SEED, verbose=-1)
        try:
            lgbm1.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        except:
            lgbm1.fit(X_train, y_train)
        models.append(('lgbm_conservative', lgbm1))
        lgbm2 = LGBMClassifier(n_estimators=800, learning_rate=0.03, max_depth=8, num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.4, reg_lambda=0.4, random_state=RANDOM_SEED, verbose=-1)
        try:
            lgbm2.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        except:
            lgbm2.fit(X_train, y_train)
        models.append(('lgbm_deep', lgbm2))
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=40, min_samples_leaf=15, max_features='sqrt', random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    models.append(('random_forest', rf))
    return models

def evaluate_ensemble(models, X_val, y_val, X_test):
    f1_scores = []
    probs_val = []
    probs_test = []
    print('\nModel performance')
    for name, model in models:
        pred_val = model.predict_proba(X_val)[:, 1]
        pred_test = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_val, (pred_val >= 0.5).astype(int))
        f1_scores.append(f1)
        probs_val.append(pred_val)
        probs_test.append(pred_test)
        print(f'{name:20s}: F1={f1:.4f}')
    f1_array = np.array(f1_scores)
    weights = f1_array ** 2 / (f1_array ** 2).sum()
    print(f'Weights: {dict(zip([m[0] for m in models], weights))}')
    ensemble_val = np.average(np.vstack(probs_val), axis=0, weights=weights)
    ensemble_test = np.average(np.vstack(probs_test), axis=0, weights=weights)
    return (ensemble_val, ensemble_test, weights)

def optimize_threshold(y_true, probs):
    best_f1 = 0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    print(f'\nOptimal threshold: {best_thr:.3f} (F1={best_f1:.4f})')
    return (best_thr, best_f1)

def main():
    print('Training heart disease predictor')
    train, test = load_data()
    print('Dataset info')
    print(f'Train: {train.shape}, Target: {train['cardio'].value_counts().to_dict()}')
    print(f'Test: {test.shape}')
    if 'target' in test.columns:
        non_null = test['target'].notna().sum()
        print(f"Test has 'target' column: {non_null} non-null values")
    print('Data cleaning')
    train_clean = clean_train_aggressively(train)
    print('Feature engineering')
    X_train_raw = engineer_features(train_clean, source='train')
    X_test_raw = engineer_features(test, source='test')
    y_train = train_clean['cardio'].values
    print(f'Engineered features: {X_train_raw.shape[1]}')
    print(f'Feature columns: {list(X_train_raw.columns)}')
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'age_sex_interact', 'age_chol_interact', 'bp_age_interact']
    cat_cols = [c for c in X_train_raw.columns if c not in cont_cols]
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_raw[cont_cols].fillna(X_train_raw[cont_cols].median()))
    X_test_cont = scaler.transform(X_test_raw[cont_cols].fillna(X_train_raw[cont_cols].median()))
    X_train = np.hstack([X_train_cont, X_train_raw[cat_cols].values])
    X_test = np.hstack([X_test_cont, X_test_raw[cat_cols].values])
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train)
    print('Training models')
    models = train_models(X_tr, y_tr, X_val, y_val)
    print('Evaluating ensemble')
    ensemble_val, ensemble_test, weights = evaluate_ensemble(models, X_val, y_val, X_test)
    best_thr, best_f1 = optimize_threshold(y_val, ensemble_val)
    print('Final predictions')
    predictions = (ensemble_test >= best_thr).astype(int)
    pos_rate = predictions.mean()
    print(f'Positive rate: {pos_rate:.2%}')
    if pos_rate < 0.05 or pos_rate > 0.95:
        print('  Predictions are degenerate, using median threshold')
        best_thr = np.median(ensemble_test)
        predictions = (ensemble_test >= best_thr).astype(int)
        print(f'Adjusted threshold: {best_thr:.3f}, new positive rate: {predictions.mean():.2%}')
    elif pos_rate < 0.2 or pos_rate > 0.8:
        print(f'Warning: positive rate {pos_rate:.2%} seems unusual (expected 30-60%)')
        print(f'Consider using median threshold: {np.median(ensemble_test):.3f}')
    submission = pd.DataFrame({'id': test['id'].values, 'cardio': predictions})
    print(f'Submission class counts: {submission['cardio'].value_counts().to_dict()}')
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f'Saved to: {SUBMISSION_OUT}')
    print('Done')
if __name__ == '__main__':
    main()

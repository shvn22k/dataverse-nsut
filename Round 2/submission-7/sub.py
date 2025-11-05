import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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

def standardize_gender(df: pd.DataFrame, source: str) -> pd.Series:
    if 'sex' in df.columns:
        return df['sex'].astype(int)
    if 'gender' in df.columns:
        if source == 'train':
            return (df['gender'] == 2).astype(int)
        else:
            return df['gender'].astype(int)
    return pd.Series(0, index=df.index)

def engineer_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    d = df.copy()
    if 'sex' in d.columns:
        d['age_years'] = d['age'].astype(float)
    else:
        d['age_years'] = d['age'] / 365.0
    d['gender_male'] = standardize_gender(d, source)
    h = d.get('height', pd.Series(np.nan, index=d.index))
    w = d.get('weight', pd.Series(np.nan, index=d.index))
    d['bmi'] = w / (h / 100.0) ** 2
    ap_hi = d.get('ap_hi', d.get('trestbps', pd.Series(np.nan, index=d.index)))
    ap_lo = d.get('ap_lo', pd.Series(np.nan, index=d.index))
    d['resting_bp'] = ap_hi
    d['pulse_pressure'] = ap_hi - ap_lo
    d['mean_arterial_pressure'] = ap_lo + d['pulse_pressure'] / 3.0
    if 'chol' in d.columns:
        d['chol_mgdl'] = pd.to_numeric(d['chol'], errors='coerce')
    else:
        d['chol_mgdl'] = d.get('cholesterol', pd.Series(1, index=d.index)).map({1: 180, 2: 240, 3: 300}).astype(float)
    d['chol_mgdl'] = d['chol_mgdl'].fillna(180.0)
    if 'fbs' in d.columns:
        d['fbs'] = pd.to_numeric(d['fbs'], errors='coerce').fillna(0).astype(int)
    else:
        d['fbs'] = (d.get('gluc', pd.Series(1, index=d.index)) >= 2).astype(int)
    for c in ['smoke', 'alco', 'active']:
        if c not in d.columns:
            d[c] = 0
    d['sedentary'] = 1 - d['active']
    d['lifestyle_risk'] = d['smoke'] + d['alco'] - d['active']
    if 'cholesterol' in d.columns:
        chol_level = pd.to_numeric(d['cholesterol'], errors='coerce').fillna(1).astype(int)
    else:
        chol_mg = d.get('chol', pd.Series(np.nan, index=d.index))
        chol_level = pd.Series(np.where(chol_mg >= 240, 3, np.where(chol_mg >= 200, 2, 1)), index=d.index)
        chol_level = pd.to_numeric(chol_level, errors='coerce').fillna(1).astype(int)
    if 'gluc' in d.columns:
        gluc_level = pd.to_numeric(d['gluc'], errors='coerce').fillna(1).astype(int)
    else:
        gluc_level = pd.Series(np.where(d['fbs'] == 1, 2, 1), index=d.index).astype(int)
    d['metabolic_risk'] = (chol_level + gluc_level).astype(int)
    d['bp_difference'] = d['pulse_pressure']
    cols = ['age_years', 'gender_male', 'bmi', 'resting_bp', 'pulse_pressure', 'mean_arterial_pressure', 'chol_mgdl', 'fbs', 'smoke', 'alco', 'active', 'sedentary', 'lifestyle_risk', 'metabolic_risk', 'bp_difference']
    d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
    for c in cols:
        if d[c].isnull().any():
            if c in ['gender_male', 'fbs', 'smoke', 'alco', 'active', 'sedentary']:
                d[c] = d[c].fillna(0)
            else:
                d[c] = d[c].fillna(d[c].median())
    return d[cols]

def engineer_to_test_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    d = df.copy()
    age = d['age'] / 365.0 if source == 'train' else pd.to_numeric(d['age'], errors='coerce')
    sex = standardize_gender(d, source)
    ap_hi = d.get('ap_hi', pd.Series(np.nan, index=d.index))
    ap_lo = d.get('ap_lo', pd.Series(np.nan, index=d.index))
    trestbps = pd.to_numeric(d.get('trestbps', ap_hi), errors='coerce')
    if 'chol' in d.columns:
        chol = pd.to_numeric(d['chol'], errors='coerce')
    else:
        chol = d.get('cholesterol', pd.Series(1, index=d.index)).map({1: 180, 2: 240, 3: 300}).astype(float)
    fbs = pd.to_numeric(d.get('fbs', (d.get('gluc', pd.Series(1, index=d.index)) >= 2).astype(int)), errors='coerce')
    if 'restecg' in d.columns:
        restecg = pd.to_numeric(d['restecg'], errors='coerce')
    else:
        ratio = ap_hi / (ap_lo + 1e-06)
        rc = pd.cut(ratio, bins=[0, 1.1, 1.4, 10], labels=[0, 1, 2])
        restecg = rc.cat.codes.replace(-1, 1)
    if 'thalach' in d.columns:
        thalach = pd.to_numeric(d['thalach'], errors='coerce')
    else:
        active = d.get('active', pd.Series(1, index=d.index)).replace({0: 0.9, 1: 1.0})
        thalach = (220 - age) * active
    if 'exang' in d.columns:
        exang = pd.to_numeric(d['exang'], errors='coerce')
    else:
        has_htn = (ap_hi >= 140) | (ap_lo >= 90)
        inactive = 1 - d.get('active', pd.Series(1, index=d.index))
        exang = ((inactive == 1) & (has_htn == True)).astype(int) if hasattr(has_htn, 'astype') else 0
    if 'oldpeak' in d.columns:
        oldpeak = pd.to_numeric(d['oldpeak'], errors='coerce')
    else:
        oldpeak = np.maximum(0.0, ap_hi - ap_lo - 60.0) / 10.0
    if 'slope' in d.columns:
        slope = pd.to_numeric(d['slope'], errors='coerce')
    else:
        r2 = (ap_hi - ap_lo) / (age + 1e-06)
        sc = pd.cut(r2, bins=[-1, 0.1, 0.3, 10], labels=[0, 1, 2])
        slope = sc.cat.codes.replace(-1, 1)
    if 'ca' in d.columns:
        ca = pd.to_numeric(d['ca'], errors='coerce')
    else:
        h = d.get('height', pd.Series(np.nan, index=d.index))
        w = d.get('weight', pd.Series(np.nan, index=d.index))
        bmi_obese = (w / (h / 100.0) ** 2 >= 30).astype(float)
        ca = np.clip(bmi_obese.fillna(0) + (d.get('cholesterol', pd.Series(1, index=d.index)) > 1).astype(int) + (d.get('gluc', pd.Series(1, index=d.index)) > 1).astype(int), 0, 3)
    if 'thal' in d.columns:
        thal = pd.to_numeric(d['thal'], errors='coerce')
    else:
        thal = (d.get('cholesterol', pd.Series(1, index=d.index)) > 1).astype(int)
    if 'cp' in d.columns:
        cp = pd.to_numeric(d['cp'], errors='coerce')
    else:
        cp = pd.Series(0, index=d.index)
        cp[ap_hi.between(120, 129) & (ap_lo < 80)] = 1
        cp[ap_hi.between(130, 139) | ap_lo.between(80, 89)] = 2
        cp[(ap_hi >= 140) | (ap_lo >= 90)] = 3
    out = pd.DataFrame({'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal})
    for c in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(int)
    for c in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def train_predict_heart(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    bin_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    scaler = StandardScaler()
    for c in cont_cols:
        med = pd.to_numeric(train_df[c], errors='coerce').median()
        train_df[c] = pd.to_numeric(train_df[c], errors='coerce').fillna(med)
        test_df[c] = pd.to_numeric(test_df[c], errors='coerce').fillna(med)
    Xc = scaler.fit_transform(train_df[cont_cols])
    Xt_c = scaler.transform(test_df[cont_cols])
    X = np.hstack([Xc, train_df[bin_cols].values])
    X_test = np.hstack([Xt_c, test_df[bin_cols].values])
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    pva = clf.predict_proba(X_va)[:, 1]
    best_thr, best_f1 = (0.5, 0)
    for t in np.arange(0.1, 0.9, 0.01):
        f = f1_score(y_va, (pva >= t).astype(int))
        if f > best_f1:
            best_f1, best_thr = (f, t)
    print(f'Validation best F1 {best_f1:.4f} at thr {best_thr:.2f}')
    clf.fit(X, y)
    ptest = clf.predict_proba(X_test)[:, 1]
    return (ptest >= best_thr).astype(int)

def train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series) -> np.ndarray:

    def to_heart(df: pd.DataFrame, source: str) -> pd.DataFrame:
        return engineer_to_test_schema(df, source)
    heart_train = to_heart(train_df.assign(age=train_df['age']), source='train')
    heart_test = to_heart(test_df.assign(age=test_df['age']), source='test')
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    bin_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    scaler = StandardScaler()
    for c in cont_cols:
        med = heart_train[c].median()
        heart_train[c] = pd.to_numeric(heart_train[c], errors='coerce').fillna(med)
        heart_test[c] = pd.to_numeric(heart_test[c], errors='coerce').fillna(med)
    Xc = scaler.fit_transform(heart_train[cont_cols])
    Xt_c = scaler.transform(heart_test[cont_cols])
    X = np.hstack([Xc, heart_train[bin_cols].values])
    X_test = np.hstack([Xt_c, heart_test[bin_cols].values])
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    pva = clf.predict_proba(X_va)[:, 1]
    best_thr, best_f1 = (0.5, 0)
    for t in np.arange(0.1, 0.9, 0.01):
        f = f1_score(y_va, (pva >= t).astype(int))
        if f > best_f1:
            best_f1, best_thr = (f, t)
    print(f'Validation best F1 {best_f1:.4f} at thr {best_thr:.2f}')
    clf.fit(X, y)
    ptest = clf.predict_proba(X_test)[:, 1]
    return (ptest >= best_thr).astype(int)

def main():
    train, test = load_data()
    X_train = engineer_to_test_schema(train, source='train')
    X_test = engineer_to_test_schema(test, source='test')
    print('Final engineered training dataset (head):')
    print(X_train.head())
    y = train['cardio'].astype(int)
    preds = train_predict_heart(X_train, X_test, y)
    sub = pd.DataFrame({'id': test['id'], 'cardio': preds.astype(int)})
    print(f'Positives in submission: {(sub.cardio == 1).sum()} / {len(sub)}')
    sub.to_csv(SUBMISSION_OUT, index=False)
    print(f'Saved: {SUBMISSION_OUT}')
if __name__ == '__main__':
    main()

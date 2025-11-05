import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TRAIN_PATH = 'C:\\\\Projects\\\\dataverse-nsut\\\\Round 2\\\\data\\\\main.csv'
TEST_PATH = 'C:\\\\Projects\\\\dataverse-nsut\\\\Round 2\\\\data\\\\test.csv'
SUBMISSION_OUT = 'submission.csv'

def col_ok(df, cols):
    return all((c in df.columns for c in cols))

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f'Train {train.shape}, Test {test.shape}')
    return (train, test)

def clean_and_flag(df):
    df = df.copy()
    if col_ok(df, ['ap_hi', 'ap_lo']):
        df = df[df['ap_hi'] > df['ap_lo']]
    if 'height' in df.columns:
        df = df[(df['height'] >= 130) & (df['height'] <= 220)]
    if 'weight' in df.columns:
        df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
    if 'ap_hi' in df.columns:
        df['ap_hi_extreme'] = ((df['ap_hi'] < 70) | (df['ap_hi'] > 250)).astype(int)
    if 'ap_lo' in df.columns:
        df['ap_lo_extreme'] = ((df['ap_lo'] < 40) | (df['ap_lo'] > 150)).astype(int)
    if col_ok(df, ['height', 'weight']):
        df['bmi_raw'] = df['weight'] / (df['height'] / 100) ** 2
        df['bmi_outlier'] = ((df['bmi_raw'] < 15) | (df['bmi_raw'] > 50)).astype(int)
    return df

def impute_values(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].median())
        elif df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df

def engineer_features(df):
    df = df.copy()
    if 'age' in df.columns:
        df['age_years'] = df['age'] / 365.25
        df['age_squared'] = df['age_years'] ** 2
        df['age_group_young'] = (df['age_years'] < 45).astype(int)
        df['age_group_middle'] = ((df['age_years'] >= 45) & (df['age_years'] <= 60)).astype(int)
        df['age_group_senior'] = (df['age_years'] > 60).astype(int)
    if 'gender' in df.columns:
        df['gender_male'] = (df['gender'] == 2).astype(int)
    if col_ok(df, ['height', 'weight']):
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
        df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
        df['bmi_normal'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)
        df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
        df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
    if col_ok(df, ['ap_hi', 'ap_lo']):
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['mean_arterial_pressure'] = df['ap_lo'] + df['pulse_pressure'] / 3.0
        df['bp_ratio'] = df['ap_hi'] / (df['ap_lo'] + 1e-06)
    if col_ok(df, ['ap_hi', 'ap_lo']):
        df['bp_normal'] = ((df['ap_hi'] < 120) & (df['ap_lo'] < 80)).astype(int)
        df['bp_elevated'] = (df['ap_hi'].between(120, 129) & (df['ap_lo'] < 80)).astype(int)
        df['bp_stage1_htn'] = (df['ap_hi'].between(130, 139) | df['ap_lo'].between(80, 89)).astype(int)
        df['bp_stage2_htn'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)
        df['bp_crisis'] = ((df['ap_hi'] > 180) | (df['ap_lo'] > 120)).astype(int)
        df['has_hypertension'] = (df['bp_stage1_htn'] | df['bp_stage2_htn'] | df['bp_crisis']).astype(int)
        df['isolated_systolic_htn'] = ((df['ap_hi'] >= 140) & (df['ap_lo'] < 80)).astype(int)
        df['isolated_diastolic_htn'] = ((df['ap_lo'] >= 90) & (df['ap_hi'] < 130)).astype(int)
    if col_ok(df, ['mean_arterial_pressure', 'age_years']):
        df['bp_age_interact'] = df['mean_arterial_pressure'] * df['age_years']
    if col_ok(df, ['pulse_pressure', 'age_years']):
        df['pulse_pressure_age_ratio'] = df['pulse_pressure'] / (df['age_years'] + 1e-06)
    if 'cholesterol' in df.columns:
        df['chol_normal'] = (df['cholesterol'] == 1).astype(int)
        df['chol_high'] = (df['cholesterol'] == 2).astype(int)
        df['chol_very_high'] = (df['cholesterol'] == 3).astype(int)
        df['chol_abnormal'] = (df['cholesterol'] > 1).astype(int)
    if 'gluc' in df.columns:
        df['gluc_normal'] = (df['gluc'] == 1).astype(int)
        df['gluc_high'] = (df['gluc'] == 2).astype(int)
        df['gluc_very_high'] = (df['gluc'] == 3).astype(int)
        df['gluc_abnormal'] = (df['gluc'] > 1).astype(int)
        df['has_diabetes'] = (df['gluc'] == 3).astype(int)
    if col_ok(df, ['bmi_obese', 'has_hypertension', 'chol_abnormal', 'gluc_abnormal']):
        df['metabolic_risk_score'] = df['bmi_obese'] + df['has_hypertension'] + df['chol_abnormal'] + df['gluc_abnormal']
        df['has_metabolic_syndrome'] = (df['metabolic_risk_score'] >= 3).astype(int)
    for col in ['smoke', 'alco', 'active']:
        if col not in df.columns:
            df[col] = 0
    df['lifestyle_risk_score'] = df['smoke'] * 2 + df['alco'] * 1 - df['active'] * 1
    df['high_risk_lifestyle'] = ((df['smoke'] == 1) & (df['alco'] == 1) & (df['active'] == 0)).astype(int)
    if col_ok(df, ['age_years', 'gender_male']):
        df['age_gender_interact'] = df['age_years'] * df['gender_male']
    if col_ok(df, ['age_years', 'bmi']):
        df['age_bmi_interact'] = df['age_years'] * df['bmi']
    if col_ok(df, ['age_years', 'smoke']):
        df['age_smoke_interact'] = df['age_years'] * df['smoke']
    if col_ok(df, ['age_years', 'chol_abnormal']):
        df['age_chol_interact'] = df['age_years'] * df['chol_abnormal']
    if col_ok(df, ['bmi', 'mean_arterial_pressure']):
        df['bmi_bp_interact'] = df['bmi'] * df['mean_arterial_pressure']
    if col_ok(df, ['bmi', 'chol_abnormal']):
        df['bmi_chol_interact2'] = df['bmi'] * df['chol_abnormal']
    if col_ok(df, ['bmi', 'gluc_abnormal']):
        df['bmi_gluc_interact'] = df['bmi'] * df['gluc_abnormal']
    if col_ok(df, ['smoke', 'has_hypertension']):
        df['smoke_bp_interact'] = df['smoke'] * df['has_hypertension']
    if col_ok(df, ['smoke', 'chol_abnormal']):
        df['smoke_chol_interact'] = df['smoke'] * df['chol_abnormal']
    if col_ok(df, ['gender_male', 'has_hypertension']):
        df['male_high_bp'] = df['gender_male'] * df['has_hypertension']
    if 'metabolic_risk_score' in df.columns and 'gender_male' in df.columns:
        df['female_metabolic_risk'] = (1 - df['gender_male']) * df['metabolic_risk_score']
    if True:
        df['framingham_risk'] = 0
        if 'age_years' in df.columns:
            df['framingham_risk'] += df['age_years'] * 0.1
        if 'gender_male' in df.columns:
            df['framingham_risk'] += df['gender_male'] * 2.0
        df['framingham_risk'] += df['smoke'] * 3.0
        if 'has_hypertension' in df.columns:
            df['framingham_risk'] += df['has_hypertension'] * 2.0
        if 'chol_abnormal' in df.columns:
            df['framingham_risk'] += df['chol_abnormal'] * 1.5
        if 'has_diabetes' in df.columns:
            df['framingham_risk'] += df['has_diabetes'] * 2.0
        if 'bmi_obese' in df.columns:
            df['framingham_risk'] += df['bmi_obese'] * 1.0
        cv = 0
        cv += df['bp_normal'] if 'bp_normal' in df.columns else 0
        cv += df['chol_normal'] if 'chol_normal' in df.columns else 0
        cv += df['gluc_normal'] if 'gluc_normal' in df.columns else 0
        cv += df['bmi_normal'] if 'bmi_normal' in df.columns else 0
        cv += 1 - df['smoke']
        cv += df['active']
        cv += (1 - df['alco']) * 0.5
        df['cv_health_score'] = cv.astype(float)
    for col in ['ap_hi', 'ap_lo', 'pulse_pressure', 'weight', 'bmi']:
        if col in df.columns:
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q1, q99)
    for col in ['weight', 'bmi']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(np.maximum(df[col], 0))
    return df

def one_hot_encode(df):
    cat_cols = []
    if set(['age_group_young', 'age_group_middle', 'age_group_senior']).issubset(df.columns):
        pass
    if set(['bp_normal', 'bp_elevated', 'bp_stage1_htn', 'bp_stage2_htn', 'bp_crisis']).issubset(df.columns):
        df['bp_stage'] = 0
        df.loc[df['bp_elevated'] == 1, 'bp_stage'] = 1
        df.loc[df['bp_stage1_htn'] == 1, 'bp_stage'] = 2
        df.loc[df['bp_stage2_htn'] == 1, 'bp_stage'] = 3
        df.loc[df['bp_crisis'] == 1, 'bp_stage'] = 4
        cat_cols.append('bp_stage')
    for c in ['cholesterol', 'gluc']:
        if c in df.columns:
            cat_cols.append(c)
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def correlation_prune(df, target_col='cardio', thresh=0.95):
    cols = [c for c in df.columns if c != target_col]
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > thresh)]
    keep = [c for c in cols if c not in drop_cols]
    keep += [target_col] if target_col in df.columns else []
    if drop_cols:
        print(f'Correlation prune dropping: {len(drop_cols)} cols')
    return df[keep]

def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if thr is None or len(thr) == 0:
        return 0.5
    idx = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[idx], 0.05, 0.95))

def train_models(X_train, y_train, X_val, y_val):
    models = []
    val_probs = {}
    dt_configs = [{'max_depth': 4, 'min_samples_split': 50, 'min_samples_leaf': 25}, {'max_depth': 6, 'min_samples_split': 50, 'min_samples_leaf': 25}, {'max_depth': 8, 'min_samples_split': 50, 'min_samples_leaf': 20}, {'max_depth': 10, 'min_samples_split': 100, 'min_samples_leaf': 30}]
    for i, cfg in enumerate(dt_configs):
        dt = DecisionTreeClassifier(max_depth=cfg['max_depth'], min_samples_split=cfg['min_samples_split'], min_samples_leaf=cfg['min_samples_leaf'], class_weight='balanced', random_state=RANDOM_SEED + i)
        dt.fit(X_train, y_train)
        models.append((f'dt_{i}', dt, None))
        val_probs[f'dt_{i}'] = dt.predict_proba(X_val)[:, 1]
    return (models, val_probs)

def predict_test(models, X_test):
    probs = {}
    for name, model, scaler in models:
        if model is None:
            probs[name] = np.zeros(len(X_test))
            continue
        if scaler is not None:
            Xt = scaler.transform(X_test)
            probs[name] = model.predict_proba(Xt)[:, 1]
        else:
            probs[name] = model.predict_proba(X_test)[:, 1]
    return probs

class MLP(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

def train_nn(X_train, y_train, X_val, y_val, seed=42):
    if not HAS_TORCH:
        return (None, None)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    val_x = torch.tensor(Xva, dtype=torch.float32).to(device)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    best_auc, best_state, patience, no_improve = (-1.0, None, 8, 0)
    for epoch in range(60):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x).squeeze(1).cpu().numpy()
            val_prob = 1 / (1 + np.exp(-val_logits))
            auc = roc_auc_score(y_val, val_prob)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return (model, scaler)

def predict_nn(model, scaler, X_data):
    if model is None:
        return np.zeros(len(X_data))
    Xn = scaler.transform(X_data)
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(Xn, dtype=torch.float32).to(device)
        logits = model(x).squeeze(1).cpu().numpy()
        prob = 1 / (1 + np.exp(-logits))
    return prob

def main():
    train, test = load_data()
    train = clean_and_flag(train)
    test = clean_and_flag(test)
    train = impute_values(train)
    test = impute_values(test)
    train = engineer_features(train)
    test = engineer_features(test)
    train = one_hot_encode(train)
    test = one_hot_encode(test)
    common = [c for c in train.columns if c in test.columns]
    if 'cardio' in common:
        common.remove('cardio')
    X = train[common]
    y = train['cardio']
    X_test = test[common]
    train_cp = correlation_prune(pd.concat([X, y], axis=1), target_col='cardio', thresh=0.95)
    feats = [c for c in train_cp.columns if c != 'cardio']
    X = X[feats]
    X_test = X_test[feats]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    models, val_probs = train_models(X_tr, y_tr, X_va, y_va)
    nn_model, nn_scaler = train_nn(X_tr, y_tr, X_va, y_va)
    if nn_model is not None:
        val_probs['nn'] = predict_nn(nn_model, nn_scaler, X_va)
    auc_scores = {k: roc_auc_score(y_va, v) for k, v in val_probs.items()}
    print('Val AUC:', {k: round(v, 4) for k, v in auc_scores.items()})
    weights = np.maximum(0.0, np.array(list(auc_scores.values())) - 0.5)
    weights = weights / (weights.sum() + 1e-09)
    ens_val = np.zeros(len(y_va))
    for w, k in zip(weights, val_probs.keys()):
        ens_val += w * val_probs[k]
    thr = youden_threshold(y_va, ens_val)
    print(f'Ensemble thr (Youden): {thr:.3f}')
    test_probs_map = predict_test(models, X_test)
    ens_test = np.zeros(len(X_test))
    for w, k in zip(weights, test_probs_map.keys()):
        ens_test += w * test_probs_map[k]
    if ens_test.max() - ens_test.min() < 1e-06:
        thr = np.median(ens_test)
    y_pred = (ens_test >= thr).astype(int)
    sub = pd.DataFrame({'id': test['id'] if 'id' in test.columns else np.arange(len(X_test)), 'cardio': y_pred})
    print(f'Positives in submission: {(sub.cardio == 1).sum()} / {len(sub)}')
    sub.to_csv(SUBMISSION_OUT, index=False)
    print(f'Saved: {SUBMISSION_OUT}')
if __name__ == '__main__':
    main()

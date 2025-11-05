import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report, accuracy_score, precision_score, recall_score

try:
	import torch
	import torch.nn as nn
	from torch.utils.data import TensorDataset, DataLoader
	HAS_TORCH = True
except Exception:
	HAS_TORCH = False

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
	from catboost import CatBoostClassifier
	HAS_CAT = True
except Exception:
	HAS_CAT = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

\
TRAIN_PATH = r'C:\Projects\dataverse-nsut\Round 2\data\main.csv'
TEST_PATH = r'C:\Projects\dataverse-nsut\Round 2\data\test.csv'
SUBMISSION_OUT = 'submission.csv'

AP_HI_RANGE = (70, 250)
AP_LO_RANGE = (40, 150)
HEIGHT_RANGE = (130, 220)
WEIGHT_RANGE = (30, 200)
BMI_RANGE = (15, 50)

def load_data():
	train = pd.read_csv(TRAIN_PATH)
	test = pd.read_csv(TEST_PATH)
	print(f'Train shape: {train.shape}  Test shape: {test.shape}')
	return train, test

def col_ok(df, cols):
	return all(c in df.columns for c in cols)

def clean_outliers(df):
	keep = pd.Series(True, index=df.index)
	if 'ap_hi' in df.columns:
		keep &= (df['ap_hi'].between(AP_HI_RANGE[0], AP_HI_RANGE[1]))
	if 'ap_lo' in df.columns:
		keep &= (df['ap_lo'].between(AP_LO_RANGE[0], AP_LO_RANGE[1]))
	if col_ok(df, ['ap_hi','ap_lo']):
		keep &= (df['ap_hi'] > df['ap_lo'])
	if 'height' in df.columns:
		keep &= (df['height'].between(HEIGHT_RANGE[0], HEIGHT_RANGE[1]))
	if 'weight' in df.columns:
		keep &= (df['weight'].between(WEIGHT_RANGE[0], WEIGHT_RANGE[1]))
	if col_ok(df, ['height','weight']):
		bmi = df['weight'] / ((df['height'] / 100) ** 2)
		keep &= (bmi.between(BMI_RANGE[0], BMI_RANGE[1]))
	if col_ok(df, ['ap_hi','ap_lo']):
		pp = df['ap_hi'] - df['ap_lo']
		keep &= (pp.between(20, 100))
	if 'age' in df.columns and col_ok(df, ['cholesterol','gluc']):
		age_years_tmp = (df['age'] // 365).astype(int)
		keep &= ~(((df['cholesterol'] == 3) & (df['gluc'] == 3) & (age_years_tmp < 35)))
	if col_ok(df, ['height','weight']):
		bmi2 = df['weight'] / ((df['height'] / 100) ** 2)
		keep &= ~((bmi2 < 16) | (bmi2 > 45))
	df2 = df[keep].copy()
	print(f'Cleaning dropped {len(df) - len(df2)} rows out of {len(df)}')
	return df2

def engineer_features(df, chol_mean=None):
	if 'age' in df.columns:
		df['age_years'] = (df['age'] // 365).astype(int)
	if col_ok(df, ['height','weight']):
		df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
	if col_ok(df, ['ap_hi','ap_lo']):
		df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
		df['bp_high'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)
		df['ap_hi_to_lo'] = df['ap_hi'] / (df['ap_lo'] + 1e-6)
		df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
	if col_ok(df, ['age_years','bmi']):
		df['age_bmi_interact'] = df['age_years'] * df['bmi']
	if col_ok(df, ['bmi','cholesterol']):
		df['bmi_chol_interact'] = df['bmi'] * df['cholesterol']
	if 'cholesterol' in df.columns:
		if chol_mean is None:
			chol_mean = df['cholesterol'].mean()
		df['chol_m_diff'] = df['cholesterol'] - chol_mean
	if 'ap_hi' in df.columns:
		df['ap_hi_sq'] = df['ap_hi'] ** 2
	if 'bp_high' in df.columns and 'cholesterol' in df.columns and 'gluc' in df.columns:
		df['risk_score'] = ((df['bp_high'] == 1).astype(int) + (df['cholesterol'] == 3).astype(int) + (df['gluc'] == 3).astype(int))
           
	met_bmi = (df['bmi'] > 30).astype(int) if 'bmi' in df.columns else 0
	met_chol = (df['cholesterol'] == 3).astype(int) if 'cholesterol' in df.columns else 0
	met_gluc = (df['gluc'] == 3).astype(int) if 'gluc' in df.columns else 0
	met_inactive = (df['active'] == 0).astype(int) if 'active' in df.columns else 0
	df['metabolic_risk_score'] = met_bmi + met_chol + met_gluc + met_inactive
	if col_ok(df, ['ap_hi','ap_lo']):
		stage = np.zeros(len(df), dtype=int)
		stage[(df['ap_hi'] >= 130) | (df['ap_lo'] >= 80)] = 1
		stage[(df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)] = 2
		stage[(df['ap_hi'] >= 180) | (df['ap_lo'] >= 120)] = 3
		df['hypertension_stage'] = stage
	if col_ok(df, ['age_years','gender']):
		df['age_x_gender'] = df['age_years'] * df['gender']
		df['is_older_female'] = ((df['age_years'] > 55) & (df['gender'] == 2)).astype(int)
		df['is_older_male'] = ((df['age_years'] > 55) & (df['gender'] == 1)).astype(int)
	if 'ap_lo' in df.columns:
		df['ap_lo_squared'] = df['ap_lo'] ** 2
	if 'age_years' in df.columns:
		df['age_years_squared'] = df['age_years'] ** 2
	if 'pulse_pressure' in df.columns:
		df['sqrt_pulse_pressure'] = np.sqrt(np.maximum(df['pulse_pressure'], 0))
	return df, chol_mean

def add_group_deviation(train_fe, test_fe):
	if 'age_years' in train_fe.columns and 'gender' in train_fe.columns:
		train_fe['age_group'] = pd.cut(train_fe['age_years'], bins=[29,40,50,60,200], labels=['29-40','40-50','50-60','60+'], include_lowest=True)
		grp = train_fe.groupby(['age_group','gender'])
		bmi_mean_map = grp['bmi'].mean() if 'bmi' in train_fe.columns else None
		bp_mean_map = grp['ap_hi'].mean() if 'ap_hi' in train_fe.columns else None
		if bmi_mean_map is not None:
			train_fe['bmi_deviation'] = train_fe['bmi'] - train_fe.set_index(['age_group','gender']).index.map(bmi_mean_map).values
		if bp_mean_map is not None:
			train_fe['bp_deviation'] = train_fe['ap_hi'] - train_fe.set_index(['age_group','gender']).index.map(bp_mean_map).values
		if 'age_years' in test_fe.columns and 'gender' in test_fe.columns:
			test_fe['age_group'] = pd.cut(test_fe['age_years'], bins=[29,40,50,60,200], labels=['29-40','40-50','50-60','60+'], include_lowest=True)
			if bmi_mean_map is not None and 'bmi' in test_fe.columns:
				keys = list(zip(test_fe['age_group'], test_fe['gender']))
				test_fe['bmi_deviation'] = test_fe['bmi'] - pd.Series(keys).map(bmi_mean_map).values
			if bp_mean_map is not None and 'ap_hi' in test_fe.columns:
				keys2 = list(zip(test_fe['age_group'], test_fe['gender']))
				test_fe['bp_deviation'] = test_fe['ap_hi'] - pd.Series(keys2).map(bp_mean_map).values
	return train_fe, test_fe

def get_feature_names():
	return [
	 'ap_hi','bp_high','ap_hi_to_lo','ap_lo','pulse_pressure','age_bmi_interact','bmi_chol_interact',
	 'age_years','cholesterol','bmi','weight','gluc','active','gender','alco','smoke','height','map',
	 'chol_m_diff','ap_hi_sq','risk_score','metabolic_risk_score','hypertension_stage','age_x_gender',
	 'is_older_female','is_older_male','ap_lo_squared','age_years_squared','sqrt_pulse_pressure',
	 'bmi_deviation','bp_deviation'
	]

def base_feature_candidates():
	return ['age','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','gender','height','weight']

def youden_threshold(y_true, y_prob):
	fpr, tpr, thr = roc_curve(y_true, y_prob)
	if thr is None or len(thr) == 0:
		return 0.5
	j = tpr - fpr
	idx = int(np.argmax(j))
	best_t = float(np.clip(thr[idx], 0.05, 0.95))
	return best_t

def adversarial_weights(X_train, X_test, y):
	try:
		clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, num_leaves=31, random_state=RANDOM_SEED)
	except Exception:
		return np.ones(len(X_train), dtype=float)
	comb = pd.concat([X_train, X_test], axis=0, ignore_index=True)
	labels = np.concatenate([np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)])
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
	preds = np.zeros(len(comb))
	for tr, va in sKF_split(len(comb), labels, s=RANDOM_SEED, n_splits=3):
		clf.fit(comb.iloc[tr], labels[tr])
		preds[va] = clf.predict_proba(comb.iloc[va])[:,1]
	train_p = preds[:len(X_train)]
	weights = 1.0 - train_p
	weights /= (weights.mean() + 1e-9)
	print(f'Adversarial AUC: {roc_auc_score(labels, preds):.4f}; weights mean normalized to 1.0')
	return weights

def sKF_split(n, y, s, n_splits=5):
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
	for tr, va in skf.split(np.zeros(n), y):
		yield tr, va

def train_models_cv(X, y, X_test, seeds=(42, 2025), n_splits=5):
	models = []
	model_names = []
	oof = {}
	test_probs = {}
	feature_importances = []
	for seed in seeds:
		print(f'CV seed {seed}')
		kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
		for name in ['logreg','rf','xgb_deep','xgb_shallow','lgbm','catboost','nn']:
			oof.setdefault(name, np.zeros(len(X)))
			test_probs.setdefault(name, [])
		val_counts = np.zeros(len(X))
                                     
		try:
			aw = adversarial_weights(X, X_test, y)
		except Exception:
			aw = np.ones(len(X), dtype=float)
		for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
			X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
			y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
			w_tr = aw[tr_idx]
           
			
   scaler = StandardScaler()
			X_tr_s = scaler.fit_transform(X_tr)
			X_va_s = scaler.transform(X_va)
			X_te_s = scaler.transform(X_test)
			mlr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=seed)
			mlr.fit(X_tr_s, y_tr)
			p_va = mlr.predict_proba(X_va_s)[:,1]
			p_te = mlr.predict_proba(X_te_s)[:,1]
			oof['logreg'][va_idx] += p_va
			val_counts[va_idx] += 1
			test_probs['logreg'].append(p_te)
          
			rf = RandomForestClassifier(n_estimators=400, max_depth=15, min_samples_split=20, class_weight='balanced', random_state=seed, n_jobs=-1)
			rf.fit(X_tr, y_tr, sample_weight=w_tr)
			p_va = rf.predict_proba(X_va)[:,1]
			p_te = rf.predict_proba(X_test)[:,1]
			oof['rf'][va_idx] += p_va
			test_probs['rf'].append(p_te)
                
			if HAS_XGB:
				xgb1 = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=9, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, gamma=1.0, reg_alpha=0.1, reg_lambda=1.0, random_state=seed, use_label_encoder=False, eval_metric='logloss')
				try:
					xgb1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False, sample_weight=w_tr)
				except TypeError:
					xgb1.fit(X_tr, y_tr, sample_weight=w_tr)
				p_va = xgb1.predict_proba(X_va)[:,1]
				p_te = xgb1.predict_proba(X_test)[:,1]
				oof['xgb_deep'][va_idx] += p_va
				test_probs['xgb_deep'].append(p_te)
                   
			if HAS_XGB:
				xgb2 = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, min_child_weight=10, subsample=0.9, colsample_bytree=0.9, gamma=0.5, random_state=seed, use_label_encoder=False, eval_metric='logloss')
				try:
					xgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50, verbose=False, sample_weight=w_tr)
				except TypeError:
					xgb2.fit(X_tr, y_tr, sample_weight=w_tr)
				p_va = xgb2.predict_proba(X_va)[:,1]
				p_te = xgb2.predict_proba(X_test)[:,1]
				oof['xgb_shallow'][va_idx] += p_va
				test_probs['xgb_shallow'].append(p_te)
            
			if HAS_LGBM:
				lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=8, num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=0.3, random_state=seed)
				try:
					lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False, sample_weight=w_tr)
				except TypeError:
					lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
				p_va = lgbm.predict_proba(X_va)[:,1]
				p_te = lgbm.predict_proba(X_test)[:,1]
				oof['lgbm'][va_idx] += p_va
				test_probs['lgbm'].append(p_te)
                 
				feature_importances.append(pd.Series(lgbm.feature_importances_, index=X.columns))
                
			if HAS_CAT:
				cat_cols = ['gender','cholesterol','gluc','smoke','alco','active','hypertension_stage']
				cat_idx = [i for i, c in enumerate(X_tr.columns) if c in cat_cols]
				cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=8, l2_leaf_reg=3, border_count=254, random_state=seed, loss_function='Logloss', eval_metric='AUC', verbose=False)
				try:
					cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx, early_stopping_rounds=50, verbose=False, sample_weight=w_tr)
				except TypeError:
					cat.fit(X_tr, y_tr, cat_features=cat_idx, verbose=False, sample_weight=w_tr)
				p_va = cat.predict_proba(X_va)[:,1]
				p_te = cat.predict_proba(X_test)[:,1]
				oof['catboost'][va_idx] += p_va
				test_probs['catboost'].append(p_te)
                                   
			if HAS_TORCH:
                                                   
				scaler_nn = StandardScaler()
				X_tr_nn = scaler_nn.fit_transform(X_tr)
				X_va_nn = scaler_nn.transform(X_va)
				X_te_nn = scaler_nn.transform(X_test)
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
				class MLP(nn.Module):
					def __init__(self, in_dim):
						super().__init__()
						self.net = nn.Sequential(
						 nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
						 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
						 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
						 nn.Linear(64, 1)
						)
					def forward(self, x):
						return self.net(x)
				model_nn = MLP(X_tr_nn.shape[1]).to(device)
				optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)
				criterion = nn.BCEWithLogitsLoss()
				train_ds = TensorDataset(torch.tensor(X_tr_nn, dtype=torch.float32), torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1))
				val_ds = TensorDataset(torch.tensor(X_va_nn, dtype=torch.float32), torch.tensor(y_va.values, dtype=torch.float32).unsqueeze(1))
				train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
				val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
				best_auc, best_state, patience, no_improve = -1.0, None, 8, 0
				for epoch in range(50):
					model_nn.train()
					for xb, yb in train_loader:
						xb = xb.to(device); yb = yb.to(device)
						optimizer.zero_grad()
						logits = model_nn(xb)
						loss = criterion(logits, yb)
						loss.backward()
						optimizer.step()
          
					model_nn.eval()
					with torch.no_grad():
						vb = torch.tensor(X_va_nn, dtype=torch.float32).to(device)
						val_logits = model_nn(vb).squeeze(1).cpu().numpy()
						val_prob = 1/(1+np.exp(-val_logits))
						auc = roc_auc_score(y_va, val_prob)
					if auc > best_auc:
						best_auc = auc
						best_state = {k: v.cpu().clone() for k, v in model_nn.state_dict().items()}
						no_improve = 0
					else:
						no_improve += 1
					if no_improve >= patience:
						break
				if best_state is not None:
					model_nn.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                 
				with torch.no_grad():
					vb = torch.tensor(X_va_nn, dtype=torch.float32).to(device)
					val_logits = model_nn(vb).squeeze(1).cpu().numpy()
					p_va = 1/(1+np.exp(-val_logits))
					tb = torch.tensor(X_te_nn, dtype=torch.float32).to(device)
					test_logits = model_nn(tb).squeeze(1).cpu().numpy()
					p_te = 1/(1+np.exp(-test_logits))
				oof['nn'][va_idx] += p_va
				test_probs['nn'].append(p_te)
                              
		for k in oof:
			oof[k] /= np.maximum(1, val_counts)
                                             
	for k, lst in test_probs.items():
		if len(lst) > 0:
			test_probs[k] = np.mean(np.vstack(lst), axis=0)
		else:
			test_probs[k] = np.zeros(len(X_test))
	models.append(oof)
	model_names = list(oof.keys())
                 
	\
 avg_oof = {k: np.mean([m[k] for m in models], axis=0) for k in model_names}
                                                                                                                                                  
	avg_test = test_probs
                               
	auc_map = {k: roc_auc_score(y, avg_oof[k]) for k in model_names}
	print('OOF AUC:', {k: round(v,4) for k,v in auc_map.items()})
                                         
	imp_avg = None
	if len(feature_importances) > 0:
		imp_avg = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
	return avg_oof, avg_test, auc_map, imp_avg, model_names

def stack_and_calibrate(oof_map, test_map, y):
                      
	model_names = list(oof_map.keys())
	X_meta = np.vstack([oof_map[m] for m in model_names]).T
	X_meta_test = np.vstack([test_map[m] for m in model_names]).T
	meta = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED)
	meta.fit(X_meta, y)
	y_oof = meta.predict_proba(X_meta)[:,1]
	iso = IsotonicRegression(out_of_bounds='clip')
	iso.fit(y_oof, y)
	y_oof_cal = iso.transform(y_oof)
	best_thr = youden_threshold(y, y_oof_cal)
	print(f'Meta OOF AUC: {roc_auc_score(y, y_oof_cal):.4f}, thr={best_thr:.3f}')
       
	y_test_prob = meta.predict_proba(X_meta_test)[:,1]
	y_test_prob_cal = iso.transform(y_test_prob)
	return y_test_prob_cal, best_thr

def main():
	train, test = load_data()
	train = clean_outliers(train)
	test = clean_outliers(test)
	train_fe, chol_mean = engineer_features(train)
	test_fe, _ = engineer_features(test, chol_mean)
	train_fe, test_fe = add_group_deviation(train_fe, test_fe)
               
	if train_fe.isnull().any().any():
		train_fe = train_fe.fillna(train_fe.median(numeric_only=True))
	if test_fe.isnull().any().any():
		test_fe = test_fe.fillna(train_fe.median(numeric_only=True))
           
	feats = [f for f in get_feature_names() if f in train_fe.columns and f in test_fe.columns]
	print('Using features:', feats)
	X = train_fe[feats]
	y = train_fe['cardio']
	X_test = test_fe[feats]
                                                   
	oof_map, test_map, auc_map, imp_avg, model_names = train_models_cv(X, y, X_test, seeds=(42, 2025), n_splits=5)
                                                            
	if imp_avg is not None:
		cut = int(max(1, round(0.2 * len(imp_avg))))
		to_drop = set(imp_avg.sort_values(ascending=True).index[:cut])
		keep_feats = [f for f in feats if f not in to_drop]
		print(f'Pruning {len(to_drop)} features. New feature count: {len(keep_feats)}')
		X = train_fe[keep_feats]
		X_test = test_fe[keep_feats]
                           
		oof_map, test_map, auc_map, _, model_names = train_models_cv(X, y, X_test, seeds=(42, 2025), n_splits=5)
                              
	weights = np.maximum(0.0, np.array([auc_map[m] for m in model_names]) - 0.5)
	weights = weights / (weights.sum() + 1e-9)
	ens_oof = np.zeros(len(y))
	ens_test = np.zeros(len(X_test))
	for w, m in zip(weights, model_names):
		ens_oof += w * oof_map[m]
		ens_test += w * test_map[m]
	print('Ensemble OOF AUC:', roc_auc_score(y, ens_oof))
                                  
	y_test_prob_cal, thr = stack_and_calibrate(oof_map, test_map, y)
                                              
	if (y_test_prob_cal.max() - y_test_prob_cal.min()) < 1e-6:
		print('Calibration degenerate; using ensemble probabilities')
		y_test_prob_cal = ens_test
		thr = youden_threshold(y, ens_oof)
                   
	y_pred = (y_test_prob_cal >= thr).astype(int)
	sub = pd.DataFrame({'id': test_fe['id'] if 'id' in test_fe.columns else np.arange(len(X_test)), 'cardio': y_pred})
	print(f'Pred positives: {(sub.cardio==1).sum()} / {len(sub)}')
	sub.to_csv(SUBMISSION_OUT, index=False)
	print(f'Saved: {SUBMISSION_OUT}')

if __name__ == '__main__':
	main()


import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json
import joblib
warnings.filterwarnings('ignore')
print('=' * 80)
print('DEEP FEATURE ENGINEERING - NUCLEAR OPTION')
print('=' * 80)
print('\n[STEP 1] Loading data...')
df = pd.read_csv('../data/main.csv')
test_df = pd.read_csv('../data/test.csv')
print(f'Train: {df.shape}, Test: {test_df.shape}')
print('\n[STEP 2] Basic preprocessing...')
df_proc = df.copy()
test_proc = test_df.copy()
numeric_cols = df_proc.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_proc[col].isnull().sum() > 0:
        median_val = df_proc[col].median()
        df_proc[col].fillna(median_val, inplace=True)
        if col in test_proc.columns:
            test_proc[col].fillna(median_val, inplace=True)
constant_features = [col for col in numeric_cols if col not in ['Label', 'sha256'] and df_proc[col].nunique() == 1]
if constant_features:
    df_proc.drop(columns=constant_features, inplace=True)
    test_proc.drop(columns=[c for c in constant_features if c in test_proc.columns], inplace=True)
    print(f'Removed {len(constant_features)} constant features')
print('\n[STEP 3] Advanced Feature Engineering - Phase 1: Statistical Features...')
numeric_features = [col for col in df_proc.columns if col not in ['Label', 'sha256'] and df_proc[col].dtype in ['int64', 'float64']]
print(f'Starting with {len(numeric_features)} numeric features')
print('Creating row-level statistical features...')
df_proc['row_sum'] = df_proc[numeric_features].sum(axis=1)
df_proc['row_mean'] = df_proc[numeric_features].mean(axis=1)
df_proc['row_std'] = df_proc[numeric_features].std(axis=1)
df_proc['row_median'] = df_proc[numeric_features].median(axis=1)
df_proc['row_min'] = df_proc[numeric_features].min(axis=1)
df_proc['row_max'] = df_proc[numeric_features].max(axis=1)
df_proc['row_range'] = df_proc['row_max'] - df_proc['row_min']
df_proc['row_skew'] = df_proc[numeric_features].skew(axis=1)
df_proc['row_kurt'] = df_proc[numeric_features].kurtosis(axis=1)
df_proc['row_zeros'] = (df_proc[numeric_features] == 0).sum(axis=1)
df_proc['row_negatives'] = (df_proc[numeric_features] < 0).sum(axis=1)
df_proc['row_positives'] = (df_proc[numeric_features] > 0).sum(axis=1)
test_proc['row_sum'] = test_proc[numeric_features].sum(axis=1)
test_proc['row_mean'] = test_proc[numeric_features].mean(axis=1)
test_proc['row_std'] = test_proc[numeric_features].std(axis=1)
test_proc['row_median'] = test_proc[numeric_features].median(axis=1)
test_proc['row_min'] = test_proc[numeric_features].min(axis=1)
test_proc['row_max'] = test_proc[numeric_features].max(axis=1)
test_proc['row_range'] = test_proc['row_max'] - test_proc['row_min']
test_proc['row_skew'] = test_proc[numeric_features].skew(axis=1)
test_proc['row_kurt'] = test_proc[numeric_features].kurtosis(axis=1)
test_proc['row_zeros'] = (test_proc[numeric_features] == 0).sum(axis=1)
test_proc['row_negatives'] = (test_proc[numeric_features] < 0).sum(axis=1)
test_proc['row_positives'] = (test_proc[numeric_features] > 0).sum(axis=1)
print('Creating feature interactions...')
top_features = df_proc[numeric_features].var().nlargest(15).index.tolist()
interaction_count = 0
for i in range(len(top_features)):
    for j in range(i + 1, min(i + 4, len(top_features))):
        feat1, feat2 = (top_features[i], top_features[j])
        df_proc[f'{feat1}_x_{feat2}'] = df_proc[feat1] * df_proc[feat2]
        df_proc[f'{feat1}_div_{feat2}'] = df_proc[feat1] / (df_proc[feat2] + 1e-08)
        df_proc[f'{feat1}_plus_{feat2}'] = df_proc[feat1] + df_proc[feat2]
        df_proc[f'{feat1}_minus_{feat2}'] = df_proc[feat1] - df_proc[feat2]
        test_proc[f'{feat1}_x_{feat2}'] = test_proc[feat1] * test_proc[feat2]
        test_proc[f'{feat1}_div_{feat2}'] = test_proc[feat1] / (test_proc[feat2] + 1e-08)
        test_proc[f'{feat1}_plus_{feat2}'] = test_proc[feat1] + test_proc[feat2]
        test_proc[f'{feat1}_minus_{feat2}'] = test_proc[feat1] - test_proc[feat2]
        interaction_count += 4
print(f'Created {interaction_count} interaction features')
print('\n[STEP 4] Advanced Feature Engineering - Phase 2: Transformations...')
print('Creating mathematical transformations...')
transformation_features = top_features[:10]
for feat in transformation_features:
    df_proc[f'{feat}_log'] = np.log1p(np.abs(df_proc[feat]))
    df_proc[f'{feat}_log2'] = np.log2(np.abs(df_proc[feat]) + 1)
    df_proc[f'{feat}_sqrt'] = np.sqrt(np.abs(df_proc[feat]))
    df_proc[f'{feat}_square'] = df_proc[feat] ** 2
    df_proc[f'{feat}_cube'] = df_proc[feat] ** 3
    df_proc[f'{feat}_sin'] = np.sin(df_proc[feat])
    df_proc[f'{feat}_cos'] = np.cos(df_proc[feat])
    test_proc[f'{feat}_log'] = np.log1p(np.abs(test_proc[feat]))
    test_proc[f'{feat}_log2'] = np.log2(np.abs(test_proc[feat]) + 1)
    test_proc[f'{feat}_sqrt'] = np.sqrt(np.abs(test_proc[feat]))
    test_proc[f'{feat}_square'] = test_proc[feat] ** 2
    test_proc[f'{feat}_cube'] = test_proc[feat] ** 3
    test_proc[f'{feat}_sin'] = np.sin(test_proc[feat])
    test_proc[f'{feat}_cos'] = np.cos(test_proc[feat])
print('Creating binning features...')
for feat in top_features[:8]:
    df_proc[f'{feat}_qbin'] = pd.qcut(df_proc[feat], q=5, labels=False, duplicates='drop')
    test_proc[f'{feat}_qbin'] = pd.cut(test_proc[feat], bins=pd.qcut(df_proc[feat], q=5, duplicates='drop').cat.categories, labels=False, include_lowest=True)
    df_proc[f'{feat}_ebin'] = pd.cut(df_proc[feat], bins=5, labels=False)
    test_proc[f'{feat}_ebin'] = pd.cut(test_proc[feat], bins=pd.cut(df_proc[feat], bins=5).cat.categories, labels=False, include_lowest=True)
print('\n[STEP 5] Advanced Feature Engineering - Phase 3: Clustering & Dimensionality...')
print('Creating clustering features...')
clustering_features = top_features[:20]
for n_clusters in [3, 5, 8]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_proc[f'kmeans_{n_clusters}'] = kmeans.fit_predict(df_proc[clustering_features])
    test_proc[f'kmeans_{n_clusters}'] = kmeans.predict(test_proc[clustering_features])
print('Creating dimensionality reduction features...')
pca = PCA(n_components=10, random_state=42)
pca_features = pca.fit_transform(df_proc[clustering_features])
for i in range(10):
    df_proc[f'pca_{i}'] = pca_features[:, i]
    test_proc[f'pca_{i}'] = pca.transform(test_proc[clustering_features])[:, i]
ica = FastICA(n_components=5, random_state=42, max_iter=1000)
ica_features = ica.fit_transform(df_proc[clustering_features])
for i in range(5):
    df_proc[f'ica_{i}'] = ica_features[:, i]
    test_proc[f'ica_{i}'] = ica.transform(test_proc[clustering_features])[:, i]
print('\n[STEP 6] Advanced Feature Engineering - Phase 4: Target-based Features...')
print('Creating target-based features...')
target_features = top_features[:5]
for feat in target_features:
    target_mean = df_proc.groupby(feat)['Label'].mean()
    df_proc[f'{feat}_target_mean'] = df_proc[feat].map(target_mean)
    test_proc[f'{feat}_target_mean'] = test_proc[feat].map(target_mean).fillna(df_proc['Label'].mean())
    target_std = df_proc.groupby(feat)['Label'].std()
    df_proc[f'{feat}_target_std'] = df_proc[feat].map(target_std)
    test_proc[f'{feat}_target_std'] = test_proc[feat].map(target_std).fillna(df_proc['Label'].std())
print('\n[STEP 7] Advanced Feature Selection - BGSA Inspired...')
all_features = [col for col in df_proc.columns if col not in ['Label', 'sha256']]
print(f'Total features after engineering: {len(all_features)}')
print('Handling NaN values in engineered features...')
for col in all_features:
    if df_proc[col].isnull().sum() > 0:
        if df_proc[col].dtype in ['int64', 'float64']:
            median_val = df_proc[col].median()
            df_proc[col].fillna(median_val, inplace=True)
            test_proc[col].fillna(median_val, inplace=True)
        else:
            mode_val = df_proc[col].mode()[0] if not df_proc[col].mode().empty else 0
            df_proc[col].fillna(mode_val, inplace=True)
            test_proc[col].fillna(mode_val, inplace=True)
print('Running multiple feature selection methods...')
mi_scores = mutual_info_classif(df_proc[all_features], df_proc['Label'], random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': all_features, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
f_scores = f_classif(df_proc[all_features], df_proc['Label'])[0]
f_df = pd.DataFrame({'feature': all_features, 'f_score': f_scores}).sort_values('f_score', ascending=False)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(df_proc[all_features], df_proc['Label'])
rf_importance = rf.feature_importances_
rf_df = pd.DataFrame({'feature': all_features, 'rf_importance': rf_importance}).sort_values('rf_importance', ascending=False)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(df_proc[all_features], df_proc['Label'])
xgb_importance = xgb_model.feature_importances_
xgb_df = pd.DataFrame({'feature': all_features, 'xgb_importance': xgb_importance}).sort_values('xgb_importance', ascending=False)
print('Combining feature selection scores...')
combined_scores = pd.DataFrame({'feature': all_features})
combined_scores = combined_scores.merge(mi_df[['feature', 'mi_score']], on='feature')
combined_scores = combined_scores.merge(f_df[['feature', 'f_score']], on='feature')
combined_scores = combined_scores.merge(rf_df[['feature', 'rf_importance']], on='feature')
combined_scores = combined_scores.merge(xgb_df[['feature', 'xgb_importance']], on='feature')
combined_scores['mi_norm'] = (combined_scores['mi_score'] - combined_scores['mi_score'].min()) / (combined_scores['mi_score'].max() - combined_scores['mi_score'].min())
combined_scores['f_norm'] = (combined_scores['f_score'] - combined_scores['f_score'].min()) / (combined_scores['f_score'].max() - combined_scores['f_score'].min())
combined_scores['rf_norm'] = (combined_scores['rf_importance'] - combined_scores['rf_importance'].min()) / (combined_scores['rf_importance'].max() - combined_scores['rf_importance'].min())
combined_scores['xgb_norm'] = (combined_scores['xgb_importance'] - combined_scores['xgb_importance'].min()) / (combined_scores['xgb_importance'].max() - combined_scores['xgb_importance'].min())
combined_scores['ensemble_score'] = 0.3 * combined_scores['mi_norm'] + 0.2 * combined_scores['f_norm'] + 0.25 * combined_scores['rf_norm'] + 0.25 * combined_scores['xgb_norm']
TOP_N = 80
selected_features = combined_scores.nlargest(TOP_N, 'ensemble_score')['feature'].tolist()
print(f'Selected {len(selected_features)} features using ensemble scoring')
print(f'Top 10 features:')
for feat in selected_features[:10]:
    score = combined_scores[combined_scores['feature'] == feat]['ensemble_score'].values[0]
    print(f'  {feat[:50]:<50} {score:.4f}')
print('\n[STEP 8] Preparing data for training...')
X = df_proc[selected_features]
y = df_proc['Label']
X_test = test_proc[selected_features]
print(f'Final feature count: {X.shape[1]}')
print('\n[STEP 9] Cross-validation with proven 99.57% approach...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\nFold {fold}/5')
    X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
    y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
    xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=70, max_depth=6, feature_fraction=0.8, bagging_fraction=0.8, reg_alpha=0.1, reg_lambda=2, random_state=42, n_jobs=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    cat_model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=2, random_seed=42, verbose=False)
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_pred)
    ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    oof_predictions[val_idx] = ensemble_pred
    cv_scores.append(ensemble_auc)
    print(f'  XGB: {xgb_auc:.6f} | LGB: {lgb_auc:.6f} | CAT: {cat_auc:.6f} | ENS: {ensemble_auc:.6f}')
mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
oof_auc = roc_auc_score(y, oof_predictions)
oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
print('\n' + '=' * 80)
print('DEEP FEATURE ENGINEERING RESULTS')
print('=' * 80)
print(f'Mean CV AUC: {mean_cv:.6f} (+/- {std_cv:.6f})')
print(f'OOF AUC: {oof_auc:.6f}')
print(f'OOF Accuracy: {oof_acc:.6f} ({oof_acc * 100:.2f}%)')
print(f'\nBaseline (99.57%): 0.995700')
print(f'Difference: {oof_auc - 0.9957:+.6f}')
print('=' * 80)
print('\n[STEP 10] Training final models...')
print('  XGBoost...')
xgb_final = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2, random_state=42, n_jobs=-1)
xgb_final.fit(X, y)
print('  LightGBM...')
lgb_final = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=70, max_depth=6, feature_fraction=0.8, bagging_fraction=0.8, reg_alpha=0.1, reg_lambda=2, random_state=42, n_jobs=-1)
lgb_final.fit(X, y)
print('  CatBoost...')
cat_final = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=2, random_seed=42, verbose=False)
cat_final.fit(X, y)
print('\n[STEP 11] Generating predictions...')
xgb_pred = xgb_final.predict_proba(X_test)[:, 1]
lgb_pred = lgb_final.predict_proba(X_test)[:, 1]
cat_pred = cat_final.predict_proba(X_test)[:, 1]
ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
submission = pd.DataFrame({'sha256': test_df['sha256'], 'Label': (ensemble_pred > 0.5).astype(int)})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_deep_features_{timestamp}.csv'
submission.to_csv(filename, index=False)
print('\n[STEP 12] Saving results...')
results = {'timestamp': timestamp, 'approach': 'deep_feature_engineering', 'total_features_created': len(all_features), 'selected_features': len(selected_features), 'cv_auc_mean': float(mean_cv), 'cv_auc_std': float(std_cv), 'oof_auc': float(oof_auc), 'oof_accuracy': float(oof_acc), 'feature_engineering': {'statistical_features': 12, 'interaction_features': interaction_count, 'transformation_features': len(transformation_features) * 7, 'binning_features': len(top_features[:8]) * 2, 'clustering_features': 3 * 3, 'pca_features': 10, 'ica_features': 5, 'target_features': len(target_features) * 2}}
with open(f'deep_features_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
joblib.dump(xgb_final, f'xgb_deep_{timestamp}.pkl')
joblib.dump(lgb_final, f'lgb_deep_{timestamp}.pkl')
joblib.dump(cat_final, f'cat_deep_{timestamp}.pkl')
print('\n' + '=' * 80)
print('DEEP FEATURE ENGINEERING COMPLETE!')
print('=' * 80)
print(f'\nSubmission: {filename}')
print(f'Results: deep_features_results_{timestamp}.json')
print(f'\nSamples: {len(submission):,}')
print(f'  Malware: {submission['Label'].sum():,} ({submission['Label'].sum() / len(submission) * 100:.2f}%)')
print(f'  Benign: {(1 - submission['Label']).sum():,} ({(1 - submission['Label']).sum() / len(submission) * 100:.2f}%)')
print(f'\nFeature Engineering Summary:')
print(f'  Total features created: {len(all_features)}')
print(f'  Selected features: {len(selected_features)}')
print(f'  Statistical features: 12')
print(f'  Interaction features: {interaction_count}')
print(f'  Transformation features: {len(transformation_features) * 7}')
print(f'  Clustering features: 9')
print(f'  Dimensionality features: 15')
print(f'  Target-based features: {len(target_features) * 2}')
print(f'\nPerformance:')
print(f'  CV AUC: {oof_auc:.6f}')
print(f'  Accuracy: {oof_acc * 100:.2f}%')
print(f'\nExpected leaderboard: ~{oof_auc * 100:.2f}%')
print('=' * 80)

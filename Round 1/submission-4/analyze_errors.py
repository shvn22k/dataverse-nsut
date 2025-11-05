import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
print('=' * 70)
print('ERROR ANALYSIS - Understanding Misclassifications')
print('=' * 70)
df = pd.read_csv('../data/main.csv')
target_col = 'Label'
id_col = 'sha256'
df_processed = df.copy()
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
constant_features = []
for col in numeric_cols:
    if col not in [target_col, id_col]:
        if df_processed[col].nunique() == 1:
            constant_features.append(col)
if len(constant_features) > 0:
    df_processed.drop(columns=constant_features, inplace=True)
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
        feat1 = top_10[i]
        feat2 = top_10[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
print(f'\nFeatures: {X.shape[1]}')
print('\nTraining model for error analysis...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
oof_ids = df['sha256'].values
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'  Fold {fold}/5')
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    model = lgb.LGBMClassifier(n_estimators=1000, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
oof_binary = (oof_predictions > 0.5).astype(int)
errors_idx = np.where(oof_binary != y)[0]
correct_idx = np.where(oof_binary == y)[0]
print('\n' + '=' * 70)
print('ERROR STATISTICS')
print('=' * 70)
print(f'Total samples: {len(y)}')
print(f'Correct: {len(correct_idx)} ({len(correct_idx) / len(y) * 100:.2f}%)')
print(f'Errors: {len(errors_idx)} ({len(errors_idx) / len(y) * 100:.2f}%)')
false_positives = np.where((oof_binary == 1) & (y == 0))[0]
false_negatives = np.where((oof_binary == 0) & (y == 1))[0]
print(f'\nError breakdown:')
print(f'  False Positives (predicted malware, actually benign): {len(false_positives)}')
print(f'  False Negatives (predicted benign, actually malware): {len(false_negatives)}')
print('\n' + '=' * 70)
print('FEATURE PATTERNS IN ERRORS')
print('=' * 70)
X_errors = X.iloc[errors_idx]
X_correct = X.iloc[correct_idx]
print('\nTop 20 features with biggest difference (errors vs correct):')
feature_diffs = []
for col in X.columns:
    error_mean = X_errors[col].mean()
    correct_mean = X_correct[col].mean()
    diff = abs(error_mean - correct_mean)
    feature_diffs.append({'Feature': col, 'Error_Mean': error_mean, 'Correct_Mean': correct_mean, 'Abs_Diff': diff})
diff_df = pd.DataFrame(feature_diffs).sort_values('Abs_Diff', ascending=False)
print(diff_df.head(20).to_string(index=False))
print('\n' + '=' * 70)
print('PREDICTION CONFIDENCE ANALYSIS')
print('=' * 70)
error_probs = oof_predictions[errors_idx]
correct_probs = oof_predictions[correct_idx]
print(f'\nErrors:')
print(f'  Mean confidence: {np.mean(np.abs(error_probs - 0.5)):.4f}')
print(f'  Median confidence: {np.median(np.abs(error_probs - 0.5)):.4f}')
print(f'  Close to decision boundary (0.45-0.55): {np.sum((error_probs > 0.45) & (error_probs < 0.55))}')
print(f'\nCorrect predictions:')
print(f'  Mean confidence: {np.mean(np.abs(correct_probs - 0.5)):.4f}')
print(f'  Median confidence: {np.median(np.abs(correct_probs - 0.5)):.4f}')
print('\n' + '=' * 70)
print('ERROR DISTRIBUTION BY PREDICTION CONFIDENCE')
print('=' * 70)
bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for low, high in bins:
    in_bin = (oof_predictions >= low) & (oof_predictions < high)
    bin_errors = np.sum(in_bin & (oof_binary != y))
    bin_total = np.sum(in_bin)
    if bin_total > 0:
        error_rate = bin_errors / bin_total * 100
        print(f'  [{low:.1f}-{high:.1f}): {bin_total:5d} samples, {bin_errors:4d} errors ({error_rate:.2f}%)')
error_analysis = pd.DataFrame({'sha256': oof_ids[errors_idx], 'true_label': y.iloc[errors_idx].values, 'predicted_label': oof_binary[errors_idx], 'prediction_prob': oof_predictions[errors_idx]})
error_analysis.to_csv('error_analysis.csv', index=False)
print(f'\nError analysis saved to error_analysis.csv')
print('\n' + '=' * 70)
print('KEY INSIGHTS')
print('=' * 70)
print('1. Check which features differ most between errors and correct predictions')
print('2. Are errors concentrated near decision boundary (0.5)?')
print('3. Are there more FP or FN? This tells us if we need to adjust threshold')
print('=' * 70)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import warnings
warnings.filterwarnings('ignore')
print('Loading data...')
df = pd.read_csv('C:\\Projects\\dataverse-nsut\\data\\main.csv')
test_df = pd.read_csv('C:\\Projects\\dataverse-nsut\\data\\test.csv')
print(f'Train shape: {df.shape}, Test shape: {test_df.shape}')
test_ids = test_df['sha256'].values
df = df.drop(columns=['sha256'])
print('Handling missing values...')
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
print('Handling missing values in test data...')
for col in test_df.columns:
    if col in df.columns and test_df[col].dtype in ['float64', 'int64']:
        median_val = df[col].median()
        test_df[col] = test_df[col].fillna(median_val)
print('Clipping outliers...')
for col in df.select_dtypes(include=['float64']).columns:
    lower, upper = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lower, upper)
    if col in test_df.columns:
        test_df[col] = test_df[col].clip(lower, upper)
y = df['Label'].values
feature_columns = [col for col in df.columns if col != 'Label']
X = df[feature_columns]
print(f'Features: {len(feature_columns)}')
print(f'Target distribution: {np.bincount(y)}')
print('Creating clustering-based features...')

def create_clustering_features(X_data, X_test_data=None, n_clusters=5):
    features = []
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_data)
    features.append(cluster_labels)
    distances = kmeans.transform(X_data)
    for i in range(n_clusters):
        features.append(distances[:, i])
    if X_test_data is not None:
        test_cluster_labels = kmeans.predict(X_test_data)
        test_distances = kmeans.transform(X_test_data)
        test_features = [test_cluster_labels]
        for i in range(n_clusters):
            test_features.append(test_distances[:, i])
        return (np.column_stack(features), np.column_stack(test_features), kmeans)
    else:
        return (np.column_stack(features), None, kmeans)
print('Creating K-means features (5 clusters)...')
X_kmeans, X_test_kmeans, kmeans_model = create_clustering_features(X, test_df[feature_columns], n_clusters=5)
print('Creating K-means features (10 clusters)...')
X_kmeans_10, X_test_kmeans_10, kmeans_model_10 = create_clustering_features(X, test_df[feature_columns], n_clusters=10)
print('Creating K-means features (15 clusters)...')
X_kmeans_15, X_test_kmeans_15, kmeans_model_15 = create_clustering_features(X, test_df[feature_columns], n_clusters=15)
print('Creating PCA features...')
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(test_df[feature_columns])
print('Creating PCA features (20 components)...')
pca_20 = PCA(n_components=20, random_state=42)
X_pca_20 = pca_20.fit_transform(X)
X_test_pca_20 = pca_20.transform(test_df[feature_columns])
print('Combining all engineered features...')
X_engineered = np.hstack([X.values, X_kmeans, X_kmeans_10, X_kmeans_15, X_pca, X_pca_20])
X_test_engineered = np.hstack([test_df[feature_columns].values, X_test_kmeans, X_test_kmeans_10, X_test_kmeans_15, X_test_pca, X_test_pca_20])
print(f'Engineered features shape: {X_engineered.shape}')
print(f'Test engineered features shape: {X_test_engineered.shape}')
print('Scaling features...')
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_engineered)
X_test_scaled = scaler.transform(X_test_engineered)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape}, Val: {X_val.shape}')
print('Creating KNN ensemble...')
knn_models = [('knn_3_euclidean', KNeighborsClassifier(n_neighbors=3, metric='euclidean')), ('knn_5_euclidean', KNeighborsClassifier(n_neighbors=5, metric='euclidean')), ('knn_7_euclidean', KNeighborsClassifier(n_neighbors=7, metric='euclidean')), ('knn_3_manhattan', KNeighborsClassifier(n_neighbors=3, metric='manhattan')), ('knn_5_manhattan', KNeighborsClassifier(n_neighbors=5, metric='manhattan')), ('knn_7_manhattan', KNeighborsClassifier(n_neighbors=7, metric='manhattan')), ('knn_3_minkowski', KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)), ('knn_5_minkowski', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)), ('knn_7_minkowski', KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=3)), ('knn_3_cosine', KNeighborsClassifier(n_neighbors=3, metric='cosine')), ('knn_5_cosine', KNeighborsClassifier(n_neighbors=5, metric='cosine')), ('knn_7_cosine', KNeighborsClassifier(n_neighbors=7, metric='cosine'))]
knn_ensemble = VotingClassifier(estimators=knn_models, voting='soft')
print('Training KNN ensemble with cross-validation...')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_ensemble.fit(X_train, y_train)
print('Evaluating on validation set...')
val_pred = knn_ensemble.predict(X_val)
val_pred_proba = knn_ensemble.predict_proba(X_val)[:, 1]
val_accuracy = accuracy_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_pred_proba)
val_f1 = f1_score(y_val, val_pred)
print(f'Validation Results:')
print(f'  Accuracy: {val_accuracy:.4f}')
print(f'  AUC: {val_auc:.4f}')
print(f'  F1: {val_f1:.4f}')
print('Making test predictions...')
test_pred = knn_ensemble.predict(X_test_scaled)
test_pred_proba = knn_ensemble.predict_proba(X_test_scaled)[:, 1]
print(f'Test prediction distribution: {np.bincount(test_pred)}')
print('Creating submission file...')
submission = pd.DataFrame({'sha256': test_ids, 'Label': test_pred})
submission.to_csv('submission-18/B7JYI.csv', index=False)
print(f'Submission saved with {len(submission)} predictions')
prob_submission = pd.DataFrame({'sha256': test_ids, 'Label': test_pred_proba})
prob_submission.to_csv('submission-18/probabilities.csv', index=False)
print('Probabilities saved for analysis')
print('Saving models and scalers...')
joblib.dump(knn_ensemble, 'submission-18/knn_ensemble.pkl')
joblib.dump(scaler, 'submission-18/scaler.pkl')
joblib.dump(kmeans_model, 'submission-18/kmeans_5.pkl')
joblib.dump(kmeans_model_10, 'submission-18/kmeans_10.pkl')
joblib.dump(kmeans_model_15, 'submission-18/kmeans_15.pkl')
joblib.dump(pca, 'submission-18/pca_10.pkl')
joblib.dump(pca_20, 'submission-18/pca_20.pkl')
print('\nClustering + KNN approach completed!')
print('Files saved:')
print('- submission-18/knn_ensemble.pkl (KNN ensemble model)')
print('- submission-18/scaler.pkl (feature scaler)')
print('- submission-18/kmeans_*.pkl (clustering models)')
print('- submission-18/pca_*.pkl (PCA models)')
print('- submission-18/B7JYI.csv (submission file)')
print('- submission-18/probabilities.csv (prediction probabilities)')
print('\nAnalyzing feature importance...')
print(f'Total engineered features: {X_engineered.shape[1]}')
print(f'Original features: {len(feature_columns)}')
print(f'K-means 5 clusters: {X_kmeans.shape[1]}')
print(f'K-means 10 clusters: {X_kmeans_10.shape[1]}')
print(f'K-means 15 clusters: {X_kmeans_15.shape[1]}')
print(f'PCA 10 components: {X_pca.shape[1]}')
print(f'PCA 20 components: {X_pca_20.shape[1]}')
print('\nClustering + KNN ensemble approach completed!')
print('This approach uses:')
print('- Multiple K-means clustering (5, 10, 15 clusters)')
print('- PCA dimensionality reduction (10, 20 components)')
print('- 12 different KNN models with various metrics')
print('- Soft voting ensemble for final prediction')

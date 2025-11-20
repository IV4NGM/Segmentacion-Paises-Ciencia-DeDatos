# Importamos las bibliotecas necesarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier

# ==================================================
# PREPROCESAMIENTO DE DATOS
# ==================================================

# Leemos los datos
df = pd.read_csv(".//Datos//CountriesData.csv")

# Ajustamos los índices

df_final = df.copy()
df_final.set_index("country", inplace=True)

# Visualizamos los datos

print(df_final.info())

print(df_final.head())

print(f"Datos faltantes por columna:")
missing_summary = df_final.isnull().sum().sort_values(ascending=False)
print(missing_summary)

print("Estadísticas descriptivas:")
print(df_final.describe())

# IMPUTACIÓN DE DATOS

# Escogemos las características con más de la mitad de los datos

threshold = len(df_final) / 2
df_reduced = df_final.loc[:, df_final.isnull().sum() < threshold]

print(df_reduced.info())

# Ahora analizamos los datos por filas (países)

missing_values_per_row = df_reduced.isnull().sum(axis=1)
total_columns = df_reduced.shape[1]
missing_percentage_per_row = (missing_values_per_row / total_columns) * 100

print(missing_percentage_per_row.sort_values(ascending=False).head(n=16))

# Eliminamos los países de los que se tiene menos del 50% de los datos

initial_rows = df_reduced.shape[0]
df_filtered = df_reduced[missing_percentage_per_row <= 50]

rows_removed = initial_rows - df_filtered.shape[0]
print(f"Número inicial de países: {initial_rows}")
print(f"Número de países filtrados: {df_filtered.shape[0]}")
print(f"Número de países eliminados: {rows_removed}")

print(df_filtered.info())

print(df_filtered.head())

# IMPUTACIÓN DE DATOS

# Escalamiento para aplicar imputación por KNN
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtered)
df_scaled = pd.DataFrame(df_scaled, columns=df_filtered.columns, index=df_filtered.index)

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_scaled)
df_imputed = pd.DataFrame(df_imputed, columns=df_scaled.columns, index=df_scaled.index)

print(df_imputed.info())

print(df_imputed.head())

# ==================================================
# ANÁLISIS DE DATOS
# ==================================================

# Matriz de correlación
plt.figure(figsize=(16, 14))
correlation_matrix = df_imputed.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f', mask=mask,
            cbar_kws={"shrink": .8})
plt.title('Matriz de correlación de indicadores de desarrollo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Clasificación (aprendizaje no supervisado)

df_scaled = df_imputed.copy() # Los datos ya habían sido estandarizados previamente

# Reducción de dimensionalidad con PCA
pca = PCA()
pca_result = pca.fit_transform(df_scaled)

loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df_scaled.columns)

# Gráficas de la varianza explicada
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(), marker='o', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.axhline(y=0.85, color='g', linestyle='--', label='85% Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Component Variance')
plt.grid(True)

plt.subplot(1, 3, 3)
# Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, marker='o', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)

plt.tight_layout()
plt.show()

# Gráficas de los loadings
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
sns.barplot(x=loadings_df['PC1'].sort_values(ascending=False).index,
            y=loadings_df['PC1'].sort_values(ascending=False).values, palette='viridis', hue=loadings_df['PC1'].sort_values(ascending=False).index, legend=False)
plt.title('Loadings for PC1')
plt.xticks(rotation=90)
plt.ylabel('Loading Value')

plt.subplot(1, 3, 2)
sns.barplot(x=loadings_df['PC2'].sort_values(ascending=False).index,
            y=loadings_df['PC2'].sort_values(ascending=False).values, palette='viridis', hue=loadings_df['PC2'].sort_values(ascending=False).index, legend=False)
plt.title('Loadings for PC2')
plt.xticks(rotation=90)
plt.ylabel('Loading Value')

plt.subplot(1, 3, 3)
sns.barplot(x=loadings_df['PC3'].sort_values(ascending=False).index,
            y=loadings_df['PC3'].sort_values(ascending=False).values, palette='viridis', hue=loadings_df['PC3'].sort_values(ascending=False).index, legend=False)
plt.title('Loadings for PC3')
plt.xticks(rotation=90)
plt.ylabel('Loading Value')

plt.tight_layout()
plt.show()

print(f"Varianza explicada por los primeros 2 PCs: {pca.explained_variance_ratio_[:2].sum():.3f}")
print(f"Varianza explicada por los primeros 3 PCs: {pca.explained_variance_ratio_[:3].sum():.3f}")

# Determinar el número óptimo de clusters
silhouette_scores = []
inertia_scores = []
k_range = range(2, 8)

plt.figure(figsize=(15, 5))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    inertia_scores.append(kmeans.inertia_)

# Gráficas de los scores de silueta y el método del codo
plt.subplot(1, 2, 1)
plt.plot(k_range, silhouette_scores, marker='o', linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters - Silhouette Score')
plt.grid(True)

# Codo
plt.subplot(1, 2, 2)
plt.plot(k_range, inertia_scores, marker='o', linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal Clusters')
plt.grid(True)

plt.tight_layout()
plt.show()

# Escogemos el número óptimo de clusters basado en el score de silueta
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Número óptimo de clusters: {optimal_k} (Score de silueta: {max(silhouette_scores):.3f})")

# Aplicamos KMeans con el número óptimo de clusters

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_scaled)

df_imputed['cluster'] = cluster_labels
df_scaled['cluster'] = cluster_labels

print("Distribución de clusters:")
cluster_dist = df_imputed['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    print(f"Cluster {cluster_id}: {count} países")

# Visualizamos los clusters en 2D usando los dos primeros componentes principales
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(df_scaled.drop('cluster', axis=1))

plt.figure(figsize=(14, 10))

scatter = plt.scatter(pca_result_2d[:, 0], pca_result_2d[:, 1], 
                     c=cluster_labels, cmap='tab10', alpha=0.8, s=80, 
                     edgecolors='w', linewidth=0.5)

plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Country Clusters in 2D PCA Space', fontsize=14, fontweight='bold')

for i, country in enumerate(df_imputed.index):
    if i % 15 == 0:
        plt.annotate(country, (pca_result_2d[i, 0], pca_result_2d[i, 1]), 
                    fontsize=8, alpha=0.8, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualizamos los clusters en 2D usando el primer y tercer componente principal
plt.figure(figsize=(14, 10))

scatter = plt.scatter(pca_result[:, 0], pca_result[:, 2],
                     c=cluster_labels, cmap='tab10', alpha=0.8, s=80,
                     edgecolors='w', linewidth=0.5)

plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
plt.title('Country Clusters in 2D PCA Space (PC1 vs PC3)', fontsize=14, fontweight='bold')

for i, country in enumerate(df_imputed.index):
    if i % 15 == 0: # Annotate every 15th country for readability
        plt.annotate(country, (pca_result[i, 0], pca_result[i, 2]),
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Análisis de características de los clusters
cluster_summary_scaled = df_imputed.groupby('cluster').mean()

# Transformar el resumen escalado de los clusters para obtener valores en la escala original
cluster_summary_unscaled = pd.DataFrame(
    scaler.inverse_transform(cluster_summary_scaled),
    columns=cluster_summary_scaled.columns,
    index=cluster_summary_scaled.index
)
print(cluster_summary_unscaled)

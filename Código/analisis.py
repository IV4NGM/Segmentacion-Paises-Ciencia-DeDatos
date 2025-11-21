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
df_scaled = pd.DataFrame(
    df_scaled, columns=df_filtered.columns, index=df_filtered.index
)

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

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    fmt=".2f",
    mask=mask,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Matriz de correlación de indicadores de desarrollo", fontsize=16, fontweight="bold"
)
plt.tight_layout()
plt.show()

# Clasificación (aprendizaje no supervisado)

df_scaled = df_imputed.copy()  # Los datos ya habían sido estandarizados previamente

# Reducción de dimensionalidad con PCA
pca = PCA()
pca_result = pca.fit_transform(df_scaled)

loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=df_scaled.columns,
)

# Gráficas de la varianza explicada
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_.cumsum(),
    marker="o",
    linewidth=2,
)
plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
plt.axhline(y=0.85, color="g", linestyle="--", label="85% Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Individual Component Variance")
plt.grid(True)

plt.subplot(1, 3, 3)
# Scree plot
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker="o",
    linewidth=2,
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(True)

plt.tight_layout()
plt.show()

# Gráficas de los loadings
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
sns.barplot(
    x=loadings_df["PC1"].sort_values(ascending=False).index,
    y=loadings_df["PC1"].sort_values(ascending=False).values,
    palette="viridis",
    hue=loadings_df["PC1"].sort_values(ascending=False).index,
    legend=False,
)
plt.title("Loadings for PC1")
plt.xticks(rotation=90)
plt.ylabel("Loading Value")

plt.subplot(1, 3, 2)
sns.barplot(
    x=loadings_df["PC2"].sort_values(ascending=False).index,
    y=loadings_df["PC2"].sort_values(ascending=False).values,
    palette="viridis",
    hue=loadings_df["PC2"].sort_values(ascending=False).index,
    legend=False,
)
plt.title("Loadings for PC2")
plt.xticks(rotation=90)
plt.ylabel("Loading Value")

plt.subplot(1, 3, 3)
sns.barplot(
    x=loadings_df["PC3"].sort_values(ascending=False).index,
    y=loadings_df["PC3"].sort_values(ascending=False).values,
    palette="viridis",
    hue=loadings_df["PC3"].sort_values(ascending=False).index,
    legend=False,
)
plt.title("Loadings for PC3")
plt.xticks(rotation=90)
plt.ylabel("Loading Value")

plt.tight_layout()
plt.show()

print(
    f"Varianza explicada por los primeros 2 PCs: {pca.explained_variance_ratio_[:2].sum():.3f}"
)
print(
    f"Varianza explicada por los primeros 3 PCs: {pca.explained_variance_ratio_[:3].sum():.3f}"
)

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
plt.plot(k_range, silhouette_scores, marker="o", linewidth=2)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Optimal Number of Clusters - Silhouette Score")
plt.grid(True)

# Codo
plt.subplot(1, 2, 2)
plt.plot(k_range, inertia_scores, marker="o", linewidth=2)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Within-cluster sum of squares)")
plt.title("Elbow Method for Optimal Clusters")
plt.grid(True)

plt.tight_layout()
plt.show()

# Escogemos el número óptimo de clusters basado en el score de silueta
optimal_k = k_range[np.argmax(silhouette_scores)]
print(
    f"Número óptimo de clusters: {optimal_k} (Score de silueta: {max(silhouette_scores):.3f})"
)

# Aplicamos KMeans con el número óptimo de clusters

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_scaled)

df_imputed["cluster"] = cluster_labels
df_scaled["cluster"] = cluster_labels

print("Distribución de clusters:")
cluster_dist = df_imputed["cluster"].value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    print(f"Cluster {cluster_id}: {count} países")

# Visualizamos los clusters en 2D usando los dos primeros componentes principales
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(df_scaled.drop("cluster", axis=1))

plt.figure(figsize=(14, 10))

scatter = plt.scatter(
    pca_result_2d[:, 0],
    pca_result_2d[:, 1],
    c=cluster_labels,
    cmap="tab10",
    alpha=0.8,
    s=80,
    edgecolors="w",
    linewidth=0.5,
)

plt.colorbar(scatter, label="Cluster")
plt.xlabel(
    f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)"
)
plt.ylabel(
    f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)"
)
plt.title("Country Clusters in 2D PCA Space", fontsize=14, fontweight="bold")

for i, country in enumerate(df_imputed.index):
    if i % 15 == 0:
        plt.annotate(
            country,
            (pca_result_2d[i, 0], pca_result_2d[i, 1]),
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualizamos los clusters en 2D usando el primer y tercer componente principal
plt.figure(figsize=(14, 10))

scatter = plt.scatter(
    pca_result[:, 0],
    pca_result[:, 2],
    c=cluster_labels,
    cmap="tab10",
    alpha=0.8,
    s=80,
    edgecolors="w",
    linewidth=0.5,
)

plt.colorbar(scatter, label="Cluster")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"Principal Component 3 ({pca.explained_variance_ratio_[2]:.2%} variance)")
plt.title(
    "Country Clusters in 2D PCA Space (PC1 vs PC3)", fontsize=14, fontweight="bold"
)

for i, country in enumerate(df_imputed.index):
    if i % 15 == 0:  # Annotate every 15th country for readability
        plt.annotate(
            country,
            (pca_result[i, 0], pca_result[i, 2]),
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Análisis de características de los clusters
cluster_summary_scaled = df_imputed.groupby("cluster").mean()

# Transformar el resumen escalado de los clusters para obtener valores en la escala original
cluster_summary_unscaled = pd.DataFrame(
    scaler.inverse_transform(cluster_summary_scaled),
    columns=cluster_summary_scaled.columns,
    index=cluster_summary_scaled.index,
)
print(cluster_summary_unscaled)

# Clustering jerárquico

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Evitar errores si la columna usada para clusters tiene distinto nombre.
# Eliminamos columnas de etiquetas si existen antes de calcular linkage.
cols_to_drop = [
    c for c in ["cluster", "cluster_name", "hc_cluster"] if c in df_scaled.columns
]
X_hc = df_scaled.drop(columns=cols_to_drop) if cols_to_drop else df_scaled

linked_data = linkage(X_hc, method="ward")

hc_classifier = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
hc_labels = hc_classifier.fit_predict(X_hc)

df_scaled["hc_cluster"] = hc_labels

# Plot the dendrogram
plt.figure(figsize=(15, 8))
dendrogram(
    linked_data, orientation="top", distance_sort="descending", show_leaf_counts=True
)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Sample Index (or Cluster Size)")
plt.ylabel("Distance")
plt.show()

from sklearn.metrics import silhouette_score

hc_silhouette_scores = []
k_range_hc = range(2, 8)

# Preparar matriz de características eliminando columnas de etiqueta si existen
label_cols = [
    c for c in ["cluster", "cluster_name", "hc_cluster"] if c in df_scaled.columns
]
X_hc_eval = df_scaled.drop(columns=label_cols) if label_cols else df_scaled

for k in k_range_hc:
    hc_model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_cluster_labels = hc_model.fit_predict(X_hc_eval)
    silhouette_avg = silhouette_score(X_hc_eval, hc_cluster_labels)
    hc_silhouette_scores.append(silhouette_avg)

print("Silhouette scores for Hierarchical Clustering:", hc_silhouette_scores)

plt.figure(figsize=(10, 6))
plt.plot(k_range_hc, hc_silhouette_scores, marker="o", linewidth=2)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Hierarchical Clustering: Silhouette Score vs. Number of Clusters")
plt.grid(True)
plt.show()

optimal_k_hc = k_range_hc[np.argmax(hc_silhouette_scores)]

hc_optimal_model = AgglomerativeClustering(n_clusters=optimal_k_hc, linkage="ward")
# Eliminar dinámicamente columnas de etiquetas si existen antes de predecir
label_cols_opt = [
    c
    for c in ["cluster", "cluster_name", "hc_cluster", "hc_cluster_optimal"]
    if c in df_scaled.columns
]
X_hc_opt = df_scaled.drop(columns=label_cols_opt) if label_cols_opt else df_scaled
hc_optimal_labels = hc_optimal_model.fit_predict(X_hc_opt)

df_scaled["hc_cluster_optimal"] = hc_optimal_labels

print(f"Agglomerative Clustering applied with {optimal_k_hc} optimal clusters.")
print("Distribution of Optimal Hierarchical Clusters:")
print(df_scaled["hc_cluster_optimal"].value_counts())

plt.figure(figsize=(14, 10))

scatter = plt.scatter(
    pca_result_2d[:, 0],
    pca_result_2d[:, 1],
    c=hc_optimal_labels,
    cmap="tab10",
    alpha=0.8,
    s=80,
    edgecolors="w",
    linewidth=0.5,
)

plt.colorbar(scatter, label="Optimal Hierarchical Cluster")
plt.xlabel(
    f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)"
)
plt.ylabel(
    f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)"
)
plt.title(
    "Optimal Hierarchical Country Clusters in 2D PCA Space",
    fontsize=14,
    fontweight="bold",
)

for i, country in enumerate(df_imputed.index):
    if i % 15 == 0:  # Annotate every 15th country for readability
        plt.annotate(
            country,
            (pca_result_2d[i, 0], pca_result_2d[i, 1]),
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Clustering espectral
from sklearn.cluster import SpectralClustering

spectral_cluster_labels = {}
k_range_spectral = range(2, 8)

for k in k_range_spectral:
    spectral_model = SpectralClustering(
        n_clusters=k, random_state=42, affinity="nearest_neighbors"
    )
    labels = spectral_model.fit_predict(
        df_scaled.drop(["cluster", "hc_cluster", "hc_cluster_optimal"], axis=1)
    )
    spectral_cluster_labels[k] = labels

print("Spectral Clustering models fitted and labels stored for k from 2 to 7.")

spectral_silhouette_scores = []

for k in k_range_spectral:
    labels = spectral_cluster_labels[k]
    silhouette_avg = silhouette_score(
        df_scaled.drop(["cluster", "hc_cluster", "hc_cluster_optimal"], axis=1), labels
    )
    spectral_silhouette_scores.append(silhouette_avg)

print("Silhouette scores for Spectral Clustering:", spectral_silhouette_scores)

plt.figure(figsize=(10, 6))
plt.plot(k_range_spectral, spectral_silhouette_scores, marker="o", linewidth=2)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Spectral Clustering: Silhouette Score vs. Number of Clusters")
plt.grid(True)
plt.show()

optimal_k_spectral = k_range_spectral[np.argmax(spectral_silhouette_scores)]
print(
    f"Optimal number of clusters for Spectral Clustering: {optimal_k_spectral} (Silhouette Score: {max(spectral_silhouette_scores):.3f})"
)

spectral_optimal_model = SpectralClustering(
    n_clusters=optimal_k_spectral, random_state=42, affinity="nearest_neighbors"
)
spectral_optimal_labels = spectral_optimal_model.fit_predict(
    df_scaled.drop(["cluster", "hc_cluster", "hc_cluster_optimal"], axis=1)
)

df_scaled["spectral_cluster_optimal"] = spectral_optimal_labels

print(f"Spectral Clustering applied with {optimal_k_spectral} optimal clusters.")
print("Distribution of Optimal Spectral Clusters:")
print(df_scaled["spectral_cluster_optimal"].value_counts())

plt.figure(figsize=(14, 10))

scatter = plt.scatter(
    pca_result_2d[:, 0],
    pca_result_2d[:, 1],
    c=spectral_optimal_labels,
    cmap="tab10",
    alpha=0.8,
    s=80,
    edgecolors="w",
    linewidth=0.5,
)

plt.colorbar(scatter, label="Optimal Spectral Cluster")
plt.xlabel(
    f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)"
)
plt.ylabel(
    f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)"
)
plt.title(
    "Optimal Spectral Country Clusters in 2D PCA Space", fontsize=14, fontweight="bold"
)

for i, country in enumerate(df_imputed.index):
    if i % 15 == 0:  # Annotate every 15th country for readability
        plt.annotate(
            country,
            (pca_result_2d[i, 0], pca_result_2d[i, 1]),
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Comparación de modelos de clasificación no supervisada

print("--- Clustering Method Comparison ---")

# K-Means
print(f"\nK-Means Clustering:")
print(
    f"Optimal number of clusters: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})"
)
print("Cluster Distribution:")
print(df_imputed["cluster"].value_counts().sort_index())

# Hierarchical Clustering
print(f"\nHierarchical Clustering:")
print(
    f"Optimal number of clusters: {optimal_k_hc} (Silhouette Score: {max(hc_silhouette_scores):.3f})"
)
print("Cluster Distribution:")
print(df_scaled["hc_cluster_optimal"].value_counts().sort_index())

# Spectral Clustering
print(f"\nSpectral Clustering:")
print(
    f"Optimal number of clusters: {optimal_k_spectral} (Silhouette Score: {max(spectral_silhouette_scores):.3f})"
)
print("Cluster Distribution:")
print(df_scaled["spectral_cluster_optimal"].value_counts().sort_index())

# APRENDIZAJE SUPERVISADO PARA PREDECIR CLUSTERS

df_scaled = df_scaled.drop(
    ["hc_cluster", "hc_cluster_optimal", "spectral_cluster_optimal"], axis=1
)

X = df_scaled.drop("cluster", axis=1)
y = df_scaled["cluster"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tamaño de conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño de conjunto de prueba: {X_test.shape}")
print(f"Distribución de clases en el conjunto de entrenamiento:")
print(y_train.value_counts().sort_index())

# Random Forest

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

print("Resultados Random Forest:")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"Cluster {i}" for i in range(optimal_k)],
    yticklabels=[f"Cluster {i}" for i in range(optimal_k)],
)
plt.title(
    "Matriz de Confusión - Predicción de Clusters", fontsize=14, fontweight="bold"
)
plt.ylabel("Cluster Verdadero")
plt.xlabel("Cluster Predicho")
plt.show()

# Importancia de características
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": rf_classifier.feature_importances_}
).sort_values("importance", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=feature_importance.head(15), x="importance", y="feature", palette="viridis"
)
plt.title(
    "Características más importantes Random Forest", fontsize=14, fontweight="bold"
)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

print("Top 10 most important features for cluster prediction:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Cross validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
print(f"Cross-validation scores: {[f'{score:.3f}' for score in cv_scores]}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Red neuronal

nn_classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
)

nn_classifier.fit(X_train, y_train)

# Evaluación de la red neuronal
nn_train_score = nn_classifier.score(X_train, y_train)
nn_test_score = nn_classifier.score(X_test, y_test)

print("Desempeño de la Red Neuronal:")
print(f"Training Accuracy: {nn_train_score:.3f}")
print(f"Test Accuracy: {nn_test_score:.3f}")

# Comparación con Random Forest
rf_test_score = rf_classifier.score(X_test, y_test)
print(f"\nRandom Forest Test Accuracy: {rf_test_score:.3f}")

y_pred_nn = nn_classifier.predict(X_test)

print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn, zero_division=0))

# Plot model comparison
plt.figure(figsize=(10, 6))
models = ["Random Forest", "Neural Network"]
accuracies = [rf_test_score, nn_test_score]
colors = ["skyblue", "lightcoral"]

bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1.0)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

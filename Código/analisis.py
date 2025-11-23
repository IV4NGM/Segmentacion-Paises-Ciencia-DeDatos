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

print("Varianza explicada acumulada por componentes principales:")
print(pca.explained_variance_ratio_.cumsum())

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

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

X = df_scaled.select_dtypes(include=[np.number]).copy()

for col in ["cluster", "hc_cluster_optimal", "spectral_cluster_optimal"]:
    if col in X.columns:
        X = X.drop(columns=[col])

X = X.fillna(X.median())

X_values = X.values

labels_dict = {}
if "cluster" in df_scaled.columns:
    labels_dict["kmeans"] = df_scaled["cluster"].astype(int).values

if "hc_cluster_optimal" in df_scaled.columns:
    labels_dict["hierarchical"] = df_scaled["hc_cluster_optimal"].astype(int).values

if "spectral_cluster_optimal" in df_scaled.columns:
    labels_dict["spectral"] = df_scaled["spectral_cluster_optimal"].astype(int).values

results = []
for name, labels in labels_dict.items():
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        print(f"Skipping metrics for {name}: only {n_clusters} cluster(s) present.")
        continue
    try:
        sil = silhouette_score(X_values, labels)
    except Exception as e:
        sil = np.nan
        print(f"Silhouette error for {name}: {e}")
    try:
        db = davies_bouldin_score(X_values, labels)
    except Exception as e:
        db = np.nan
        print(f"Davies–Bouldin error for {name}: {e}")
    try:
        ch = calinski_harabasz_score(X_values, labels)
    except Exception as e:
        ch = np.nan
        print(f"Calinski–Harabasz error for {name}: {e}")

    results.append(
        {
            "method": name,
            "n_clusters": n_clusters,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
        }
    )

metrics_df = pd.DataFrame(results).set_index("method")
print("\nInternal validity metrics:")
print(metrics_df)

if not metrics_df.empty:
    ranking = metrics_df.copy()
    ranking["silhouette_rank"] = ranking["silhouette"].rank(
        ascending=False, method="min"
    )
    ranking["db_rank"] = ranking["davies_bouldin"].rank(ascending=True, method="min")
    ranking["ch_rank"] = ranking["calinski_harabasz"].rank(
        ascending=False, method="min"
    )
    ranking["avg_rank"] = ranking[["silhouette_rank", "db_rank", "ch_rank"]].mean(
        axis=1
    )
    print("\nRanking summary (lower avg_rank = better overall):")
    print(ranking.sort_values("avg_rank"))

# Análisis de características de los clusters

df_imputed_total = df_imputed.copy()
df_scaled_total = df_scaled.copy()

# # En la columna de cluster usamos los labels del mejor método (kmeans en este caso)
# df_imputed_total["cluster"] = cluster_labels
# df_scaled_total["cluster"] = cluster_labels

cluster_summary_scaled = df_imputed_total.groupby("cluster").mean()

# Análisis de las características económicas
cluster_summary_scaled = df_imputed_total.groupby("cluster").mean()

cols = list(cluster_summary_scaled.columns)
try:
    if hasattr(scaler, "feature_names_in_"):
        all_features = list(scaler.feature_names_in_)
    else:
        all_features = list(df_filtered.columns)

    indices = [all_features.index(c) for c in cols]
    means = scaler.mean_[indices]
    scales = scaler.scale_[indices]

    unscaled_values = cluster_summary_scaled.values * scales + means
    cluster_summary_unscaled = pd.DataFrame(
        unscaled_values, columns=cols, index=cluster_summary_scaled.index
    )
except Exception:
    cluster_summary_unscaled = pd.DataFrame(
        scaler.inverse_transform(cluster_summary_scaled),
        columns=cluster_summary_scaled.columns,
        index=cluster_summary_scaled.index,
    )

print(cluster_summary_unscaled)

clustersnumber = len(cluster_summary_unscaled)


# # ==================================================
# # VISUALIZACIÓN DE RESULTADOS
# # ==================================================
# import pycountry
# import difflib
# import pandas as pd
# import plotly.express as px
# import os

# # Optional GeoPandas installation
# try:
#     import geopandas as gpd
#     import matplotlib.pyplot as plt

#     GEOPANDAS_AVAILABLE = True
# except Exception:
#     GEOPANDAS_AVAILABLE = False

# OUTPUT_DIR = "maps_output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------
# # 1. Prepare DataFrame
# # -----------------------------
# df_map = df_imputed_total.copy()
# df_map = df_map.reset_index()  # country becomes a column
# df_map.rename(columns={"index": "country"}, inplace=True)

# # -----------------------------
# # 2. Override dictionary
# # -----------------------------
# OVERRIDES = {
#     "United States": "USA",
#     "Russian Federation": "RUS",
#     "Czech Republic": "CZE",
#     "South Korea": "KOR",
#     "North Korea": "PRK",
#     "Democratic Republic of the Congo": "COD",
#     "Republic of Congo": "COG",
#     "Ivory Coast": "CIV",
#     "Syria": "SYR",
#     "Iran": "IRN",
#     "Venezuela": "VEN",
#     "Tanzania": "TZA",
#     "Laos": "LAO",
# }


# # -----------------------------
# # 3. Converter name → ISO3
# # -----------------------------
# def name_to_iso3(name, overrides=OVERRIDES):
#     if pd.isna(name):
#         return None
#     s = str(name).strip()

#     # Override first
#     if s in overrides:
#         return overrides[s]

#     # Best pycountry attempt
#     try:
#         country = pycountry.countries.lookup(s)
#         return country.alpha_3
#     except Exception:
#         pass

#     # Fuzzy match
#     try:
#         all_names = [c.name for c in pycountry.countries]
#         close = difflib.get_close_matches(s, all_names, n=1, cutoff=0.82)
#         if close:
#             c = pycountry.countries.get(name=close[0])
#             return c.alpha_3
#     except Exception:
#         pass

#     return None


# # -----------------------------
# # 4. Build ISO3 column
# # -----------------------------
# unique_names = df_map["country"].unique()
# name_iso_map = {n: name_to_iso3(n) for n in unique_names}
# df_map["iso3"] = df_map["country"].map(name_iso_map)

# # Save mapping for inspection
# pd.DataFrame.from_dict(name_iso_map, orient="index", columns=["iso3"]).rename_axis(
#     "country"
# ).to_csv(f"{OUTPUT_DIR}/name_to_iso3_map.csv")

# # Print unmapped
# unmapped = [n for n, iso in name_iso_map.items() if iso is None]
# print(
#     f"Mapped {len(unique_names)-len(unmapped)} out of {len(unique_names)} country names."
# )
# if unmapped:
#     print("Unmapped names:", unmapped[:40])

# # -----------------------------
# # 5. Interactive Plotly map
# # -----------------------------
# plot_df = df_map.dropna(subset=["iso3"])
# if not plot_df.empty:
#     plot_df["cluster_str"] = plot_df["cluster"].astype(str)

#     fig = px.choropleth(
#         plot_df,
#         locations="iso3",
#         color="cluster_str",
#         hover_name="country",
#         title="Country Clusters - General Indicators",
#         locationmode="ISO-3",
#         labels={"cluster_str": "Cluster"},
#     )

#     out_html = f"{OUTPUT_DIR}/general_country_clusters_{clustersnumber}_clusters.html"
#     fig.write_html(out_html)
#     print("Interactive map saved:", out_html)

# # -----------------------------
# # 7. Print clusters neatly
# # -----------------------------
# print("\n--- Countries by Cluster ---")
# for c in sorted(df_map["cluster"].unique()):
#     members = df_map[df_map["cluster"] == c]["country"].tolist()
#     print(f"\n=== Cluster {c} ({len(members)} countries) ===")
#     for country in members:
#         print(" -", country)

# # Save each cluster to text file
# for c in sorted(df_map["cluster"].unique()):
#     members = df_map[df_map["cluster"] == c]["country"].tolist()
#     with open(
#         f"{OUTPUT_DIR}/general_cluster_{c}_countries_{clustersnumber}_clusters.txt",
#         "w",
#     ) as f:
#         f.write("\n".join(members))


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

# REGRESIÓN LINEAL PARA PREDECIR INDICADORES

# GDP per cápita

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats


X_reg = df_scaled.drop(["gdp_per_capita", "cluster"], axis=1, errors="ignore")
df_unscaled = pd.DataFrame(
    scaler.inverse_transform(df_scaled.drop("cluster", axis=1)),
    columns=df_scaled.drop("cluster", axis=1).columns,
    index=df_scaled.index,
)
y_reg = df_unscaled["gdp_per_capita"]

X_reg = X_reg.loc[:, X_reg.nunique() > 1]

print(f"Predictors: {X_reg.shape[1]} indicators")
print(f"Target: GDP per capita")
print(f"Sample size: {X_reg.shape[0]} countries")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

y_pred_reg = lr_model.predict(X_test_reg)

# Evaluar el modelo
r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = np.mean(np.abs(y_test_reg - y_pred_reg))

print(f"\nRegression Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"MAE: ${mae:,.0f}")
print(f"Mean GDP: ${y_reg.mean():,.0f}")

# Análisis de coeficientes
feature_coef = pd.DataFrame(
    {
        "feature": X_reg.columns,
        "coefficient": lr_model.coef_,
        "abs_coefficient": np.abs(lr_model.coef_),
    }
).sort_values("abs_coefficient", ascending=False)

print("\nTop 10 Most Influential Predictors of GDP per capita:")
print("(Positive coefficients increase GDP, negative decrease)")
for i, row in feature_coef.head(10).iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    print(f"  {direction} {row['feature']}: {row['coefficient']:,.0f}")


# Visualize regression results
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, color="blue")
plt.plot(
    [y_test_reg.min(), y_test_reg.max()],
    [y_test_reg.min(), y_test_reg.max()],
    "r--",
    lw=2,
)
plt.xlabel("Actual GDP per capita ($)")
plt.ylabel("Predicted GDP per capita ($)")
plt.title(f"Actual vs Predicted GDP\nR² = {r2:.3f}")
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals, alpha=0.7, color="green")
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted GDP per capita ($)")
plt.ylabel("Residuals ($)")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

# Plot 3: Top feature coefficients
plt.subplot(1, 3, 3)
top_features = feature_coef.head(8)
colors = ["green" if x > 0 else "red" for x in top_features["coefficient"]]
plt.barh(top_features["feature"], top_features["coefficient"], color=colors, alpha=0.7)
plt.xlabel("Coefficient Value")
plt.title("Top Feature Coefficients\n(Impact on GDP per capita)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Regresión lineal para esperanza de vida


X_reg = df_scaled.drop(["life_expectancy", "cluster"], axis=1, errors="ignore")
df_unscaled = pd.DataFrame(
    scaler.inverse_transform(df_scaled.drop("cluster", axis=1)),
    columns=df_scaled.drop("cluster", axis=1).columns,
    index=df_scaled.index,
)
y_reg = df_unscaled["life_expectancy"]

X_reg = X_reg.loc[:, X_reg.nunique() > 1]

print(f"Predictors: {X_reg.shape[1]} indicators")
print(f"Target: Life Expectancy")
print(f"Sample size: {X_reg.shape[0]} countries")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

y_pred_reg = lr_model.predict(X_test_reg)

# Evaluar el modelo
r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = np.mean(np.abs(y_test_reg - y_pred_reg))

print(f"\nRegression Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:,.1f} years")
print(f"MAE: {mae:,.1f} years")
print(f"Mean Life Expectancy: {y_reg.mean():,.1f} years")

# Análisis de coeficientes
feature_coef = pd.DataFrame(
    {
        "feature": X_reg.columns,
        "coefficient": lr_model.coef_,
        "abs_coefficient": np.abs(lr_model.coef_),
    }
).sort_values("abs_coefficient", ascending=False)

print("\nTop 10 Most Influential Predictors of Life Expectancy:")
print("(Positive coefficients increase life expectancy, negative decrease)")
for i, row in feature_coef.head(10).iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    print(f"  {direction} {row['feature']}: {row['coefficient']:,.0f}")


# Visualize regression results
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, color="blue")
plt.plot(
    [y_test_reg.min(), y_test_reg.max()],
    [y_test_reg.min(), y_test_reg.max()],
    "r--",
    lw=2,
)
plt.xlabel("Actual Life Expectancy (years)")
plt.ylabel("Predicted Life Expectancy (years)")
plt.title(f"Actual vs Predicted Life Expectancy\nR² = {r2:.3f}")
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals, alpha=0.7, color="green")
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Life Expectancy (years)")
plt.ylabel("Residuals (years)")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

# Plot 3: Top feature coefficients
plt.subplot(1, 3, 3)
top_features = feature_coef.head(8)
colors = ["green" if x > 0 else "red" for x in top_features["coefficient"]]
plt.barh(top_features["feature"], top_features["coefficient"], color=colors, alpha=0.7)
plt.xlabel("Coefficient Value")
plt.title("Top Feature Coefficients\n(Impact on Life Expectancy)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Regresión lineal dentro de clusters

# GDP per cápita por cluster
for cluster_id in range(optimal_k):
    cluster_data = df_imputed[df_imputed["cluster"] == cluster_id]

    if len(cluster_data) > 10:
        X_cluster = cluster_data.drop(
            ["gdp_per_capita", "cluster"], axis=1, errors="ignore"
        )
        y_cluster = cluster_data["gdp_per_capita"]

        X_cluster = X_cluster.loc[:, X_cluster.nunique() > 1]

        if len(X_cluster.columns) > 0 and len(cluster_data) > len(X_cluster.columns):
            lr_cluster = LinearRegression()
            lr_cluster.fit(X_cluster, y_cluster)

            r2_cluster = lr_cluster.score(X_cluster, y_cluster)

            print(f"\nCluster {cluster_id}:")
            print(f"  Countries: {len(cluster_data)}, R²: {r2_cluster:.3f}")

            # Top 5 predictors by absolute coefficient
            if len(lr_cluster.coef_) > 0:
                coefs = pd.Series(lr_cluster.coef_, index=X_cluster.columns)
                top5 = coefs.abs().sort_values(ascending=False).head(5)
                print("  Top 5 predictors:")
                for feat in top5.index:
                    coef = coefs[feat]
                    direction = "increases" if coef > 0 else "decreases"
                    print(f"    - {feat}: coef={coef:.4f} ({direction} GDP)")


# Esperanza de vida por cluster
for cluster_id in range(optimal_k):
    cluster_data = df_imputed[df_imputed["cluster"] == cluster_id]

    if len(cluster_data) > 10:
        X_cluster = cluster_data.drop(
            ["life_expectancy", "cluster"], axis=1, errors="ignore"
        )
        y_cluster = cluster_data["life_expectancy"]

        X_cluster = X_cluster.loc[:, X_cluster.nunique() > 1]

        if len(X_cluster.columns) > 0 and len(cluster_data) > len(X_cluster.columns):
            lr_cluster = LinearRegression()
            lr_cluster.fit(X_cluster, y_cluster)

            r2_cluster = lr_cluster.score(X_cluster, y_cluster)

            print(f"\nCluster {cluster_id}:")
            print(f"  Countries: {len(cluster_data)}, R²: {r2_cluster:.3f}")

            # Top 5 predictors by absolute coefficient
            if len(lr_cluster.coef_) > 0:
                coefs = pd.Series(lr_cluster.coef_, index=X_cluster.columns)
                top5 = coefs.abs().sort_values(ascending=False).head(5)
                print("  Top 5 predictors:")
                for feat in top5.index:
                    coef = coefs[feat]
                    direction = "increases" if coef > 0 else "decreases"
                    print(
                        f"    - {feat}: coef={coef:.4f} ({direction} Life Expectancy)"
                    )

# ANÁLISIS PAREJAS DE VARIABLES


X_bivariate = df_unscaled[["gdp_per_capita"]].values
y_bivariate = df_unscaled["life_expectancy"].values

lr_simple = LinearRegression()
lr_simple.fit(X_bivariate, y_bivariate)

y_pred_simple = lr_simple.predict(X_bivariate)
r2_simple = r2_score(y_bivariate, y_pred_simple)

print(f"GDP per capita vs Life Expectancy:")
print(f"  R²: {r2_simple:.3f}")
print(
    f"  Coefficient: {lr_simple.coef_[0]:.6f} (each $1k GDP → {lr_simple.coef_[0]*1000:.2f} years life expectancy)"
)
print(f"  Intercept: {lr_simple.intercept_:.2f} years")


# Visualize bivariate relationships
plt.figure(figsize=(15, 5))

# Plot 1: GDP vs Life Expectancy
plt.subplot(1, 2, 1)
plt.scatter(
    df_unscaled["gdp_per_capita"],
    df_unscaled["life_expectancy"],
    alpha=0.6,
    c=df_imputed["cluster"],
    cmap="tab10",
)

# Plot regression line
x_range = np.linspace(
    df_unscaled["gdp_per_capita"].min(), df_unscaled["gdp_per_capita"].max(), 100
)
y_range = lr_simple.predict(x_range.reshape(-1, 1))
# plt.plot(x_range, y_range, "r-", linewidth=2, label=f"R² = {r2_simple:.3f}")

plt.xlabel("GDP per capita ($)")
plt.ylabel("Life Expectancy (years)")
plt.title("GDP vs Life Expectancy")
# plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Governance vs GDP
plt.subplot(1, 2, 2)
plt.scatter(
    df_unscaled["government_effectiveness"],
    df_unscaled["gdp_per_capita"],
    alpha=0.6,
    c=df_imputed["cluster"],
    cmap="tab10",
)

# Simple regression line
X_gov = df_unscaled[["government_effectiveness"]].values
y_gov = df_unscaled["gdp_per_capita"].values
lr_gov = LinearRegression()
lr_gov.fit(X_gov, y_gov)
r2_gov = lr_gov.score(X_gov, y_gov)

x_range = np.linspace(X_gov.min(), X_gov.max(), 100)
y_range = lr_gov.predict(x_range.reshape(-1, 1))
# plt.plot(x_range, y_range, "r-", linewidth=2, label=f"R² = {r2_gov:.3f}")

plt.xlabel("Government Effectiveness")
plt.ylabel("GDP per capita ($)")
plt.title("Governance vs Economic Development")
# plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Correlaciones entre variables importantes

key_pairs = [
    ("gdp_per_capita", "life_expectancy"),
    ("gdp_per_capita", "government_effectiveness"),
    ("life_expectancy", "infant_mortality"),
]

for var1, var2 in key_pairs:
    if var1 in df_unscaled.columns and var2 in df_unscaled.columns:
        # Remove missing values
        data_pair = df_unscaled[[var1, var2]].dropna()
        if len(data_pair) > 10:
            corr_coef, p_value = stats.pearsonr(data_pair[var1], data_pair[var2])
            significance = (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            )
            print(f"{var1} vs {var2}:")
            print(f"  Correlation: {corr_coef:.3f} {significance}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Sample size: {len(data_pair)} countries")

# Síntesis de resultados

results = {
    "dataset_info": {
        "n_countries": len(df_imputed),
        "n_indicators": len(df_imputed.columns) - 1,
        "optimal_clusters": optimal_k,
    },
    "model_performance": {
        "random_forest_cv": cv_scores.mean(),
        "random_forest_test": rf_test_score,
        "neural_network_test": nn_test_score,
    },
}

print(f"\nRESUMEN DEL CONJUNTO DE DATOS:")
print(f"   • Países analizados: {results['dataset_info']['n_countries']}")
print(f"   • Indicadores de desarrollo: {results['dataset_info']['n_indicators']}")
print(f"   • Número óptimo de clusters: {results['dataset_info']['optimal_clusters']}")

print(f"\nRENDIMIENTO DEL MODELO:")
print(
    f"   • Random Forest CV Accuracy: {results['model_performance']['random_forest_cv']:.3f}"
)
print(
    f"   • Random Forest Test Accuracy: {results['model_performance']['random_forest_test']:.3f}"
)
print(
    f"   • Neural Network Test Accuracy: {results['model_performance']['neural_network_test']:.3f}"
)

# Análisis de países que son mejores y peores en desarrollo por cluster

# Calculate relative performance
if "cluster" not in df_unscaled.columns and "cluster" in df_imputed.columns:
    df_unscaled["cluster"] = df_imputed["cluster"]

df_unscaled["economic_rank"] = df_unscaled["gdp_per_capita"].rank()
df_unscaled["health_rank"] = df_unscaled["life_expectancy"].rank()
df_unscaled["development_gap"] = (
    df_unscaled["economic_rank"] - df_unscaled["health_rank"]
)
gap_by_cluster = df_unscaled.groupby("cluster")["development_gap"].mean()
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
# Plot development gap by cluster
plt.bar(
    range(len(gap_by_cluster)),
    gap_by_cluster.values,
    color=plt.cm.Set3(range(len(gap_by_cluster))),
    alpha=0.7,
)
plt.xlabel("Cluster")
plt.ylabel("Average Development Gap (Economic - Health Rank)")
plt.title("Development Gap by Cluster")
plt.xticks(range(len(gap_by_cluster)), [f"C{i}" for i in gap_by_cluster.index])
plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)

plt.subplot(1, 3, 2)
# Identify outliers (countries with large gaps)
threshold = df_unscaled["development_gap"].std() * 1.5
outliers_positive = df_unscaled[df_unscaled["development_gap"] > threshold]
outliers_negative = df_unscaled[df_unscaled["development_gap"] < -threshold]

print(f"Countries with much better economic than health development:")
for country in outliers_positive.index[:5]:
    print(f"  {country}: gap = {outliers_positive.loc[country, 'development_gap']:.1f}")

print(f"\nCountries with much better health than economic development:")
for country in outliers_negative.index[:5]:
    print(f"  {country}: gap = {outliers_negative.loc[country, 'development_gap']:.1f}")

# Scatter plot with highlights
colors_scatter = ["gray"] * len(df_unscaled)
sizes = [30] * len(df_unscaled)

for idx in outliers_positive.index:
    pos = df_unscaled.index.get_loc(idx)
    colors_scatter[pos] = "red"
    sizes[pos] = 100

for idx in outliers_negative.index:
    pos = df_unscaled.index.get_loc(idx)
    colors_scatter[pos] = "green"
    sizes[pos] = 100

plt.scatter(
    df_unscaled["gdp_per_capita"],
    df_unscaled["life_expectancy"],
    c=colors_scatter,
    s=sizes,
    alpha=0.7,
)
plt.xlabel("GDP per capita ($)")
plt.ylabel("Life Expectancy (years)")
plt.title(
    "Development Mismatches\nRed: Over-performing economically, Green: Over-performing health"
)

plt.subplot(1, 3, 3)
# Analyze what drives mismatches
if len(outliers_positive) > 0:
    mismatch_features = []
    for feature in [
        "government_effectiveness",
        "health_expenditure",
        "control_corruption",
    ]:
        if feature in df_unscaled.columns:
            positive_mean = outliers_positive[feature].mean()
            negative_mean = (
                outliers_negative[feature].mean() if len(outliers_negative) > 0 else 0
            )
            overall_mean = df_unscaled[feature].mean()

            mismatch_features.append(
                (feature, positive_mean - overall_mean, negative_mean - overall_mean)
            )

    if len(mismatch_features) > 0:
        features, pos_diffs, neg_diffs = zip(*mismatch_features)
        x = np.arange(len(features))

        plt.bar(x - 0.2, pos_diffs, 0.4, label="Economic Over-performers", alpha=0.7)
        if len(outliers_negative) > 0:
            plt.bar(x + 0.2, neg_diffs, 0.4, label="Health Over-performers", alpha=0.7)
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.xlabel("Features")
        plt.ylabel("Difference from Overall Mean")
        plt.title("Characteristics of Mismatched Countries")
        plt.xticks(x, features, rotation=45)
        plt.legend()

plt.tight_layout()
plt.show()


# Scatter plot with highlights
# Base colors: color points by their cluster label (if available), otherwise use gray
sizes = [30] * len(df_unscaled)
if "cluster" in df_unscaled.columns:
    # Ensure alignment
    cluster_series = df_unscaled["cluster"].reindex(df_unscaled.index).astype(int)
    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(int(c) % cmap.N) for c in cluster_series]
else:
    base_colors = ["gray"] * len(df_unscaled)

colors_scatter = list(base_colors)

for idx in outliers_positive.index:
    pos = df_unscaled.index.get_loc(idx)
    colors_scatter[pos] = "red"
    sizes[pos] = 100

for idx in outliers_negative.index:
    pos = df_unscaled.index.get_loc(idx)
    colors_scatter[pos] = "green"
    sizes[pos] = 100

plt.scatter(
    df_unscaled["gdp_per_capita"],
    df_unscaled["life_expectancy"],
    c=colors_scatter,
    s=sizes,
    alpha=0.7,
)
plt.xlabel("GDP per capita ($)")
plt.ylabel("Life Expectancy (years)")
plt.title(
    "Development Mismatches\nRed: Over-performing economically, Green: Over-performing health"
)
plt.show()

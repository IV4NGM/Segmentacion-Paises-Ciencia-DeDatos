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

economic_indicators = [
    "gdp_per_capita",
    "unemployment",
    "gdp_growth",
    "inflation",
    "exports_gdp",
]

df_economic = df_imputed[economic_indicators]

# Reducción de dimensionalidad con PCA
pca = PCA()
pca_result = pca.fit_transform(df_economic)

loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=df_economic.columns,
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


# Determinar el número óptimo de clusters
silhouette_scores = []
inertia_scores = []
k_range = range(2, 8)

plt.figure(figsize=(15, 5))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_economic)
    silhouette_avg = silhouette_score(df_economic, cluster_labels)
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

clustersnumber = optimal_k

kmeans = KMeans(n_clusters=clustersnumber, random_state=42, n_init=10)
df_imputed_fixedclusters = df_imputed[economic_indicators].copy()
df_scaled_fixedclusters = df_scaled[economic_indicators].copy()

cluster_labels = kmeans.fit_predict(df_imputed[economic_indicators])

# Add cluster labels to dataframes
df_imputed_fixedclusters["cluster"] = cluster_labels
df_scaled_fixedclusters["cluster"] = cluster_labels

print("Cluster distribution:")
cluster_dist = df_imputed_fixedclusters["cluster"].value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    print(f"Cluster {cluster_id}: {count} countries")

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
# df_map = df_imputed_fixedclusters.copy()  # <<--- YOUR DATAFRAME
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
#         title="Country Clusters - Economic Indicators",
#         locationmode="ISO-3",
#         labels={"cluster_str": "Cluster"},
#     )

#     out_html = f"{OUTPUT_DIR}/economic_country_clusters_{clustersnumber}_clusters.html"
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
#         f"{OUTPUT_DIR}/economic_cluster_{c}_countries_{clustersnumber}_clusters.txt",
#         "w",
#     ) as f:
#         f.write("\n".join(members))


# Visualizamos los clusters en 2D usando los dos primeros componentes principales
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(df_imputed_fixedclusters.drop("cluster", axis=1))

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

step = 25
for cluster_id in sorted(df_imputed_fixedclusters["cluster"].unique()):
    members = df_imputed_fixedclusters[
        df_imputed_fixedclusters["cluster"] == cluster_id
    ].index.tolist()
    for j, country in enumerate(members):
        if j % step == 0:
            pos = df_imputed_fixedclusters.index.get_loc(country)
            plt.annotate(
                country,
                (pca_result_2d[pos, 0], pca_result_2d[pos, 1]),
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

step = 25
for cluster_id in sorted(df_imputed_fixedclusters["cluster"].unique()):
    members = df_imputed_fixedclusters[
        df_imputed_fixedclusters["cluster"] == cluster_id
    ].index.tolist()
    for j, country in enumerate(members):
        if j % step == 0:
            pos = df_imputed_fixedclusters.index.get_loc(country)
            plt.annotate(
                country,
                (pca_result_2d[pos, 0], pca_result_2d[pos, 1]),
                fontsize=8,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Análisis de características de los clusters

df_imputed_total = df_imputed.copy()
df_scaled_total = df_scaled.copy()

df_imputed_total["cluster"] = cluster_labels
df_scaled_total["cluster"] = cluster_labels

cluster_summary_scaled = df_imputed_total.groupby("cluster").mean()

# Transformar el resumen escalado de los clusters para obtener valores en la escala original
cluster_summary_unscaled = pd.DataFrame(
    scaler.inverse_transform(cluster_summary_scaled),
    columns=cluster_summary_scaled.columns,
    index=cluster_summary_scaled.index,
)
print(cluster_summary_unscaled)

# Análisis de las características económicas
cluster_summary_scaled = df_imputed_fixedclusters.groupby("cluster").mean()

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

df_imputed = df_imputed_fixedclusters.copy()
df_scaled = df_imputed.copy()


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

step = 25
for cluster_id in sorted(df_imputed["cluster"].unique()):
    members = df_imputed[df_imputed["cluster"] == cluster_id].index.tolist()
    for j, country in enumerate(members):
        if j % step == 0:
            pos = df_imputed_fixedclusters.index.get_loc(country)
            plt.annotate(
                country,
                (pca_result_2d[pos, 0], pca_result_2d[pos, 1]),
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

step = 25
for cluster_id in sorted(df_imputed["cluster"].unique()):
    members = df_imputed[df_imputed["cluster"] == cluster_id].index.tolist()
    for j, country in enumerate(members):
        if j % step == 0:
            pos = df_imputed_fixedclusters.index.get_loc(country)
            plt.annotate(
                country,
                (pca_result_2d[pos, 0], pca_result_2d[pos, 1]),
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


# APRENDIZAJE SUPERVISADO PARA PREDECIR CLUSTERS

df_scaled = df_scaled.drop(
    ["hc_cluster", "hc_cluster_optimal", "spectral_cluster_optimal"], axis=1
)

X = df_scaled.drop("cluster", axis=1)
y = df_scaled["cluster"]

# Eliminamos los países con clusters minoritarios para evitar problemas de desbalanceo extremo
cluster_counts = y.value_counts()
min_cluster_size = 5  # Umbral mínimo de tamaño de cluster
valid_clusters = cluster_counts[cluster_counts >= min_cluster_size].index
X = X[y.isin(valid_clusters)]
y = y[y.isin(valid_clusters)]

# Actualizamos el optimal_k basado en los clusters válidos
optimal_k = len(valid_clusters)

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

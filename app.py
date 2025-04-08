import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Klasterisasi Wilayah Rawan Deforestasi dengan K-Means")
st.markdown("""
### 🔃 Membaca Dataset
""")
df = pd.read_excel("Dataset_ML_AI.xlsx")

# tampilkan preview
st.markdown("""
## 🔍 Data Awal
Berikut adalah tampilan awal dari dataset yang diunggah:
""")
st.dataframe(df.head())

# Membuat daftar kolom tc_loss_ha_2001 hingga tc_loss_ha_2023
tc_loss_columns = [f'tc_loss_ha_{year}' for year in range(2001, 2024)]

# Menambahkan kolom tc_loss_ha_total yang merupakan jumlah dari semua tahun
df['tc_loss_ha_total'] = df[tc_loss_columns].sum(axis=1)

# Menampilkan beberapa baris pertama, termasuk kolom threshold dan tc_loss_ha_total
st.markdown("""
### 📊 Ringkasan Threshold & Kehilangan Total
""")
st.dataframe(df[['subnational1', 'threshold', 'area_ha', 'tc_loss_ha_total']].head())

# Menampilkan semua nama kolom dalam DataFrame
st.markdown("""
### 📋 Daftar Nama Kolom Dataset
""")
st.dataframe(pd.DataFrame(df.columns, columns=['Column Names']))

# Memisahkan data berdasarkan nilai threshold
threshold_values = [0, 10, 15, 20, 25, 30, 50, 75]  # Tentukan nilai threshold sesuai keinginan

# Membuat dictionary untuk menampung data yang sudah dipisah
threshold_data = {}

for value in threshold_values:
    threshold_data[value] = df[df['threshold'] == value]

# Menampilkan hasil untuk setiap threshold
st.markdown("""
#### 🔢 Data Threshold 0
""")
st.dataframe(threshold_data[0].head())
st.markdown("""
#### 🔢 Data Threshold 10
""")
st.dataframe(threshold_data[10].head())
st.markdown("""
#### 🔢 Data Threshold 15
""")
st.dataframe(threshold_data[15].head())
st.markdown("""
#### 🔢 Data Threshold 20
""")
st.dataframe(threshold_data[20].head())
st.markdown("""
#### 🔢 Data Threshold 25
""")
st.dataframe(threshold_data[25].head())
st.markdown("""
#### 🔢 Data Threshold 30
""")
st.dataframe(threshold_data[30].head())
st.markdown("""
#### 🔢 Data Threshold 50
""")
st.dataframe(threshold_data[50].head())
st.markdown("""
#### 🔢 Data Threshold 75
""")
st.dataframe(threshold_data[75].head())

# Pembersihan data dan pemilihan fitur untuk clustering
features = ["area_ha", "tc_loss_ha_total"]
df = df.dropna(subset=features)


# ----------------- THRESHOLD 0 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 0
### 📌 Data Mentah
""")
threshold_0_raw = threshold_data[0][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_0_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_0_norm = threshold_0_raw[features].copy()
threshold_0_norm = (threshold_0_norm - threshold_0_norm.min()) / (threshold_0_norm.max() - threshold_0_norm.min()) * 9 + 1
st.dataframe(threshold_0_norm.describe())
st.dataframe(threshold_0_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_0_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_0_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_0_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_0_norm, labels, k):
    return threshold_0_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_0_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_0_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_0_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_0_norm['subnational1'] = threshold_data[0]['subnational1'].values
threshold_0_final = threshold_0_norm.reset_index().rename(columns={'index': 'No'})
threshold_0_final = threshold_0_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_0_final)

# Simpan daftar wilayah per cluster
threshold_0_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_0_norm[threshold_0_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_0_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_0_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


# ----------------- THRESHOLD 10 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 10
### 📌 Data Mentah
""")
threshold_10_raw = threshold_data[10][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_10_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_10_norm = threshold_10_raw[features].copy()
threshold_10_norm = (threshold_10_norm - threshold_10_norm.min()) / (threshold_10_norm.max() - threshold_10_norm.min()) * 9 + 1
st.dataframe(threshold_10_norm.describe())
st.dataframe(threshold_10_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_10_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_10_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_10_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_10_norm, labels, k):
    return threshold_10_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_10_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_10_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_10_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_10_norm['subnational1'] = threshold_data[10]['subnational1'].values
threshold_10_final = threshold_10_norm.reset_index().rename(columns={'index': 'No'})
threshold_10_final = threshold_10_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_10_final)

# Simpan daftar wilayah per cluster
threshold_10_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_10_norm[threshold_10_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_10_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_10_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


    # ----------------- THRESHOLD 15 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 15
### 📌 Data Mentah
""")
threshold_15_raw = threshold_data[15][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_15_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_15_norm = threshold_15_raw[features].copy()
threshold_15_norm = (threshold_15_norm - threshold_15_norm.min()) / (threshold_15_norm.max() - threshold_15_norm.min()) * 9 + 1
st.dataframe(threshold_15_norm.describe())
st.dataframe(threshold_15_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_15_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_15_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_15_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_15_norm, labels, k):
    return threshold_15_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_15_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_15_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_15_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_15_norm['subnational1'] = threshold_data[15]['subnational1'].values
threshold_15_final = threshold_15_norm.reset_index().rename(columns={'index': 'No'})
threshold_15_final = threshold_15_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_15_final)

# Simpan daftar wilayah per cluster
threshold_15_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_15_norm[threshold_15_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_15_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_15_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


    # ----------------- THRESHOLD 20 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 20
### 📌 Data Mentah
""")
threshold_20_raw = threshold_data[20][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_20_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_20_norm = threshold_20_raw[features].copy()
threshold_20_norm = (threshold_20_norm - threshold_20_norm.min()) / (threshold_20_norm.max() - threshold_20_norm.min()) * 9 + 1
st.dataframe(threshold_20_norm.describe())
st.dataframe(threshold_20_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_20_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_20_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_20_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_20_norm, labels, k):
    return threshold_20_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_20_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_20_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_20_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_20_norm['subnational1'] = threshold_data[20]['subnational1'].values
threshold_20_final = threshold_20_norm.reset_index().rename(columns={'index': 'No'})
threshold_20_final = threshold_20_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_20_final)

# Simpan daftar wilayah per cluster
threshold_20_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_20_norm[threshold_20_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_20_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_20_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


    # ----------------- THRESHOLD 25 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 25
### 📌 Data Mentah
""")
threshold_25_raw = threshold_data[25][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_25_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_25_norm = threshold_25_raw[features].copy()
threshold_25_norm = (threshold_25_norm - threshold_25_norm.min()) / (threshold_25_norm.max() - threshold_25_norm.min()) * 9 + 1
st.dataframe(threshold_25_norm.describe())
st.dataframe(threshold_25_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_25_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_25_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_25_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_25_norm, labels, k):
    return threshold_25_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_25_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_25_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_25_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_25_norm['subnational1'] = threshold_data[25]['subnational1'].values
threshold_25_final = threshold_25_norm.reset_index().rename(columns={'index': 'No'})
threshold_25_final = threshold_25_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_25_final)

# Simpan daftar wilayah per cluster
threshold_25_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_25_norm[threshold_25_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_25_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_25_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


    # ----------------- THRESHOLD 30 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 30
### 📌 Data Mentah
""")
threshold_30_raw = threshold_data[30][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_30_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_30_norm = threshold_30_raw[features].copy()
threshold_30_norm = (threshold_30_norm - threshold_30_norm.min()) / (threshold_30_norm.max() - threshold_30_norm.min()) * 9 + 1
st.dataframe(threshold_30_norm.describe())
st.dataframe(threshold_30_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_30_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_30_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_30_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_30_norm, labels, k):
    return threshold_30_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_30_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_30_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_30_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_30_norm['subnational1'] = threshold_data[30]['subnational1'].values
threshold_30_final = threshold_30_norm.reset_index().rename(columns={'index': 'No'})
threshold_30_final = threshold_30_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_30_final)

# Simpan daftar wilayah per cluster
threshold_30_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_30_norm[threshold_30_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_30_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_30_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


    # ----------------- THRESHOLD 50 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 50
### 📌 Data Mentah
""")
threshold_50_raw = threshold_data[50][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_50_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_50_norm = threshold_50_raw[features].copy()
threshold_50_norm = (threshold_50_norm - threshold_50_norm.min()) / (threshold_50_norm.max() - threshold_50_norm.min()) * 9 + 1
st.dataframe(threshold_50_norm.describe())
st.dataframe(threshold_50_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_50_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_50_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_50_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_50_norm, labels, k):
    return threshold_50_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_50_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_50_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_50_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_50_norm['subnational1'] = threshold_data[50]['subnational1'].values
threshold_50_final = threshold_50_norm.reset_index().rename(columns={'index': 'No'})
threshold_50_final = threshold_50_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_50_final)

# Simpan daftar wilayah per cluster
threshold_50_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_50_norm[threshold_50_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_50_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_50_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)


# ----------------- THRESHOLD 75 -----------------
st.markdown("""
---
## 🔵 Analisis Threshold 75
### 📌 Data Mentah
""")
threshold_75_raw = threshold_data[75][features + ['subnational1']].copy().dropna().reset_index(drop=True)
st.dataframe(threshold_75_raw)

# Menormalisasi data ke rentang 1 hingga 10
st.markdown("### 🔃 Normalisasi Data")
threshold_75_norm = threshold_75_raw[features].copy()
threshold_75_norm = (threshold_75_norm - threshold_75_norm.min()) / (threshold_75_norm.max() - threshold_75_norm.min()) * 9 + 1
st.dataframe(threshold_75_norm.describe())
st.dataframe(threshold_75_norm)

# Menghitung inertia untuk berbagai jumlah cluster
st.markdown("### 📉 Perhitungan Inertia - Elbow Method")
X_norm = threshold_75_norm.to_numpy()
max_iter = 100
inertia_values = []
k_range = range(1, 11)  # Uji untuk jumlah cluster dari 1 hingga 10

for k in k_range:
    # Inisialisasi centroid secara acak
    centroids = X_norm[np.random.choice(X_norm.shape[0], k, replace=False)]

    # Iterasi K-Means
    for _ in range(max_iter):
        # Hitung jarak dari setiap titik ke centroid
        distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
        new_centroids = np.array([X_norm[clusters == i].mean(axis=0) for i in range(k)])

        # Jika centroid tidak berubah, berarti konvergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia
    inertia = np.sum((X_norm - centroids[clusters]) ** 2)
    inertia_values.append(inertia)

# Membuat DataFrame dari hasil inertia
inertia_df = pd.DataFrame({
    'Cluster (k)': list(k_range),
    'Inertia': inertia_values
})

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(inertia_df)

# Plot Elbow Method untuk Inertia
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
st.pyplot(plt)

# Fungsi untuk menentukan centroid acak
st.markdown("### 🎯 Centroid Acak & Labeling")
def random_centroids(data, k):
    numeric_data = data[["area_ha", "tc_loss_ha_total"]]
    centroids = []
    for _ in range(k):
        random_point = numeric_data.sample(n=1).values.flatten()
        centroids.append(random_point)
    return np.array(centroids)

centroids = random_centroids(threshold_75_norm, 5)
centroids_df = pd.DataFrame(centroids.T, index=["area_ha", "tc_loss_ha_total"])
st.dataframe(centroids_df)

# Fungsi untuk menentukan label berdasarkan jarak ke centroid
def get_labels(data, centroids):
    X = data[["area_ha", "tc_loss_ha_total"]].values
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

labels = get_labels(threshold_75_norm, centroids)

# Menampilkan jumlah data per cluster
cluster_counts = pd.Series(labels, name="Cluster").value_counts().sort_index()

# Ubah ke DataFrame
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Jumlah Data']

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Fungsi untuk menghitung centroid baru
def new_centroids(threshold_75_norm, labels, k):
    return threshold_75_norm.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Pilih kolom kehilangan lahan (tc_loss_ha) dan kolom area
tc_loss_columns = [col for col in threshold_75_norm.columns if "tc_loss_ha_total" in col]

st.markdown("### 🔁 Iterasi K-Means & Visualisasi")
X = threshold_75_norm[["area_ha", "tc_loss_ha_total"]].values

# Tentukan jumlah cluster
k = 5
np.random.seed(42)

# Inisialisasi centroid secara acak (memilih k titik data secara acak)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Iterasi K-Means
max_iter = 100
iteration = 0
for _ in range(max_iter):
    iteration += 1

    # Hitung jarak dari setiap titik ke centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Menghitung jarak Euclidean ke setiap centroid
    clusters = np.argmin(distances, axis=1)  # Pilih cluster terdekat berdasarkan jarak minimum

    # Hitung centroid baru berdasarkan rata-rata titik dalam setiap cluster
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Jika centroid tidak berubah, berarti konvergen
    if np.all(centroids == new_centroids):
        # Update iterasi terakhir jika konvergen
        iteration_str = f"Iteration {iteration} (Converged)"
        break
    centroids = new_centroids

# Tambahkan hasil clustering ke dataset
threshold_75_norm["Cluster"] = clusters

# Menampilkan jumlah data di tiap cluster
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Buat DataFrame dari hasil clustering
cluster_counts_df = pd.DataFrame({
    "Cluster": list(cluster_counts.keys()),
    "Jumlah Data": list(cluster_counts.values())
})

# Tampilkan sebagai tabel
st.dataframe(cluster_counts_df)

# Plot hasil clustering
plt.figure(figsize=(8, 6))

# Menggunakan colormap untuk menghasilkan warna secara otomatis
colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

for i in range(k):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f'Cluster {i}', color=colors[i])

# Menampilkan centroid
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Menampilkan plot dengan judul Iterasi
plt.title(iteration_str if 'iteration_str' in locals() else f'Iteration {iteration}')
plt.xlabel('Area HA')
plt.ylabel('Total TC Loss HA')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Menyimpan centroids ke dalam DataFrame
st.markdown("### 🧭 Tabel Centroid Akhir")
centroids_df = pd.DataFrame(centroids, columns=["area_ha", "tc_loss_ha_total"])

# Tambahkan kolom "Cluster" sebagai label indeks
centroids_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(centroids_df))])

# Format dua desimal
pd.set_option('display.float_format', '{:.2f}'.format)

# Tampilkan sebagai tabel
st.dataframe(centroids_df)

st.markdown("### 🗺️ Distribusi Wilayah per Cluster")
threshold_75_norm['subnational1'] = threshold_data[75]['subnational1'].values
threshold_75_final = threshold_75_norm.reset_index().rename(columns={'index': 'No'})
threshold_75_final = threshold_75_final[['No', 'area_ha', 'tc_loss_ha_total', 'Cluster', 'subnational1']]
st.dataframe(threshold_75_final)

# Simpan daftar wilayah per cluster
threshold_75_clusters = {}

for cluster_num in range(k):
    cluster_data = threshold_75_norm[threshold_75_norm["Cluster"] == cluster_num]["subnational1"].tolist()
    threshold_75_clusters[cluster_num] = cluster_data

# Konversi dictionary ke DataFrame
clustered_wilayah_list = []

for cluster_num, wilayah_list in threshold_75_clusters.items():
    for idx, wilayah in enumerate(wilayah_list, 1):
        clustered_wilayah_list.append({
            "Cluster": cluster_num,
            "No": idx,
            "Subnational1": wilayah
        })

# Buat DataFrame dari list dict
clustered_wilayah_df = pd.DataFrame(clustered_wilayah_list)

# Tampilkan sebagai tabel
st.dataframe(clustered_wilayah_df)
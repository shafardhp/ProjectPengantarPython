import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(layout="wide", page_title="Dashboard Penyewaan Sepeda")

# Load data
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# === Sidebar: Filter ===
st.sidebar.header("Filter Data")

# Rentang tanggal
date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(day_df['dteday'].min(), day_df['dteday'].max()),
    min_value=day_df['dteday'].min(),
    max_value=day_df['dteday'].max()
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = day_df[
        (day_df['dteday'] >= pd.to_datetime(start_date)) &
        (day_df['dteday'] <= pd.to_datetime(end_date))
    ]
else:
    filtered_df = day_df.copy()

# Filter cuaca
selected_weather = st.sidebar.multiselect(
    "Pilih Kondisi Cuaca",
    options=[1, 2, 3],
    default=[1, 2, 3],
    format_func=lambda x: {1: "Cerah", 2: "Mendung", 3: "Hujan"}[x]
)
filtered_df = filtered_df[filtered_df['weathersit'].isin(selected_weather)]

# Filter musim
selected_seasons = st.sidebar.multiselect(
    "Pilih Musim",
    options=[1, 2, 3, 4],
    default=[1, 2, 3, 4],
    format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4:"Winter"}[x]
)
filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]


# === Judul dan Kredit ===
st.title("📊 Dashboard Penyewaan Sepeda")

if st.button("Credit"):
    st.write("Dibuat oleh Kelompok 3:")
    st.write("1. Dinastisya Vasha Agysta (M0722034)")
    st.write("2. Mayisya Najmuts Zahra A (M0722048)")
    st.write("3. Shafa Ardhana Putri S (M0722072)")
    st.write("Sumber Data: Dataset Rental Sepeda")
st.markdown("##")

# === Tampilan Data ===
st.subheader("Data Harian")
st.dataframe(day_df.head())

st.subheader("Data Per Jam")
st.dataframe(hour_df.head())

# === Statistik Deskriptif ===
st.subheader("📌 Statistik Deskriptif")
st.dataframe(filtered_df.describe())

# === Visualisasi Eksploratif ===
st.subheader("📈 Visualisasi Eksploratif")

# Rata-rata penyewaan per musim
filtered_df['season'] = filtered_df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4:'Winter'})
season_avg = filtered_df.groupby('season')['cnt'].mean().reset_index()

# Warna berdasarkan palet gambar
palette_custom = {
    'Spring': '#b07b6f',   # mocha mousse
    'Summer': '#6e3945',   # rumors
    'Fall': '#dda73a',     # true joy
    'Winter': '#b0c0b5'    # quietude
}

fig1, ax1 = plt.subplots()
sns.barplot(x='season', y='cnt', data=season_avg, palette=palette_custom, ax=ax1)
ax1.set_title('Rata-rata Penyewaan per Musim')
st.pyplot(fig1)

# Suhu vs Jumlah Penyewaan
fig2, ax2 = plt.subplots()
sns.regplot(x='temp', y='cnt', data=filtered_df, scatter_kws={'alpha': 0.5}, ax=ax2)
ax2.set_title("Suhu vs Jumlah Penyewaan")
st.pyplot(fig2)

# Distribusi Penyewaan berdasarkan Cuaca
weather_avg = filtered_df.groupby('weathersit')['cnt'].mean().reset_index()
weather_avg['weathersit'] = weather_avg['weathersit'].map({1: 'Cerah', 2: 'Mendung', 3: 'Hujan'})

fig3, ax3 = plt.subplots()
ax3.pie(weather_avg['cnt'], labels=weather_avg['weathersit'], autopct='%1.1f%%', startangle=90)
ax3.set_title("Distribusi Penyewaan Berdasarkan Cuaca")
st.pyplot(fig3)

# Scatterplot: Suhu vs Kelembaban
fig4, ax4 = plt.subplots()
sns.scatterplot(data=filtered_df, x='temp', y='hum', hue='cnt', palette='coolwarm', ax=ax4)
ax4.set_title("Suhu vs Kelembaban (dengan Jumlah Penyewaan)")
st.pyplot(fig4)

# === Clustering Analisis ===
st.subheader("📊 Clustering Jumlah Penyewaan")

cluster_data = filtered_df[['temp', 'hum', 'windspeed', 'cnt']].copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
filtered_df['cluster'] = kmeans.fit_predict(scaled)

fig5, ax5 = plt.subplots()
sns.scatterplot(data=filtered_df, x='temp', y='cnt', hue='cluster', palette='Set2', ax=ax5)
ax5.set_title("Clustering Penyewaan Berdasarkan Suhu dan Jumlah Penyewaan")
st.pyplot(fig5)

# === Tren Harian dan Jam ===
st.subheader("🕒 Pola Penyewaan Harian dan Jam")

hour_filtered = hour_df[
    (hour_df['dteday'] >= pd.to_datetime(start_date)) &
    (hour_df['dteday'] <= pd.to_datetime(end_date)) &
    (hour_df['weathersit'].isin(selected_weather))
]

# Rata-rata penyewaan per jam
avg_hour = hour_filtered.groupby('hr')['cnt'].mean().reset_index()

fig6, ax6 = plt.subplots()
sns.lineplot(data=avg_hour, x='hr', y='cnt', marker='o', ax=ax6)
ax6.set_title("Rata-rata Penyewaan per Jam")
ax6.set_xticks(range(0, 24))
st.pyplot(fig6)

# Heatmap jam vs hari
hour_filtered['weekday'] = hour_filtered['weekday'].map({
    0: 'Minggu', 1: 'Senin', 2: 'Selasa', 3: 'Rabu', 4: 'Kamis', 5: 'Jumat', 6: 'Sabtu'
})
pivot_table = hour_filtered.pivot_table(values='cnt', index='hr', columns='weekday', aggfunc='mean')

fig7, ax7 = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_table, cmap='YlGnBu', ax=ax7)
ax7.set_title("Heatmap Jumlah Penyewaan per Jam dan Hari")
st.pyplot(fig7)
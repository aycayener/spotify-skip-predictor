import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import json
import plotly.express as px

# 1. Sayfa Yapılandırması
st.set_page_config(page_title="Spotify Mini ML", layout="wide")

DATA_PATH = Path("data/processed/listens.parquet")
MODEL_PATH = Path("models/skip_model.pkl")
META_PATH = Path("models/skip_model_meta.json")


# 2. Veri ve Model Yükleme
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def load_model():
    model, meta = None, None
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Model yükleme hatası: {e}")

    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            st.warning(f"Meta dosyası okunamadı: {e}")

    return model, meta


df = load_data()
model, meta = load_model()

st.title("🎧 Spotify Dinleme Analizi & ML Tahmini")

if df is None:
    st.error(f"⚠️ Veri dosyası bulunamadı: `{DATA_PATH}`.")
    st.stop()

# Sütun İsimleri
artist_col = "master_metadata_album_artist_name" if "master_metadata_album_artist_name" in df.columns else "creator_name"
track_col = "master_metadata_track_name" if "master_metadata_track_name" in df.columns else "item_name"
skip_col = "is_skip" if "is_skip" in df.columns else "skipped"

# Üst Metrikler
col1, col2, col3 = st.columns(3)
col1.metric("Toplam Dinleme", f"{len(df):,}")
if artist_col in df.columns:
    col2.metric("Sanatçı Sayısı", f"{df[artist_col].nunique():,}")
if skip_col in df.columns:
    skip_rate = (df[skip_col].astype(bool).sum() / len(df)) * 100
    col3.metric("Ortalama Skip Oranı", f"%{skip_rate:.1f}")

st.divider()

# Sekmeler
tab_dna, tab_insights, tab_ml = st.tabs(["📊 DNA", "🔎 İçgörü", "🧠 Skip Tahmini"])

# --- 1. SEKME: DNA ---
with tab_dna:
    st.subheader("Müzik Dinleme Alışkanlıkları")
    col_dna1, col_dna2 = st.columns(2)

    with col_dna1:
        if artist_col in df.columns:
            st.write("**En Çok Dinlenen 10 Sanatçı**")
            top_artists = df[artist_col].value_counts().head(10).reset_index()
            top_artists.columns = ["Sanatçı", "Dinleme Sayısı"]
            fig_art = px.bar(top_artists, x="Sanatçı", y="Dinleme Sayısı", color="Dinleme Sayısı",
                             color_continuous_scale="Viridis")
            st.plotly_chart(fig_art, use_container_width=True)

    with col_dna2:
        if track_col in df.columns:
            st.write("**En Çok Dinlenen 10 Parça**")
            top_tracks = df[track_col].value_counts().head(10).reset_index()
            top_tracks.columns = ["Parça", "Dinleme Sayısı"]
            fig_trk = px.bar(top_tracks, x="Parça", y="Dinleme Sayısı", color="Dinleme Sayısı",
                             color_continuous_scale="Plasma")
            st.plotly_chart(fig_trk, use_container_width=True)

# --- 2. SEKME: İçgörü ---
with tab_insights:
    st.subheader("Zaman ve Davranış Analizi")
    if "hour" in df.columns:
        hourly_counts = df["hour"].value_counts().sort_index().reset_index()
        hourly_counts.columns = ["Saat", "Dinleme Sayısı"]
        fig_hour = px.line(hourly_counts, x="Saat", y="Dinleme Sayısı", title="Günün Saatlerine Göre Dinleme Dağılımı",
                           markers=True)
        st.plotly_chart(fig_hour, use_container_width=True)

    if skip_col in df.columns and "part_of_day" in df.columns:
        st.write("**Günün Zamanına Göre Atlama (Skip) Oranı**")
        skip_part = df.groupby("part_of_day")[skip_col].mean().reset_index()
        skip_part[skip_col] = skip_part[skip_col] * 100
        skip_part.columns = ["Günün Vakti", "Skip Oranı (%)"]
        fig_part = px.bar(skip_part, x="Günün Vakti", y="Skip Oranı (%)", color="Skip Oranı (%)",
                          color_continuous_scale="Reds")
        st.plotly_chart(fig_part, use_container_width=True)

# --- 3. SEKME: ML Skip Tahmini ---
with tab_ml:
    st.subheader("Şarkı Atlama (Skip) Tahmini")

    if model is None:
        st.warning("⚠️ Model (`models/skip_model.pkl`) yüklenemedi.")
    else:
        # Modelin eğitiminde kullanılan sütunlar:
        num_cols = meta.get("num_cols",
                            ["hour", "dayofweek", "play_minutes", "session_pos", "session_size", "is_shuffle",
                             "is_offline"])
        cat_cols = meta.get("cat_cols",
                            ["part_of_day", "platform", "reason_start", "reason_end", "conn_country", "content_type"])
        expected_features = num_cols + cat_cols

        st.info("Veri setinizden rastgele bir dinleme satırı seçerek modelin tahminini test edin:")

        # Kullanıcının test edebileceği örnek satır seçimi
        sample_idx = st.slider("Test Edilecek Kayıt İndeksi", 0, min(len(df) - 1, 1000), 0)
        selected_row = df.iloc[[sample_idx]]

        # Seçilen kaydın önemli bilgilerini göster
        show_cols = [c for c in [track_col, artist_col, "play_minutes", skip_col] if c in df.columns]
        st.write("**Seçilen Dinleme Bilgisi:**")
        st.dataframe(selected_row[show_cols])

        if st.button("🔮 Atlama İhtimalini Hesapla"):
            try:
                # Modele SADECE eğitimde gördüğü 13 sütunu ve doğru tiplerde veriyoruz:
                X_input = selected_row[expected_features].copy()

                for c in num_cols:
                    X_input[c] = pd.to_numeric(X_input[c], errors="coerce").fillna(0)
                for c in cat_cols:
                    X_input[c] = X_input[c].fillna("unknown").astype(str)

                # Tahmin hesaplama
                proba = float(model.predict_proba(X_input)[0, 1])
                st.metric(label="Tahmini Skip Olasılığı", value=f"%{proba * 100:.1f}")

                if proba >= 0.5:
                    st.error("⚠️ Sonuç: Büyük ihtimalle bu şarkıyı geçeceksiniz (Skip)!")
                else:
                    st.success("✅ Sonuç: Bu şarkıyı atlamadan sonuna kadar dinleyeceksiniz.")

                if skip_col in selected_row.columns:
                    actual = bool(selected_row[skip_col].values[0])
                    st.caption(f"📌 Gerçekte ne olmuştu?: {'Atlandı (Skip)' if actual else 'Sonuna kadar dinlendi'}")

            except Exception as e:
                st.error(f"Tahmin sırasında hata oluştu: {e}")
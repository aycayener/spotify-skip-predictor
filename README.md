# 🎧 Spotify Dinleme Analizi & ML Atlama (Skip) Tahmini

Spotify **Genişletilmiş Dinleme Geçmişi (Extended Streaming History)** verilerini kullanarak dinleme alışkanlıklarını analiz eden ve bir parçanın atlanıp atlanmayacağını (**skip**) tahmin eden uçtan uca veri bilimi ve makine öğrenimi projesi.

---

## 📌 Projenin Amacı ve Hikayesi

Spotify üzerindeki dinleme geçmişi üzerinden şu sorulara cevap aranmıştır:
1. **Dinleme Alışkanlıkları (DNA):** En çok hangi sanatçıları ve parçaları, ne kadar süreyle dinliyorum?
2. **Davranışsal İçgörüler:** Müzik dinleme yoğunluğum günün hangi saatlerinde artıyor? Hangi zaman aralıklarında şarkı atlama eğilimim daha yüksek?
3. **Makine Öğrenimi ile Skip Tahmini:** Bir dinleme oturumunun dinamikleri (zaman, oturum içi sıra, shuffle, platform vb.) şarkının atlanma olasılığını nasıl etkiliyor?

---

## 🛠️ Proje Mimarisi & Yetkinlikler

- **Veri Ön İşleme & Pipeline (`src/data_prep.py`):**
  - Ham Spotify JSON kayıtlarının okunması ve standart şemaya dönüştürülmesi
  - Zaman damgalarının yerel saate (`Europe/Istanbul`) çevrilmesi ve zaman öznitelikleri (`hour`, `dayofweek`, `part_of_day`) üretimi
  - Dinleme oturumlarının (sessionization) 30 dakikalık zaman farkı eşiği ile tespiti (`session_pos`, `session_size`)
  - İşlenmiş verinin yüksek performanslı **Apache Parquet** formatında saklanması

- **Makine Öğrenimi Modeli (`src/train_model.py`):**
  - Sayısal değişkenler için `StandardScaler`, kategorik değişkenler için `OneHotEncoder` içeren Scikit-Learn `ColumnTransformer` ön işleme hattı
  - Dengesiz sınıf dağılımına karşı dengelenmiş ağırlıklı (`class_weight='balanced'`) **Lojistik Regresyon** sınıflandırıcısı
  - Zaman bazlı veri bölme (train/test split) ve model metrikleri (F1-Skor, ROC-AUC)

- **Etkileşimli Arayüz (`app.py`):**
  - Streamlit ile geliştirilmiş 3 sekmeli kullanıcı arayüzü:
    - **📊 DNA:** En popüler sanatçı ve parça dağılımları, toplam dinleme süresi metrikleri
    - **🔎 İçgörü:** Günün saatlerine göre dinleme grafiği ve günün vakitlerine göre skip oranları
    - **🧠 Skip Tahmini:** Modelin beklediği girdileri dinamik olarak alıp atlama olasılığını (`predict_proba`) hesaplayan tahmin paneli

---

## 📁 Proje Dizin Yapısı

```text
├── data/
│   ├── raw/                # Spotify'dan gelen ham Streaming_History_Audio_*.json dosyaları (Gitignore)
│   └── processed/          # Üretilen listens.parquet dosyası (Gitignore)
├── models/
│   ├── skip_model.pkl      # Eğitilmiş Scikit-Learn pipeline modeli
│   └── skip_model_meta.json# Kullanılan öznitelik (feature) meta verileri
├── src/
│   ├── data_prep.py        # Ham veriyi işleyip parquet oluşturan script
│   └── train_model.py      # Modeli eğiten ve pkl olarak kaydeden script
├── app.py                  # Streamlit web arayüzü
├── requirements.txt        # Gerekli Python kütüphaneleri
├── .gitignore              # Kişisel verilerin GitHub'a sızmasını engelleyen kural dosyası
└── README.md               # Proje dokümantasyonu

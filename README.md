# Spotify 1–2 Günlük Mini ML Projesi
Amaç: Kendi Streaming History verinle (Spotify JSON) **davranışı anlamak** ve küçük bir ML modeliyle **skip (atlama)** olasılığını tahmin etmek.

## Ne öğreneceksin?
- Feature engineering: zaman bağlamı + dinleme davranışı + session
- ML Pipeline: one-hot + LogisticRegression (hızlı, stabil)
- Basit model değerlendirme: F1, ROC-AUC
- Streamlit ile 3 ekranlı canlı sunum

## Kurulum (PyCharm veya terminal)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma sırası
```bash
python src/data_prep.py
python src/train_model.py
streamlit run app.py
```

## Sunum hikayesi (kısa)
1) Problem: "Hangi koşullarda şarkı atlıyorum?"
2) Veri: streaming history olay kayıtları
3) Feature engineering: saat, gün, hafta sonu, shuffle, oturum içi sıra...
4) Model: skip olasılığı tahmini + içgörüler
5) Çözüm: Dinleme alışkanlığım için eyleme dönük sonuçlar

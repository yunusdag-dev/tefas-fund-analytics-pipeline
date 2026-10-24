# TEFAS Yatırım Fonları Performans, Risk Analizi ve Portföy Simülasyonu (ETL Pipeline)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![SQLAlchemy](https://img.shields.io/badge/Database-PostgreSQL%20%7C%20SQLite-336791.svg)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75.svg)
![Tests](https://img.shields.io/badge/Tests-pytest%207%2F7%20passing-brightgreen.svg)

Türkiye Elektronik Fon Alım Satım Platformu (**TEFAS**) bünyesinde işlem gören yatırım fonlarının günlük performans/risk metriklerini (**Sharpe Oranı, Sortino Oranı, Yıllıklandırılmış Volatilite, Maksimum Düşüş / Max Drawdown**) hesaplayan ve `Faker` ile üretilen Türk yatırımcı profillerinin portföy kar/zarar (**PnL & ROI**) analizini uçtan uca gerçekleştiren **Python 3.12 Veri Mühendisliği (ETL) ve Finansal Analitik** projesidir.

---

## Mimari ve Proje Yapısı

```text
data_engineering_project/
├── config/
│   └── settings.py            # TEFAS fon tanımları, veritabanı URI ve finansal parametreler
├── src/
│   ├── extractor.py           # TEFAS Canlı API istemcisi + Kalibre edilmiş GBM simülasyon motoru
│   ├── generator.py           # Faker (tr_TR) ile Türk yatırımcı ve BUY/SELL işlem üreticisi
│   ├── transformer.py         # Veri temizleme, Sharpe, Sortino, Max Drawdown ve PnL hesaplayıcı
│   ├── loader.py              # SQLAlchemy (PostgreSQL / SQLite) veritabanı yükleyici ve SQL sorguları
│   ├── visualizer.py          # 4 panelli interaktif Plotly HTML Dashboard üreticisi
│   └── pipeline.py            # Uçtan uca ETL orkestratörü
├── notebooks/
│   └── tefas_analysis.ipynb   # Keşifsel veri analizi (EDA) ve görselleştirme çalışma defteri
├── tests/
│   └── test_pipeline.py       # 7 kapsamlı birim ve veritabanı entegrasyon testi (pytest)
├── reports/
│   └── tefas_dashboard.html   # İnteraktif Plotly finansal analitik paneli
├── .github/workflows/ci.yml   # GitHub Actions otomatik test ve pipeline CI akışı
└── requirements.txt           # Proje bağımlılıkları
```

---

## İncelenen TEFAS Yatırım Fonları

| Fon Kodu | Fon Unvanı | Kurucu | Kategori | SPK Risk Değeri |
| :--- | :--- | :--- | :--- | :---: |
| **TI2** | İş Portföy BIST 100 Dışı Şirketler Hisse Senedi Fonu | İş Portföy | Hisse Senedi | 7 |
| **MAC** | Marmara Capital Portföy Hisse Senedi Fonu | Marmara Capital | Hisse Senedi | 7 |
| **TCD** | Tacirler Portföy Değişken Fon | Tacirler Portföy | Değişken | 6 |
| **IPB** | İstanbul Portföy Birinci Değişken Fon | İstanbul Portföy | Değişken | 6 |
| **AFT** | Ak Portföy Yeni Teknolojiler Yabancı Hisse Senedi Fonu | Ak Portföy | Yabancı Hisse | 6 |
| **YAY** | Yapı Kredi Portföy Yabancı Teknoloji Sektörü Hisse Senedi Fonu | Yapı Kredi Portföy | Yabancı Hisse | 6 |
| **TTA** | İş Portföy Altın Fonu | İş Portföy | Kıymetli Madenler | 5 |
| **TI1** | İş Portföy Para Piyasası (TL) Fonu | İş Portföy | Para Piyasası | 1 |

---

## Kurulum ve Çalıştırma

### 1. Sanal Ortamın Oluşturulması ve Paketlerin Kurulumu
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Uçtan Uca ETL Pipeline ve Dashboard Çalıştırma
```powershell
python -m src.pipeline
```
> **Not:** Varsayılan olarak sıfır yapılandırma ile yerel `data/tefas_analytics.db` (SQLite) veritabanına yazar ve `reports/tefas_dashboard.html` dosyasını üretir. PostgreSQL kullanmak için `DATABASE_URL="postgresql+psycopg2://kullanici:sifre@localhost:5432/tefas_db"` ortam değişkenini tanımlamanız yeterlidir.

### 3. Otomatik Testlerin Çalıştırılması
```powershell
pytest -v
```

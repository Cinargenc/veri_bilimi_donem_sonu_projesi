#  Kitsune Network Attack Dataset - ARP MitM ML Pipeline

##  Proje Açıklaması

Kitsune Network Attack Dataset'indeki **ARP Man-in-the-Middle (MitM)** saldırısı üzerinde **8 farklı makine öğrenmesi algoritmasını** karşılaştıran kapsamlı bir veri bilimi pipeline'ı.

**Saldırı Tanımı:** Ettercap aracı kullanılarak ARP poisoning ile LAN trafiğinin dinlenmesi (Confidentiality ihlali).

> **Not:** İleride Fuzzing saldırı veri seti de eklenebilir.

##  Karşılaştırılan Algoritmalar

| # | Algoritma | Tür | Açıklama |
|---|-----------|-----|----------|
| 1 | **Logistic Regression** | Baseline | Temel karşılaştırma modeli |
| 2 | **KNN (K=5)** | Instance-based | Ölçeklemeden faydalanır |
| 3 | **SVM (RBF)** | Kernel-based | Güçlü sınıflandırıcı |
| 4 | **Decision Tree** | Tree-based | Hızlı ve yorumlanabilir |
| 5 | **Random Forest** | Ensemble | Güçlü karşılaştırma modeli |
| 6 | **Naive Bayes** | Probabilistic | Hızlı baseline |
| 7 | **ANN (Keras Dense)** | Neural Network | Derin öğrenme modeli |
| 8 | **CNN (1D)** | Neural Network | Tabular veri üzerinde Conv1D |

##  Proje Yapısı

```
veri_bilimi_projesi/
├── data/
│   ├── ARP_MitM_dataset.csv   # 115 feature (N x 115 matris)
│   └── ARP_MitM_labels.csv    # Etiketler (0=Benign, 1=Attack)
├── results/
│   └── metrics.csv             # Model metrikleri
├── plots/
│   ├── cm_*.png                # Confusion matrix görselleri
│   ├── model_comparison_*.png  # Karşılaştırma grafikleri
│   └── training_times.png      # Eğitim süreleri
├── pipeline.py                 # Ana pipeline betiği
├── requirements.txt            # Python bağımlılıkları
└── README.md
```

##  Kurulum ve Kullanım

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Veri setini data/ dizinine yerleştir:
#   - ARP_MitM_dataset.csv
#   - ARP_MitM_labels.csv

# Pipeline'ı çalıştır
python pipeline.py
```

##  Pipeline Adımları

1. **Veri Yükleme** → `ARP_MitM_dataset.csv` (features) + `ARP_MitM_labels.csv` (labels)
2. **Veri Keşfi** → Boyut, sınıf dağılımı, veri tipleri
3. **Eksik Veri Kontrolü** → inf → NaN dönüşümü, median imputation
4. **Stratified Sampling** → Verinin %30'u (sınıf dengesi korunarak)
5. **StandardScaler** → Feature ölçekleme
6. **Train-Test Split** → %80 train / %20 test
7. **Model Eğitimi** → 8 algoritma
8. **Değerlendirme** → Accuracy, Precision, Recall, F1-Score, AUC-ROC
9. **Görselleştirme** → Confusion matrix, radar chart, bar chart
10. **Kaydetme** → `results/metrics.csv`

##  Çıktı Metrikleri

- **Accuracy**: Doğru sınıflandırma oranı
- **Precision**: Pozitif tahmin kesinliği
- **Recall**: Gerçek pozitifleri yakalama oranı
- **F1-Score**: Precision ve Recall harmonik ortalaması
- **AUC-ROC**: ROC eğrisi altındaki alan
- **Confusion Matrix**: TP, TN, FP, FN detayları
- **Train/Predict Time**: Eğitim ve tahmin süreleri

##  Veri Seti Hakkında

- **Kaynak**: [Kitsune Network Attack Dataset (UCI)](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset)
- **Saldırı**: ARP MitM — Ettercap ile ARP poisoning saldırısı
- **Features**: 115 istatistiksel ağ trafiği özniteliği (AfterImage, 5 zaman penceresi)
- **Label**: Binary (0 = Benign, 1 = Attack — MitM üzerinden geçen paketler)
- **Referans**: Y. Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018

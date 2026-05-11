"""
╔══════════════════════════════════════════════════════════════════════╗
║   Kitsune Network Attack Dataset - Mirai Botnet Saldırı Analizi   ║
║   Karşılaştırmalı ML Model Pipeline'ı                             ║
╚══════════════════════════════════════════════════════════════════════╝

Saldırı: Mirai Botnet (IoT cihazları Telnet üzerinden ele geçirme)

Algoritmalar:
  1. Linear Regression (Baseline - Logistic Regression olarak)
  2. K-Nearest Neighbors (KNN)
  3. Support Vector Machine (SVM)
  4. Decision Tree
  5. Random Forest
  6. Naive Bayes (Gaussian)
  7. Artificial Neural Network (ANN - MLPClassifier + Keras)
  8. Convolutional Neural Network (CNN - Keras, tabular reshape)

Veri: data/Mirai_dataset.csv + data/Mirai_labels.csv
Çıktı: results/metrics.csv, plots/ dizini
"""

import os
import sys
import io
import time

# Türkçe Windows (cp1254) ortamında Unicode karakterler için UTF-8 zorla
# line_buffering=True → print() çıktıları anında flush edilir
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışmak için
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# Logger Yapılandırması
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-8s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "Mirai_dataset.csv")      # 115 feature
LABELS_PATH = os.path.join(DATA_DIR, "mirai_labels.csv")        # 0/1 etiket
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.csv")

ATTACK_NAME = "Mirai Botnet"

SAMPLE_RATIO = 0.30        # Stratified sampling - verinin %30'u
TEST_SIZE = 0.20            # Train-test split oranı
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────
# 1. VERİ YÜKLEME VE KEŞİF
# ─────────────────────────────────────────────────────────────────────
def load_data(dataset_path: str, labels_path: str) -> pd.DataFrame:
    """Kitsune dataset ve labels CSV dosyalarını yükler ve birleştirir.
    
    Kitsune veri setinde her saldırı tipi için:
      - *_dataset.csv  → N x M feature matrisi (header yok)
      - *_labels.csv   → N x 1 etiket vektörü (0=benign, 1=attack)
    """
    logger.info(f"Feature dosyası yükleniyor: {dataset_path}")
    logger.info(f"Label dosyası yükleniyor : {labels_path}")

    # Dosya kontrolü
    for fpath in [dataset_path, labels_path]:
        if not os.path.exists(fpath):
            logger.error(f"Dosya bulunamadı: {fpath}")
            logger.error(f"Lütfen dataset ve labels dosyalarını "
                         f"'{os.path.dirname(dataset_path)}' dizinine yerleştirin.")
            sys.exit(1)

    # Önce label dosyasını oku (satır sayısını bilmek için)
    # Label vektörü — farklı format olabilir, otomatik algıla
    y_raw = pd.read_csv(labels_path)
    # Tek sütun varsa doğrudan al, birden fazla varsa 'x' veya son sütunu al
    if y_raw.shape[1] == 1:
        y = y_raw.copy()
        y.columns = ['label']
    elif 'x' in y_raw.columns:
        y = y_raw[['x']].rename(columns={'x': 'label'})
    else:
        y = y_raw.iloc[:, -1:].copy()
        y.columns = ['label']
    
    y['label'] = y['label'].astype(int)
    n_labels = y.shape[0]
    logger.info(f"Label boyutu  : {n_labels:,} satır")
    logger.info(f"Label dağılımı: 0(Benign)={int((y['label']==0).sum()):,} | 1(Attack)={int((y['label']==1).sum()):,}")

    # Feature matrisi — header olup olmadığını otomatik algıla
    # İlk satırı header olarak okuyup satır sayısını karşılaştır
    X = pd.read_csv(dataset_path, header=0, low_memory=False)
    if X.shape[0] == n_labels:
        # İlk satır header idi (veya sayısal pseudo-header), satır sayıları eşleşti
        logger.info("Dataset ilk satırı header olarak atlandı (satır sayıları eşleşti).")
    else:
        # Header yok, ilk satır da veri
        X = pd.read_csv(dataset_path, header=None, low_memory=False)
        if X.shape[0] != n_labels:
            logger.error(f"Feature ({X.shape[0]:,}) ve label ({n_labels:,}) satır sayıları uyuşmuyor!")
            sys.exit(1)
    
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    X.columns = feature_cols
    logger.info(f"Feature boyutu: {X.shape[0]:,} satır × {X.shape[1]} sütun")

    # Satır sayısı son kontrolü
    if X.shape[0] != y.shape[0]:
        logger.error(f"Feature ({X.shape[0]:,}) ve label ({y.shape[0]:,}) satır sayıları uyuşmuyor!")
        sys.exit(1)

    # Birleştir
    df = pd.concat([X, y], axis=1)

    logger.info(f"Birleşik veri: {df.shape[0]:,} satır × {df.shape[1]} sütun")
    logger.info(f"Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Veri keşfi ve istatistik özeti."""
    print("\n" + "═" * 70)
    print("  VERİ KEŞFİ (Exploratory Data Analysis)")
    print("═" * 70)

    print(f"\n  Toplam örnek sayısı : {df.shape[0]:>12,}")
    print(f"  Feature sayısı      : {df.shape[1] - 1:>12}")
    print(f"  Hedef değişken      : label")

    print("\n  ┌─── Sınıf Dağılımı ───┐")
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    for label, count in label_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  │ Sınıf {label}: {count:>10,} ({pct:5.1f}%) {bar}")
    print(f"  └{'─' * 22}┘")

    print(f"\n  Veri tipleri:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {dtype}: {count} sütun")


# ─────────────────────────────────────────────────────────────────────
# 2. EKSİK VERİ KONTROLÜ
# ─────────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik değerleri kontrol eder ve işler."""
    print("\n" + "═" * 70)
    print("  EKSİK VERİ KONTROLÜ")
    print("═" * 70)

    # inf değerleri NaN'a çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    total_missing = missing.sum()

    if total_missing == 0:
        print("  ✓ Eksik veri bulunamadı!")
    else:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        print(f"\n  ✗ {len(missing_cols)} sütunda eksik veri tespit edildi:")
        print(f"    Toplam eksik değer: {total_missing:,}")
        for col in missing_cols.index[:10]:
            print(f"    • {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
        if len(missing_cols) > 10:
            print(f"    ... ve {len(missing_cols) - 10} sütun daha")

        # Eksik verileri median ile doldur
        logger.info("Eksik veriler median ile dolduruluyor...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Kategorik sütunlar varsa mode ile doldur
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                if col != 'label':
                    df[col].fillna(df[col].mode()[0], inplace=True)

        remaining = df.isnull().sum().sum()
        print(f"  ✓ İşlem sonrası eksik veri: {remaining}")

    return df


# ─────────────────────────────────────────────────────────────────────
# 3. STRATİFİED SAMPLING
# ─────────────────────────────────────────────────────────────────────
def stratified_sample(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Sınıf dağılımını koruyarak veriyi örnekler."""
    print("\n" + "═" * 70)
    print("  STRATİFİED SAMPLING")
    print("═" * 70)

    original_size = len(df)
    target_size = int(original_size * ratio)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - ratio), random_state=RANDOM_STATE)
    X = df.drop('label', axis=1)
    y = df['label']

    for sample_idx, _ in sss.split(X, y):
        df_sampled = df.iloc[sample_idx].reset_index(drop=True)

    print(f"  Orijinal veri  : {original_size:>12,} satır")
    print(f"  Örneklem oranı : {ratio * 100:.0f}%")
    print(f"  Örneklem boyutu: {len(df_sampled):>12,} satır")

    print(f"\n  Örneklem sınıf dağılımı:")
    sample_counts = df_sampled['label'].value_counts().sort_index()
    for label, count in sample_counts.items():
        pct = count / len(df_sampled) * 100
        print(f"    Sınıf {label}: {count:>10,} ({pct:5.1f}%)")

    return df_sampled


# ─────────────────────────────────────────────────────────────────────
# 4. ÖN İŞLEME VE TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────────────
def preprocess_and_split(df: pd.DataFrame):
    """Feature/label ayırma, ölçekleme ve train-test split."""
    print("\n" + "═" * 70)
    print("  ÖN İŞLEME & TRAIN-TEST SPLIT")
    print("═" * 70)

    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values

    # Label encoding (eğer string ise)
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info(f"Label encoding uygulandı: {le.classes_}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"  Train seti: {X_train.shape[0]:>10,} örnek")
    print(f"  Test seti : {X_test.shape[0]:>10,} örnek")
    print(f"  Feature   : {X_train.shape[1]:>10} boyut")

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  ✓ StandardScaler uygulandı")
    print(f"    Train mean ≈ {X_train_scaled.mean():.6f}, std ≈ {X_train_scaled.std():.4f}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ─────────────────────────────────────────────────────────────────────
# 5. MODEL TANIMLARI
# ─────────────────────────────────────────────────────────────────────
def get_sklearn_models() -> dict:
    """Scikit-learn modellerini döner."""
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            n_jobs=-1
        ),
        "KNN (K=5)": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        "SVM (RBF Kernel)": SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=RANDOM_STATE,
            probability=True,
            max_iter=5000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            random_state=RANDOM_STATE,
            min_samples_split=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_split=5
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "ANN (MLPClassifier)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            batch_size=256
        ),
    }
    return models


# ─────────────────────────────────────────────────────────────────────
# 6. KERAS ANN MODELİ
# ─────────────────────────────────────────────────────────────────────
def build_keras_ann(input_dim: int):
    """Keras Dense (ANN) model oluşturur."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # GPU bellek büyümesine izin ver
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name="Keras_ANN")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except ImportError:
        logger.warning("TensorFlow bulunamadı → Keras ANN atlanıyor.")
        return None


# ─────────────────────────────────────────────────────────────────────
# 7. KERAS CNN MODELİ (Tabular → 1D Reshape)
# ─────────────────────────────────────────────────────────────────────
def build_keras_cnn(input_dim: int):
    """Tabular veri için 1D CNN modeli oluşturur."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name="Keras_CNN_1D")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except ImportError:
        logger.warning("TensorFlow bulunamadı → Keras CNN atlanıyor.")
        return None


# ─────────────────────────────────────────────────────────────────────
# 8. MODEL EĞİTİMİ VE DEĞERLENDİRME
# ─────────────────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Model performans metriklerini hesaplar."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['AUC-ROC'] = np.nan
    else:
        metrics['AUC-ROC'] = np.nan

    return metrics


def train_sklearn_models(models: dict, X_train, X_test, y_train, y_test) -> list:
    """Scikit-learn modellerini eğitir ve değerlendirir."""
    results = []

    print("\n" + "═" * 70)
    print("  MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("═" * 70)

    SVM_SUBSAMPLE = 30000  # SVM RBF için alt-örneklem limiti

    for name, model in models.items():
        print(f"\n  ▸ {name}")
        print(f"    {'─' * 50}")

        # SVM büyük veride çok yavaş (O(n²~n³)), alt-örnekle
        if 'SVM' in name and X_train.shape[0] > SVM_SUBSAMPLE:
            logger.info(f"SVM için {SVM_SUBSAMPLE:,} örnekle alt-örnekleme yapılıyor...")
            from sklearn.model_selection import StratifiedShuffleSplit as SSS
            sss = SSS(n_splits=1, train_size=SVM_SUBSAMPLE, random_state=RANDOM_STATE)
            idx, _ = next(sss.split(X_train, y_train))
            X_tr, y_tr = X_train[idx], y_train[idx]
            print(f"    ⚠ SVM alt-örneklem: {len(X_tr):,} / {X_train.shape[0]:,}")
        else:
            X_tr, y_tr = X_train, y_train

        start_time = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - start_time

        # Tahmin
        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred

        # Olasılık (AUC-ROC için)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)

        # Metrikler
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['Model'] = name
        metrics['Train Time (s)'] = round(train_time, 2)
        metrics['Predict Time (s)'] = round(pred_time, 4)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['Confusion Matrix'] = str(cm.tolist())

        results.append(metrics)

        # Sonuçları yazdır
        print(f"    Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"    Precision : {metrics['Precision']:.4f}")
        print(f"    Recall    : {metrics['Recall']:.4f}")
        print(f"    F1-Score  : {metrics['F1-Score']:.4f}")
        print(f"    AUC-ROC   : {metrics['AUC-ROC']:.4f}" if not np.isnan(metrics['AUC-ROC']) else "    AUC-ROC   : N/A")
        print(f"    Eğitim    : {train_time:.2f}s | Tahmin: {pred_time:.4f}s")

        # Confusion matrix kaydet
        save_confusion_matrix(cm, name)

    return results


def train_keras_models(X_train, X_test, y_train, y_test) -> list:
    """Keras ANN ve CNN modellerini eğitir."""
    results = []
    input_dim = X_train.shape[1]

    # ──── Keras ANN ────
    print(f"\n  ▸ Keras ANN (Dense)")
    print(f"    {'─' * 50}")

    ann_model = build_keras_ann(input_dim)
    if ann_model is not None:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        start_time = time.time()
        history = ann_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time

        start_pred = time.time()
        y_prob = ann_model.predict(X_test, verbose=0).flatten()
        pred_time = time.time() - start_pred
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['Model'] = 'Keras ANN (Dense)'
        metrics['Train Time (s)'] = round(train_time, 2)
        metrics['Predict Time (s)'] = round(pred_time, 4)

        cm = confusion_matrix(y_test, y_pred)
        metrics['Confusion Matrix'] = str(cm.tolist())
        results.append(metrics)

        print(f"    Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"    Precision : {metrics['Precision']:.4f}")
        print(f"    Recall    : {metrics['Recall']:.4f}")
        print(f"    F1-Score  : {metrics['F1-Score']:.4f}")
        print(f"    AUC-ROC   : {metrics['AUC-ROC']:.4f}")
        print(f"    Eğitim    : {train_time:.2f}s | Tahmin: {pred_time:.4f}s")
        print(f"    Epochs    : {len(history.history['loss'])}")

        save_confusion_matrix(cm, "Keras ANN (Dense)")
        save_training_history(history, "Keras_ANN")

    # ──── Keras CNN (1D) ────
    print(f"\n  ▸ Keras CNN (1D - Tabular Reshape)")
    print(f"    {'─' * 50}")

    cnn_model = build_keras_cnn(input_dim)
    if cnn_model is not None:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # Tabular veriyi 1D-CNN için reshape et: (samples, features, 1)
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        start_time = time.time()
        history = cnn_model.fit(
            X_train_cnn, y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time

        start_pred = time.time()
        y_prob = cnn_model.predict(X_test_cnn, verbose=0).flatten()
        pred_time = time.time() - start_pred
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['Model'] = 'Keras CNN (1D)'
        metrics['Train Time (s)'] = round(train_time, 2)
        metrics['Predict Time (s)'] = round(pred_time, 4)

        cm = confusion_matrix(y_test, y_pred)
        metrics['Confusion Matrix'] = str(cm.tolist())
        results.append(metrics)

        print(f"    Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"    Precision : {metrics['Precision']:.4f}")
        print(f"    Recall    : {metrics['Recall']:.4f}")
        print(f"    F1-Score  : {metrics['F1-Score']:.4f}")
        print(f"    AUC-ROC   : {metrics['AUC-ROC']:.4f}")
        print(f"    Eğitim    : {train_time:.2f}s | Tahmin: {pred_time:.4f}s")
        print(f"    Epochs    : {len(history.history['loss'])}")

        save_confusion_matrix(cm, "Keras CNN (1D)")
        save_training_history(history, "Keras_CNN")

    return results


# ─────────────────────────────────────────────────────────────────────
# 9. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Confusion matrix'i heatmap olarak kaydeder."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Benign (0)', 'Attack (1)'],
        yticklabels=['Benign (0)', 'Attack (1)'],
        ax=ax, linewidths=0.5, linecolor='white',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    ax.set_title(f'Confusion Matrix\n{model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Tahmin (Predicted)', fontsize=12, labelpad=10)
    ax.set_ylabel('Gerçek (Actual)', fontsize=12, labelpad=10)

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    filepath = os.path.join(PLOTS_DIR, f"cm_{safe_name}.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_training_history(history, model_name: str) -> None:
    """Keras eğitim geçmişini grafiğe kaydeder."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[0].set_title(f'{model_name} - Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
    axes[1].set_title(f'{model_name} - Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    filepath = os.path.join(PLOTS_DIR, f"history_{model_name}.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Tüm modellerin karşılaştırmalı grafiklerini oluşturur."""

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # ──── 1. Bar Chart - Metrik Karşılaştırması ────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = sns.color_palette("viridis", len(results_df))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2][idx % 2]
        bars = ax.barh(
            results_df['Model'], results_df[metric],
            color=colors, edgecolor='white', linewidth=0.5
        )
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.05)
        ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.4, label='0.90 eşik')

        # Değerleri barların üzerine yaz
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Model Karşılaştırması - Performans Metrikleri',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison_metrics.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ──── 2. Radar Chart ────
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Döngü tamamla

    color_cycle = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'],
                color=color_cycle[i], markersize=6)
        ax.fill(angles, values, alpha=0.05, color=color_cycle[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_title('Model Karşılaştırması - Radar Chart',
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison_radar.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ──── 3. Eğitim Süresi Karşılaştırması ────
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(results_df['Model'], results_df['Train Time (s)'],
                   color=sns.color_palette("magma", len(results_df)),
                   edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, results_df['Train Time (s)']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}s', va='center', fontsize=11, fontweight='bold')

    ax.set_title('Model Eğitim Süreleri', fontsize=14, fontweight='bold')
    ax.set_xlabel('Süre (saniye)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "training_times.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Tüm grafikler kaydedildi → {PLOTS_DIR}")


# ─────────────────────────────────────────────────────────────────────
# 10. SONUÇLARI KAYDETME
# ─────────────────────────────────────────────────────────────────────
def save_results(results: list) -> pd.DataFrame:
    """Sonuçları CSV dosyasına kaydeder."""
    df = pd.DataFrame(results)

    # Sütun sıralaması
    col_order = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
                 'AUC-ROC', 'Train Time (s)', 'Predict Time (s)', 'Confusion Matrix']
    df = df[[c for c in col_order if c in df.columns]]

    # F1-Score'a göre sırala
    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    # CSV kaydet
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(METRICS_PATH, index=False, encoding='utf-8-sig')
    logger.info(f"Metrikler kaydedildi → {METRICS_PATH}")

    return df


def print_final_report(results_df: pd.DataFrame) -> None:
    """Nihai karşılaştırma raporunu yazdırır."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  NİHAİ KARŞILAŞTIRMA RAPORU".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Tablo formatı
    display_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Train Time (s)']].copy()
    display_df = display_df.round(4)

    try:
        from tabulate import tabulate
        print(tabulate(display_df, headers='keys', tablefmt='fancy_grid',
                       showindex=True, numalign='right'))
    except ImportError:
        print(display_df.to_string(index=True))

    # En iyi model
    best = results_df.iloc[0]
    print(f"\n  🏆 En İyi Model: {best['Model']}")
    print(f"     F1-Score : {best['F1-Score']:.4f}")
    print(f"     Accuracy : {best['Accuracy']:.4f}")
    print(f"     AUC-ROC  : {best['AUC-ROC']:.4f}" if not np.isnan(best['AUC-ROC']) else "     AUC-ROC  : N/A")


# ─────────────────────────────────────────────────────────────────────
# ANA PIPELINE
# ─────────────────────────────────────────────────────────────────────
def main():
    """Ana pipeline fonksiyonu."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  KITSUNE NETWORK ATTACK DATASET".center(68) + "║")
    print("║" + f"  Saldırı: {ATTACK_NAME}".center(68) + "║")
    print("║" + "  ML Karşılaştırma Pipeline'ı".center(68) + "║")
    print("║" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Dizinleri oluştur
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Veri Yükleme (feature + label ayrı dosyalar)
    df = load_data(DATASET_PATH, LABELS_PATH)

    # 2. Veri Keşfi
    explore_data(df)

    # 3. Eksik Veri Kontrolü
    df = handle_missing_values(df)

    # 4. Stratified Sampling (%30)
    df_sampled = stratified_sample(df, SAMPLE_RATIO)

    # 5. Ön İşleme ve Train-Test Split
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df_sampled)

    # 6. Scikit-learn Modelleri
    sklearn_models = get_sklearn_models()
    results = train_sklearn_models(sklearn_models, X_train, X_test, y_train, y_test)

    # 7. Keras Modelleri (ANN + CNN)
    keras_results = train_keras_models(X_train, X_test, y_train, y_test)
    results.extend(keras_results)

    # 8. Sonuçları Kaydet
    results_df = save_results(results)

    # 9. Görselleştirme
    plot_model_comparison(results_df)

    # 10. Final Raporu
    print_final_report(results_df)

    print(f"\n  📁 Metrikler  : {METRICS_PATH}")
    print(f"  📊 Grafikler  : {PLOTS_DIR}")
    print(f"\n  ✓ Pipeline tamamlandı!\n")


if __name__ == "__main__":
    main()

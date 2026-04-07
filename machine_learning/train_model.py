import pandas as pd
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    print("Membaca dataset universitas (Fase Pelatihan Akomodasi)...")
    
    # Path dataset yang baru
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'data_universitas.csv')
    df = pd.read_csv(dataset_path)
    
    # Target variabel kita sekarang adalah memprediksi Estimasi Biaya Akomodasi
    # karena ini akan digabungkan dengan Biaya Studi di proses optimisasi (placement)
    target = 'Historis Biaya Akomodasi / mhs'
    
    # Fitur-fitur yang digunakan (kita buang nama universitas agar model tidak menghafal)
    drop_columns = ['Universitas Rekanan', target]
    
    X = df.drop(columns=drop_columns)
    y = df[target]
    
    # Semua kolom yang bertipe teks (object) adalah Categorical Features
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print("Categorical features detected:", cat_features)
    
    print("Membagi data latih dan data uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Melatih Model CatBoost Regressor (Prediksi Biaya)...")
    # Menggunakan Regressor karena target kita adalah Angka (Rupiah), bukan Kategori (Puas/Tidak Puas)
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        verbose=100,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    print("Mengevaluasi Model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (Kesalahan rata-rata per mahasiswa): Rp {mae:,.2f}")
    print(f"R-Squared (Akurasi Prediksi): {r2:.2f}")
    
    print("Menyimpan model...")
    model_path = os.path.join(os.path.dirname(__file__), 'catboost_akomodasi.pkl')
    joblib.dump(model, model_path)
    print(f"Model berhasil disimpan di: {model_path}")

if __name__ == '__main__':
    main()

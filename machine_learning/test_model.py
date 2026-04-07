import pandas as pd
import joblib
import os

def main():
    print("Memuat model 'catboost_akomodasi.pkl'...")
    model_path = os.path.join(os.path.dirname(__file__), 'catboost_akomodasi.pkl')
    model = joblib.load(model_path)
    print("Model berhasil dimuat!\n")
    
    # Membuat contoh universitas fiktif yang baru buka untuk simulasi
    print("Membuat contoh data universitas tujuan baru...")
    sample_data = {
        'Negara': ['Australia'],
        'Program (GC)': ['CS, IR, IBM'],
        'Jenis (SE/SA)': ['SA'],
        'Kuota per batch': [20],
        'Biaya studi (1 semester)': [35000000]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    print("Data Universitas Baru:")
    for key, value in sample_data.items():
        print(f" - {key}: {value[0]}")
    print("\n--------------------------")
    
    print("Memprediksi Realisasi Biaya Akomodasi...\n")
    prediksi = model.predict(df_sample)
    
    hasil = prediksi[0]
    # Agar angka formatnya terbaca jelas sbg Rupiah
    hasil_rupiah = f"Rp {int(hasil):,}".replace(",", ".")
    
    print(f"🎓 HASIL PREDIKSI AI: Mahasiswa yang pergi ke Universitas ini diperkirakan \nmembutuhkan batas budget akomodasi sebesar {hasil_rupiah} per orang.")

if __name__ == '__main__':
    main()

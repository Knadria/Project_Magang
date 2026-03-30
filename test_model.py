import pandas as pd
import joblib

def main():
    print("Memuat model 'catboost_model.pkl'...")
    model = joblib.load('catboost_model.pkl')
    print("Model berhasil dimuat!\n")
    
    # Membuat contoh data mahasiswa yang ingin melanjutkan studi
    print("Membuat contoh data pendaftar baru...")
    sample_data = {
        'age': [21],
        'gender': ['Male'],
        'Country': ['USA'],
        'university': ['Harvard University'],
        'program_level': ['Bachelor'],
        'Program': ['Computer Science'],
        'year_of_study': [1],
        'tuition_usd': [45000],
        'scholarship': ['Yes'],  # Anggap dia dapat beasiswa
        'online_classes': ['No'],
        'City': ['Cambridge'],
        'University': ['Harvard University'],
        'Level': ['Bachelor'],
        'Duration_Years': [4.0],
        'Tuition_USD': [45000],
        'Living_Cost_Index': [83.5],
        'Rent_USD': [2200],
        'Visa_Fee_USD': [160],
        'Insurance_USD': [1500],
        'Exchange_Rate': [1.0]
    }
    
    # Mengubah ke format DataFrame pandas karena model mengharapkan DataFrame
    df_sample = pd.DataFrame(sample_data)
    
    print("Data Pendaftar:")
    for key, value in sample_data.items():
        print(f" - {key}: {value[0]}")
    print("\n--------------------------")
    
    # Melakukan prediksi
    print("Memprediksi Efficiency Score\n")
    prediksi = model.predict(df_sample)
    
    # Mengartikan hasil prediksi
    hasil = prediksi[0]
    if type(hasil) is list or type(hasil) is tuple or type(hasil).__name__ == 'ndarray':
        hasil = hasil[0] # Ambil angka di dalam array jika ada
        
    keterangan = {
        4: 'Sangat Efisien',
        3: 'Efisien',
        2: 'Biasa Saja',
        1: 'Kurang Efisien',
        0: 'Sangat Tidak Efisien'
    }
    
    print(f"🎓 HASIL PREDIKSI AI: Score {hasil} -> {keterangan.get(hasil, 'Tidak diketahui')}")

if __name__ == '__main__':
    main()

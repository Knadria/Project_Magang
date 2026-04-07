import pandas as pd
import joblib
import os

def load_data_and_model():
    base_dir = os.path.dirname(__file__)
    
    print("[1] Memuat model CatBoost dan Dataset...")
    model_path = os.path.join(base_dir, 'catboost_akomodasi.pkl')
    model = joblib.load(model_path)
    
    univ_path = os.path.join(base_dir, 'dataset', 'data_universitas.csv')
    mhs_path = os.path.join(base_dir, 'dataset', 'data_mahasiswa_batch.csv')
    
    df_univ = pd.read_csv(univ_path)
    df_mhs = pd.read_csv(mhs_path)
    
    return model, df_univ, df_mhs

def optimize_placement():
    model, df_univ, df_mhs = load_data_and_model()
    
    # 1. Menyiapkan Data Universitas & Prediksi Cost menggunakan CatBoost
    print("[2] Menghitung Proyeksi Biaya per Universitas menggunakan AI...")
    univ_dict = {}
    
    # Prediksi biaya akomodasi untuk setiap universitas
    features_for_prediction = df_univ.drop(columns=['Universitas Rekanan', 'Historis Biaya Akomodasi / mhs'])
    df_univ['Prediksi_Akomodasi'] = model.predict(features_for_prediction)
    
    for _, row in df_univ.iterrows():
        univ_name = row['Universitas Rekanan']
        biaya_studi = row['Biaya studi (1 semester)']
        prediksi_akomodasi = max(0, int(row['Prediksi_Akomodasi'])) # Memastikan tidak minus
        total_biaya_mhs = biaya_studi + prediksi_akomodasi
        
        univ_dict[univ_name] = {
            'kuota_sisa': row['Kuota per batch'],
            'tipe': row['Jenis (SE/SA)'],
            'biaya_studi': biaya_studi,
            'limit_akomodasi': prediksi_akomodasi,
            'total_biaya': total_biaya_mhs,
            'syarat_ielts': 6.5 if row['Negara'] in ['United States', 'Australia', 'UK', 'Canada'] else 6.0
        }
        
    # 2. Persiapan Parameter Keuangan & Mahasiswa
    BUDGET_PER_MHS = 50_000_000
    TOTAL_MHS = len(df_mhs)
    TOTAL_BUDGET = BUDGET_PER_MHS * TOTAL_MHS
    
    print(f"\n[INFO] Total Pendaftar: {TOTAL_MHS} Mahasiswa")
    print(f"[INFO] Limit Total Budget dari Finance: Rp {TOTAL_BUDGET:,}")
    
    # Sort Mahasiswa berdasarkan IPK tertinggi (Prioritas Penempatan)
    df_mhs = df_mhs.sort_values(by='IPK', ascending=False).reset_index(drop=True)
    
    # 3. ALGORITMA ALOKASI (Iterasi setiap mahasiswa)
    print("\n[3] Memulai Algoritma Alokasi (Greedy Allocation by GPA)...")
    
    placements = []
    total_biaya_terpakai = 0
    mhs_tidak_dapat_kuota = 0
    
    for _, mhs in df_mhs.iterrows():
        allocated = False
        mhs_name = mhs['Nama']
        mhs_gpa = mhs['IPK']
        mhs_ielts = mhs['IELTS']
        
        # Mengecek Preferensi 1, 2, dan 3 secara berurutan
        preferensi = [mhs['Preferensi_1'], mhs['Preferensi_2'], mhs['Preferensi_3']]
        
        for pref in preferensi:
            # Skip jika univ tidak ada di database
            if pref not in univ_dict: continue
                
            univ_info = univ_dict[pref]
            
            # CEK KENDALA (CONSTRAINTS)
            cek_kuota = univ_info['kuota_sisa'] > 0
            cek_ielts = mhs_ielts >= univ_info['syarat_ielts']
            
            # Cek limit budget: Apakah jika anak ini masuk sini, budget sisa masih cukup aman?
            # Kita sangat ketat di budget
            cek_budget = (total_biaya_terpakai + univ_info['total_biaya']) <= TOTAL_BUDGET
            
            if cek_kuota and cek_ielts and cek_budget:
                # Alokasikan!
                univ_dict[pref]['kuota_sisa'] -= 1
                total_biaya_terpakai += univ_info['total_biaya']
                
                placements.append({
                    'Student_ID': mhs['Student_ID'],
                    'Nama': mhs_name,
                    'IPK': mhs_gpa,
                    'Universitas_Tujuan': pref,
                    'Tipe_Program': univ_info['tipe'],
                    'Biaya_Studi': univ_info['biaya_studi'],
                    'Limit_Akomodasi': univ_info['limit_akomodasi'],
                    'Total_Biaya_Mhs': univ_info['total_biaya']
                })
                allocated = True
                break # Berhenti mengecek preferensi selanjutnya karena sudah dapat
        
        if not allocated:
            # Jika preferensi 1-3 penuh/syarat tidak cukup, kita lempar ke universitas SE ter-murah yg masih ada kuota
            # Sebagai "Sisa Kuota" placement
            available_univs = [u for u, info in univ_dict.items() if info['kuota_sisa'] > 0 and 
                               (total_biaya_terpakai + info['total_biaya']) <= TOTAL_BUDGET and
                               mhs_ielts >= info['syarat_ielts']]
            
            if available_univs:
                # Pilih yang paling murah
                cheapest_univ = min(available_univs, key=lambda x: univ_dict[x]['total_biaya'])
                univ_info = univ_dict[cheapest_univ]
                
                univ_dict[cheapest_univ]['kuota_sisa'] -= 1
                total_biaya_terpakai += univ_info['total_biaya']
                
                placements.append({
                    'Student_ID': mhs['Student_ID'],
                    'Nama': mhs_name,
                    'IPK': mhs_gpa,
                    'Universitas_Tujuan': cheapest_univ + " (Fallback)",
                    'Tipe_Program': univ_info['tipe'],
                    'Biaya_Studi': univ_info['biaya_studi'],
                    'Limit_Akomodasi': univ_info['limit_akomodasi'],
                    'Total_Biaya_Mhs': univ_info['total_biaya']
                })
            else:
                mhs_tidak_dapat_kuota += 1
                placements.append({
                    'Student_ID': mhs['Student_ID'],
                    'Nama': mhs_name,
                    'IPK': mhs_gpa,
                    'Universitas_Tujuan': "TIDAK DITEMPATKAN",
                    'Tipe_Program': "-",
                    'Biaya_Studi': 0,
                    'Limit_Akomodasi': 0,
                    'Total_Biaya_Mhs': 0
                })
                
    # 4. REPORTING LOGIC
    base_dir = os.path.dirname(__file__)
    df_hasil = pd.DataFrame(placements)
    os.makedirs(os.path.join(base_dir, 'output'), exist_ok=True)
    out_path = os.path.join(base_dir, 'output', 'hasil_optimisasi_penempatan.csv')
    df_hasil.to_csv(out_path, index=False)
    
    print("\n==================================")
    print(" 🏁 RINGKASAN HASIL OPTIMISASI 🏁")
    print("==================================")
    print(f"Total Mahasiswa yang Berhasil Ditempatkan : {TOTAL_MHS - mhs_tidak_dapat_kuota} dari {TOTAL_MHS}")
    print(f"Total Budget Terpakai                     : Rp {total_biaya_terpakai:,}")
    print(f"Sisa / Penghematan Budget                 : Rp {TOTAL_BUDGET - total_biaya_terpakai:,}")
    
    se_count = len(df_hasil[df_hasil['Tipe_Program'] == 'SE'])
    sa_count = len(df_hasil[df_hasil['Tipe_Program'] == 'SA'])
    print(f"Alokasi ke Jalur SE (Bebas Biaya Studi)   : {se_count} Mahasiswa")
    print(f"Alokasi ke Jalur SA (BINUS Bayar Studi)   : {sa_count} Mahasiswa")
    print(f"\n✅ Data Excel Daftar Penempatan berhasil disimpan ke: {out_path}")


if __name__ == "__main__":
    optimize_placement()

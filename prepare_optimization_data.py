import pandas as pd
import numpy as np
import random
import os

def generate_dummy_data():
    print("Membaca dataset awal...")
    file_path = os.path.join('machine_learning', 'dataset_untuk_training.csv')
    try:
        df_awal = pd.read_csv(file_path)
    except FileNotFoundError:
        df_awal = pd.read_csv('dataset_untuk_training.csv')

    # 1. GENERATE DATA UNIVERSITAS
    print("Membuat Dataset Universitas...")
    # Ambil data unik Universitas dan Negaranya
    univ_data = df_awal[['University', 'Country']].drop_duplicates().reset_index(drop=True)
    
    programs = ['CS', 'IR', 'IBM', 'CS, IR', 'CS, IBM', 'IR, IBM', 'CS, IR, IBM']
    jenis_kerjasama = ['SE', 'SA']
    
    # Biaya studi kita hitung proporsional (rata-rata Tuition USD * 15000 kurs) / 2 untuk 1 semester
    avg_tuition = df_awal.groupby('University')['Tuition_USD'].mean().reset_index()
    univ_data = pd.merge(univ_data, avg_tuition, on='University', how='left')
    
    # Rata-rata akomodasi = (Rent_USD * 6 bulan + Living_Cost_Index * 30 * 6 bulan) * 15000 (Dummy formula)
    # Agar model ML (CatBoost) kita punya data historis untuk diprediksi
    avg_rent = df_awal.groupby('University')['Rent_USD'].mean().reset_index()
    univ_data = pd.merge(univ_data, avg_rent, on='University', how='left')
    
    univ_data['Program (GC)'] = [random.choice(programs) for _ in range(len(univ_data))]
    univ_data['Jenis (SE/SA)'] = [random.choice(jenis_kerjasama) for _ in range(len(univ_data))]
    univ_data['Kuota per batch'] = [random.randint(5, 20) for _ in range(len(univ_data))]
    
    # Logical Rule: Jika SE, biaya studi = 0. Jika SA, biaya studi > 0
    univ_data['Biaya studi (1 semester)'] = univ_data.apply(
        lambda row: 0 if row['Jenis (SE/SA)'] == 'SE' else int(row['Tuition_USD'] * 15000 / 2), 
        axis=1
    )
    
    univ_data['Historis Biaya Akomodasi / mhs'] = (univ_data['Rent_USD'] * 6 * 15000).fillna(15000000).astype(int)
    
    # Clean up columns kita sesuaikan dgn nama tabel dosen
    df_univ = univ_data[['University', 'Country', 'Program (GC)', 'Jenis (SE/SA)', 'Kuota per batch', 'Biaya studi (1 semester)', 'Historis Biaya Akomodasi / mhs']]
    df_univ.columns = ['Universitas Rekanan', 'Negara', 'Program (GC)', 'Jenis (SE/SA)', 'Kuota per batch', 'Biaya studi (1 semester)', 'Historis Biaya Akomodasi / mhs']
    
    
    # 2. GENERATE DATA MAHASISWA
    print("Membuat Dataset Mahasiswa (Pendaftar Batch ini)...")
    jumlah_mhs = 80
    mhs_programs = ['CS', 'IBM', 'IR']
    
    student_ids = [f"BINUS{str(i).zfill(3)}" for i in range(1, jumlah_mhs + 1)]
    student_names = [f"Student_{i}" for i in range(1, jumlah_mhs + 1)]
    mhs_prog = [random.choice(mhs_programs) for _ in range(jumlah_mhs)]
    gpas = [round(random.uniform(2.5, 4.0), 2) for _ in range(jumlah_mhs)]
    ielts_scores = [random.choice([5.5, 6.0, 6.5, 7.0, 7.5, 8.0]) for _ in range(jumlah_mhs)]
    
    # Buat preferensi mhs (Harus milih universitas yang ada program mereka)
    pref1, pref2, pref3 = [], [], []
    for prog in mhs_prog:
        # Cari Univ yang punya substring program ini
        eligible_univs = df_univ[df_univ['Program (GC)'].str.contains(prog)]['Universitas Rekanan'].tolist()
        if len(eligible_univs) >= 3:
            p1, p2, p3 = random.sample(eligible_univs, 3)
        else:
            # Fallback jika kurang dr 3
            p1, p2, p3 = eligible_univs[0], eligible_univs[0], eligible_univs[0]
        pref1.append(p1)
        pref2.append(p2)
        pref3.append(p3)

    df_mhs = pd.DataFrame({
        'Student_ID': student_ids,
        'Nama': student_names,
        'Program': mhs_prog,
        'IPK': gpas,
        'IELTS': ielts_scores,
        'Preferensi_1': pref1,
        'Preferensi_2': pref2,
        'Preferensi_3': pref3
    })

    # Simpan ke CSV
    out_dir = os.path.join('machine_learning', 'dataset')
    os.makedirs(out_dir, exist_ok=True)
    df_univ.to_csv(os.path.join(out_dir, 'data_universitas.csv'), index=False)
    df_mhs.to_csv(os.path.join(out_dir, 'data_mahasiswa_batch.csv'), index=False)
    
    print("\n✅ SELESAI! File dummy berhasil dibuat di folder 'machine_learning/dataset':")
    print("   1. data_universitas.csv")
    print("   2. data_mahasiswa_batch.csv")

if __name__ == '__main__':
    generate_dummy_data()

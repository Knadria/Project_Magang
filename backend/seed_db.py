import pandas as pd
import os
from database import SessionLocal, init_db, University, Student

def seed_database():
    print("Inisialisasi Database...")
    init_db()
    db = SessionLocal()
    
    # Menghapus isian database lama bila ada
    db.query(University).delete()
    db.query(Student).delete()
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    print("Membaca file data_universitas.csv ...")
    univ_path = os.path.join(base_dir, 'machine_learning', 'dataset', 'data_universitas.csv')
    df_univ = pd.read_csv(univ_path)
    
    for _, row in df_univ.iterrows():
        univ = University(
            name=row['Universitas Rekanan'],
            country=row['Negara'],
            programs=row['Program (GC)'],
            type=row['Jenis (SE/SA)'],
            quota=int(row['Kuota per batch']),
            tuition_fee=float(row['Biaya studi (1 semester)']),
            historical_accomodation=float(row['Historis Biaya Akomodasi / mhs'])
        )
        db.add(univ)
        
    print("Membaca file data_mahasiswa_batch.csv ...")
    mhs_path = os.path.join(base_dir, 'machine_learning', 'dataset', 'data_mahasiswa_batch.csv')
    df_mhs = pd.read_csv(mhs_path)
    
    for _, row in df_mhs.iterrows():
        mhs = Student(
            student_id=row['Student_ID'],
            name=row['Nama'],
            program=row['Program'],
            gpa=float(row['IPK']),
            ielts=float(row['IELTS'])
            # Note: Kita TIDAK mengisi pref 1,2,3 di database SEKARANG.
            # Karena di skenario aplikasi WEB, Preferences dibiarkan kosong, 
            # lalu diisi ketika mahasiswa mensubmit form melalui portal web!
        )
        db.add(mhs)
        
    db.commit()
    db.close()
    print("✅ Sedding Selesai! Tabel University dan Student berhasil diisi di app.db")

if __name__ == '__main__':
    seed_database()

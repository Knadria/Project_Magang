from sqlalchemy.orm import Session
from backend.database import Student, University
import pandas as pd
import joblib
import os

catboost_model = None
try:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'machine_learning', 'catboost_akomodasi.pkl')
    catboost_model = joblib.load(model_path)
except Exception:
    pass

def run_optimization(db: Session, budget_per_mhs: int = 50_000_000):
    univs = db.query(University).all()
    students = db.query(Student).order_by(Student.gpa.desc()).all()
    
    univ_dict = {}
    
    # Menghitung proyeksi akomodasi & sisa kuota
    for u in univs:
        prediksi_akomodasi = u.historical_accomodation
        if catboost_model:
            df_temp = pd.DataFrame({
                'Negara': [u.country],
                'Program (GC)': [u.programs],
                'Jenis (SE/SA)': [u.type],
                'Kuota per batch': [u.quota],
                'Biaya studi (1 semester)': [u.tuition_fee]
            })
            pred = catboost_model.predict(df_temp)[0]
            prediksi_akomodasi = max(0, float(pred))
            
        total_biaya = u.tuition_fee + prediksi_akomodasi
        
        univ_dict[u.name] = {
            'kuota_sisa': u.quota,
            'tipe': u.type,
            'biaya_studi': u.tuition_fee,
            'limit_akomodasi': prediksi_akomodasi,
            'total_biaya': total_biaya,
            'syarat_ielts': 6.5 if u.country in ['United States', 'Australia', 'UK', 'Canada'] else 6.0
        }
        
    TOTAL_BUDGET = budget_per_mhs * len(students)
    total_biaya_terpakai = 0
    mhs_tidak_dapat_kuota = 0
    placements = []
    
    # 1. PRE-ALOKASI (Siswa yang sudah dikunci oleh Admin)
    unlocked_students = []
    for mhs in students:
        if mhs.allocated_univ:
            u_name = mhs.allocated_univ.replace(" (Fallback)", "")
            if u_name in univ_dict:
                univ_dict[u_name]['kuota_sisa'] -= 1
            total_biaya_terpakai += mhs.allocated_cost
            
            placements.append({
                'Student_ID': mhs.student_id,
                'Nama': mhs.name,
                'IPK': mhs.gpa,
                'Universitas_Tujuan': f"{mhs.allocated_univ} [🔒 TERKUNCI]",
                'Tipe_Program': univ_dict[u_name]['tipe'] if u_name in univ_dict else '-',
                'Limit_Akomodasi': univ_dict[u_name]['limit_akomodasi'] if u_name in univ_dict else 0,
                'Total_Biaya': mhs.allocated_cost,
                'is_locked': True
            })
        else:
            unlocked_students.append(mhs)
            
    # 2. ALGORITMA ALOKASI GREEDY (Untuk siswa yang belum terkunci)
    for mhs in unlocked_students:
        allocated = False
        preferensi = [mhs.pref_1, mhs.pref_2, mhs.pref_3]
        
        for pref in preferensi:
            # Lewati jika preferensi kosong atau ga valid
            if not pref or pref not in univ_dict: 
                continue
                
            univ_info = univ_dict[pref]
            cek_kuota = univ_info['kuota_sisa'] > 0
            cek_ielts = mhs.ielts >= univ_info['syarat_ielts']
            cek_budget = (total_biaya_terpakai + univ_info['total_biaya']) <= TOTAL_BUDGET
            
            if cek_kuota and cek_ielts and cek_budget:
                univ_dict[pref]['kuota_sisa'] -= 1
                total_biaya_terpakai += univ_info['total_biaya']
                
                placements.append({
                    'Student_ID': mhs.student_id,
                    'Nama': mhs.name,
                    'IPK': mhs.gpa,
                    'Universitas_Tujuan': pref,
                    'Tipe_Program': univ_info['tipe'],
                    'Limit_Akomodasi': univ_info['limit_akomodasi'],
                    'Total_Biaya': univ_info['total_biaya'],
                    'is_locked': False
                })
                allocated = True
                break
                
        # Jika semua preferensi ditolak (Fallback ke Universitas SE termurah)
        if not allocated:
            available_univs = [u for u, info in univ_dict.items() if info['kuota_sisa'] > 0 and 
                               (total_biaya_terpakai + info['total_biaya']) <= TOTAL_BUDGET and
                               mhs.ielts >= info['syarat_ielts']]
            
            if available_univs:
                cheapest_univ = min(available_univs, key=lambda x: univ_dict[x]['total_biaya'])
                univ_info = univ_dict[cheapest_univ]
                
                univ_dict[cheapest_univ]['kuota_sisa'] -= 1
                total_biaya_terpakai += univ_info['total_biaya']
                
                placements.append({
                    'Student_ID': mhs.student_id,
                    'Nama': mhs.name,
                    'IPK': mhs.gpa,
                    'Universitas_Tujuan': cheapest_univ + " (Fallback)",
                    'Tipe_Program': univ_info['tipe'],
                    'Limit_Akomodasi': univ_info['limit_akomodasi'],
                    'Total_Biaya': univ_info['total_biaya'],
                    'is_locked': False
                })
            else:
                mhs_tidak_dapat_kuota += 1
                placements.append({
                    'Student_ID': mhs.student_id,
                    'Nama': mhs.name,
                    'IPK': mhs.gpa,
                    'Universitas_Tujuan': "TIDAK DITEMPATKAN",
                    'Tipe_Program': "-",
                    'Limit_Akomodasi': 0,
                    'Total_Biaya': 0,
                    'is_locked': False
                })
                
    # Menghitung Statistik
    se_count = sum(1 for p in placements if p['Tipe_Program'] == 'SE')
    sa_count = sum(1 for p in placements if p['Tipe_Program'] == 'SA')
    
    return {
        "summary": {
            "total_mahasiswa": len(students),
            "berhasil_ditempatkan": len(students) - mhs_tidak_dapat_kuota,
            "budget_limit": TOTAL_BUDGET,
            "budget_terpakai": total_biaya_terpakai,
            "budget_sisa": TOTAL_BUDGET - total_biaya_terpakai,
            "alokasi_se": se_count,
            "alokasi_sa": sa_count
        },
        "placements": placements
    }

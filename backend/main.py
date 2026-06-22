from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import SessionLocal, Student, University, init_db 
import joblib
import pandas as pd
import os
import io

app = FastAPI(title="Placement Optimization Backend")

@app.on_event("startup")
def startup():
    init_db()

# Mengizinkan koneksi dari Frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Model (Hanya untuk keperluan endpoint Rekomendasi/Estimasi Biaya)
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'machine_learning', 'catboost_akomodasi.pkl')
try:
    catboost_model = joblib.load(model_path)
except Exception as e:
    catboost_model = None

# Dependency Database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from backend.optimizer import run_optimization

# Pydantic Schemas
class LoginRequest(BaseModel):
    student_id: str
    name: str

class PreferenceSubmit(BaseModel):
    student_id: str
    pref_1: str
    pref_2: str
    pref_3: str

class AdminLogin(BaseModel):
    username: str
    password: str

class StudentInput(BaseModel):
    student_id: str
    name: str
    program: str
    gpa: float
    ielts: float

class UnivInput(BaseModel):
    name: str
    country: str
    programs: str
    type: str # SE/SA
    quota: int
    tuition_fee: float
    historical_accomodation: float

class StudentUpdate(BaseModel):
    name: str
    program: str
    gpa: float
    ielts: float

class UnivUpdate(BaseModel):
    name: str
    country: str
    programs: str
    type: str
    quota: int
    tuition_fee: float
    historical_accomodation: float

# Global State untuk melacak setting budget terakhir
GLOBAL_BUDGET_LIMIT = 50_000_000

class BudgetUpdate(BaseModel):
    budget: int

@app.get("/api/system/budget")
def get_system_budget():
    return {"budget_limit": GLOBAL_BUDGET_LIMIT}

@app.post("/api/system/budget")
def update_system_budget(req: BudgetUpdate):
    global GLOBAL_BUDGET_LIMIT
    GLOBAL_BUDGET_LIMIT = req.budget
    return {"message": "Budget limit berhasil diubah", "budget_limit": GLOBAL_BUDGET_LIMIT}



@app.post("/api/student/login")
def login_student(req: LoginRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(
        Student.student_id == req.student_id.strip(), 
        Student.name == req.name.strip()
    ).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan. Periksa ID dan Nama.")
    
    return {
        "success": True,
        "student_id": student.student_id,
        "name": student.name,
        "program": student.program,
        "gpa": student.gpa,
        "ielts": student.ielts,
        "status_form": "Sudah Mengisi" if student.pref_1 else "Belum Mengisi",
        "allocated_univ": student.allocated_univ,
        "cancel_request": student.cancel_request
    }

@app.get("/api/universities/recommend")
def get_university_recommendations(program: str, gpa: float = 0.0, ielts: float = 0.0, db: Session = Depends(get_db)):
    """Mengembalikan semua universitas yang cocok dengan program mahasiswa.
    Jika gpa dan ielts diberikan, 4 universitas teratas diurutkan berdasarkan skor AI
    yang mempertimbangkan kecocokan nilai mahasiswa dengan profil universitas.
    """
    univs = db.query(University).filter(University.programs.contains(program)).all()
    
    result = []
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
            
        total_estimate = u.tuition_fee + prediksi_akomodasi
        
        # Hitung skor AI berdasarkan profil mahasiswa:
        # Mahasiswa IPK/IELTS tinggi → lebih cocok ke kampus kompetitif (kuota kecil, biaya lebih tinggi)
        # Mahasiswa IPK/IELTS rendah → lebih cocok ke kampus dengan kuota besar
        ai_score = 0.0
        if gpa > 0 or ielts > 0:
            # Normalisasi: IPK max 4.0, IELTS max 9.0
            norm_gpa = gpa / 4.0
            norm_ielts = ielts / 9.0
            student_strength = (norm_gpa * 0.6 + norm_ielts * 0.4)  # bobot IPK lebih besar
            
            # Kampus dengan kuota kecil = lebih prestisius, cocok untuk mahasiswa kuat
            quota_factor = max(0, 1 - (u.quota / 20.0))  # kuota < 20 = prestisius
            
            # Biaya reasonable relatif terhadar kekuatan mahasiswa
            # Mahasiswa kuat → lebih berani memilih kampus mahal
            cost_factor = 1 - min(1.0, total_estimate / 50_000_000) * (1 - student_strength)
            
            ai_score = (student_strength * 0.5 + quota_factor * 0.3 + cost_factor * 0.2)
            
        result.append({
            "id": u.id,
            "name": u.name,
            "country": u.country,
            "type": u.type,
            "quota": u.quota,
            "programs": u.programs,
            "tuition_fee": u.tuition_fee,
            "predicted_accomodation": prediksi_akomodasi,
            "estimated_total_cost": total_estimate,
            "ai_score": round(ai_score, 4)
        })
        
    # Sort dari termurah by default
    result.sort(key=lambda x: x["estimated_total_cost"])
    return result

@app.post("/api/student/submit_preferences")
def submit_preferences(req: PreferenceSubmit, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
        
    student.pref_1 = req.pref_1
    student.pref_2 = req.pref_2
    student.pref_3 = req.pref_3
    db.commit()
    
    return {"message": "Preferensi berhasi disimpan!"}

@app.post("/api/admin/optimize_placement")
def trigger_optimization(budget_per_mhs: int = 50_000_000, db: Session = Depends(get_db)):
    result = run_optimization(db, budget_per_mhs)
    return {"message": "Optimisasi Selesai", "data": result}
    
@app.post("/api/admin/login")
def admin_login_endpoint(req: AdminLogin):
    if req.username == "admin_1" and req.password == "binus_admin":
        return {"success": True, "token": "admin-global-class-token"}
    raise HTTPException(status_code=401, detail="Akses ditolak! Username atau Password salah.")

@app.post("/api/admin/add_student")
def add_student_manual(req: StudentInput, db: Session = Depends(get_db)):
    cek = db.query(Student).filter(Student.student_id == req.student_id).first()
    if cek:
        raise HTTPException(status_code=400, detail="Student ID sudah terdaftar.")
    new_mhs = Student(
        student_id=req.student_id, name=req.name, program=req.program, 
        gpa=req.gpa, ielts=req.ielts
    )
    db.add(new_mhs)
    db.commit()
    return {"message": "Data Mahasiswa berhasil ditambahkan!"}

@app.post("/api/admin/upload_students")
async def upload_students_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berformat CSV")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        count = 0
        for _, row in df.iterrows():
            cek = db.query(Student).filter(Student.student_id == str(row.get('Student_ID'))).first()
            if not cek:
                new_mhs = Student(
                    student_id=str(row.get('Student_ID')), name=str(row.get('Nama')),
                    program=str(row.get('Program')), gpa=float(row.get('IPK')), ielts=float(row.get('IELTS'))
                )
                db.add(new_mhs)
                count += 1
        db.commit()
        return {"message": f"Sukses mengunggah. {count} data mahasiswa baru ditambahkan!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca format CSV: {e}")

@app.post("/api/admin/add_university")
def add_univ_manual(req: UnivInput, db: Session = Depends(get_db)):
    new_univ = University(
        name=req.name, country=req.country, programs=req.programs,
        type=req.type, quota=req.quota, tuition_fee=req.tuition_fee, 
        historical_accomodation=req.historical_accomodation
    )
    db.add(new_univ)
    db.commit()
    return {"message": "Data Kampus berhasil ditambahkan!"}

@app.post("/api/admin/upload_universities")
async def upload_univ_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
     if not file.filename.endswith('.csv'):
         raise HTTPException(status_code=400, detail="File harus berformat CSV")
     contents = await file.read()
     try:
         df = pd.read_csv(io.BytesIO(contents))
         count = 0
         for _, row in df.iterrows():
             new_univ = University(
                 name=str(row.get('Universitas Rekanan')),
                 country=str(row.get('Negara')),
                 programs=str(row.get('Program (GC)')),
                 type=str(row.get('Jenis (SE/SA)')),
                 quota=int(row.get('Kuota per batch', 0)),
                 tuition_fee=float(row.get('Biaya studi (1 semester)', 0)),
                 historical_accomodation=float(row.get('Historis Biaya Akomodasi / mhs', 0))
             )
             db.add(new_univ)
             count += 1
         db.commit()
         return {"message": f"Sukses mengunggah. {count} universitas baru ditambahkan!"}
     except Exception as e:
         raise HTTPException(status_code=500, detail=f"Gagal membaca format CSV: {e}")

@app.get("/api/admin/students")
def list_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    return [
        {
            "student_id": s.student_id,
            "name": s.name,
            "program": s.program,
            "gpa": s.gpa,
            "ielts": s.ielts,
            "pref_1": s.pref_1,
            "pref_2": s.pref_2,
            "pref_3": s.pref_3,
            "allocated_univ": s.allocated_univ,
            "allocated_cost": s.allocated_cost,
            "cancel_request": s.cancel_request,
        }
        for s in students
    ]

@app.put("/api/admin/students/{student_id}")
def update_student(student_id: str, req: StudentUpdate, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.name = req.name
    student.program = req.program
    student.gpa = req.gpa
    student.ielts = req.ielts
    db.commit()
    return {"message": "Data Mahasiswa berhasil diperbarui!"}

@app.delete("/api/admin/students/{student_id}")
def delete_student(student_id: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    db.delete(student)
    db.commit()
    return {"message": "Data Mahasiswa berhasil dihapus!"}

@app.get("/api/admin/universities")
def list_universities(db: Session = Depends(get_db)):
    univs = db.query(University).all()
    return [
        {
            "id": u.id,
            "name": u.name,
            "country": u.country,
            "programs": u.programs,
            "type": u.type,
            "quota": u.quota,
            "tuition_fee": u.tuition_fee,
            "historical_accomodation": u.historical_accomodation,
        }
        for u in univs
    ]

@app.put("/api/admin/universities/{univ_id}")
def update_university(univ_id: int, req: UnivUpdate, db: Session = Depends(get_db)):
    univ = db.query(University).filter(University.id == univ_id).first()
    if not univ:
        raise HTTPException(status_code=404, detail="Universitas tidak ditemukan")
    univ.name = req.name
    univ.country = req.country
    univ.programs = req.programs
    univ.type = req.type
    univ.quota = req.quota
    univ.tuition_fee = req.tuition_fee
    univ.historical_accomodation = req.historical_accomodation
    db.commit()
    return {"message": "Data Universitas berhasil diperbarui!"}

@app.delete("/api/admin/universities/{univ_id}")
def delete_university(univ_id: int, db: Session = Depends(get_db)):
    univ = db.query(University).filter(University.id == univ_id).first()
    if not univ:
        raise HTTPException(status_code=404, detail="Universitas tidak ditemukan")
    db.delete(univ)
    db.commit()
    return {"message": "Data Universitas berhasil dihapus!"}

class TargetStudent(BaseModel):
    student_id: str

class RejectCancelRequest(BaseModel):
    student_id: str
    reason: str

class RejectPlacementRequest(BaseModel):
    student_id: str
    reason: str

class PlacementLock(BaseModel):
    student_id: str
    univ_name: str
    total_cost: float

class BulkPlacementLock(BaseModel):
    placements: list[PlacementLock]

@app.post("/api/student/request_cancel")
def request_cancel(req: TargetStudent, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.cancel_request = True
    db.commit()
    return {"message": "Pengajuan pembatalan berhasil dikirim ke Admin."}

@app.get("/api/admin/cancel_requests")
def get_cancel_requests(db: Session = Depends(get_db)):
    students = db.query(Student).filter(Student.cancel_request == True).all()
    return [{"student_id": s.student_id, "name": s.name, "program": s.program} for s in students]

@app.post("/api/admin/approve_cancel")
def approve_cancel(req: TargetStudent, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.pref_1 = None
    student.pref_2 = None
    student.pref_3 = None
    student.cancel_request = False
    db.commit()
    return {"message": "Pembatalan disetujui. Data preferensi mahasiswa telah direset."}

@app.post("/api/admin/reject_cancel")
def reject_cancel(req: RejectCancelRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.cancel_request = False
    db.commit()
    return {"message": f"Pengajuan pembatalan untuk {student.name} ditolak. Alasan: {req.reason}"}

@app.post("/api/admin/reject_placement")
def reject_placement(req: RejectPlacementRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.allocated_univ = None
    student.allocated_cost = 0.0
    student.cancel_request = False
    db.commit()
    return {"message": f"Penempatan {student.name} ditolak. Alasan: {req.reason}"}

@app.post("/api/admin/approve_placement")
def approve_placement(req: PlacementLock, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.allocated_univ = req.univ_name
    student.allocated_cost = req.total_cost
    student.cancel_request = False # Just in case
    db.commit()
    return {"message": f"Penempatan {student.name} ke {req.univ_name} berhasil dikunci!"}

@app.post("/api/admin/approve_all_placements")
def approve_all_placements(req: BulkPlacementLock, db: Session = Depends(get_db)):
    for p in req.placements:
        student = db.query(Student).filter(Student.student_id == p.student_id).first()
        if student and not student.allocated_univ:
            student.allocated_univ = p.univ_name
            student.allocated_cost = p.total_cost
            student.cancel_request = False
    db.commit()
    return {"message": f"Seluruh {len(req.placements)} penempatan berhasil dikunci permanen!"}

@app.post("/api/admin/reset_placement/{student_id}")
def reset_placement(student_id: str, db: Session = Depends(get_db)):
    """Reset alokasi mahasiswa yang sudah di-approve (cancel penempatan dari sisi admin)."""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student tidak ditemukan")
    student.allocated_univ = None
    student.allocated_cost = 0.0
    student.cancel_request = False
    db.commit()
    return {"message": f"Alokasi {student.name} berhasil direset. Mahasiswa bisa dialokasikan ulang."}

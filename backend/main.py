from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import SessionLocal, Student, University
import joblib
import pandas as pd
import os
import io

app = FastAPI(title="Placement Optimization Backend")

# Mengizinkan koneksi dari Frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
def get_university_recommendations(program: str, db: Session = Depends(get_db)):
    # Mencari universitas yang programnya cocok dengan mahasiswa
    univs = db.query(University).filter(University.programs.contains(program)).all()
    
    result = []
    for u in univs:
        # Jika AI model tersedia, lakukan prediksi biaya akomodasi (On-the-fly)
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
            
        result.append({
            "id": u.id,
            "name": u.name,
            "country": u.country,
            "type": u.type,
            "quota": u.quota,
            "tuition_fee": u.tuition_fee,
            "predicted_accomodation": prediksi_akomodasi,
            "estimated_total_cost": total_estimate
        })
        
    # Sort dari termurah
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

@app.get("/api/admin/universities")
def get_all_universities(db: Session = Depends(get_db)):
    univs = db.query(University).all()
    return univs

@app.get("/api/admin/students")
def get_all_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    return students

class TargetStudent(BaseModel):
    student_id: str

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

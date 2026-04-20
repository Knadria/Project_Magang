from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import SessionLocal, Student, University
import joblib
import pandas as pd
import os

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
        "status_form": "Sudah Mengisi" if student.pref_1 else "Belum Mengisi"
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


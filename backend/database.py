from sqlalchemy import Column, Integer, String, Float, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class University(Base):
    __tablename__ = "universities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    country = Column(String)
    programs = Column(String) # e.g. "CS, IR, IBM"
    type = Column(String) # "SE" or "SA"
    quota = Column(Integer)
    tuition_fee = Column(Float)
    historical_accomodation = Column(Float)

class Student(Base):
    __tablename__ = "students"
    
    student_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    program = Column(String)
    gpa = Column(Float)
    ielts = Column(Float)
    
    # Akan diisi saat pendaftaran (portal apply)
    pref_1 = Column(String, nullable=True)
    pref_2 = Column(String, nullable=True)
    pref_3 = Column(String, nullable=True)
    allocated_univ = Column(String, nullable=True)
    allocated_cost = Column(Float, default=0.0)
    cancel_request = Column(Boolean, default=False)

# Membuat engine sqlite di dalam folder backend
DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

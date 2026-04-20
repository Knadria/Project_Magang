"use client";
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function AdminDataManagement() {
  const router = useRouter();
  const [tab, setTab] = useState<'student' | 'university'>('student');
  
  // Status Messages
  const [msg, setMsg] = useState({ text: '', type: '' });
  const [loading, setLoading] = useState(false);

  // Form Student
  const [sId, setSId] = useState('');
  const [sName, setSName] = useState('');
  const [sProg, setSProg] = useState('Computer Science');
  const [sGpa, setSGpa] = useState('');
  const [sIelts, setSielts] = useState('');
  const [studentCSV, setStudentCSV] = useState<File | null>(null);

  // Form Univ
  const [uName, setUName] = useState('');
  const [uCountry, setUCountry] = useState('');
  const [uProgs, setUProgs] = useState('Computer Science');
  const [uType, setUType] = useState('SE');
  const [uQuota, setUQuota] = useState('');
  const [uFee, setUFee] = useState('');
  const [uAcc, setUAcc] = useState('');
  const [univCSV, setUnivCSV] = useState<File | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('admin_token');
    if (token !== "admin-global-class-token") router.push('/admin');
  }, [router]);

  const handleManualStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/add_student', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: sId, name: sName, program: sProg, gpa: parseFloat(sGpa), ielts: parseFloat(sIelts) })
      });
      const data = await res.json();
      if(res.ok) {
         setMsg({text: data.message, type: 'success'});
         setSId(''); setSName(''); setSGpa(''); setSielts('');
      } else setMsg({text: data.detail, type: 'error'});
    } catch(e) {
      setMsg({text: 'Koneksi ke backend gagal', type: 'error'});
    }
    setLoading(false);
  };

  const handleUploadStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentCSV) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", studentCSV);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/upload_students', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if(res.ok) {
         setMsg({text: data.message, type: 'success'});
         setStudentCSV(null);
      } else setMsg({text: data.detail, type: 'error'});
    } catch(e) {
      setMsg({text: 'Gagal mengupload file.', type: 'error'});
    }
    setLoading(false);
  };

  // Sama dengan Universitas (Ringkasan)
  const handleManualUniv = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/add_university', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
           name: uName, country: uCountry, programs: uProgs, type: uType, 
           quota: parseInt(uQuota), tuition_fee: parseFloat(uFee), historical_accomodation: parseFloat(uAcc) 
        })
      });
      const data = await res.json();
      if(res.ok) {
         setMsg({text: data.message, type: 'success'});
         setUName(''); setUCountry(''); setUQuota(''); setUAcc(''); setUFee('');
      } else setMsg({text: data.detail, type: 'error'});
    } catch(e) {
      setMsg({text: 'Koneksi ke backend gagal', type: 'error'});
    }
    setLoading(false);
  };

  const handleUploadUniv = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!univCSV) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", univCSV);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/upload_universities', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if(res.ok) {
         setMsg({text: data.message, type: 'success'});
         setUnivCSV(null);
      } else setMsg({text: data.detail, type: 'error'});
    } catch(e) {
      setMsg({text: 'Gagal mengupload file.', type: 'error'});
    }
    setLoading(false);
  };


  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 p-8 font-sans">
      <div className="max-w-4xl mx-auto mb-8 border-b border-slate-700 pb-4 flex justify-between items-center">
         <div>
           <h1 className="text-3xl font-bold text-white mb-2">Manajemen Data</h1>
           <p className="text-slate-400">Tambahkan Data Pelamar atau Kampus Rekanan</p>
         </div>
         <button onClick={() => router.push('/admin/dashboard')} className="text-sm bg-slate-800 hover:bg-slate-700 border border-slate-600 px-4 py-2 rounded-lg text-white font-medium">
           ⬅ Kembali ke Dasboard
         </button>
      </div>

      <div className="max-w-4xl mx-auto flex gap-4 mb-6">
         <button 
           onClick={() => {setTab('student'); setMsg({text:'', type:''});}}
           className={`px-6 py-3 rounded-lg font-bold transition-all ${tab === 'student' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
         >
           📝 Kelola Mahasiswa
         </button>
         <button 
           onClick={() => {setTab('university'); setMsg({text:'', type:''});}}
           className={`px-6 py-3 rounded-lg font-bold transition-all ${tab === 'university' ? 'bg-purple-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
         >
           🏫 Kelola Kampus
         </button>
      </div>

      <div className="max-w-4xl mx-auto">
        {msg.text && (
           <div className={`p-4 mb-6 rounded-lg font-medium border ${msg.type === 'error' ? 'bg-red-900/30 border-red-500/50 text-red-400' : 'bg-green-900/30 border-green-500/50 text-green-400'}`}>
              {msg.type === 'error' ? '❌ ' : '✅ '} {msg.text}
           </div>
        )}

        {/* =============== MAHASISWA TAB =============== */}
        {tab === 'student' && (
           <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Manual Input */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
                 <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">Input Manual Baru</h2>
                 <form onSubmit={handleManualStudent} className="space-y-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Student ID</label>
                      <input type="text" required value={sId} onChange={e=>setSId(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Nama Lengkap</label>
                      <input type="text" required value={sName} onChange={e=>setSName(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Program Studi</label>
                      <input type="text" required value={sProg} onChange={e=>setSProg(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                       <div>
                         <label className="block text-sm text-slate-400 mb-1">IPK (0.0 - 4.0)</label>
                         <input type="number" step="0.01" required value={sGpa} onChange={e=>setSGpa(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                       </div>
                       <div>
                         <label className="block text-sm text-slate-400 mb-1">Skor IELTS</label>
                         <input type="number" step="0.5" required value={sIelts} onChange={e=>setSielts(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                       </div>
                    </div>
                    <button type="submit" disabled={loading} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded mt-2">Simpan Mahasiswa</button>
                 </form>
              </div>

              {/* Upload CSV */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-fit">
                 <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">Upload File Dataset (CSV)</h2>
                 <form onSubmit={handleUploadStudent} className="space-y-4">
                    <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:bg-slate-700/50 transition-colors">
                       <input type="file" accept=".csv" onChange={(e) => setStudentCSV(e.target.files ? e.target.files[0] : null)} className="w-full text-slate-400 text-sm" required />
                    </div>
                    <p className="text-xs text-slate-500">*Pastikan header kolom excel sesuai template: Student_ID, Nama, Program, IPK, IELTS.</p>
                    <button type="submit" disabled={loading || !studentCSV} className="w-full bg-slate-700 hover:bg-slate-600 border border-slate-500 text-white font-bold py-2 rounded">Unggah dan Gabungkan Data</button>
                 </form>
              </div>
           </div>
        )}

        {/* =============== KAMPUS TAB =============== */}
        {tab === 'university' && (
           <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
                 <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">Input Manual Univ. Rekanan</h2>
                 <form onSubmit={handleManualUniv} className="space-y-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Nama Universitas</label>
                      <input type="text" required value={uName} onChange={e=>setUName(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Negara</label>
                      <input type="text" required value={uCountry} onChange={e=>setUCountry(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                       <div>
                         <label className="block text-sm text-slate-400 mb-1">Jalur (SE/SA)</label>
                         <select value={uType} onChange={e=>setUType(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"><option>SE</option><option>SA</option></select>
                       </div>
                       <div>
                         <label className="block text-sm text-slate-400 mb-1">Kuota per Batch</label>
                         <input type="number" required value={uQuota} onChange={e=>setUQuota(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                       </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                       <div>
                         <label className="block text-sm text-slate-400 mb-1">Biaya Studi (USD)</label>
                         <input type="number" required value={uFee} onChange={e=>setUFee(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                       </div>
                       <div>
                         <label className="block text-sm text-slate-400 mb-1">Historis Akomodasi</label>
                         <input type="number" required value={uAcc} onChange={e=>setUAcc(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white" />
                       </div>
                    </div>
                    <button type="submit" disabled={loading} className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 rounded mt-2">Simpan Universitas</button>
                 </form>
              </div>

              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-fit">
                 <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">Upload File Dataset (CSV)</h2>
                 <form onSubmit={handleUploadUniv} className="space-y-4">
                    <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:bg-slate-700/50 transition-colors">
                       <input type="file" accept=".csv" onChange={(e) => setUnivCSV(e.target.files ? e.target.files[0] : null)} className="w-full text-slate-400 text-sm" required />
                    </div>
                    <p className="text-xs text-slate-500">*Pastikan header kolom excel sesuai template: Universitas Rekanan, Negara, Jenis (SE/SA), Kuota per batch, Biaya studi (1 semester), Historis Biaya Akomodasi / mhs.</p>
                    <button type="submit" disabled={loading || !univCSV} className="w-full bg-slate-700 hover:bg-slate-600 border border-slate-500 text-white font-bold py-2 rounded">Unggah dan Gabungkan Data</button>
                 </form>
              </div>
           </div>
        )}

      </div>
    </div>
  );
}

"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

interface PlacementResult {
  summary: {
    total_mahasiswa: number;
    berhasil_ditempatkan: number;
    budget_limit: number;
    budget_terpakai: number;
    budget_sisa: number;
    alokasi_se: number;
    alokasi_sa: number;
  };
  placements: Array<{
    Student_ID: string;
    Nama: string;
    IPK: number;
    Universitas_Tujuan: string;
    Tipe_Program: string;
    Limit_Akomodasi: number;
    Total_Biaya: number;
    is_locked?: boolean;
  }>;
}

interface CancelRequest {
  student_id: string;
  name: string;
  program: string;
}

export default function AdminDashboardPage() {
  const router = useRouter();
  const [budget, setBudget] = useState('50000000');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PlacementResult | null>(null);
  const [cancelRequests, setCancelRequests] = useState<CancelRequest[]>([]);
  const [error, setError] = useState('');

  const fetchCancelRequests = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/cancel_requests');
      const data = await res.json();
      setCancelRequests(data);
    } catch(e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const token = localStorage.getItem('admin_token');
    if (token !== "admin-global-class-token") {
       router.push('/admin');
    } else {
       fetchCancelRequests();
    }
  }, [router]);

  const formatRupiah = (number: number) => {
    return new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(number);
  };

  const runOptimization = async () => {
    if (!window.confirm(`Apakah Anda yakin ingin menjalankan alokasi dengan budget Rp ${formatRupiah(parseInt(budget))} per mahasiswa? \nTindakan ini dapat memakan waktu beberapa detik.`)) {
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      // 1. Simpan Limit Budget ke Global State Backend
      await fetch('http://127.0.0.1:8000/api/system/budget', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ budget: parseInt(budget) })
      });

      // 2. Jalankan Optimisasi
      const res = await fetch(`http://127.0.0.1:8000/api/admin/optimize_placement?budget_per_mhs=${budget}`, {
        method: 'POST'
      });
      
      const data = await res.json();
      if (res.ok) {
        setResult(data.data);
        alert("Optimisasi alokasi berhasil dijalankan!");
      } else {
        setError("Gagal menjalankan algoritma: " + (data.detail || 'Terjadi kesalahan'));
      }
    } catch(e) {
      setError("Gagal terhubung ke Backend API.");
    }
    setLoading(false);
  };

  const logout = () => {
    localStorage.removeItem('admin_token');
    router.push('/admin');
  };

  const approveCancel = async (student_id: string) => {
    if(!window.confirm("Apakah Anda yakin ingin menyetujui pembatalan ini? Form preferensi mahasiswa akan direset.")) return;
    try {
      await fetch('http://127.0.0.1:8000/api/admin/approve_cancel', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ student_id })
      });
      alert("Pembatalan disetujui!");
      fetchCancelRequests();
    } catch(e) {}
  };

  const approveSinglePlacement = async (p: any) => {
    if(!window.confirm(`Kunci penempatan ${p.Nama} di ${p.Universitas_Tujuan}?`)) return;
    try {
      await fetch('http://127.0.0.1:8000/api/admin/approve_placement', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ student_id: p.Student_ID, univ_name: p.Universitas_Tujuan, total_cost: p.Total_Biaya })
      });
      alert(`Berhasil mengunci ${p.Nama}`);
      runOptimization(); // Refresh data
    } catch(e) {}
  };

  const approveAllPlacements = async () => {
    if(!result) return;
    if(!window.confirm("Kunci SELURUH penempatan di tabel ini secara permanen?")) return;
    try {
      const unlocked = result.placements.filter(p => !p.is_locked).map(p => ({
        student_id: p.Student_ID, univ_name: p.Universitas_Tujuan, total_cost: p.Total_Biaya
      }));
      await fetch('http://127.0.0.1:8000/api/admin/approve_all_placements', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ placements: unlocked })
      });
      alert("Seluruh penempatan berhasil dikunci!");
      runOptimization(); // Refresh data
    } catch(e) {}
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 p-8 font-sans">
      <div className="max-w-6xl mx-auto flex justify-between items-center mb-8 border-b border-slate-700 pb-4">
         <div>
           <h1 className="text-3xl font-bold text-white mb-2">Halaman Admin Global Class</h1>
           <p className="text-slate-400">Dashboard Eksekusi Algoritma Alokasi (Greedy Placement + AI)</p>
         </div>
         <div className="flex gap-4 items-center">
           <button onClick={() => router.push('/admin/data')} className="text-sm bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white font-medium transition-colors">
             Kelola Database
           </button>
           <button onClick={logout} className="text-sm text-red-400 hover:text-red-300 underline">Logout</button>
         </div>
      </div>

      {cancelRequests.length > 0 && (
        <div className="bg-orange-900/30 border border-orange-500/50 p-6 rounded-xl shadow-xl max-w-6xl mx-auto mb-8 animate-in fade-in slide-in-from-top-4">
          <h2 className="text-xl font-bold text-orange-400 mb-4 flex items-center gap-2">⚠️ Antrean Pengajuan Pembatalan</h2>
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-orange-950/50 text-orange-400 text-xs uppercase">
              <tr><th className="px-4 py-2">NIM</th><th className="px-4 py-2">Nama</th><th className="px-4 py-2">Program</th><th className="px-4 py-2 text-right">Aksi</th></tr>
            </thead>
            <tbody>
              {cancelRequests.map((req, i) => (
                <tr key={i} className="border-t border-orange-900/50 hover:bg-orange-900/20">
                  <td className="px-4 py-3">{req.student_id}</td><td className="px-4 py-3 font-bold">{req.name}</td><td className="px-4 py-3">{req.program}</td>
                  <td className="px-4 py-3 text-right">
                    <button onClick={() => approveCancel(req.student_id)} className="bg-orange-600 hover:bg-orange-500 text-white px-3 py-1 rounded shadow text-xs font-bold transition-colors">
                      Izinkan Batal
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl max-w-6xl mx-auto mb-8">
        <h2 className="text-xl font-bold text-white mb-4">Parameter Optimisasi</h2>
        <div className="flex flex-col md:flex-row gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Batas Budget Maximum per Mahasiswa (Rupiah)
            </label>
            <input
              type="number"
              className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
            />
          </div>
          <button 
            onClick={runOptimization}
            disabled={loading}
            className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-8 rounded-lg transition-colors flex items-center justify-center min-w-[200px] h-[50px]"
          >
            {loading ? 'Sedang Memproses AI...' : '⚡ Jalankan Alokasi'}
          </button>
        </div>
        {error && <p className="text-red-400 mt-4">{error}</p>}
      </div>

      {result && (
        <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
              <p className="text-sm text-slate-400 mb-1">Total Mahasiswa Diproses</p>
              <h3 className="text-3xl font-bold text-white">{result.summary.total_mahasiswa}</h3>
              <p className="text-xs text-green-400 mt-2">{result.summary.berhasil_ditempatkan} Berhasil Ditempatkan</p>
            </div>
            
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
              <p className="text-sm text-slate-400 mb-1">Total Limit Budget (Finance)</p>
              <h3 className="text-2xl font-bold text-blue-400">{formatRupiah(result.summary.budget_limit)}</h3>
            </div>
            
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
              <p className="text-sm text-slate-400 mb-1">Realisasi Budget Terpakai</p>
              <h3 className="text-2xl font-bold text-orange-400">{formatRupiah(result.summary.budget_terpakai)}</h3>
              <p className="text-xs text-green-400 mt-2">Penghematan: {formatRupiah(result.summary.budget_sisa)}</p>
            </div>
            
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
              <p className="text-sm text-slate-400 mb-1">Distribusi Jalur (SE vs SA)</p>
              <div className="flex gap-4 mt-2">
                <div>
                  <span className="text-2xl font-bold text-purple-400">{result.summary.alokasi_se}</span>
                  <span className="text-xs text-slate-500 ml-1">SE</span>
                </div>
                <div>
                  <span className="text-2xl font-bold text-emerald-400">{result.summary.alokasi_sa}</span>
                  <span className="text-xs text-slate-500 ml-1">SA</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-xl">
             <div className="p-4 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
               <div>
                  <h3 className="font-bold text-white">Daftar Hasil Penempatan Final Mahasiswa</h3>
                  <p className="text-xs text-slate-400">Tinjau draft algoritma di bawah ini dan kunci pilihan mereka.</p>
               </div>
               <button onClick={approveAllPlacements} className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded shadow transition-colors">
                 Approve Semua Draft
               </button>
             </div>
             <div className="overflow-x-auto">
               <table className="w-full text-left text-sm text-slate-300">
                 <thead className="bg-slate-900/50 text-slate-400 text-xs uppercase">
                   <tr>
                     <th className="px-6 py-3">Student</th>
                     <th className="px-6 py-3">IPK</th>
                     <th className="px-6 py-3">Penempatan Tujuan</th>
                     <th className="px-6 py-3">Jalur</th>
                     <th className="px-6 py-3">Limit Akomodasi (AI)</th>
                     <th className="px-6 py-3">Total Cost per Mhs</th>
                     <th className="px-6 py-3">Aksi</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-slate-700/50">
                   {result.placements.map((p, i) => (
                     <tr key={i} className={`hover:bg-slate-750 transition-colors ${p.is_locked ? 'bg-blue-900/10' : ''}`}>
                       <td className="px-6 py-4 font-medium text-white">{p.Nama} <br/><span className="text-xs text-slate-500">{p.Student_ID}</span></td>
                       <td className="px-6 py-4"><span className="bg-slate-700 px-2 py-1 rounded text-blue-400">{p.IPK}</span></td>
                       <td className="px-6 py-4 font-bold">{p.Universitas_Tujuan}</td>
                       <td className="px-6 py-4">
                         <span className={`px-2 py-1 rounded-full text-xs font-bold ${p.Tipe_Program === 'SE' ? 'bg-purple-500/20 text-purple-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                           {p.Tipe_Program}
                         </span>
                       </td>
                       <td className="px-6 py-4 text-orange-400">{formatRupiah(p.Limit_Akomodasi)}</td>
                       <td className="px-6 py-4 font-bold text-blue-400">{formatRupiah(p.Total_Biaya)}</td>
                       <td className="px-6 py-4">
                         {p.is_locked || p.Universitas_Tujuan === "TIDAK DITEMPATKAN" ? (
                           <span className="text-xs font-bold text-slate-500">{p.is_locked ? '🔒 TERKUNCI' : '-'}</span>
                         ) : (
                           <button onClick={() => approveSinglePlacement(p)} className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 rounded shadow text-xs font-bold">
                             Approve
                           </button>
                         )}
                       </td>
                     </tr>
                   ))}
                 </tbody>
               </table>
             </div>
          </div>
          
        </div>
      )}
    </div>
  );
}

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
  }>;
}

export default function AdminDashboardPage() {
  const router = useRouter();
  const [budget, setBudget] = useState('50000000');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PlacementResult | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('admin_token');
    if (token !== "admin-global-class-token") {
       router.push('/admin');
    }
  }, [router]);

  const formatRupiah = (number: number) => {
    return new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(number);
  };

  const runOptimization = async () => {
    setLoading(true);
    setError('');
    
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/optimize_placement?budget_per_mhs=${budget}`, {
        method: 'POST'
      });
      
      const data = await res.json();
      if (res.ok) {
        setResult(data.data);
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
                  <p className="text-xs text-slate-400">Diurutkan berdasarkan skor IPK untuk perebutan kursi SE.</p>
               </div>
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
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-slate-700/50">
                   {result.placements.map((p, i) => (
                     <tr key={i} className="hover:bg-slate-750 transition-colors">
                       <td className="px-6 py-4 font-medium text-white">{p.Nama} <br/><span className="text-xs text-slate-500">{p.Student_ID}</span></td>
                       <td className="px-6 py-4"><span className="bg-slate-700 px-2 py-1 rounded text-blue-400">{p.IPK}</span></td>
                       <td className="px-6 py-4">{p.Universitas_Tujuan}</td>
                       <td className="px-6 py-4">
                         <span className={`px-2 py-1 rounded-full text-xs font-bold ${p.Tipe_Program === 'SE' ? 'bg-purple-500/20 text-purple-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                           {p.Tipe_Program}
                         </span>
                       </td>
                       <td className="px-6 py-4 text-orange-400">{formatRupiah(p.Limit_Akomodasi)}</td>
                       <td className="px-6 py-4 font-bold text-blue-400">{formatRupiah(p.Total_Biaya)}</td>
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

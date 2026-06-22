"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

interface StudentData {
  student_id: string;
  name: string;
  program: string;
  gpa: number;
  ielts: number;
  status_form: string;
}

interface UniversityData {
  id: number;
  name: string;
  country: string;
  type: string;
  quota: number;
  tuition_fee: number;
  predicted_accomodation: number;
  estimated_total_cost: number;
}

export default function StudentDashboard() {
  const router = useRouter();
  const [student, setStudent] = useState<StudentData | null>(null);
  const [universities, setUniversities] = useState<UniversityData[]>([]);
  const [loading, setLoading] = useState(true);

  // State untuk form preferensi
  const [budgetLimit, setBudgetLimit] = useState(50000000);
  const [pref1, setPref1] = useState('');
  const [pref2, setPref2] = useState('');
  const [pref3, setPref3] = useState('');
  const [submitStatus, setSubmitStatus] = useState({ loading: false, msg: '', type: '' });

  useEffect(() => {
    // Ambil data student dari local storage saat render
    const data = localStorage.getItem('student_data');
    if (!data) {
      router.push('/');
      return;
    }

    const parsedData = JSON.parse(data);
    setStudent(parsedData);

    // Ambil batas budget dari backend
    fetch('http://127.0.0.1:8000/api/system/budget')
      .then(res => res.json())
      .then(data => setBudgetLimit(data.budget_limit))
      .catch(err => console.error("Gagal mengambil setting budget", err));

    // Ambil rekomendasi universitas berdasarkan jurusan mereka
    fetch(`http://127.0.0.1:8000/api/universities/recommend?program=${parsedData.program}`)
      .then(res => res.json())
      .then(data => {
        setUniversities(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Gagal mengambil data rekomendasi:", err);
        setLoading(false);
      });
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pref1 || !pref2 || !pref3) {
      setSubmitStatus({ loading: false, msg: 'Pilih semua 3 preferensi!', type: 'error' });
      return;
    }
    if (new Set([pref1, pref2, pref3]).size < 3) {
      setSubmitStatus({ loading: false, msg: 'Pilihan preferensi tidak boleh sama!', type: 'error' });
      return;
    }

    if (!window.confirm("Apakah Anda yakin dengan 3 pilihan universitas ini? Pilihan yang sudah disimpan akan diproses oleh Admin.")) {
      return;
    }

    setSubmitStatus({ loading: true, msg: '', type: '' });
    try {
      const res = await fetch('http://127.0.0.1:8000/api/student/submit_preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id: student.student_id,
          pref_1: pref1,
          pref_2: pref2,
          pref_3: pref3
        })
      });

      if (res.ok) {
        setSubmitStatus({ loading: false, msg: 'Berhasil menyimpan preferensi Anda!', type: 'success' });
        setStudent({ ...student, status_form: 'Sudah Mengisi' });
        alert("Berhasil menyimpan preferensi kampus! Semoga Anda lolos seleksi.");
      } else {
        setSubmitStatus({ loading: false, msg: 'Gagal mengirim data!', type: 'error' });
      }
    } catch (e) {
      setSubmitStatus({ loading: false, msg: 'Server Backend mati.', type: 'error' });
    }
  };

  const formatRupiah = (number: number) => {
    return new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(number);
  };

  if (loading || !student) {
    return <div className="min-h-screen bg-slate-900 flex items-center justify-center text-white">Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans pb-10">
      {/* Header Profile */}
      <header className="bg-slate-800 border-b border-slate-700 p-6 sticky top-0 z-10 shadow-lg">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Selamat Datang, {student.name}</h1>
            <p className="text-slate-400 text-sm">ID: {student.student_id} • Jurusan: {student.program}</p>
          </div>
          <div className="flex gap-4">
            <div className="bg-slate-700 px-4 py-2 rounded-lg text-center">
              <span className="block text-xs uppercase tracking-wider text-slate-400">IPK (GPA)</span>
              <span className="font-bold text-blue-400 text-lg">{student.gpa}</span>
            </div>
            <div className="bg-slate-700 px-4 py-2 rounded-lg text-center">
              <span className="block text-xs uppercase tracking-wider text-slate-400">IELTS Score</span>
              <span className="font-bold text-green-400 text-lg">{student.ielts}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-6 mt-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Kolom Kiri: AI Recommendation Cards */}
          <div className="lg:col-span-2">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <span>✨</span> Rekomendasi Kampus Cerdas
            </h2>
            <p className="text-sm text-slate-400 mb-6">
              AI kami telah memfilter universitas yang cocok dengan jurusan Anda ({student.program}) dan memprediksi total biaya akomodasinya untuk membantu keputusan Anda.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {universities.map((u, i) => (
                <div key={i} className="bg-slate-800 rounded-xl overflow-hidden shadow-lg border border-slate-700 hover:border-blue-500 transition-colors p-5 relative group">
                  <div className={`absolute top-0 right-0 px-3 py-1 text-xs font-bold rounded-bl-lg ` + (u.type === 'SE' ? 'bg-purple-600 text-white' : 'bg-emerald-600 text-white')}>
                    Jalur {u.type}
                  </div>
                  <h3 className="text-lg font-bold text-white mb-1 pr-12">{u.name}</h3>
                  <p className="text-slate-400 text-sm mb-4">📍 {u.country}</p>

                  <div className="space-y-2 mt-4 pt-4 border-t border-slate-700/50">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Biaya Studi:</span>
                      <span>{u.tuition_fee === 0 ? 'Gratis' : formatRupiah(u.tuition_fee)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Estimasi Akomodasi:</span>
                      <span className="text-orange-400">{formatRupiah(u.predicted_accomodation)}</span>
                    </div>
                    <div className="flex justify-between text-sm font-bold pt-2 border-t border-slate-700/50">
                      <span className="text-white">Total Prediksi:</span>
                      <span className="text-blue-400">{formatRupiah(u.estimated_total_cost)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Kolom Kanan: Form Pemilihan Preferensi */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800 rounded-xl shadow-lg border border-slate-700 p-6 sticky top-28">
              <h2 className="text-xl font-bold text-white mb-6">Status Penempatan</h2>

              {/* JIKA SUDAH DI-APPROVE ADMIN (DIKUNCI) */}
              {student.allocated_univ ? (
                <div className="bg-gradient-to-b from-blue-900/50 to-blue-800/20 border border-blue-500/50 rounded-xl p-6 text-center shadow-2xl animate-in fade-in slide-in-from-bottom-4">
                  <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 border border-blue-400">
                    <span className="text-3xl">🎉</span>
                  </div>
                  <h3 className="text-green-400 font-bold mb-2">SELAMAT!</h3>
                  <p className="text-slate-300 text-sm mb-4">Anda telah resmi lolos seleksi dan ditempatkan oleh Global Class di:</p>
                  <div className="bg-slate-900/50 p-4 rounded-lg border border-blue-500/30">
                     <span className="block font-bold text-blue-400 text-lg">{student.allocated_univ}</span>
                     <span className="text-xs text-slate-500 mt-1 block">(Status Terkunci Permanen)</span>
                  </div>
                  <button type="button" onClick={() => { localStorage.clear(); router.push('/'); }} className="cursor-pointer w-full mt-6 bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded transition-colors">
                    Keluar (Log Out)
                  </button>
                </div>
              ) : 
              
              /* JIKA SEDANG MENGAJUKAN BATAL (CANCEL REQUEST) */
              student.cancel_request ? (
                <div className="bg-orange-500/10 border border-orange-500/50 rounded-xl p-6 text-center">
                  <div className="text-4xl mb-4 animate-pulse">⏳</div>
                  <h3 className="text-orange-400 font-bold mb-2">Menunggu Persetujuan Admin</h3>
                  <p className="text-slate-300 text-sm mb-4">Anda telah mengajukan pembatalan pilihan kampus. Silakan tunggu Admin Global Class untuk mereset data Anda sebelum Anda bisa memilih kembali.</p>
                  <button type="button" onClick={() => { localStorage.clear(); router.push('/'); }} className="cursor-pointer w-full mt-2 text-sm text-slate-400 hover:text-white transition-colors">
                    Keluar (Log Out)
                  </button>
                </div>
              ) :
              
              /* JIKA SUDAH MENGISI TAPI BELUM DIKUNCI / BELUM BATAL */
              student.status_form === 'Sudah Mengisi' ? (
                <div>
                  <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 mb-6">
                    <p className="text-green-400 text-sm font-medium text-center">✅ Preferensi Anda sudah direkam dan sedang dalam antrean pemrosesan Algoritma.</p>
                  </div>
                  <button 
                    onClick={async () => {
                      if(window.confirm("Apakah Anda yakin ingin membatalkan pilihan? Anda harus menunggu Admin mereset data Anda.")) {
                         await fetch('http://127.0.0.1:8000/api/student/request_cancel', {
                            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({student_id: student.student_id})
                         });
                         setStudent({...student, cancel_request: true});
                         alert("Pengajuan pembatalan terkirim!");
                      }
                    }} 
                    className="cursor-pointer w-full bg-slate-700 hover:bg-red-900/50 hover:text-red-400 border border-slate-600 text-slate-300 font-bold py-3 px-4 rounded-lg transition-all"
                  >
                    Ajukan Pembatalan Pilihan
                  </button>
                  <button type="button" onClick={() => { localStorage.clear(); router.push('/'); }} className="cursor-pointer w-full mt-4 text-sm text-slate-400 hover:text-white transition-colors">
                    Keluar (Log Out)
                  </button>
                </div>
              ) : 
              
              /* JIKA BELUM MENGISI SAMA SEKALI (FORM NORMAL) */
              (
              <form onSubmit={handleSubmit} className="space-y-5">
                <p className="text-xs text-blue-400 bg-blue-900/20 p-3 rounded border border-blue-500/30">
                  *Sistem Kompetisi: Anda <b>bebas memilih kampus melebihi limit subsidi ({formatRupiah(budgetLimit)})</b>. Namun jika banyak yang memilih kampus mahal, sistem akan memprioritaskan mahasiswa dengan IPK tertinggi. IPK pas-pasan mungkin akan dilempar ke preferensi 2 atau 3 Anda.
                </p>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Preferensi 1 (Paling Diidamkan)</label>
                  <select value={pref1} onChange={e => setPref1(e.target.value)} required className="w-full bg-slate-700 border border-slate-600 text-white rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="">-- Pilih Universitas --</option>
                    {universities.map((u, i) => (
                      <option key={i} value={u.name}>
                        {u.name} (Jalur {u.type}) {u.estimated_total_cost > budgetLimit ? '⚠️ Mahal' : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Preferensi 2</label>
                  <select value={pref2} onChange={e => setPref2(e.target.value)} required className="w-full bg-slate-700 border border-slate-600 text-white rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="">-- Pilih Universitas --</option>
                    {universities.map((u, i) => (
                      <option key={i} value={u.name}>
                        {u.name} (Jalur {u.type}) {u.estimated_total_cost > budgetLimit ? '⚠️ Mahal' : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Preferensi 3</label>
                  <select value={pref3} onChange={e => setPref3(e.target.value)} required className="w-full bg-slate-700 border border-slate-600 text-white rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="">-- Pilih Universitas --</option>
                    {universities.map((u, i) => (
                      <option key={i} value={u.name}>
                        {u.name} (Jalur {u.type}) {u.estimated_total_cost > budgetLimit ? '⚠️ Mahal' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                {submitStatus.msg && (
                  <div className={`p-3 rounded-lg text-sm border ${submitStatus.type === 'error' ? 'bg-red-500/10 border-red-500/50 text-red-400' : 'bg-green-500/10 border-green-500/50 text-green-400'}`}>
                    {submitStatus.msg}
                  </div>
                )}

                <button type="submit" disabled={submitStatus.loading} className="cursor-pointer w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors">
                  {submitStatus.loading ? 'Menyimpan...' : 'Simpan Preferensi'}
                </button>
                <button type="button" onClick={() => { localStorage.clear(); router.push('/'); }} className="cursor-pointer w-full mt-2 text-sm text-slate-400 hover:text-white transition-colors">
                  Keluar (Log Out)
                </button>
              </form>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

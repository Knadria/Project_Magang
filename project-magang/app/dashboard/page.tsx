"use client";
import { useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';

interface StudentData {
  student_id: string;
  name: string;
  program: string;
  gpa: number;
  ielts: number;
  status_form: string;
  allocated_univ?: string | null;
  cancel_request?: boolean;
}

interface UniversityData {
  id: number;
  name: string;
  country: string;
  type: string;
  quota: number;
  programs: string;
  tuition_fee: number;
  predicted_accomodation: number;
  estimated_total_cost: number;
  ai_score: number;
}

export default function StudentDashboard() {
  const router = useRouter();
  const [student, setStudent] = useState<StudentData | null>(null);
  const [universities, setUniversities] = useState<UniversityData[]>([]);
  const [loading, setLoading] = useState(true);
  const [openCountries, setOpenCountries] = useState<Record<string, boolean>>({});

  const [pref1, setPref1] = useState('');
  const [pref2, setPref2] = useState('');
  const [pref3, setPref3] = useState('');
  const [submitStatus, setSubmitStatus] = useState({ loading: false, msg: '', type: '' });

  useEffect(() => {
    const data = localStorage.getItem('student_data');
    if (!data) { router.push('/'); return; }

    const parsedData: StudentData = JSON.parse(data);
    setStudent(parsedData);

    fetch(`http://127.0.0.1:8000/api/universities/recommend?program=${parsedData.program}&gpa=${parsedData.gpa}&ielts=${parsedData.ielts}`)
      .then(res => res.json())
      .then(data => {
        setUniversities(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [router]);

  // 4 rekomendasi AI teratas: urutkan berdasarkan ai_score tertinggi
  const aiRecommendations = useMemo(() => {
    return [...universities]
      .sort((a, b) => b.ai_score - a.ai_score)
      .slice(0, 4);
  }, [universities]);

  // Group semua universitas berdasarkan negara
  const byCountry = useMemo(() => {
    const groups: Record<string, UniversityData[]> = {};
    universities.forEach(u => {
      if (!groups[u.country]) groups[u.country] = [];
      groups[u.country].push(u);
    });
    // Sort negara A-Z
    return Object.fromEntries(
      Object.entries(groups).sort(([a], [b]) => a.localeCompare(b))
    );
  }, [universities]);

  const toggleCountry = (country: string) => {
    setOpenCountries(prev => ({ ...prev, [country]: !prev[country] }));
  };

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
    if (!window.confirm("Apakah Anda yakin dengan 3 pilihan universitas ini? Pilihan yang sudah disimpan akan diproses oleh Admin.")) return;

    setSubmitStatus({ loading: true, msg: '', type: '' });
    try {
      const res = await fetch('http://127.0.0.1:8000/api/student/submit_preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: student!.student_id, pref_1: pref1, pref_2: pref2, pref_3: pref3 })
      });
      if (res.ok) {
        setSubmitStatus({ loading: false, msg: 'Berhasil menyimpan preferensi Anda!', type: 'success' });
        setStudent(prev => prev ? { ...prev, status_form: 'Sudah Mengisi' } : prev);
        alert("Berhasil menyimpan preferensi kampus! Semoga Anda lolos seleksi.");
      } else {
        setSubmitStatus({ loading: false, msg: 'Gagal mengirim data!', type: 'error' });
      }
    } catch {
      setSubmitStatus({ loading: false, msg: 'Server Backend mati.', type: 'error' });
    }
  };

  if (loading || !student) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400">Memuat data kampus...</p>
        </div>
      </div>
    );
  }

  const aiMatchPercent = (score: number) => Math.round(score * 100);

  // Semua opsi universitas untuk dropdown preferensi (tanpa info biaya/jalur)
  const allUnivOptions = [...universities].sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans pb-16">

      {/* ─── Header ─── */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-5 sticky top-0 z-10 shadow-lg">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Selamat Datang, {student.name}</h1>
            <p className="text-slate-400 text-sm mt-0.5">ID: {student.student_id} &nbsp;•&nbsp; Jurusan: {student.program}</p>
          </div>
          <div className="flex gap-3">
            <div className="bg-slate-700 px-4 py-2 rounded-xl text-center min-w-[80px]">
              <span className="block text-xs uppercase tracking-wider text-slate-400 mb-0.5">IPK</span>
              <span className="font-bold text-blue-400 text-xl">{student.gpa.toFixed(2)}</span>
            </div>
            <div className="bg-slate-700 px-4 py-2 rounded-xl text-center min-w-[80px]">
              <span className="block text-xs uppercase tracking-wider text-slate-400 mb-0.5">IELTS</span>
              <span className="font-bold text-green-400 text-xl">{student.ielts}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 mt-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* ─── Kolom Kiri: Daftar Kampus ─── */}
          <div className="lg:col-span-2 space-y-8">

            {/* === SECTION 1: AI Recommendations === */}
            <section>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">✨</span>
                <div>
                  <h2 className="text-xl font-bold text-white">Rekomendasi Kampus Cerdas</h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    4 kampus terbaik dipilih AI berdasarkan IPK ({student.gpa}) dan IELTS ({student.ielts}) Anda
                  </p>
                </div>
              </div>

              {aiRecommendations.length === 0 ? (
                <p className="text-slate-500 text-sm">Tidak ada kampus yang cocok dengan jurusan Anda.</p>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                  {aiRecommendations.map((u, i) => {
                    const matchPct = aiMatchPercent(u.ai_score);
                    const matchColor =
                      matchPct >= 75 ? 'text-green-400' :
                      matchPct >= 50 ? 'text-yellow-400' : 'text-orange-400';
                    const barColor =
                      matchPct >= 75 ? 'bg-green-500' :
                      matchPct >= 50 ? 'bg-yellow-500' : 'bg-orange-500';
                    return (
                      <div key={u.id}
                        className="bg-slate-800 rounded-2xl border border-slate-700 hover:border-blue-500/60 transition-all p-5 relative overflow-hidden group shadow-xl">
                        {/* Rank badge */}
                        <div className="absolute top-3 left-3 w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs font-black text-white shadow">
                          #{i + 1}
                        </div>

                        <div className="mt-5">
                          <h3 className="font-bold text-white text-base leading-tight mb-1">{u.name}</h3>
                          <p className="text-slate-400 text-sm mb-4">📍 {u.country}</p>

                          {/* AI Match Bar */}
                          <div className="mb-4">
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-xs text-slate-400">Kecocokan AI</span>
                              <span className={`text-sm font-bold ${matchColor}`}>{matchPct}%</span>
                            </div>
                            <div className="w-full bg-slate-700 rounded-full h-1.5">
                              <div className={`h-1.5 rounded-full transition-all ${barColor}`} style={{ width: `${matchPct}%` }}></div>
                            </div>
                          </div>

                          <div className="text-xs text-slate-500 pt-3 border-t border-slate-700/60 flex items-center gap-1">
                            <span>🎓</span>
                            <span>Kuota {u.quota} mahasiswa</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            {/* === SECTION 2: Semua Kampus per Negara === */}
            <section>
              <div className="flex items-center gap-3 mb-4">
                <span className="text-2xl">🌏</span>
                <div>
                  <h2 className="text-xl font-bold text-white">Semua Kampus Rekanan</h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {universities.length} kampus tersedia · dikelompokkan per negara
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                {Object.entries(byCountry).map(([country, univList]) => (
                  <div key={country} className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                    {/* Dropdown Header */}
                    <button
                      onClick={() => toggleCountry(country)}
                      className="w-full flex items-center justify-between px-5 py-4 hover:bg-slate-700/40 transition-colors cursor-pointer"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-bold text-white">{country}</span>
                        <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded-full">
                          {univList.length} kampus
                        </span>
                      </div>
                      <span className={`text-slate-400 text-lg transition-transform duration-300 ${openCountries[country] ? 'rotate-180' : ''}`}>▼</span>
                    </button>

                    {/* Dropdown Content */}
                    {openCountries[country] && (
                      <div className="border-t border-slate-700 divide-y divide-slate-700/60">
                        {univList.map(u => (
                          <div key={u.id} className="px-5 py-3 flex items-center justify-between hover:bg-slate-700/20 transition-colors">
                            <div>
                              <p className="font-semibold text-white text-sm">{u.name}</p>
                              <p className="text-xs text-slate-400 mt-0.5">Kuota: {u.quota} mahasiswa</p>
                            </div>
                            <div className="text-right shrink-0 ml-4">
                              {aiRecommendations.some(r => r.id === u.id) && (
                                <span className="inline-block mb-1 text-xs bg-blue-600/20 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-full">✨ AI Pick</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}

                {Object.keys(byCountry).length === 0 && (
                  <div className="text-center py-12 text-slate-500">
                    Belum ada kampus rekanan yang tersedia untuk jurusan Anda.
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* ─── Kolom Kanan: Status & Form Preferensi ─── */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800 rounded-2xl shadow-xl border border-slate-700 p-6 sticky top-28">
              <h2 className="text-xl font-bold text-white mb-5">Status Penempatan</h2>

              {/* SUDAH DI-APPROVE */}
              {student.allocated_univ ? (
                <div className="bg-gradient-to-b from-blue-900/50 to-blue-800/20 border border-blue-500/50 rounded-xl p-6 text-center shadow-2xl">
                  <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 border border-blue-400">
                    <span className="text-3xl">🎉</span>
                  </div>
                  <h3 className="text-green-400 font-bold mb-2">SELAMAT!</h3>
                  <p className="text-slate-300 text-sm mb-4">Anda telah resmi lolos seleksi dan ditempatkan di:</p>
                  <div className="bg-slate-900/50 p-4 rounded-lg border border-blue-500/30">
                    <span className="block font-bold text-blue-400 text-base">{student.allocated_univ}</span>
                    <span className="text-xs text-slate-500 mt-1 block">Status Terkunci Permanen</span>
                  </div>
                  <button onClick={() => { localStorage.clear(); router.push('/'); }}
                    className="cursor-pointer w-full mt-5 bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 px-4 rounded-lg transition-colors text-sm">
                    Keluar (Log Out)
                  </button>
                </div>

              /* MENUNGGU PERSETUJUAN BATAL */
              ) : student.cancel_request ? (
                <div className="bg-orange-500/10 border border-orange-500/50 rounded-xl p-6 text-center">
                  <div className="text-4xl mb-4 animate-pulse">⏳</div>
                  <h3 className="text-orange-400 font-bold mb-2">Menunggu Persetujuan Admin</h3>
                  <p className="text-slate-300 text-sm mb-4">Anda telah mengajukan pembatalan. Tunggu Admin Global Class mereset data Anda.</p>
                  <button onClick={() => { localStorage.clear(); router.push('/'); }}
                    className="cursor-pointer w-full mt-2 text-sm text-slate-400 hover:text-white transition-colors">
                    Keluar (Log Out)
                  </button>
                </div>

              /* SUDAH ISI, BELUM DIKUNCI */
              ) : student.status_form === 'Sudah Mengisi' ? (
                <div>
                  <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 mb-5">
                    <p className="text-green-400 text-sm font-medium text-center">
                      ✅ Preferensi Anda sudah direkam dan sedang dalam antrean pemrosesan.
                    </p>
                  </div>
                  <button
                    onClick={async () => {
                      if (window.confirm("Apakah Anda yakin ingin membatalkan pilihan? Anda harus menunggu Admin mereset data Anda.")) {
                        await fetch('http://127.0.0.1:8000/api/student/request_cancel', {
                          method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ student_id: student.student_id })
                        });
                        setStudent(prev => prev ? { ...prev, cancel_request: true } : prev);
                        alert("Pengajuan pembatalan terkirim!");
                      }
                    }}
                    className="cursor-pointer w-full bg-slate-700 hover:bg-red-900/50 hover:text-red-400 border border-slate-600 text-slate-300 font-bold py-3 px-4 rounded-lg transition-all">
                    Ajukan Pembatalan Pilihan
                  </button>
                  <button onClick={() => { localStorage.clear(); router.push('/'); }}
                    className="cursor-pointer w-full mt-3 text-sm text-slate-400 hover:text-white transition-colors">
                    Keluar (Log Out)
                  </button>
                </div>

              /* BELUM MENGISI — FORM PREFERENSI */
              ) : (
                <form onSubmit={handleSubmit} className="space-y-5">
                  <p className="text-xs text-blue-400 bg-blue-900/20 p-3 rounded-lg border border-blue-500/30 leading-relaxed">
                    💡 Sistem kompetisi: Jika banyak mahasiswa memilih kampus yang sama, sistem akan memprioritaskan mahasiswa dengan IPK tertinggi. Pilihan bisa bergeser ke preferensi 2 atau 3.
                  </p>

                  {/* Preferensi 1 */}
                  <div>
                    <label className="block text-sm font-semibold text-slate-300 mb-2">
                      Preferensi 1 <span className="text-blue-400 font-normal">(Paling Diidamkan)</span>
                    </label>
                    <select
                      value={pref1}
                      onChange={e => setPref1(e.target.value)}
                      required
                      className="w-full bg-slate-700 border border-slate-600 text-white rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:outline-none text-sm"
                    >
                      <option value="">-- Pilih Universitas --</option>
                      {allUnivOptions.map(u => (
                        <option key={u.id} value={u.name}>{u.name}</option>
                      ))}
                    </select>
                  </div>

                  {/* Preferensi 2 */}
                  <div>
                    <label className="block text-sm font-semibold text-slate-300 mb-2">Preferensi 2</label>
                    <select
                      value={pref2}
                      onChange={e => setPref2(e.target.value)}
                      required
                      className="w-full bg-slate-700 border border-slate-600 text-white rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:outline-none text-sm"
                    >
                      <option value="">-- Pilih Universitas --</option>
                      {allUnivOptions.map(u => (
                        <option key={u.id} value={u.name}>{u.name}</option>
                      ))}
                    </select>
                  </div>

                  {/* Preferensi 3 */}
                  <div>
                    <label className="block text-sm font-semibold text-slate-300 mb-2">Preferensi 3</label>
                    <select
                      value={pref3}
                      onChange={e => setPref3(e.target.value)}
                      required
                      className="w-full bg-slate-700 border border-slate-600 text-white rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:outline-none text-sm"
                    >
                      <option value="">-- Pilih Universitas --</option>
                      {allUnivOptions.map(u => (
                        <option key={u.id} value={u.name}>{u.name}</option>
                      ))}
                    </select>
                  </div>

                  {submitStatus.msg && (
                    <div className={`p-3 rounded-lg text-sm border ${submitStatus.type === 'error' ? 'bg-red-500/10 border-red-500/50 text-red-400' : 'bg-green-500/10 border-green-500/50 text-green-400'}`}>
                      {submitStatus.msg}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={submitStatus.loading}
                    className="cursor-pointer w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:opacity-60"
                  >
                    {submitStatus.loading ? 'Menyimpan...' : 'Simpan Preferensi'}
                  </button>
                  <button
                    type="button"
                    onClick={() => { localStorage.clear(); router.push('/'); }}
                    className="cursor-pointer w-full text-sm text-slate-400 hover:text-white transition-colors"
                  >
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

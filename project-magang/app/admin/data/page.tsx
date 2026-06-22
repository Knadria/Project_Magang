"use client";
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

interface StudentRecord {
  student_id: string;
  name: string;
  program: string;
  gpa: number;
  ielts: number;
  pref_1?: string | null;
  pref_2?: string | null;
  pref_3?: string | null;
  allocated_univ?: string | null;
  allocated_cost?: number;
  cancel_request: boolean;
}

interface UniversityRecord {
  id: number;
  name: string;
  country: string;
  programs: string;
  type: string;
  quota: number;
  tuition_fee: number;
  historical_accomodation: number;
}

export default function AdminDataManagement() {
  const router = useRouter();
  const [tab, setTab] = useState<'student' | 'university'>('student');

  const [msg, setMsg] = useState({ text: '', type: '' });
  const [loading, setLoading] = useState(false);

  // Student form state
  const [sId, setSId] = useState('');
  const [sName, setSName] = useState('');
  const [sProg, setSProg] = useState('CS');
  const [sGpa, setSGpa] = useState('');
  const [sIelts, setSielts] = useState('');
  const [studentCSV, setStudentCSV] = useState<File | null>(null);
  const [editingStudentId, setEditingStudentId] = useState<string | null>(null);
  const [editingStudentAllocated, setEditingStudentAllocated] = useState<string | null>(null);
  const [students, setStudents] = useState<StudentRecord[]>([]);
  const [showStudentList, setShowStudentList] = useState(false);

  // University form state
  const [uId, setUId] = useState<number | null>(null);
  const [uName, setUName] = useState('');
  const [uCountry, setUCountry] = useState('');
  const [uProgs, setUProgs] = useState('CS');
  const [uType, setUType] = useState('SE');
  const [uQuota, setUQuota] = useState('');
  const [uFee, setUFee] = useState('');
  const [uAcc, setUAcc] = useState('');
  const [univCSV, setUnivCSV] = useState<File | null>(null);
  const [editingUniversityId, setEditingUniversityId] = useState<number | null>(null);
  const [universities, setUniversities] = useState<UniversityRecord[]>([]);
  const [showUniversityList, setShowUniversityList] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('admin_token');
    if (token !== 'admin-global-class-token') {
      router.push('/admin');
      return;
    }
    fetchAdminData();
  }, [router]);

  const fetchAdminData = async () => {
    await Promise.all([fetchStudents(), fetchUniversities()]);
  };

  const fetchStudents = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/students");
      if (res.ok) {
        const data = await res.json();
        setStudents(data);
      }
    } catch (error) {
      console.error(error);
    }
  };

  const fetchUniversities = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/universities");
      if (res.ok) {
        const data = await res.json();
        setUniversities(data);
      }
    } catch (error) {
      console.error(error);
    }
  };

  const clearStudentForm = () => {
    setSId('');
    setSName('');
    setSProg('CS');
    setSGpa('');
    setSielts('');
    setEditingStudentId(null);
    setEditingStudentAllocated(null);
  };

  const clearUniversityForm = () => {
    setUId(null);
    setUName('');
    setUCountry('');
    setUProgs('CS');
    setUType('SE');
    setUQuota('');
    setUFee('');
    setUAcc('');
    setEditingUniversityId(null);
  };

  const showMessage = (text: string, type: 'success' | 'error') => {
    setMsg({ text, type });
  };

  const handleSaveStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const payload = {
        name: sName,
        program: sProg,
        gpa: parseFloat(sGpa),
        ielts: parseFloat(sIelts),
      };

      let res;
      if (editingStudentId) {
        res = await fetch(`http://127.0.0.1:8000/api/admin/students/${editingStudentId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else {
        res = await fetch(`http://127.0.0.1:8000/api/admin/add_student`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ student_id: sId, ...payload }),
        });
      }

      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        clearStudentForm();
        fetchStudents();
      } else {
        showMessage(data.detail || 'Terjadi kesalahan saat menyimpan data mahasiswa.', 'error');
      }
    } catch (error) {
      showMessage('Koneksi ke backend gagal', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentCSV) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', studentCSV);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/upload_students', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        setStudentCSV(null);
        fetchStudents();
      } else {
        showMessage(data.detail || 'Gagal mengupload file.', 'error');
      }
    } catch (error) {
      showMessage('Gagal mengupload file.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleEditStudent = (student: StudentRecord) => {
    setEditingStudentId(student.student_id);
    setEditingStudentAllocated(student.allocated_univ || null);
    setSId(student.student_id);
    setSName(student.name);
    setSProg(student.program);
    setSGpa(String(student.gpa));
    setSielts(String(student.ielts));
    setTab('student');
    setMsg({ text: '', type: '' });
    // scroll ke atas form
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleDeleteStudent = async (student_id: string) => {
    if (!window.confirm('Hapus data mahasiswa ini?')) return;
    setLoading(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/students/${student_id}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        fetchStudents();
      } else {
        showMessage(data.detail || 'Gagal menghapus mahasiswa.', 'error');
      }
    } catch (error) {
      showMessage('Koneksi ke backend gagal', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleResetPlacement = async (student_id: string, name: string) => {
    if (!window.confirm(`Batalkan alokasi penempatan untuk ${name}?\nMahasiswa akan bisa dialokasikan ulang.`)) return;
    setLoading(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/reset_placement/${student_id}`, {
        method: 'POST',
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        setEditingStudentAllocated(null);
        fetchStudents();
      } else {
        showMessage(data.detail || 'Gagal mereset alokasi.', 'error');
      }
    } catch (error) {
      showMessage('Koneksi ke backend gagal', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleManualUniv = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const payload = {
        name: uName,
        country: uCountry,
        programs: uProgs,
        type: uType,
        quota: parseInt(uQuota),
        tuition_fee: parseFloat(uFee),
        historical_accomodation: parseFloat(uAcc),
      };

      let res;
      if (editingUniversityId) {
        res = await fetch(`http://127.0.0.1:8000/api/admin/universities/${editingUniversityId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else {
        res = await fetch(`http://127.0.0.1:8000/api/admin/add_university`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }

      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        clearUniversityForm();
        fetchUniversities();
      } else {
        showMessage(data.detail || 'Terjadi kesalahan saat menyimpan data universitas.', 'error');
      }
    } catch (error) {
      showMessage('Koneksi ke backend gagal', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadUniv = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!univCSV) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', univCSV);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/upload_universities', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        setUnivCSV(null);
        fetchUniversities();
      } else {
        showMessage(data.detail || 'Gagal mengupload file.', 'error');
      }
    } catch (error) {
      showMessage('Gagal mengupload file.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleEditUniv = (univ: UniversityRecord) => {
    setEditingUniversityId(univ.id);
    setUId(univ.id);
    setUName(univ.name);
    setUCountry(univ.country);
    setUProgs(univ.programs);
    setUType(univ.type);
    setUQuota(String(univ.quota));
    setUFee(String(univ.tuition_fee));
    setUAcc(String(univ.historical_accomodation));
    setTab('university');
    setMsg({ text: '', type: '' });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleDeleteUniv = async (univId: number) => {
    if (!window.confirm('Hapus data universitas ini?')) return;
    setLoading(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/universities/${univId}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (res.ok) {
        showMessage(data.message, 'success');
        fetchUniversities();
      } else {
        showMessage(data.detail || 'Gagal menghapus universitas.', 'error');
      }
    } catch (error) {
      showMessage('Koneksi ke backend gagal', 'error');
    } finally {
      setLoading(false);
    }
  };

  const formatRupiah = (number: number) =>
    new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(number);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 p-8 font-sans">
      {/* Header */}
      <div className="max-w-5xl mx-auto mb-8 border-b border-slate-700 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Manajemen Data Admin</h1>
          <p className="text-slate-400">CRUD Mahasiswa dan Universitas untuk tim admin.</p>
        </div>
        <button
          onClick={() => router.push('/admin/dashboard')}
          className="text-sm bg-slate-800 hover:bg-slate-700 border border-slate-600 px-4 py-2 rounded-lg text-white font-medium cursor-pointer"
        >
          ⬅ Kembali ke Dashboard
        </button>
      </div>

      {/* Tab Selector */}
      <div className="max-w-5xl mx-auto flex gap-4 mb-6">
        <button
          onClick={() => { setTab('student'); setMsg({ text: '', type: '' }); }}
          className={`px-6 py-3 rounded-lg font-bold transition-all cursor-pointer ${tab === 'student' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
        >
          📝 Kelola Mahasiswa
        </button>
        <button
          onClick={() => { setTab('university'); setMsg({ text: '', type: '' }); }}
          className={`px-6 py-3 rounded-lg font-bold transition-all cursor-pointer ${tab === 'university' ? 'bg-purple-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
        >
          🏫 Kelola Kampus
        </button>
      </div>

      <div className="max-w-5xl mx-auto">
        {/* Alert Message */}
        {msg.text && (
          <div className={`p-4 mb-6 rounded-lg font-medium border ${msg.type === 'error' ? 'bg-red-900/30 border-red-500/50 text-red-400' : 'bg-green-900/30 border-green-500/50 text-green-400'}`}>
            {msg.type === 'error' ? '❌ ' : '✅ '} {msg.text}
          </div>
        )}

        {/* ===================== TAB: MAHASISWA ===================== */}
        {tab === 'student' && (
          <>
            {/* Form + Upload */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
              {/* Form Input */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
                <h2 className="text-xl font-bold text-white mb-4 border-b border-slate-700 pb-2">
                  {editingStudentId ? 'Edit Mahasiswa' : 'Input Manual Baru'}
                </h2>

                {/* Banner alokasi aktif */}
                {editingStudentId && editingStudentAllocated && (
                  <div className="mb-4 p-3 bg-blue-900/40 border border-blue-500/50 rounded-lg flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs text-blue-300 font-semibold uppercase tracking-wide mb-1">🔒 Sudah Ditempatkan</p>
                      <p className="text-sm text-white font-bold">{editingStudentAllocated}</p>
                    </div>
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => handleResetPlacement(editingStudentId, sName)}
                      className="cursor-pointer shrink-0 bg-orange-600 hover:bg-orange-500 text-white text-xs font-bold px-3 py-2 rounded-lg transition-colors"
                    >
                      🔓 Cancel Alokasi
                    </button>
                  </div>
                )}

                <form onSubmit={handleSaveStudent} className="space-y-4">
                  <div>
                    <label className="block text-sm text-slate-400 mb-1">Student ID</label>
                    <input
                      type="text"
                      required
                      disabled={Boolean(editingStudentId)}
                      value={sId}
                      onChange={e => setSId(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white disabled:opacity-60"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-slate-400 mb-1">Nama Lengkap</label>
                    <input
                      type="text"
                      required
                      value={sName}
                      onChange={e => setSName(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-slate-400 mb-1">Program Studi</label>
                    <select
                      required
                      value={sProg}
                      onChange={e => setSProg(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                    >
                      <option value="CS">CS</option>
                      <option value="IR">IR</option>
                      <option value="IBM">IBM</option>
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">IPK (0.0 - 4.0)</label>
                      <input
                        type="number"
                        step="0.01"
                        required
                        value={sGpa}
                        onChange={e => setSGpa(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Skor IELTS</label>
                      <input
                        type="number"
                        step="0.5"
                        required
                        value={sIelts}
                        onChange={e => setSielts(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                      />
                    </div>
                  </div>
                  <div className="flex gap-3 items-center">
                    <button type="submit" disabled={loading} className="cursor-pointer w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded">
                      {editingStudentId ? 'Simpan Perubahan' : 'Simpan Mahasiswa'}
                    </button>
                    {editingStudentId && (
                      <button type="button" onClick={clearStudentForm} className="cursor-pointer w-full bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 rounded">
                        Batal
                      </button>
                    )}
                  </div>
                </form>
              </div>

              {/* Upload CSV */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-fit">
                <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">Upload File Dataset (CSV)</h2>
                <form onSubmit={handleUploadStudent} className="space-y-4">
                  <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:bg-slate-700/50 transition-colors">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={e => setStudentCSV(e.target.files ? e.target.files[0] : null)}
                      className="w-full text-slate-400 text-sm"
                      required
                    />
                  </div>
                  <p className="text-xs text-slate-500">*Pastikan header kolom excel sesuai template: Student_ID, Nama, Program, IPK, IELTS.</p>
                  <button type="submit" disabled={loading || !studentCSV} className="cursor-pointer w-full bg-slate-700 hover:bg-slate-600 border border-slate-500 text-white font-bold py-2 rounded">
                    Unggah dan Gabungkan Data
                  </button>
                </form>
              </div>
            </div>

            {/* Daftar Mahasiswa - Dropdown */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-xl mb-6">
              <button
                onClick={() => setShowStudentList(!showStudentList)}
                className="w-full p-4 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center hover:bg-slate-700/30 transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg">📝</span>
                  <div className="text-left">
                    <h3 className="font-bold text-white">Daftar Mahasiswa</h3>
                    <p className="text-xs text-slate-400">{students.length} mahasiswa terdaftar</p>
                  </div>
                </div>
                <span className={`text-slate-400 transition-transform duration-300 text-xl ${showStudentList ? 'rotate-180' : ''}`}>▼</span>
              </button>

              {showStudentList && (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-left text-sm text-slate-300">
                    <thead className="bg-slate-900/50 text-slate-400 text-xs uppercase">
                      <tr>
                        <th className="px-4 py-3">ID</th>
                        <th className="px-4 py-3">Nama</th>
                        <th className="px-4 py-3">Program</th>
                        <th className="px-4 py-3">IPK</th>
                        <th className="px-4 py-3">IELTS</th>
                        <th className="px-4 py-3">Status Alokasi</th>
                        <th className="px-4 py-3">Aksi</th>
                      </tr>
                    </thead>
                    <tbody>
                      {students.map(student => (
                        <tr key={student.student_id} className={`border-t border-slate-700 hover:bg-slate-900/60 transition-colors ${student.allocated_univ ? 'bg-blue-900/5' : ''}`}>
                          <td className="px-4 py-3 text-slate-100 font-mono text-xs">{student.student_id}</td>
                          <td className="px-4 py-3 font-medium">{student.name}</td>
                          <td className="px-4 py-3">{student.program}</td>
                          <td className="px-4 py-3">{student.gpa.toFixed(2)}</td>
                          <td className="px-4 py-3">{student.ielts}</td>
                          <td className="px-4 py-3">
                            {student.allocated_univ ? (
                              <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-bold bg-blue-500/20 text-blue-400">
                                🔒 {student.allocated_univ}
                              </span>
                            ) : student.pref_1 ? (
                              <span className="px-2 py-1 rounded-full text-xs font-bold bg-yellow-500/20 text-yellow-400">
                                ⏳ Menunggu Alokasi
                              </span>
                            ) : (
                              <span className="px-2 py-1 rounded-full text-xs font-bold bg-slate-600/50 text-slate-400">
                                — Belum Mengisi
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex gap-2 flex-wrap">
                              <button
                                onClick={() => handleEditStudent(student)}
                                className="cursor-pointer bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors"
                              >
                                Edit
                              </button>
                              <button
                                onClick={() => handleDeleteStudent(student.student_id)}
                                className="cursor-pointer bg-red-600 hover:bg-red-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors"
                              >
                                Hapus
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                      {students.length === 0 && (
                        <tr>
                          <td colSpan={7} className="px-4 py-8 text-center text-slate-500">Belum ada data mahasiswa.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}

        {/* ===================== TAB: UNIVERSITAS ===================== */}
        {tab === 'university' && (
          <>
            {/* Form + Upload */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
              {/* Form Input */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
                <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">
                  {editingUniversityId ? 'Edit Universitas' : 'Input Manual Univ. Rekanan'}
                </h2>
                <form onSubmit={handleManualUniv} className="space-y-4">
                  <div>
                    <label className="block text-sm text-slate-400 mb-1">Nama Universitas</label>
                    <input
                      type="text"
                      required
                      value={uName}
                      onChange={e => setUName(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-slate-400 mb-1">Negara</label>
                    <input
                      type="text"
                      required
                      value={uCountry}
                      onChange={e => setUCountry(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Program</label>
                      <input
                        type="text"
                        required
                        value={uProgs}
                        onChange={e => setUProgs(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Jalur (SE/SA)</label>
                      <select
                        value={uType}
                        onChange={e => setUType(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                      >
                        <option value="SE">SE</option>
                        <option value="SA">SA</option>
                      </select>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Kuota per Batch</label>
                      <input
                        type="number"
                        required
                        value={uQuota}
                        onChange={e => setUQuota(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Biaya Studi (IDR)</label>
                      <input
                        type="number"
                        required
                        value={uFee}
                        onChange={e => setUFee(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm text-slate-400 mb-1">Historis Akomodasi</label>
                    <input
                      type="number"
                      required
                      value={uAcc}
                      onChange={e => setUAcc(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white"
                    />
                  </div>
                  <div className="flex gap-3 items-center">
                    <button type="submit" disabled={loading} className="cursor-pointer w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 rounded">
                      {editingUniversityId ? 'Simpan Perubahan' : 'Simpan Universitas'}
                    </button>
                    {editingUniversityId && (
                      <button type="button" onClick={clearUniversityForm} className="cursor-pointer w-full bg-slate-700 hover:bg-slate-600 text-white font-bold py-2 rounded">
                        Batal
                      </button>
                    )}
                  </div>
                </form>
              </div>

              {/* Upload CSV */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-fit">
                <h2 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-2">Upload File Dataset (CSV)</h2>
                <form onSubmit={handleUploadUniv} className="space-y-4">
                  <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:bg-slate-700/50 transition-colors">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={e => setUnivCSV(e.target.files ? e.target.files[0] : null)}
                      className="w-full text-slate-400 text-sm"
                      required
                    />
                  </div>
                  <p className="text-xs text-slate-500">*Pastikan header kolom excel sesuai template: Universitas Rekanan, Negara, Program (GC), Jenis (SE/SA), Kuota per batch, Biaya studi (1 semester), Historis Biaya Akomodasi / mhs.</p>
                  <button type="submit" disabled={loading || !univCSV} className="cursor-pointer w-full bg-slate-700 hover:bg-slate-600 border border-slate-500 text-white font-bold py-2 rounded">
                    Unggah dan Gabungkan Data
                  </button>
                </form>
              </div>
            </div>

            {/* Daftar Universitas - Dropdown */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-xl mb-6">
              <button
                onClick={() => setShowUniversityList(!showUniversityList)}
                className="w-full p-4 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center hover:bg-slate-700/30 transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg">🏫</span>
                  <div className="text-left">
                    <h3 className="font-bold text-white">Daftar Universitas Rekanan</h3>
                    <p className="text-xs text-slate-400">{universities.length} kampus terdaftar</p>
                  </div>
                </div>
                <span className={`text-slate-400 transition-transform duration-300 text-xl ${showUniversityList ? 'rotate-180' : ''}`}>▼</span>
              </button>

              {showUniversityList && (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-left text-sm text-slate-300">
                    <thead className="bg-slate-900/50 text-slate-400 text-xs uppercase">
                      <tr>
                        <th className="px-4 py-3">Nama</th>
                        <th className="px-4 py-3">Negara</th>
                        <th className="px-4 py-3">Program</th>
                        <th className="px-4 py-3">Jalur</th>
                        <th className="px-4 py-3">Kuota</th>
                        <th className="px-4 py-3">Biaya Pendidikan</th>
                        <th className="px-4 py-3">Biaya Akomodasi</th>
                        <th className="px-4 py-3">Aksi</th>
                      </tr>
                    </thead>
                    <tbody>
                      {universities.map(univ => (
                        <tr key={univ.id} className="border-t border-slate-700 hover:bg-slate-900/60 transition-colors">
                          <td className="px-4 py-3 font-semibold text-slate-100">{univ.name}</td>
                          <td className="px-4 py-3">
                            <span className="bg-slate-700 px-2 py-1 rounded text-xs">{univ.country}</span>
                          </td>
                          <td className="px-4 py-3">{univ.programs}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-bold ${univ.type === 'SE' ? 'bg-purple-500/20 text-purple-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                              {univ.type}
                            </span>
                          </td>
                          <td className="px-4 py-3">{univ.quota}</td>
                          <td className="px-4 py-3 text-blue-400">{formatRupiah(univ.tuition_fee)}</td>
                          <td className="px-4 py-3 text-orange-400">{formatRupiah(univ.historical_accomodation)}</td>
                          <td className="px-4 py-3">
                            <div className="flex gap-2 flex-wrap">
                              <button
                                onClick={() => handleEditUniv(univ)}
                                className="cursor-pointer bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors"
                              >
                                Edit
                              </button>
                              <button
                                onClick={() => handleDeleteUniv(univ.id)}
                                className="cursor-pointer bg-red-600 hover:bg-red-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors"
                              >
                                Hapus
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                      {universities.length === 0 && (
                        <tr>
                          <td colSpan={8} className="px-4 py-8 text-center text-slate-500">Belum ada data universitas.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

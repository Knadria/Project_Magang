"use client";
import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const [studentId, setStudentId] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const res = await fetch('http://127.0.0.1:8000/api/student/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: studentId, name }),
      });

      const data = await res.json();

      if (res.ok) {
        // Simpan data di local storage untuk dashboard
        localStorage.setItem('student_data', JSON.stringify(data));
        router.push('/dashboard');
      } else {
        setError(data.detail || 'Terjadi kesalahan saat login.');
      }
    } catch (err) {
      setError('Gagal menghubungi server. Pastikan Backend API sudah menyala.');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-slate-800 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-sm border border-slate-700">
        <div className="p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">BINUS Global Class</h1>
            <p className="text-slate-400">Portal Penempatan Study Abroad</p>
          </div>

          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Student ID
              </label>
              <input
                type="text"
                required
                placeholder="Contoh: BINUS001"
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Nama Lengkap
              </label>
              <input
                type="text"
                required
                placeholder="Contoh: Student_1"
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="cursor-pointer w-full focus:animate-none hover:animate-none bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors duration-200 flex justify-center items-center h-12"
            >
              {loading ? (
                <span className="w-6 h-6 border-2 border-white/20 border-t-white rounded-full animate-spin"></span>
              ) : (
                'Masuk ke Portal'
              )}
            </button>
            <div className='flex justify-center mt-3 gap-2'>
              <button 
                  type="button" 
                  onClick={() => router.push('/admin')} 
                  className="cursor-pointer text-xs text-slate-400 underline hover:text-slate-200">
                Masuk sebagai Tim Global Class (Admin)
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

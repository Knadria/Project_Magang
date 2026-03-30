import pandas as pd

# 1. Load Data
costs = pd.read_csv('dataset/International_Education_Costs.csv')
survey = pd.read_csv('dataset/world_university_survey_dataset.csv')

# 2. Pre-processing Sederhana (Menyamakan Nama Kolom)
survey = survey.rename(columns={'country': 'Country', 'field_of_study': 'Program'})

# 3. Merging (Menggabungkan berdasarkan Negara dan Bidang Studi)
# Ini akan mencocokkan profil mahasiswa dengan data biaya di negara tersebut
df_final = pd.merge(survey, costs, on=['Country', 'Program'], how='inner')

# 4. Membuat Target Variabel (Efficiency Score)
# Kita ubah data teks kepuasan menjadi angka untuk dipelajari AI
satisfaction_map = {
    'Very Satisfied': 4,
    'Satisfied': 3,
    'Neutral': 2,
    'Dissatisfied': 1,
    'Very Dissatisfied': 0
}
df_final['efficiency_score'] = df_final['overall_satisfaction'].map(satisfaction_map)

# 5. Simpan Hasil
df_final.to_csv('dataset_untuk_training.csv', index=False)
print("Data berhasil digabung! Total baris:", len(df_final))
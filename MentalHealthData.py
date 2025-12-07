import numpy as np
import pandas as pd

# Définition du nombre de lignes (75000 lignes)
N_ROWS = 75000 # Augmenté à 75k
print(f"Tentative de génération de {N_ROWS} lignes de données...")

# Fixer la graine pour la reproductibilité
np.random.seed(75000)

# --- 1. Génération des Identifiants et Démographie de Base ---
user_ids = np.arange(1001, 1001 + N_ROWS)
ages = np.clip(np.random.normal(loc=38, scale=14, size=N_ROWS).astype(int), a_min=18, a_max=85)
genders = np.random.choice(['Female', 'Male', 'Non-binary', 'Other'], size=N_ROWS, p=[0.48, 0.45, 0.05, 0.02])
edu_levels = np.random.choice(['High School', 'University', 'Masters', 'Ph.D.'], size=N_ROWS, p=[0.10, 0.40, 0.35, 0.15])
employment_status = np.random.choice(['Employed', 'Student', 'Self-employed', 'Unemployed', 'Retired'], size=N_ROWS, p=[0.55, 0.15, 0.15, 0.10, 0.05])

# --- 2. Variables Clés (Comportementales et Socio-économiques) ---
work_hours = np.zeros(N_ROWS)
employed_mask = (employment_status == 'Employed') | (employment_status == 'Self-employed')
work_hours[employed_mask] = np.clip(np.random.normal(loc=50, scale=12, size=employed_mask.sum()), 20, 90)
work_hours = work_hours.astype(float)

sleep_hours = np.clip(np.random.normal(7.2, 1.8, size=N_ROWS), 2.0, 12.0)
exercise_freq = np.random.randint(0, 8, size=N_ROWS) # 0 to 7

# Qualité des Relations (échelle 1 à 10, 10 = excellente)
rel_quality = np.clip(np.random.normal(loc=7.0, scale=2.0, size=N_ROWS), 1, 10)
# Tension Financière (échelle 1 à 10, 10 = tension extrême)
fin_strain = np.clip(np.random.normal(loc=4.0, scale=3.0, size=N_ROWS), 1, 10)

# --- 3. Introduction des CORRELATIONS MAXIMALES (Variables Latentes) ---

# Variable Latente 1: Indice de DÉTRESSE (Facteur Principal pour ACP/K-Means)
distress_index_base = (work_hours / 40) + (10 / rel_quality) + (fin_strain * 1.5) + (14 / (sleep_hours + 1.0)) - (exercise_freq / 7.0) * 2.0
distress_index_scaled = np.clip(distress_index_base, 5, 25)

# Variable Latente 2: Tendance à la RÉSILIENCE (Facteur Principal pour k-NN)
# FIX APPLIQUÉ : Utilisation de l'array 'edu_levels' au lieu de 'data['Education_Level']'
resilience_base = (ages / 60) + (10 - fin_strain) / 5 + np.select(
    [edu_levels == 'Ph.D.', edu_levels == 'Masters', edu_levels == 'University'],
    [3, 2, 1], default=0
)
resilience_scaled = np.clip(resilience_base, 0.1, 5)

# 3.1 Génération des scores avec un bruit ALÉATOIRE MINIMAL
GAD7_SCALE = 0.5
PHQ9_SCALE = 0.7
STRESS_SCALE = 0.2

gad7_scores = np.clip(np.random.normal(loc=distress_index_scaled * 1.2, scale=GAD7_SCALE, size=N_ROWS).astype(int), 0, 21)
phq9_scores = np.clip(np.random.normal(loc=distress_index_scaled * 1.5, scale=PHQ9_SCALE, size=N_ROWS).astype(int), 0, 27)
stress_levels = np.clip(np.random.normal(loc=distress_index_scaled * 0.4, scale=STRESS_SCALE, size=N_ROWS).astype(int), 1, 10)

# --- 4. Variable Cible pour k-NN (Classification) ---
burnout_metric = (gad7_scores + phq9_scores + stress_levels * 2.5) / resilience_scaled

burnout_risk = np.select(
    [burnout_metric <= 15, (burnout_metric > 15) & (burnout_metric <= 30), burnout_metric > 30],
    ['Low', 'Medium', 'High']
)

# --- 5. Assemblage du DataFrame (Le DataFrame 'data' est créé ici) ---
data = pd.DataFrame({
    'User_ID': user_ids, 'Age': ages, 'Gender': genders, 'Education_Level': edu_levels,
    'Employment_Status': employment_status, 'Work_Hours_Week': work_hours,
    'Sleep_Hours_Night': sleep_hours, 'Exercise_Freq_Week': exercise_freq,
    'Financial_Strain': fin_strain, 'Relationship_Quality': rel_quality,
    'GAD-7_Score': gad7_scores, 'PHQ-9_Score': phq9_scores,
    'Stress_Level_Scale': stress_levels, 'Burnout_Risk': burnout_risk # Cible k-NN
})

# --- 6. Introduction de Bruit et de Valeurs Manquantes CIBLÉES (15% pour la robustesse) ---

# 6.1 Ajout des NaN (15% aléatoire dans certaines colonnes)
nan_cols_random = ['Education_Level', 'Employment_Status', 'Work_Hours_Week', 'Sleep_Hours_Night', 'Exercise_Freq_Week']
for col in nan_cols_random:
    nan_indices = data.sample(frac=0.15, random_state=42 + nan_cols_random.index(col)).index
    data.loc[nan_indices, col] = np.nan

# 6.2 Valeurs Manquantes SÉLECTIVES (Ex: Les personnes avec un stress financier élevé ont tendance à ne pas répondre)
# 25% des personnes ayant un Financial_Strain > 8 ne remplissent pas leur score GAD-7 ou PHQ-9.
high_strain_mask = data['Financial_Strain'] > 8
selective_nan_indices = data[high_strain_mask].sample(frac=0.25, random_state=101).index
data.loc[selective_nan_indices, ['GAD-7_Score', 'PHQ-9_Score']] = np.nan


# 6.3 Ajout d'Outliers extrêmes (pour tester la robustesse de la mise à l'échelle)
outlier_indices_1 = data.sample(n=75, random_state=105).index
data.loc[outlier_indices_1, 'Work_Hours_Week'] = np.random.randint(150, 250, size=75) # Overworking outliers
data.loc[outlier_indices_1, 'Sleep_Hours_Night'] = np.random.uniform(0.1, 1.0, size=75) # Extreme sleep deprivation

# 6.4 Valeurs Illogiques (pour tester le nettoyage des données)
illogical_indices = data.sample(n=60, random_state=108).index
data.loc[illogical_indices, 'Relationship_Quality'] = -99 # Code d'erreur illogique

# --- 7. Sauvegarde du fichier CSV ---
output_filename = 'HealthMind_Mental_Health_Data_75k_MultiAlgo.csv'
data.to_csv(output_filename, index=False)

print(f"✅ Succès : {N_ROWS} lignes de données SYNTHÉTIQUES générées.")
print(f"Fichier sauvegardé sous : '{output_filename}'.")
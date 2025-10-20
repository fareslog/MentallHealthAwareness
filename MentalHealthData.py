import numpy as np
import pandas as pd

# Set the number of rows as requested (11000 rows)
N_ROWS = 11000

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Generate User_ID and Demographics ---
user_ids = np.arange(1001, 1001 + N_ROWS)
ages = np.clip(np.random.normal(loc=32, scale=15, size=N_ROWS).astype(int), a_min=16, a_max=80) 
genders = np.random.choice(['Female', 'Male', 'Non-binary', 'Other'], size=N_ROWS, p=[0.44, 0.44, 0.08, 0.04])
edu_levels = np.random.choice(['High School', 'University', 'Masters', 'Ph.D.'], size=N_ROWS, p=[0.20, 0.40, 0.30, 0.10])
employment_status = np.random.choice(['Employed', 'Student', 'Self-employed', 'Unemployed', 'Retired'], size=N_ROWS, p=[0.55, 0.20, 0.10, 0.10, 0.05])

# --- 2. Generate Behavioral and Health Metrics (Independent Variables) ---
work_hours = np.zeros(N_ROWS)
employed_mask = (employment_status == 'Employed')
self_employed_mask = (employment_status == 'Self-employed')
student_mask = (employment_status == 'Student')

# Assign work hours
work_hours[employed_mask] = np.clip(np.random.normal(42, 15, size=employed_mask.sum()), 5, 85)
work_hours[self_employed_mask] = np.clip(np.random.normal(60, 20, size=self_employed_mask.sum()), 5, 100)
work_hours[student_mask] = np.random.randint(0, 45, size=student_mask.sum())
work_hours = work_hours.astype(float)
work_hours[(employment_status == 'Unemployed') | (employment_status == 'Retired')] = 0.0

sleep_hours = np.clip(np.random.normal(6.5, 2.0, size=N_ROWS), 1.0, 12.0)
exercise_freq = np.random.randint(0, 10, size=N_ROWS)
social_support = np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'MISSING'], size=N_ROWS, p=[0.1, 0.25, 0.35, 0.2, 0.1])
coping_mechanisms = np.random.choice(['Exercise', 'Socializing', 'Isolation', 'Mindfulness', 'Gaming', 'Work', 'Reading', 'N/A'], size=N_ROWS, p=[0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1])

# --- 3. Introduce CORRELATION (Correlated Variables) ---

# 3.1 Mapping Social Support (High score = Low Support = High Stress)
support_map = {'Very Low': 4, 'Low': 3, 'Medium': 2, 'High': 1, 'MISSING': 2.5}
social_support_numeric = np.array([support_map.get(s, 2.5) for s in social_support])

# 3.2 Calculate Base Stress Index (The latent variable driving outcomes)
# The index increases with work hours, decreases with sleep hours, and increases with poor social support (high social_support_numeric value)
stress_index_base = (work_hours / 60) + (10 / (sleep_hours + 0.5)) + (social_support_numeric * 1.5)

# 3.3 Scale and clamp the index for stability
# Normalized to a range that influences the mean of the scores
stress_index_scaled = np.clip(stress_index_base, 2, 15)

# 3.4 Generate GAD-7, PHQ-9, and Stress_Level_Scale based on the scaled index
# The 'loc' (mean) for the normal distribution is now determined by the stress index
gad7_scores = np.clip(np.random.normal(loc=stress_index_scaled * 1.2, scale=4, size=N_ROWS).astype(int), 0, 21)
phq9_scores = np.clip(np.random.normal(loc=stress_index_scaled * 1.5, scale=5, size=N_ROWS).astype(int), 0, 27)
stress_levels = np.clip(np.random.normal(loc=stress_index_scaled * 0.5, scale=1.5, size=N_ROWS).astype(int), 1, 9)

# --- 4. Determine Risk_Level (Target Variable) ---
# Risk level is still based on an average of GAD-7 and PHQ-9, which are now correlated with stressors
risk_metric = (gad7_scores + phq9_scores) / 2.0
risk_level = np.select(
    [risk_metric <= 7, (risk_metric > 7) & (risk_metric <= 15), risk_metric > 15],
    ['Low', 'Medium', 'High']
)

# --- 5. Assemble the DataFrame ---
data = pd.DataFrame({
    'User_ID': user_ids, 'Age': ages, 'Gender': genders, 'Education_Level': edu_levels,
    'Employment_Status': employment_status, 'Work_Hours_Week': work_hours,
    'Sleep_Hours_Night': sleep_hours, 'Exercise_Freq_Week': exercise_freq,
    'Social_Support': social_support, 'GAD-7_Score': gad7_scores,
    'PHQ-9_Score': phq9_scores, 'Stress_Level_Scale': stress_levels,
    'Coping_Mechanism': coping_mechanisms, 'Risk_Level': risk_level
})

# --- 6. Introduce EXTREME MESSINESS (NaNs, Outliers, Illogical Data) ---

# 6.1 Increase NaN (missing values) to 15% across selected columns
nan_cols = ['Education_Level', 'Work_Hours_Week', 'Sleep_Hours_Night', 'Exercise_Freq_Week', 'Stress_Level_Scale', 'Coping_Mechanism', 'Age']
for col in nan_cols:
    nan_indices = data.sample(frac=0.15, random_state=42 + nan_cols.index(col)).index # 15% NaN
    data.loc[nan_indices, col] = np.nan

# 6.2 Add extreme Outliers: Very high work hours and very low sleep
outlier_indices_1 = data.sample(n=20, random_state=101).index
data.loc[outlier_indices_1, 'Work_Hours_Week'] = np.random.randint(140, 180, size=20)
data.loc[outlier_indices_1, 'Sleep_Hours_Night'] = np.random.uniform(0.5, 2.5, size=20)
data.loc[outlier_indices_1, 'Stress_Level_Scale'] = 9

# Outlier 3: Illogical sleep hours (negative or zero)
illogical_indices = data.sample(n=10, random_state=103).index
data.loc[illogical_indices, 'Sleep_Hours_Night'] = np.random.uniform(-10.0, 0, size=10)

# Outlier 4: Illogical exercise frequency (much too high)
illogical_freq_indices = data.sample(n=5, random_state=104).index
data.loc[illogical_freq_indices, 'Exercise_Freq_Week'] = np.random.choice([50, 99], size=5)

# 6.3 Add zero/empty strings and placeholder values
data.loc[data.sample(frac=0.07, random_state=105).index, 'Education_Level'] = 'N/A'
data.loc[data.sample(frac=0.05, random_state=106).index, 'Gender'] = 'NA'

# --- 7. Save the CSV file ---
output_filename = 'HealthMind_Mental_Health_Data_11000_Correlated_Messy.csv'
data.to_csv(output_filename, index=False)

print(f"✅ Successfully generated {N_ROWS} rows of SYNTHETIC, CORRELATED and EXTREME MESSY data.")
print(f"File saved as '{output_filename}'. The core variables now exhibit expected correlations.")
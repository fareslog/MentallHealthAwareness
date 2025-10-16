import numpy as np
import pandas as pd

# Set the number of rows as requested (increased to 5500)
N_ROWS = 5500

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Generate User_ID and Demographics ---
user_ids = np.arange(1001, 1001 + N_ROWS)
ages = np.clip(np.random.normal(loc=31, scale=13, size=N_ROWS).astype(int), a_min=16, a_max=75) # Slight shift in mean/scale
genders = np.random.choice(['Female', 'Male', 'Non-binary', 'Other'], size=N_ROWS, p=[0.44, 0.44, 0.08, 0.04])
edu_levels = np.random.choice(['High School', 'University', 'Masters', 'Ph.D.'], size=N_ROWS, p=[0.20, 0.40, 0.30, 0.10])
employment_status = np.random.choice(['Employed', 'Student', 'Self-employed', 'Unemployed', 'Retired'], size=N_ROWS, p=[0.55, 0.20, 0.10, 0.10, 0.05])

# --- 2. Generate Behavioral and Health Metrics ---
work_hours = np.zeros(N_ROWS)
employed_mask = (employment_status == 'Employed')
self_employed_mask = (employment_status == 'Self-employed')
student_mask = (employment_status == 'Student')

# Assign work hours based on employment status
work_hours[employed_mask] = np.clip(np.random.normal(42, 12, size=employed_mask.sum()), 10, 80)
work_hours[self_employed_mask] = np.clip(np.random.normal(55, 18, size=self_employed_mask.sum()), 10, 95)
work_hours[student_mask] = np.random.randint(0, 40, size=student_mask.sum())
work_hours = work_hours.astype(float)
work_hours[(employment_status == 'Unemployed') | (employment_status == 'Retired')] = 0.0

sleep_hours = np.clip(np.random.normal(6.8, 1.8, size=N_ROWS), 2.5, 11.0)
exercise_freq = np.random.randint(0, 8, size=N_ROWS) # Days per week (allowing > 7 to represent intensity/error)
social_support = np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'MISSING'], size=N_ROWS, p=[0.1, 0.25, 0.35, 0.2, 0.1]) # Introduce 'MISSING' as a non-NaN string
gad7_scores = np.random.randint(0, 22, size=N_ROWS) # GAD-7 score (0-21)
phq9_scores = np.random.randint(0, 28, size=N_ROWS) # PHQ-9 score (0-27)
stress_levels = np.random.randint(0, 10, size=N_ROWS) # Stress scale (0-9, allowing 0 and 9 for extra range)
coping_mechanisms = np.random.choice(['Exercise', 'Socializing', 'Isolation', 'Mindfulness', 'Gaming', 'Work', 'Reading', 'N/A'], size=N_ROWS, p=[0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]) # Introduce 'N/A'

# --- 3. Determine Risk_Level (Target Variable) ---
# Risk level calculation remains the same
risk_metric = (gad7_scores + phq9_scores) / 2.0
risk_level = np.select(
    [risk_metric <= 7, (risk_metric > 7) & (risk_metric <= 15), risk_metric > 15],
    ['Low', 'Medium', 'High']
)

# --- 4. Assemble the DataFrame ---
data = pd.DataFrame({
    'User_ID': user_ids, 'Age': ages, 'Gender': genders, 'Education_Level': edu_levels,
    'Employment_Status': employment_status, 'Work_Hours_Week': work_hours,
    'Sleep_Hours_Night': sleep_hours, 'Exercise_Freq_Week': exercise_freq,
    'Social_Support': social_support, 'GAD-7_Score': gad7_scores,
    'PHQ-9_Score': phq9_scores, 'Stress_Level_Scale': stress_levels,
    'Coping_Mechanism': coping_mechanisms, 'Risk_Level': risk_level
})

# --- 5. Introduce MESSINESS (NaNs, Outliers, Zeroes) ---

# 5.1 Increase NaN (missing values) to 12% across selected columns
nan_cols = ['Education_Level', 'Work_Hours_Week', 'Sleep_Hours_Night', 'Exercise_Freq_Week', 'Stress_Level_Scale', 'Coping_Mechanism', 'Age']
for col in nan_cols:
    nan_indices = data.sample(frac=0.12, random_state=42 + nan_cols.index(col)).index # Use different seed for each column
    data.loc[nan_indices, col] = np.nan

# 5.2 Add extreme Outliers (for detection practice)
# Outlier 1: Very young/old age with extreme scores (5 rows)
outlier_indices_1 = data.sample(n=5, random_state=101).index
data.loc[outlier_indices_1, 'Age'] = np.random.choice([95, 12, 100], size=5)
data.loc[outlier_indices_1, 'GAD-7_Score'] = 21
data.loc[outlier_indices_1, 'PHQ-9_Score'] = 27

# Outlier 2: Extremely high work hours (10 rows)
outlier_indices_2 = data.sample(n=10, random_state=102).index
data.loc[outlier_indices_2, 'Work_Hours_Week'] = np.random.randint(120, 160, size=10)
data.loc[outlier_indices_2, 'Sleep_Hours_Night'] = np.random.uniform(1.0, 3.0, size=10)
data.loc[outlier_indices_2, 'Stress_Level_Scale'] = 9

# 5.3 Add illogical values (e.g., negative sleep hours)
illogical_indices = data.sample(n=3, random_state=103).index
data.loc[illogical_indices, 'Sleep_Hours_Night'] = np.random.uniform(-5.0, 0, size=3) # Negative sleep hours

# 5.4 Add zero/empty strings instead of NaN in some columns (making cleaning harder)
data.loc[data.sample(frac=0.05, random_state=104).index, 'Education_Level'] = '' # Empty string
data.loc[data.sample(frac=0.03, random_state=105).index, 'Work_Hours_Week'] = 0.0 # Zero where it shouldn't be

# --- 6. Save the CSV file ---
output_filename = 'HealthMind_Mental_Health_Data_5500_Messy.csv'
data.to_csv(output_filename, index=False)

print(f"✅ Successfully generated {N_ROWS} rows of SYNTHETIC, VERY MESSY data.")
print(f"File saved as '{output_filename}'. You can now download and use this file for advanced preprocessing.")

import numpy as np
import pandas as pd

# Set the number of rows as requested
N_ROWS = 3500

# Set seed for reproducibility so the data is the same every time you run it
np.random.seed(42)

# --- 1. Generate User_ID and Demographics ---
user_ids = np.arange(1001, 1001 + N_ROWS)
ages = np.clip(np.random.normal(loc=30, scale=12, size=N_ROWS).astype(int), a_min=18, a_max=70)
genders = np.random.choice(['Female', 'Male', 'Non-binary', 'Other'], size=N_ROWS, p=[0.45, 0.45, 0.05, 0.05])
edu_levels = np.random.choice(['High School', 'University', 'Masters', 'Ph.D.'], size=N_ROWS, p=[0.25, 0.45, 0.20, 0.10])
employment_status = np.random.choice(['Employed', 'Student', 'Self-employed', 'Unemployed', 'Retired'], size=N_ROWS, p=[0.50, 0.25, 0.10, 0.10, 0.05])

# --- 2. Generate Behavioral and Health Metrics ---
work_hours = np.zeros(N_ROWS)
employed_mask = (employment_status == 'Employed')
self_employed_mask = (employment_status == 'Self-employed')
student_mask = (employment_status == 'Student')

# Assign work hours based on employment status
work_hours[employed_mask] = np.clip(np.random.normal(45, 10, size=employed_mask.sum()), 10, 70)
work_hours[self_employed_mask] = np.clip(np.random.normal(50, 15, size=self_employed_mask.sum()), 10, 80)
work_hours[student_mask] = np.random.randint(0, 30, size=student_mask.sum())
work_hours = work_hours.astype(float)
work_hours[(employment_status == 'Unemployed') | (employment_status == 'Retired')] = 0.0

sleep_hours = np.clip(np.random.normal(6.5, 1.5, size=N_ROWS), 3.0, 10.0)
exercise_freq = np.random.randint(0, 7, size=N_ROWS) # Days per week
social_support = np.random.choice(['Very Low', 'Low', 'Medium', 'High'], size=N_ROWS, p=[0.1, 0.3, 0.4, 0.2])
gad7_scores = np.random.randint(0, 22, size=N_ROWS) # GAD-7 score (0-21)
phq9_scores = np.random.randint(0, 28, size=N_ROWS) # PHQ-9 score (0-27)
stress_levels = np.random.randint(1, 9, size=N_ROWS) # Stress scale (1-8)
coping_mechanisms = np.random.choice(['Exercise', 'Socializing', 'Isolation', 'Mindfulness', 'Gaming', 'Work', 'Reading'], size=N_ROWS, p=[0.15, 0.15, 0.2, 0.15, 0.1, 0.15, 0.1])

# --- 3. Determine Risk_Level (Target Variable) ---
# Risk level is calculated based on an average of GAD-7 and PHQ-9 for simplicity
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

# --- 5. Introduce Messiness (NaN values) for Data Preprocessing ---
# Introduce NaN (missing values) across 7% of the data for selected columns
for col in ['Education_Level', 'Work_Hours_Week', 'Sleep_Hours_Night', 'Exercise_Freq_Week', 'Stress_Level_Scale', 'Coping_Mechanism']:
    nan_indices = data.sample(frac=0.07, random_state=42).index
    data.loc[nan_indices, col] = np.nan

# --- 6. Save the CSV file ---
output_filename = 'HealthMind_Mental_Health_Data_3500.csv'
data.to_csv(output_filename, index=False)

print(f"✅ Successfully generated {N_ROWS} rows of synthetic, messy data.")
print(f"File saved as '{output_filename}'. You can now download and use this file.")
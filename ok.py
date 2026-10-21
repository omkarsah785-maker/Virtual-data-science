import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

print("=" * 60)
print("WEEK 3: STATISTICAL ANALYSIS AND HYPOTHESIS TESTING")
print("=" * 60)

# ---------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop_duplicates(inplace=True)

print("\nDataset successfully loaded and preprocessed.")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

# ---------------------------------------------------------
# 2. HYPOTHESIS 1: TWO-SAMPLE INDEPENDENT T-TEST
# ---------------------------------------------------------
# Question: Did female passengers pay a significantly higher fare than male passengers?
# H0: Mean Fare(Females) = Mean Fare(Males)
# H1: Mean Fare(Females) != Mean Fare(Males)

female_fares = df[df['Sex'] == 'female']['Fare']
male_fares = df[df['Sex'] == 'male']['Fare']

t_stat, p_val_t = stats.ttest_ind(female_fares, male_fares, equal_var=False)

print("\n" + "-" * 50)
print("HYPOTHESIS 1: Two-Sample Independent T-Test (Fare vs Sex)")
print("-" * 50)
print(f"Female Mean Fare: ${female_fares.mean():.2f}")
print(f"Male Mean Fare:   ${male_fares.mean():.2f}")
print(f"T-statistic:      {t_stat:.4f}")
print(f"P-value:          {p_val_t:.4e}")

if p_val_t < 0.05:
    print("Conclusion: Reject H0. There is a statistically significant difference in fares paid by females and males.")
else:
    print
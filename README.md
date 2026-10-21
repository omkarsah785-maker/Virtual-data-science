import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Acquisition & Loading
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Data Cleaning & Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.drop_duplicates(inplace=True)

df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

# 3. Exploratory Data Analysis (EDA) & Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Visual 1: Overall Survival Rate
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', palette='Set2')
plt.title('Overall Survival Distribution (0 = No, 1 = Yes)')
plt.savefig('survival_distribution.png', bbox_inches='tight')
plt.close()

# Visual 2: Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Pclass', y=df['Survived'].astype(int), palette='viridis')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_class.png', bbox_inches='tight')
plt.close()

# Visual 3: Age Distribution by Gender & Survival
plt.figure(figsize=(7, 4))
sns.violinplot(data=df, x='Sex', y='Age', hue='Survived', split=True, palette='Pastel1')
plt.title('Age and Gender Distribution by Survival')
plt.savefig('age_distribution.png', bbox_inches='tight')
plt.close()

# Visual 4: Correlation Heatmap
plt.figure(figsize=(6, 4))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()

# Visual 5: Fare vs Embarkation Port
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='Embarked', y='Fare', hue='Survived', palette='Set3')
plt.ylim(0, 300) # Limit outliers for clear storytelling
plt.title('Fare Distribution across Embarkation Ports')
plt.savefig('fare_by_embarked.png', bbox_inches='tight')
plt.close()

print("All charts generated and saved successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Acquisition (Loading Dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- Initial Data Info ---")
print(df.info())

# 2. Data Cleaning
# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Dropping column with excessive missing data

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Correcting Data Types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

print("\n--- Cleaned Data Summary ---")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (Visualizations)

# Visualization 1: Distribution of Age (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title('Distribution of Passenger Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png')
plt.close()

# Visualization 2: Survival Count by Passenger Class (Bar Plot)
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Count')
plt.savefig('survival_by_class.png')
plt.close()

# Visualization 3: Correlation Matrix (Heatmap)
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nVisualizations saved successfully!")
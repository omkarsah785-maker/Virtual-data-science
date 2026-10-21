import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

# Visual styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

print("=" * 60)
print("WEEK 4: MACHINE LEARNING MODEL DEVELOPMENT & EVALUATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Feature Engineering / Selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# Encoding Categorical Variables
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nPreprocessing Complete!")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape:     {X_test.shape}")

# ---------------------------------------------------------
# 2. MODEL TRAINING
# ---------------------------------------------------------
# Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. MODEL EVALUATION & METRICS
# ---------------------------------------------------------
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

def print_metrics(model_name, y_true, y_pred):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

print_metrics("Logistic Regression", y_test, y_pred_lr)
print_metrics("Random Forest Classifier", y_test, y_pred_rf)

# ---------------------------------------------------------
# 4. VISUALIZATION 1: CONFUSION MATRICES
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

ConfusionMatrixDisplay(cm_lr, display_labels=['Perished', 'Survived']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title("Logistic Regression Confusion Matrix", fontweight='bold')
axes[0].grid(False)

ConfusionMatrixDisplay(cm_rf, display_labels=['Perished', 'Survived']).plot(ax=axes[1], cmap='Greens')
axes[1].set_title("Random Forest Confusion Matrix", fontweight='bold')
axes[1].grid(False)

plt.tight_layout()
plt.savefig("ml_confusion_matrix.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 5. VISUALIZATION 2: ROC CURVES COMPARISON
# ---------------------------------------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve Model Comparison', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ml_roc_curve.png", dpi=300)
plt.close()

print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print("Saved Charts: ml_confusion_matrix.png, ml_roc_curve.png")
print("=" * 60)
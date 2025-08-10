import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report
)

# Create plots folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Settings
RANDOM_STATE = 42
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 110

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("rows, cols:", df.shape)
print(df.info())
print(df.isnull().sum().sort_values(ascending=False))

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin', 'Ticket'], inplace=True, errors='ignore')

# Feature engineering
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
title_map = {
    **dict.fromkeys(['Mr'], 'Mr'),
    **dict.fromkeys(['Mrs'], 'Mrs'),
    **dict.fromkeys(['Miss'], 'Miss'),
    **dict.fromkeys(['Master'], 'Master')
}
df['Title'] = df['Title'].map(lambda t: title_map.get(t, 'Other'))
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['AgeGroup'] = pd.cut(
    df['Age'], bins=[0, 12, 18, 35, 50, 120],
    labels=['Child', 'Teen', 'YoungAdult', 'MiddleAge', 'Senior']
)

# Survival stats
female_survival = df[df.Sex == 'female'].Survived.mean()
male_survival = df[df.Sex == 'male'].Survived.mean()
pclass_survival = df.groupby('Pclass').Survived.mean()
print("Female survival:", round(female_survival, 3))
print("Male survival:", round(male_survival, 3))
print("Survival by class:\n", pclass_survival)

# Plot: Survival rate by class
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df, errorbar=None)
plt.title("Survival RATE by Class (mean of Survived)")
plt.ylim(0, 1)
plt.savefig("plots/survival_rate_by_class.png", bbox_inches='tight')

# Plot: Survival rate by sex
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None)
plt.title("Survival RATE by Sex")
plt.ylim(0, 1)
plt.savefig("plots/survival_rate_by_sex.png", bbox_inches='tight')

# Features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
X = df[features].copy()
X['Sex'] = (X['Sex'] == 'male').astype(int)
y = df['Survived']
X.fillna(0, inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Logistic Regression
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
print("\nLogistic Regression:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("ROC AUC:", round(roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]), 3))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))
print("ROC AUC:", round(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), 3))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances (Random Forest):\n", importances)

# Plot: Feature importances
plt.figure(figsize=(6, 4))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("plots/feature_importances.png", bbox_inches='tight')
print("Saved plots in 'plots/' folder")

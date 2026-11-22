import pandas as pd
import numpy as np

# Charger les données brutes
df = pd.read_csv('data/kidney_disease.csv')

# 1. Remplacer les ? par des NaN
df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)

# 2. Nettoyer les espaces dans le texte
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(['nan', ''], np.nan)

# 3. Convertir les colonnes numériques (sauf 'class')
for col in df.columns:
    if col != 'class':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Nettoyer la cible
df['class'] = df['class'].replace({'ckd\t': 'ckd', 'notckd\t': 'notckd'})
df = df[df['class'].isin(['ckd', 'notckd'])]

# 5. Sauvegarder
df.to_csv('data/clean_kidney_disease.csv', index=False)

print(f"Termine. Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")

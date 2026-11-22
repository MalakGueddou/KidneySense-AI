import pandas as pd

# Charger les données
df = pd.read_csv('../data/kidney_disease.csv')

print("Dimensions du dataset :", df.shape)
print("\n--- Les 5 premières lignes ---")
print(df.head())

print("\n--- Types de données et valeurs manquantes ---")
print(df.info())

print("\n--- Nombre de valeurs manquantes par colonne ---")
print(df.isnull().sum())

print("\n--- Distribution de la variable cible ---")
print(df['class'].value_counts())

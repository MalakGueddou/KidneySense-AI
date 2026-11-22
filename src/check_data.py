import pandas as pd

print("--- 1. Aperçu des 5 premières lignes ---")
df = pd.read_csv('data/clean_kidney_disease.csv')
print(df.head())
print("\n")

print("--- 2. Types de donnees (Dtypes) ---")
print(df.dtypes)
print("\n")

print("--- 3. Valeurs uniques de la cible (class) ---")
print(df['class'].value_counts())
print("\n")

print("--- 4. Y a-t-il encore des '?' caches dans le texte ? ---")
# On cherche la chaine de caractere '?' dans toutes les colonnes textuelles
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    if df[col].astype(str).str.contains(r'\?').any():
        print(f"Attention : Il reste des '?' dans la colonne {col}")
print("Verification des '?' terminee.")
print("\n")

print("--- 5. Statistiques descriptives (pour les colonnes numeriques) ---")
print(df.describe())

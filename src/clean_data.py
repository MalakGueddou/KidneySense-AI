import pandas as pd
import numpy as np

def clean_data_ultimate(input_path, output_path):
    print("Chargement des donnees brutes...")
    df = pd.read_csv(input_path)
    
    print("Nettoyage des caracteres invisibles (espaces, tabulations)...")
    # 1. Remplacer tous les ? par des NaN
    df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)
    
    # 2. Nettoyer les espaces dans les colonnes texte
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Remplacer les 'nan' (texte) par de vrais NaN
            df[col] = df[col].replace(['nan', ''], np.nan)
            
    # 3. Convertir les colonnes numériques
    # On essaie de convertir TOUTES les colonnes sauf 'class' en nombres
    for col in df.columns:
        if col != 'class':
            # On utilise errors='coerce' pour forcer la conversion. 
            # Si Pandas n'y arrive pas, il met un NaN (ce qui est bien pour les valeurs manquantes)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 4. Nettoyer la cible (class)
    df['class'] = df['class'].replace({'ckd\t': 'ckd', 'notckd\t': 'notckd'})
    df = df[df['class'].isin(['ckd', 'notckd'])]
    
    print("Sauvegarde...")
    df.to_csv(output_path, index=False)
    
    print(f"Nettoyage termine !")
    print(f"Dimensions finales : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print("\nTypes de donnees finaux :")
    print(df.dtypes)
    
if __name__ == "__main__":
    clean_data_ultimate('data/kidney_disease.csv', 'data/clean_kidney_disease.csv')

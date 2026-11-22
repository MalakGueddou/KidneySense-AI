import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    print("Chargement des donnees brutes...")
    df = pd.read_csv(input_path)
    
    # 1. Remplacer les '?' et les tabulations par de vrais NaN (valeurs manquantes)
    df = df.replace('?', np.nan)
    df = df.replace('\t?', np.nan)
    df = df.replace(' ?', np.nan)
    
    # 2. Nettoyer les colonnes textuelles (enlever les espaces et mettre en minuscules)
    # On identifie les colonnes de type "object" (texte)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
        
    # 3. Corriger la colonne cible (class)
    # On s'assure qu'il n'y a que 'ckd' et 'notckd'
    df['class'] = df['class'].replace({'ckd\t': 'ckd', 'notckd\t': 'notckd'})
    # On garde uniquement les lignes valides
    df = df[df['class'].isin(['ckd', 'notckd'])]
    
    # 4. Convertir les colonnes numériques qui sont restées en texte
    # (Parfois à cause des '?', Pandas lit une colonne de chiffres comme du texte)
    for col in df.columns:
        if col != 'class':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass # Si ça ne marche pas, on laisse en texte (c'est une vraie catégorie)
                
    # 5. Sauvegarder le fichier propre
    df.to_csv(output_path, index=False)
    print(f"Nettoyage termine ! Fichier sauvegarde sous : {output_path}")
    print(f"Dimensions finales : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    
    # Afficher un aperçu des valeurs manquantes restantes
    print("\nValeurs manquantes restantes par colonne :")
    print(df.isnull().sum())

if __name__ == "__main__":
    clean_data('data/kidney_disease.csv', 'data/clean_kidney_disease.csv')

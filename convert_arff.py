import pandas as pd
from scipy.io import arff
import os

# On cherche le fichier .arff dans le dossier data
file_path = 'data/chronic_kidney_disease.arff'

if not os.path.exists(file_path):
    print("Erreur : Le fichier .arff est introuvable dans data/")
    print("Vérifiez qu'il a bien été extrait du fichier .rar")
else:
    # Charger et convertir
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Décoder les colonnes catégorielles (bytes -> str)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
        
    # Sauvegarder en CSV
    df.to_csv('data/kidney_disease.csv', index=False)
    print("✅ Conversion réussie ! Le fichier data/kidney_disease.csv est prêt.")
    print(f"Dimensions du dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

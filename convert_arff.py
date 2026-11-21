import pandas as pd
import csv

def convert_arff_to_csv(arff_path, csv_path):
    print("Lecture du fichier ARFF...")
    
    # 1. Trouver où commencent les données (après le mot-clé @data)
    with open(arff_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start_idx = i + 1
            break
            
    # 2. Récupérer les noms des colonnes (après le mot-clé @attribute)
    columns = []
    for line in lines[:data_start_idx]:
        if line.strip().lower().startswith('@attribute'):
            # On extrait le nom de l'attribut
            col_name = line.strip().split()[1].strip("'\"")
            columns.append(col_name)
            
    # 3. Lire les données ligne par ligne en ignorant les lignes vides
    data = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if line: # Si la ligne n'est pas vide
            # On sépare par des virgules et on enlève les espaces autour des valeurs
            row = [val.strip().strip("'\"") for val in line.split(',')]
            data.append(row)
            
    # 4. Créer le DataFrame Pandas et sauvegarder en CSV
    df = pd.DataFrame(data, columns=columns)
    
    # Remplacer les '?' (valeurs manquantes dans ce dataset) par de vrais NaN pour Pandas
    df = df.replace('?', pd.NA)
    
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("✅ Conversion réussie ! Le fichier CSV est prêt.")
    print(f"Dimensions du dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print("\nAperçu des 3 premières lignes :")
    print(df.head(3))

if __name__ == "__main__":
    # On indique où se trouve le fichier source et où on veut le sauvegarder
    # (Adaptez le chemin si votre .arff n'est pas dans data/)
    source = 'data/chronic_kidney_disease.arff' 
    destination = 'data/kidney_disease.csv'
    convert_arff_to_csv(source, destination)

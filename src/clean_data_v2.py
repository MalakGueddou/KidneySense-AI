import json
from pathlib import Path
import numpy as np
import pandas as pd

# Bornes physiologiques pour détecter les erreurs de saisie manifestes
UCI_PHYSIO_BOUNDS = {
    "age": (1.0, 105.0),
    "bp": (40.0, 140.0),       
    "sg": (1.005, 1.025),
    "al": (0.0, 5.0),
    "su": (0.0, 5.0),
    "bgr": (40.0, 600.0),      
    "bu": (5.0, 350.0),
    "sc": (0.2, 20.0),         
    "sod": (110.0, 160.0),     
    "pot": (2.0, 8.5),         
    "hemo": (3.0, 20.0),       
    "pcv": (10.0, 60.0),       
    "wc": (1500.0, 30000.0),   
    "rc": (1.5, 8.5),          
}

UCI_NUM_COLS = list(UCI_PHYSIO_BOUNDS.keys())

UCI_BINARY_MAPS = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
    "class": {"notckd": 0, "ckd": 1},
}

def clean_uci_dataset(input_path, output_path):
    print(f"Chargement de {input_path}...")
    df = pd.read_csv(
        input_path,
        na_values=["?", "", " ", "\t", "\t?", "NA", "nan"],
        skipinitialspace=True,
    )
    df.columns = df.columns.str.strip()

    audit = {
        "rows_raw": int(len(df)),
        "cols": int(df.shape[1]),
        "outliers_replaced_with_nan": {},
        "missing_before": {},
    }

    # 1. Nettoyage des chaînes catégorielles
    for col in UCI_BINARY_MAPS.keys():
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("\t", "", regex=False)
                .str.strip().str.lower()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan, "?": np.nan})
            )

    # 2. Conversion numérique stricte
    for col in UCI_NUM_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("\t", "", regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    audit["missing_before"] = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}

    # 3. Détection & neutralisation des outliers (-> NaN)
    for col, (low, high) in UCI_PHYSIO_BOUNDS.items():
        if col in df.columns:
            mask = (df[col] < low) | (df[col] > high)
            if mask.any():
                outliers_info = [{"row_index": int(idx), "value": float(val)} for idx, val in df.loc[mask, col].items()]
                audit["outliers_replaced_with_nan"][col] = outliers_info
                df.loc[mask, col] = np.nan

    # 4. Encodage binaire propre (0 / 1) et nettoyage de la cible
    for col, mapping in UCI_BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].replace({'ckd\t': 'ckd', 'notckd\t': 'notckd'}) # Fix tab
            unknown_vals = set(df[col].dropna().unique()) - set(mapping.keys())
            if unknown_vals:
                print(f"Attention, valeurs inattendues dans {col}: {unknown_vals}")
            df[col] = df[col].map(mapping)

    # Supprimer les lignes où la cible n'a pas pu être encodée
    df = df.dropna(subset=['class'])
    df['class'] = df['class'].astype(int)

    # 5. Sauvegarder
    df = df.dropna(subset=['class', 'appet', 'pe']) # Supprime la ligne corrompue
    df.to_csv(output_path, index=False)
    
    # Rapport d'audit
    print(f"\n--- RAPPORT D'AUDIT ---")
    print(f"Lignes traitées : {len(df)}")
    print(f"Valeurs aberrantes supprimées (remplacées par NaN) :")
    for col, data in audit["outliers_replaced_with_nan"].items():
        print(f"  - {col} : {len(data)} valeurs aberrantes (ex: {data[0]['value']})")
    
    with open('data_quality_report.json', 'w') as f:
        json.dump(audit, f, indent=4)
    print("Rapport sauvegardé dans data_quality_report.json")

if __name__ == "__main__":
    # On cherche le fichier brut
    clean_uci_dataset('data/kidney_disease.csv', 'data/clean_kidney_disease.csv')

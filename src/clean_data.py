import pandas as pd
import numpy as np


def clean_data(input_path, output_path):
    print("Chargement des donnees brutes...")
    df = pd.read_csv(input_path)
    print(f"Lignes dans le fichier brut : {len(df)}")

    # 0. Supprimer la colonne id si elle existe
    if "id" in df.columns:
        df = df.drop(columns="id")

    # 1. Nettoyer les colonnes texte : minuscules, espaces et tabulations enlevés
    colonnes_texte = df.select_dtypes(exclude="number").columns
    for col in colonnes_texte:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # 2. Transformer "?" et "nan" (texte) en vraies valeurs manquantes
    df = df.replace({"?": np.nan, "nan": np.nan, "": np.nan})

    # 3. Garder seulement les lignes dont la cible est ckd ou notckd
    valides = df["class"].isin(["ckd", "notckd"])
    nb_invalides = int((~valides).sum())
    if nb_invalides > 0:
        print(f"Lignes exclues (class invalide) : {nb_invalides}")
    df = df[valides]

    # 4. Convertir en nombres les colonnes qui doivent l'être
    colonnes_numeriques = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
                           "sod", "pot", "hemo", "pcv", "wc", "rc"]
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Supprimer les doublons exacts
    avant = len(df)
    df = df.drop_duplicates()
    print(f"Doublons supprimes : {avant - len(df)}")

    # 6. Sauvegarder (les NaN sont gardés : l'imputation se fera plus tard
    #    dans le Pipeline, après la séparation train/test)
    df.to_csv(output_path, index=False)

    # 7. Rapport
    print(f"\nNettoyage termine ! Fichier sauvegarde : {output_path}")
    print(f"Dimensions finales : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum())
    print("\nDistribution de la cible :")
    print(df["class"].value_counts())


if __name__ == "__main__":
    clean_data("data/kidney_disease.csv", "data/clean_kidney_disease.csv")
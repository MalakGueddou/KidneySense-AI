import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_preprocessor(X):
    """
    Crée un pipeline de préparation des données.
    X est le DataFrame contenant les caractéristiques (features) sans la cible.
    """
    # 1. Séparer les colonnes numériques et textuelles
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # 2. Créer les étapes pour les chiffres
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Remplir les vides par la médiane
        ('scaler', StandardScaler())                   # Mettre à la même échelle
    ])
    
    # 3. Créer les étapes pour le texte
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Remplir les vides par le plus fréquent
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Transformer en 0/1
    ])
    
    # 4. Assembler le tout
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

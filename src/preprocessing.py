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
    # On garde l'imputation (pour les trous restants) et le Scaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())                   
    ])
    
    # 3. Créer les étapes pour le texte
    # On garde l'imputation et le OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    
    ])
    
    # 4. Assembler le tout
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
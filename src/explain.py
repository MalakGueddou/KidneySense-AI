import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

print("Chargement du modele...")
model = joblib.load('models/random_forest_model.pkl')

print("Chargement des donnees...")
df = pd.read_csv('data/clean_kidney_disease.csv')
X = df.drop('class', axis=1)

# 1. On transforme tout en nombres (les textes deviennent des codes numériques)
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# 2. On remplit les valeurs manquantes par 0 (uniquement pour SHAP)
X = X.fillna(0)

# 3. On prend un petit échantillon
X_sample = X.sample(50, random_state=42)

print("Calcul des valeurs SHAP...")
# On utilise l'explainer standard (plus robuste)
explainer = shap.Explainer(model.predict_proba, X_sample)
shap_values = explainer(X_sample)

print("Generation du graphique...")
plt.figure(figsize=(10, 6))

# Si shap_values a 3 dimensions (lignes, colonnes, classes), on prend la classe 1
if len(shap_values.shape) == 3:
    shap.summary_plot(shap_values[:, :, 1].values, X_sample, show=False)
else:
    shap.summary_plot(shap_values.values, X_sample, show=False)

plt.title("Impact des variables sur la prediction de la maladie renale")
plt.tight_layout()
plt.savefig('notebooks/shap_summary.png')
print("Graphique SHAP sauvegarde dans notebooks/shap_summary.png")

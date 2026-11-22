import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# On importe notre fonction de préparation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import get_preprocessor

# 1. Charger les données (déjà nettoyées)
print("Chargement des données...")
df = pd.read_csv('data/clean_kidney_disease.csv')

# CORRECTION ICI : Nettoyage agressif de la cible
df['class'] = df['class'].astype(str).str.lower().str.strip()
# On garde uniquement les lignes qui contiennent 'ckd' ou 'notckd'
df = df[df['class'].isin(['ckd', 'notckd'])]

# 2. Séparer X (features) et y (cible)
X = df.drop('class', axis=1)
y = df['class']

# 3. Séparer en train (80%) et test (20%)
# On utilise stratify=y pour garder la même proportion de malades/sains dans le train et le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Créer le préprocesseur (appris sur le train uniquement !)
preprocessor = get_preprocessor(X_train)

# 5. Définir les modèles à comparer
models = {
    "Régression Logistique": LogisticRegression(max_iter=1000),
    "Arbre de Décision": DecisionTreeClassifier(random_state=42),
    "Forêt Aléatoire": RandomForestClassifier(random_state=42)
}

# 6. Entraîner et évaluer chaque modèle
print("\n--- Comparaison des modèles ---")
for name, model in models.items():
    # On crée un pipeline qui fait : préparation -> modèle
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Entraînement
    clf.fit(X_train, y_train)
    
    # Prédiction sur le test
    y_pred = clf.predict(X_test)
    
    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='ckd')
    rec = recall_score(y_test, y_pred, pos_label='ckd')
    f1 = f1_score(y_test, y_pred, pos_label='ckd')
    
    print(f"\nModèle : {name}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall (Sensibilité) : {rec:.3f}")
    print(f"  F1-Score : {f1:.3f}")

# --- AJOUT : Génération de la matrice de confusion pour le meilleur modèle ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib # Pour sauvegarder le modèle

# On choisit la Forêt Aléatoire (Random Forest)
best_model_name = "Forêt Aléatoire"
best_model = models[best_model_name]
best_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', best_model)])

# On le ré-entraîne sur tout le train
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

# 1. Matrice de confusion
cm = confusion_matrix(y_test, y_pred_best, labels=['ckd', 'notckd'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CKD (Malade)', 'Non-CKD (Sain)'])
disp.plot(cmap='Blues')
plt.title(f"Matrice de confusion - {best_model_name}")
plt.savefig('notebooks/matrice_confusion.png')
print("Matrice de confusion sauvegardee dans notebooks/")

# 2. Sauvegarder le modèle pour Streamlit
import os
os.makedirs('models', exist_ok=True)
joblib.dump(best_clf, 'models/random_forest_model.pkl')
print("Modele sauvegarde dans models/random_forest_model.pkl")
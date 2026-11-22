import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, make_scorer,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import get_preprocessor

# 1. Charger les données nettoyées
print("Chargement des données...")
df = pd.read_csv('data/clean_kidney_disease.csv')
df['class'] = df['class'].astype(str).str.lower().str.strip()
df = df[df['class'].isin(['ckd', 'notckd'])]

# 2. Séparer X et y
X = df.drop('class', axis=1)
y = df['class']

# 3. Séparer train (80%) / test (20%), avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")

# 4. Préprocesseur (sera appris dans le Pipeline, sur le train seulement)
preprocessor = get_preprocessor(X_train)

# 5. Modèles à comparer
models = {
    "Régression Logistique": LogisticRegression(max_iter=1000),
    "Arbre de Décision": DecisionTreeClassifier(random_state=42),
    "Forêt Aléatoire": RandomForestClassifier(random_state=42),
}

def make_pipeline(model):
    return Pipeline(steps=[('preprocessor', clone(preprocessor)),
                           ('classifier', clone(model))])

# 6. Validation croisée (sur le TRAIN uniquement) pour choisir le modèle
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label='ckd'),
    'recall': make_scorer(recall_score, pos_label='ckd'),
    'f1': make_scorer(f1_score, pos_label='ckd'),
    'roc_auc': 'roc_auc',
}

print("\n--- Validation croisée 5 plis (train seulement) ---")
cv_f1 = {}
for name, model in models.items():
    res = cross_validate(make_pipeline(model), X_train, y_train,
                         cv=cv, scoring=scoring)
    print(f"\n{name}")
    for m in scoring:
        vals = res[f'test_{m}']
        print(f"  {m:10s}: {vals.mean():.3f} ± {vals.std():.3f}")
    cv_f1[name] = res['test_f1'].mean()

# 7. Le meilleur modèle est celui qui a le meilleur F1 en validation croisée
best_model_name = max(cv_f1, key=cv_f1.get)
print(f"\nMeilleur modèle (F1 en validation croisée) : {best_model_name}")

# 8. Évaluation finale sur le test (une seule fois, pour chaque modèle)
print("\n--- Évaluation sur le jeu de test ---")
fitted = {}
for name, model in models.items():
    clf = make_pipeline(model)
    clf.fit(X_train, y_train)
    fitted[name] = clf

    y_pred = clf.predict(X_test)
    proba_ckd = clf.predict_proba(X_test)[:, list(clf.classes_).index('ckd')]
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=['notckd', 'ckd']).ravel()

    print(f"\nModèle : {name}")
    print(f"  Accuracy    : {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision   : {precision_score(y_test, y_pred, pos_label='ckd'):.3f}")
    print(f"  Recall      : {recall_score(y_test, y_pred, pos_label='ckd'):.3f}")
    print(f"  Spécificité : {tn / (tn + fp):.3f}")
    print(f"  F1-Score    : {f1_score(y_test, y_pred, pos_label='ckd'):.3f}")
    print(f"  ROC-AUC     : {roc_auc_score(y_test == 'ckd', proba_ckd):.3f}")
    print(f"  Faux négatifs (CKD non détectés) : {fn}")
    print(f"  Faux positifs                    : {fp}")

# 9. Matrice de confusion du meilleur modèle
best_clf = fitted[best_model_name]
y_pred_best = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=['ckd', 'notckd'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['CKD (Malade)', 'Non-CKD (Sain)'])
disp.plot(cmap='Blues')
plt.title(f"Matrice de confusion - {best_model_name}")
plt.savefig('notebooks/matrice_confusion.png')
print("\nMatrice de confusion sauvegardee dans notebooks/")

# 10. Sauvegarder les modèles pour Streamlit
os.makedirs('models', exist_ok=True)
joblib.dump(best_clf, 'models/best_model.pkl')
joblib.dump(fitted["Forêt Aléatoire"], 'models/random_forest_model.pkl')
print("Modeles sauvegardes dans models/ (best_model.pkl et random_forest_model.pkl)")
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Configuration de la page
st.set_page_config(page_title="KidneySense AI", page_icon="🩺", layout="wide")

# Titre et avertissement
st.title("🩺 KidneySense AI")
st.subheader("Prototype académique de prédiction de la maladie rénale chronique")
st.warning("⚠️ Ceci est un projet pédagogique. Ne remplace pas un avis médical.")

# Créer des onglets
tab1, tab2, tab3 = st.tabs(["📊 Le Dataset", "🤖 Les Modèles", "🔍 Explicabilité (SHAP)"])

with tab1:
    st.header("Analyse Exploratoire des Données")
    st.write("Voici un aperçu du jeu de données nettoyé utilisé pour entraîner nos modèles.")
    
    # Charger et afficher le dataset
    df = pd.read_csv('data/clean_kidney_disease.csv')
    st.dataframe(df.head(10))
    
    st.write(f"Le dataset contient **{df.shape[0]} lignes** et **{df.shape[1]} colonnes**.")
    
    # Graphique de distribution
    st.subheader("Distribution des classes")
    fig, ax = plt.subplots()
    df['class'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre de patients")
    st.pyplot(fig)

with tab2:
    st.header("Performance des Modèles")
    st.write("Nous avons comparé 3 algorithmes de Machine Learning.")
    
    # Afficher le tableau des scores
    scores = {
        "Modèle": ["Régression Logistique", "Arbre de Décision", "Forêt Aléatoire"],
        "Accuracy": [0.988, 0.975, 0.988],
        "Recall": [0.980, 1.000, 1.000],
        "F1-Score": [0.990, 0.980, 0.990]
    }
    st.table(pd.DataFrame(scores))
    
    st.subheader("Matrice de Confusion (Forêt Aléatoire)")
    st.image('notebooks/matrice_confusion.png', use_column_width=True)

with tab3:
    st.header("Explicabilité avec SHAP")
    st.write("Pourquoi le modèle prend-il ses décisions ? SHAP nous montre quelles variables médicales influencent le plus les prédictions.")
    
    st.image('notebooks/shap_summary.png', use_column_width=True)
    
    st.info("Les variables en haut du graphique sont les plus importantes pour le modèle. Par exemple, un faible taux d'hémoglobine pousse fortement vers la prédiction 'Malade'.")

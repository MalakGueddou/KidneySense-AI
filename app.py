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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Le Dataset", "🤖 Les Modèles", "🔍 Explicabilité (SHAP)", "🩺 Prédiction"])

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
    import json
    with open('artifacts/model1_metadata.json', 'r') as f:
        meta = json.load(f)
    
    # Afficher le meilleur modèle
    st.success(f"Modèle sélectionné : {meta['best_model']}")
    
    # Créer un DataFrame à partir des métriques
    df_metrics = pd.DataFrame(meta['all_candidates_metrics'])
    
    # Garder uniquement les colonnes intéressantes
    cols_to_show = ['model', 'accuracy', 'roc_auc', 'sensitivity_recall', 'specificity', 'f1_score', 'brier_score']
    st.dataframe(df_metrics[cols_to_show].set_index('model'))
    
    # Afficher la matrice de confusion du meilleur modèle (si vous l'avez sauvegardée)
    # Sinon, vous pouvez garder l'ancienne image
    st.image('notebooks/matrice_confusion.png', use_column_width=True)

with tab3:
    st.header("Explicabilité avec SHAP")
    st.write("Pourquoi le modèle prend-il ses décisions ? SHAP nous montre quelles variables médicales influencent le plus les prédictions.")
    
    st.image('notebooks/shap_summary.png', use_column_width=True)
    
    st.info("Les variables en haut du graphique sont les plus importantes pour le modèle. Par exemple, un faible taux d'hémoglobine pousse fortement vers la prédiction 'Malade'.")
with tab4:
    st.header("Simulation de Prediction")
    st.write("Entrez les analyses d'un patient pour que le modele estime s'il est a risque.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (annees)", min_value=1, max_value=100, value=50)
            bp = st.number_input("Pression arterielle", min_value=50, max_value=200, value=80)
            hemo = st.number_input("Hemoglobine (g/dL)", min_value=3.0, max_value=20.0, value=15.0, step=0.1)
            
        with col2:
            sc = st.number_input("Creatinine serique", min_value=0.1, max_value=20.0, value=1.2, step=0.1)
            bgr = st.number_input("Glycemie", min_value=50, max_value=500, value=120)
            sod = st.number_input("Sodium", min_value=100, max_value=160, value=135)
            
        with col3:
            pot = st.number_input("Potassium", min_value=2.0, max_value=10.0, value=4.5, step=0.1)
            htn = st.selectbox("Hypertension ?", ["yes", "no"])
            dm = st.selectbox("Diabete ?", ["yes", "no"])
            
        submit = st.form_submit_button("Lancer l'analyse")
        
    if submit:
        try:
            # 1. Charger le modele
            model = joblib.load('artifacts/model1_uci_pipeline.joblib')
            
            # 2. Donnees par defaut pour les colonnes non demandees
            # ON ENVOIE DES 0 ET DES 1 POUR LES VARIABLES BINAIRES (Pas de texte)
            input_data = {
                'age': [age], 'bp': [bp], 'sg': [1.020], 'al': [0], 'su': [0],
                'rbc': [0], 'pc': [0], 'pcc': [0], 'ba': [0], 
                'bgr': [bgr], 'bu': [40], 'sc': [sc], 'sod': [sod], 'pot': [pot],
                'hemo': [hemo], 'pcv': [40], 'wc': [8000], 'rc': [5.0],
                'htn': [1 if htn == "yes" else 0], 
                'dm': [1 if dm == "yes" else 0], 
                'cad': [0], 'appet': [0],
                'pe': [0], 'ane': [0]
            }
            input_df = pd.DataFrame(input_data)
            
            # 3. Predire
            prediction = model.predict(input_df)[0]
            probabilite = model.predict_proba(input_df)[0]
            
            st.divider()
            if prediction == 1:
                st.error(f"Resultat : Risque de maladie renale chronique (CKD). Probabilite : {probabilite[1]*100:.1f}%")
            else:
                st.success(f"Resultat : Patient sain (Non-CKD). Probabilite : {probabilite[0]*100:.1f}%")
            
                         # 4. Explication simplifiee (Coefficients du modele)
            st.divider()
            st.subheader("Pourquoi cette decision ? (Facteurs influents)")
            
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            
            try:
                # On recupere les noms des features apres transformation
                feature_names = preprocessor.get_feature_names_out()
                
                # On verifie si le modele a des coefficients (Regression Logistique)
                if hasattr(classifier, "coef_"):
                    # On recupere les coefficients
                    coefs = classifier.coef_[0]
                    
                    # On cree un DataFrame
                    df_coefs = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefs
                    })
                    
                    # On trie par valeur absolue pour avoir les plus influents
                    df_coefs['Abs'] = df_coefs['Coefficient'].abs()
                    df_coefs = df_coefs.sort_values(by='Abs', ascending=True).tail(7)
                    
                    # On cree le graphique
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#d62728' if x > 0 else '#1f77b4' for x in df_coefs['Coefficient']]
                    ax.barh(df_coefs['Feature'], df_coefs['Coefficient'], color=colors)
                    ax.set_xlabel("Poids du coefficient (Rouge = Pousse vers Malade, Bleu = Pousse vers Sain)")
                    ax.set_title("Facteurs les plus influents (Coefficients du modele)")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.caption("Ces coefficients sont ceux du modele de Regression Logistique. Ils expliquent le comportement global du modele, pas seulement pour ce patient.")
                
                # Si c'est un Random Forest, on utilise les importances
                elif hasattr(classifier, "feature_importances_"):
                    importances = classifier.feature_importances_
                    df_imp = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=True).tail(7)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(df_imp['Feature'], df_imp['Importance'], color='#2ca02c')
                    ax.set_xlabel("Importance de la variable")
                    ax.set_title("Variables les plus importantes pour le modele")
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Impossible de generer le graphique d'explication : {e}")
                st.info("Le modele a quand meme fait sa prediction. Vous pouvez vous fier au resultat ci-dessus.")

            st.info("Rappel : Ceci est une simulation pedagogique.")         
        except Exception as e:
            st.error(f"Erreur : {e}")
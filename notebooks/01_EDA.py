import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger les données
df = pd.read_csv('../data/kidney_disease.csv')

# 2. Nettoyage rapide : remplacer les '?' et espaces par des NaN
df = df.replace('?', pd.NA)
df = df.replace('\t?', pd.NA)

# 3. Créer une figure avec 2 graphiques côte à côte
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Distribution de la variable cible (class)
sns.countplot(data=df, x='class', ax=axes[0], palette='Set2')
axes[0].set_title("Distribution des classes (CKD vs Non-CKD)")
axes[0].set_xlabel("Classe")
axes[0].set_ylabel("Nombre de patients")

# Graphique 2 : Carte des valeurs manquantes
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=axes[1])
axes[1].set_title("Carte des valeurs manquantes")
axes[1].set_ylabel("Index des patients")

# Ajuster l'affichage et sauvegarder l'image
plt.tight_layout()
plt.savefig('../notebooks/graphique_eda.png') # Sauvegarde l'image dans le dossier notebooks
print("Graphiques generes et sauvegardes sous le nom 'graphique_eda.png'")
plt.show()

# 4. Afficher quelques statistiques
print("\n--- Statistiques descriptives ---")
print(df.describe())

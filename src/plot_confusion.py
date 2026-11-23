import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Charger le modèle
model = joblib.load('artifacts/model1_uci_pipeline.joblib')
df = pd.read_csv('data/clean_kidney_disease.csv')
X = df.drop('class', axis=1)
y = df['class'].astype(int)

# Prédiction
y_pred = model.predict(X)

# Matrice
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-CKD', 'CKD'], 
            yticklabels=['Non-CKD', 'CKD'])
plt.title("Matrice de Confusion (Nouveau Modèle)")
plt.ylabel('Vrai')
plt.xlabel('Prédit')
plt.savefig('notebooks/matrice_confusion.png')
print("Matrice sauvegardée.")